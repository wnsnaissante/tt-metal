// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

namespace {

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kTileWidth = 32;
constexpr uint32_t kOneTile = 1;
constexpr uint32_t kInputTapCb0 = tt::CBIndex::c_0;
constexpr uint32_t kWeightTapCb0 = tt::CBIndex::c_4;
constexpr uint32_t kBiasCb = tt::CBIndex::c_8;
constexpr uint32_t kRawCurrentCb = tt::CBIndex::c_11;
constexpr uint32_t kRawNextCb = tt::CBIndex::c_12;
constexpr uint32_t kRawWeightCb = tt::CBIndex::c_14;
constexpr uint32_t kRawBiasCb = tt::CBIndex::c_15;

struct OutputTileCoord {
    uint32_t batch_idx;
    uint32_t sequence_tile_idx;
    uint32_t feature_tile_idx;
};

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    const uint32_t face_h = h & 0xF;
    const uint32_t face_w = w & 0xF;
    const uint32_t face = (h >> 4) * 2 + (w >> 4);
    return face * 256 + face_h * 16 + face_w;
}

FORCE_INLINE OutputTileCoord decode_output_tile_id(uint32_t tile_id, uint32_t sequence_tiles, uint32_t feature_tiles) {
    const uint32_t tiles_per_batch = sequence_tiles * feature_tiles;
    const uint32_t batch_tile_offset = tile_id % tiles_per_batch;
    return {
        .batch_idx = tile_id / tiles_per_batch,
        .sequence_tile_idx = batch_tile_offset / feature_tiles,
        .feature_tile_idx = batch_tile_offset % feature_tiles};
}

FORCE_INLINE uint32_t x_page_id(
    uint32_t batch_idx,
    uint32_t sequence_tile_idx,
    uint32_t feature_tile_idx,
    uint32_t batch_stride,
    uint32_t sequence_stride) {
    return batch_idx * batch_stride + sequence_tile_idx * sequence_stride + feature_tile_idx;
}

FORCE_INLINE uint32_t weight_page_id(uint32_t feature_tile_idx, uint32_t feature_stride) {
    return feature_tile_idx * feature_stride;
}

FORCE_INLINE uint32_t bias_page_id(uint32_t feature_tile_idx, uint32_t feature_stride) {
    return feature_tile_idx * feature_stride;
}

FORCE_INLINE void fill_bias_tile(
    volatile tt_l1_ptr uint16_t* bias_tile, volatile tt_l1_ptr uint16_t* raw_bias, uint32_t valid_rows) {
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            const uint16_t value = row < valid_rows ? raw_bias[get_tilized_idx(0, col)] : 0;
            bias_tile[get_tilized_idx(row, col)] = value;
        }
    }
}

FORCE_INLINE void fill_weight_tile(
    volatile tt_l1_ptr uint16_t* weight_tile, volatile tt_l1_ptr uint16_t* raw_weight, uint32_t tap) {
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            weight_tile[get_tilized_idx(row, col)] = raw_weight[get_tilized_idx(col, tap)];
        }
    }
}

FORCE_INLINE void fill_input_tile(
    volatile tt_l1_ptr uint16_t* input_tile,
    volatile tt_l1_ptr uint16_t* raw_current,
    volatile tt_l1_ptr uint16_t* raw_next,
    uint32_t tap,
    uint32_t valid_rows,
    bool has_next_tile) {
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            uint16_t value = 0;
            if (row < valid_rows) {
                const uint32_t source_row = row + tap;
                if (source_row < kTileHeight) {
                    value = raw_current[get_tilized_idx(source_row, col)];
                } else if (has_next_tile) {
                    value = raw_next[get_tilized_idx(source_row - kTileHeight, col)];
                }
            }
            input_tile[get_tilized_idx(row, col)] = value;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t kernel_size = get_compile_time_arg_val(0);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);
    const uint32_t sequence_length = get_arg_val<uint32_t>(5);
    const uint32_t sequence_tiles = get_arg_val<uint32_t>(6);
    const uint32_t input_sequence_tiles = get_arg_val<uint32_t>(7);
    const uint32_t feature_tiles = get_arg_val<uint32_t>(8);
    const uint32_t x_batch_stride = get_arg_val<uint32_t>(9);
    const uint32_t x_sequence_stride = get_arg_val<uint32_t>(10);
    const uint32_t weight_feature_stride = get_arg_val<uint32_t>(11);
    const uint32_t bias_feature_stride = get_arg_val<uint32_t>(12);

    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto weight_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_local_cb_interface(kInputTapCb0).fifo_page_size;
    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, tile_bytes);

    experimental::Noc noc;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto coord = decode_output_tile_id(start_tile + tile_idx, sequence_tiles, feature_tiles);
        const bool has_next_tile = coord.sequence_tile_idx + 1 < input_sequence_tiles;

        const uint32_t tile_sequence_start = coord.sequence_tile_idx * kTileHeight;
        const uint32_t remaining_rows =
            sequence_length > tile_sequence_start ? (sequence_length - tile_sequence_start) : 0;
        const uint32_t valid_rows = remaining_rows > kTileHeight ? kTileHeight : remaining_rows;

        experimental::CircularBuffer raw_weight_cb(kRawWeightCb);
        experimental::CircularBuffer raw_bias_cb(kRawBiasCb);
        experimental::CircularBuffer raw_current_cb(kRawCurrentCb);
        experimental::CircularBuffer raw_next_cb(kRawNextCb);

        raw_weight_cb.reserve_back(kOneTile);
        noc.async_read(
            weight_accessor,
            raw_weight_cb,
            tile_bytes,
            {.page_id = weight_page_id(coord.feature_tile_idx, weight_feature_stride)},
            {.offset_bytes = 0});

        raw_bias_cb.reserve_back(kOneTile);
        noc.async_read(
            bias_accessor,
            raw_bias_cb,
            tile_bytes,
            {.page_id = bias_page_id(coord.feature_tile_idx, bias_feature_stride)},
            {.offset_bytes = 0});

        raw_current_cb.reserve_back(kOneTile);
        noc.async_read(
            x_accessor,
            raw_current_cb,
            tile_bytes,
            {.page_id = x_page_id(
                 coord.batch_idx, coord.sequence_tile_idx, coord.feature_tile_idx, x_batch_stride, x_sequence_stride)},
            {.offset_bytes = 0});

        if (has_next_tile) {
            raw_next_cb.reserve_back(kOneTile);
            noc.async_read(
                x_accessor,
                raw_next_cb,
                tile_bytes,
                {.page_id = x_page_id(
                     coord.batch_idx,
                     coord.sequence_tile_idx + 1,
                     coord.feature_tile_idx,
                     x_batch_stride,
                     x_sequence_stride)},
                {.offset_bytes = 0});
        }

        noc.async_read_barrier();

        raw_weight_cb.push_back(kOneTile);
        raw_bias_cb.push_back(kOneTile);
        raw_current_cb.push_back(kOneTile);
        if (has_next_tile) {
            raw_next_cb.push_back(kOneTile);
        }
        raw_weight_cb.wait_front(kOneTile);
        raw_bias_cb.wait_front(kOneTile);
        raw_current_cb.wait_front(kOneTile);
        if (has_next_tile) {
            raw_next_cb.wait_front(kOneTile);
        }

        auto* raw_weight = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawWeightCb));
        auto* raw_bias = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawBiasCb));
        auto* raw_current = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawCurrentCb));
        auto* raw_next =
            has_next_tile ? reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawNextCb)) : nullptr;

        experimental::CircularBuffer bias_cb(kBiasCb);
        bias_cb.reserve_back(kOneTile);
        fill_bias_tile(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(kBiasCb)), raw_bias, valid_rows);
        bias_cb.push_back(kOneTile);

        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            const uint32_t input_cb_id = kInputTapCb0 + tap;
            const uint32_t weight_cb_id = kWeightTapCb0 + tap;

            experimental::CircularBuffer input_cb(input_cb_id);
            input_cb.reserve_back(kOneTile);
            fill_input_tile(
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(input_cb_id)),
                raw_current,
                raw_next,
                tap,
                valid_rows,
                has_next_tile);
            input_cb.push_back(kOneTile);

            experimental::CircularBuffer weight_cb(weight_cb_id);
            weight_cb.reserve_back(kOneTile);
            fill_weight_tile(
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(weight_cb_id)), raw_weight, tap);
            weight_cb.push_back(kOneTile);
        }

        raw_weight_cb.pop_front(kOneTile);
        raw_bias_cb.pop_front(kOneTile);
        raw_current_cb.pop_front(kOneTile);
        if (has_next_tile) {
            raw_next_cb.pop_front(kOneTile);
        }
    }
}
