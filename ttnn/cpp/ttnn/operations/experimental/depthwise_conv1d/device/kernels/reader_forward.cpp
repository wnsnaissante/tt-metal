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
constexpr uint32_t kRawPrefixCb = tt::CBIndex::c_12;

struct TileCoord {
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

FORCE_INLINE TileCoord decode_tile_id(uint32_t tile_id, uint32_t sequence_tiles, uint32_t feature_tiles) {
    const uint32_t tiles_per_batch = sequence_tiles * feature_tiles;
    const uint32_t tile_in_batch = tile_id % tiles_per_batch;
    return {
        .batch_idx = tile_id / tiles_per_batch,
        .sequence_tile_idx = tile_in_batch / feature_tiles,
        .feature_tile_idx = tile_in_batch % feature_tiles,
    };
}

FORCE_INLINE uint32_t x_page_id(
    uint32_t batch_idx,
    uint32_t sequence_tile_idx,
    uint32_t feature_tile_idx,
    uint32_t batch_stride,
    uint32_t sequence_stride) {
    return batch_idx * batch_stride + sequence_tile_idx * sequence_stride + feature_tile_idx;
}

FORCE_INLINE uint32_t
conv_state_page_id(uint32_t batch_idx, uint32_t feature_tile_idx, uint32_t batch_stride, uint32_t feature_stride) {
    return batch_idx * batch_stride + feature_tile_idx * feature_stride;
}

FORCE_INLINE uint32_t weight_page_id(uint32_t tap, uint32_t feature_tile_idx, uint32_t feature_tiles) {
    return tap * feature_tiles + feature_tile_idx;
}

FORCE_INLINE uint32_t bias_page_id(uint32_t feature_tile_idx) { return feature_tile_idx; }

FORCE_INLINE void fill_input_tile(
    volatile tt_l1_ptr uint16_t* input_tile,
    volatile tt_l1_ptr uint16_t* raw_prefix,
    volatile tt_l1_ptr uint16_t* raw_current,
    uint32_t tap,
    uint32_t cache_len,
    uint32_t valid_rows,
    bool prefix_is_conv_state) {
    for (uint32_t row = 0; row < kTileHeight; ++row) {
        for (uint32_t col = 0; col < kTileWidth; ++col) {
            uint16_t value = 0;
            if (row < valid_rows) {
                const int32_t source_row =
                    static_cast<int32_t>(row) + static_cast<int32_t>(tap) - static_cast<int32_t>(cache_len);
                if (source_row >= 0) {
                    value = raw_current[get_tilized_idx(static_cast<uint32_t>(source_row), col)];
                } else if (prefix_is_conv_state) {
                    const uint32_t cache_col = cache_len + source_row;
                    value = raw_prefix[get_tilized_idx(col, cache_col)];
                } else {
                    const uint32_t prev_row = static_cast<uint32_t>(static_cast<int32_t>(kTileHeight) + source_row);
                    value = raw_prefix[get_tilized_idx(prev_row, col)];
                }
            }
            input_tile[get_tilized_idx(row, col)] = value;
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t kernel_size = get_compile_time_arg_val(0);
    constexpr uint32_t cache_len = kernel_size - 1;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t conv_state_addr = get_arg_val<uint32_t>(1);
    const uint32_t weight_addr = get_arg_val<uint32_t>(2);
    const uint32_t bias_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t start_tile = get_arg_val<uint32_t>(5);
    const uint32_t sequence_length = get_arg_val<uint32_t>(6);
    const uint32_t sequence_tiles = get_arg_val<uint32_t>(7);
    const uint32_t feature_tiles = get_arg_val<uint32_t>(8);
    const uint32_t x_batch_stride = get_arg_val<uint32_t>(9);
    const uint32_t x_sequence_stride = get_arg_val<uint32_t>(10);
    const uint32_t state_batch_stride = get_arg_val<uint32_t>(11);
    const uint32_t state_feature_stride = get_arg_val<uint32_t>(12);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto conv_state_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<conv_state_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_local_cb_interface(kInputTapCb0).fifo_page_size;
    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto conv_state_accessor = TensorAccessor(conv_state_args, conv_state_addr, tile_bytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, tile_bytes);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, tile_bytes);

    experimental::Noc noc;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto coord = decode_tile_id(start_tile + tile_idx, sequence_tiles, feature_tiles);
        experimental::CircularBuffer raw_current_cb(kRawCurrentCb);
        experimental::CircularBuffer raw_prefix_cb(kRawPrefixCb);

        experimental::CircularBuffer bias_cb(kBiasCb);
        bias_cb.reserve_back(kOneTile);
        noc.async_read(
            bias_accessor, bias_cb, tile_bytes, {.page_id = bias_page_id(coord.feature_tile_idx)}, {.offset_bytes = 0});

        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            const uint32_t weight_cb_id = kWeightTapCb0 + tap;
            experimental::CircularBuffer weight_cb(weight_cb_id);
            weight_cb.reserve_back(kOneTile);
            noc.async_read(
                weight_accessor,
                weight_cb,
                tile_bytes,
                {.page_id = weight_page_id(tap, coord.feature_tile_idx, feature_tiles)},
                {.offset_bytes = 0});
        }

        noc.async_read_barrier();
        bias_cb.push_back(kOneTile);
        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            const uint32_t weight_cb_id = kWeightTapCb0 + tap;
            experimental::CircularBuffer weight_cb(weight_cb_id);
            weight_cb.push_back(kOneTile);
        }

        const bool has_prefix_tile = coord.sequence_tile_idx > 0;
        const uint32_t tile_sequence_start = coord.sequence_tile_idx * kTileHeight;
        const uint32_t remaining_rows =
            sequence_length > tile_sequence_start ? (sequence_length - tile_sequence_start) : 0;
        const uint32_t valid_rows = remaining_rows > kTileHeight ? kTileHeight : remaining_rows;

        raw_current_cb.reserve_back(kOneTile);
        noc.async_read(
            x_accessor,
            raw_current_cb,
            tile_bytes,
            {.page_id = x_page_id(
                 coord.batch_idx, coord.sequence_tile_idx, coord.feature_tile_idx, x_batch_stride, x_sequence_stride)},
            {.offset_bytes = 0});

        raw_prefix_cb.reserve_back(kOneTile);
        if (has_prefix_tile) {
            noc.async_read(
                x_accessor,
                raw_prefix_cb,
                tile_bytes,
                {.page_id = x_page_id(
                     coord.batch_idx,
                     coord.sequence_tile_idx - 1,
                     coord.feature_tile_idx,
                     x_batch_stride,
                     x_sequence_stride)},
                {.offset_bytes = 0});
        } else {
            noc.async_read(
                conv_state_accessor,
                raw_prefix_cb,
                tile_bytes,
                {.page_id = conv_state_page_id(
                     coord.batch_idx, coord.feature_tile_idx, state_batch_stride, state_feature_stride)},
                {.offset_bytes = 0});
        }

        noc.async_read_barrier();

        raw_current_cb.push_back(kOneTile);
        raw_prefix_cb.push_back(kOneTile);
        raw_current_cb.wait_front(kOneTile);
        raw_prefix_cb.wait_front(kOneTile);

        auto* raw_current = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawCurrentCb));
        auto* raw_prefix = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(kRawPrefixCb));

        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            const uint32_t input_cb_id = kInputTapCb0 + tap;

            experimental::CircularBuffer input_cb(input_cb_id);
            input_cb.reserve_back(kOneTile);
            fill_input_tile(
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(input_cb_id)),
                raw_prefix,
                raw_current,
                tap,
                cache_len,
                valid_rows,
                !has_prefix_tile);
            input_cb.push_back(kOneTile);
        }

        raw_current_cb.pop_front(kOneTile);
        raw_prefix_cb.pop_front(kOneTile);
    }
}
