// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t DTYPE_BFLOAT16 = 0;
constexpr uint32_t DTYPE_FLOAT32 = 1;

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    h = h % TILE_HEIGHT;
    w = w % TILE_WIDTH;
    uint32_t idx = 0;
    if (w >= FACE_WIDTH) {
        w -= FACE_WIDTH;
        idx += FACE_HEIGHT * FACE_WIDTH;
    }
    if (h >= FACE_HEIGHT) {
        h -= FACE_HEIGHT;
        idx += FACE_HEIGHT * TILE_WIDTH;
    }
    idx += h * FACE_WIDTH + w;
    return idx;
}

FORCE_INLINE float bits_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } cast{};
    cast.u = bits;
    return cast.f;
}

FORCE_INLINE uint32_t float_to_bits(float value) {
    union {
        uint32_t u;
        float f;
    } cast{};
    cast.f = value;
    return cast.u;
}

FORCE_INLINE float bfloat16_to_float(uint16_t bf16) { return bits_to_float(static_cast<uint32_t>(bf16) << 16); }

FORCE_INLINE float fast_exp(float x) {
    if (x > 88.0f) {
        x = 88.0f;
    } else if (x < -88.0f) {
        x = -88.0f;
    }
    union {
        uint32_t u;
        float f;
    } cast{};
    cast.u = static_cast<uint32_t>(12102203.0f * x + 1064866805.0f);
    return cast.f;
}

FORCE_INLINE float read_tile_scalar(const uint32_t l1_addr, uint32_t dtype, uint32_t row, uint32_t col) {
    const uint32_t idx = get_tilized_idx(row, col);
    if (dtype == DTYPE_FLOAT32) {
        const volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
        return bits_to_float(ptr[idx]);
    }
    const volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    return bfloat16_to_float(ptr[idx]);
}

FORCE_INLINE void pack_c_as_float_tile(
    uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t src_dtype, uint32_t state_size) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        dst[idx] = 0;
    }
    for (uint32_t row = 0; row < TILE_HEIGHT; ++row) {
        for (uint32_t col = 0; col < state_size; ++col) {
            dst[get_tilized_idx(row, col)] = float_to_bits(read_tile_scalar(src_l1_addr, src_dtype, row, col));
        }
    }
}

FORCE_INLINE void pack_decay_tile(
    uint32_t dst_l1_addr, uint32_t a_l1_addr, uint32_t a_dtype, uint32_t chunk_row, uint32_t valid_t) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        dst[idx] = 0;
    }
    for (uint32_t t = 0; t < valid_t; ++t) {
        const uint32_t bits = float_to_bits(fast_exp(read_tile_scalar(a_l1_addr, a_dtype, chunk_row, t)));
        for (uint32_t p = 0; p < TILE_HEIGHT; ++p) {
            dst[get_tilized_idx(p, t)] = bits;
        }
    }
}

FORCE_INLINE uint32_t
c_tile_id(uint32_t batch_idx, uint32_t num_chunks, uint32_t t_tiles, uint32_t chunk_idx, uint32_t t_tile) {
    return ((batch_idx * num_chunks + chunk_idx) * t_tiles) + t_tile;
}

FORCE_INLINE uint32_t a_tile_id_fixed(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t c_tiles,
    uint32_t chunk_idx,
    uint32_t t_tiles,
    uint32_t t_tile) {
    return (((batch_idx * num_heads + head_idx) * c_tiles + (chunk_idx / TILE_HEIGHT)) * t_tiles) + t_tile;
}

FORCE_INLINE uint32_t states_tile_id(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t p_tiles,
    uint32_t p_tile) {
    return ((((batch_idx * num_chunks) + chunk_idx) * num_heads + head_idx) * p_tiles) + p_tile;
}

void kernel_main() {
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t c_raw_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_c_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t decay_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t scratch_a_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t states_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t c_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t c_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(9);

    const uint32_t states_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t a_addr = get_arg_val<uint32_t>(2);
    const uint32_t unit_start = get_arg_val<uint32_t>(3);
    const uint32_t unit_count = get_arg_val<uint32_t>(4);
    const uint32_t batch_size = get_arg_val<uint32_t>(5);
    const uint32_t num_chunks = get_arg_val<uint32_t>(6);
    const uint32_t chunk_size = get_arg_val<uint32_t>(7);
    const uint32_t seq_len = get_arg_val<uint32_t>(8);
    const uint32_t num_heads = get_arg_val<uint32_t>(9);
    const uint32_t head_dim = get_arg_val<uint32_t>(10);
    const uint32_t state_size = get_arg_val<uint32_t>(11);

    (void)batch_size;
    constexpr auto states_args = TensorAccessorArgs<10>();
    constexpr auto c_args = TensorAccessorArgs<states_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto states = TensorAccessor(states_args, states_addr, states_page_size);
    const auto c = TensorAccessor(c_args, c_addr, c_page_size);
    const auto a = TensorAccessor(a_args, a_addr, a_page_size);

    const uint32_t p_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t t_tiles = (chunk_size + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t unit_local = 0; unit_local < unit_count; ++unit_local) {
        const uint32_t unit = unit_start + unit_local;
        const uint32_t batch_idx = unit / num_chunks;
        const uint32_t chunk_idx = unit % num_chunks;
        const uint32_t chunk_seq_start = chunk_idx * chunk_size;
        if (chunk_seq_start >= seq_len) {
            continue;
        }

        cb_reserve_back(states_cb_index, num_heads * p_tiles);
        uint32_t state_write_addr = get_write_ptr(states_cb_index);
        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                const uint32_t state_tile =
                    states_tile_id(batch_idx, num_chunks, chunk_idx, num_heads, head_idx, p_tiles, p_tile);
                noc_async_read_tile(state_tile, states, state_write_addr);
                state_write_addr += states_page_size;
            }
        }
        noc_async_read_barrier();
        cb_push_back(states_cb_index, num_heads * p_tiles);

        cb_reserve_back(scratch_c_cb_index, 1);
        const uint32_t scratch_c_l1_addr = get_write_ptr(scratch_c_cb_index);
        cb_reserve_back(scratch_a_cb_index, 1);
        const uint32_t scratch_a_l1_addr = get_write_ptr(scratch_a_cb_index);
        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            const uint32_t c_tile = c_tile_id(batch_idx, num_chunks, t_tiles, chunk_idx, t_tile);
            noc_async_read_tile(c_tile, c, scratch_c_l1_addr);
            noc_async_read_barrier();
            cb_reserve_back(c_raw_cb_index, 1);
            pack_c_as_float_tile(get_write_ptr(c_raw_cb_index), scratch_c_l1_addr, c_dtype, state_size);
            cb_push_back(c_raw_cb_index, 1);

            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                const uint32_t a_tile =
                    a_tile_id_fixed(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, t_tile);
                noc_async_read_tile(a_tile, a, scratch_a_l1_addr);
                noc_async_read_barrier();
                const uint32_t valid_t =
                    ((t_tile + 1) * TILE_WIDTH <= chunk_size) ? TILE_WIDTH : (chunk_size % TILE_WIDTH);
                const uint32_t use_valid_t = valid_t == 0 ? TILE_WIDTH : valid_t;
                cb_reserve_back(decay_cb_index, 1);
                pack_decay_tile(
                    get_write_ptr(decay_cb_index), scratch_a_l1_addr, a_dtype, chunk_idx % TILE_HEIGHT, use_valid_t);
                cb_push_back(decay_cb_index, 1);
            }
        }
    }
}
