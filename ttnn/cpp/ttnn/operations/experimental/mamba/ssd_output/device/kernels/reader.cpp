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
    if (h >= FACE_WIDTH) {
        h -= FACE_WIDTH;
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

FORCE_INLINE void zero_float_tile(uint32_t l1_addr) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        ptr[idx] = 0;
    }
}

FORCE_INLINE void fill_scalar_tile(uint32_t l1_addr, float value) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    const uint32_t bits = float_to_bits(value);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        ptr[idx] = bits;
    }
}

FORCE_INLINE void fill_float_row_tile(
    uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t src_dtype, uint32_t src_row, uint32_t valid_cols) {
    zero_float_tile(dst_l1_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
    for (uint32_t col = 0; col < valid_cols; ++col) {
        dst[get_tilized_idx(0, col)] = float_to_bits(read_tile_scalar(src_l1_addr, src_dtype, src_row, col));
    }
}

FORCE_INLINE void fill_c_row_tile(
    uint32_t dst_l1_addr,
    uint32_t src_l1_addr,
    uint32_t src_dtype,
    uint32_t dst_dtype,
    uint32_t src_row,
    uint32_t valid_cols) {
    if (dst_dtype == DTYPE_FLOAT32) {
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
        for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
            dst[idx] = 0;
        }
        for (uint32_t col = 0; col < valid_cols; ++col) {
            dst[get_tilized_idx(0, col)] = float_to_bits(read_tile_scalar(src_l1_addr, src_dtype, src_row, col));
        }
        return;
    }

    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        dst[idx] = 0;
    }
    if (src_dtype == DTYPE_FLOAT32) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1_addr);
        for (uint32_t col = 0; col < valid_cols; ++col) {
            const uint32_t src_bits = src[get_tilized_idx(src_row, col)];
            dst[get_tilized_idx(0, col)] = static_cast<uint16_t>(src_bits >> 16);
        }
        return;
    }
    const volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_l1_addr);
    for (uint32_t col = 0; col < valid_cols; ++col) {
        dst[get_tilized_idx(0, col)] = src[get_tilized_idx(src_row, col)];
    }
}

FORCE_INLINE void copy_float_tile(uint32_t dst_l1_addr, uint32_t src_l1_addr) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1_addr);
    const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        dst[idx] = src[idx];
    }
}

FORCE_INLINE void copy_tile_with_dtype(uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t dtype) {
    if (dtype == DTYPE_FLOAT32) {
        copy_float_tile(dst_l1_addr, src_l1_addr);
        return;
    }
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_l1_addr);
    const volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_l1_addr);
    for (uint32_t idx = 0; idx < TILE_HEIGHT * TILE_WIDTH; ++idx) {
        dst[idx] = src[idx];
    }
}

FORCE_INLINE uint32_t c_tile_id(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t t_tiles,
    uint32_t n_tiles,
    uint32_t chunk_idx,
    uint32_t t_tile,
    uint32_t n_tile) {
    return (((batch_idx * num_chunks + chunk_idx) * t_tiles + t_tile) * n_tiles) + n_tile;
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

FORCE_INLINE uint32_t y_diag_tile_id_fixed(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t chunk_size,
    uint32_t time_idx,
    uint32_t h_tiles,
    uint32_t head_idx,
    uint32_t p_tiles,
    uint32_t p_tile) {
    return (((((batch_idx * num_chunks) + chunk_idx) * chunk_size) + time_idx) * h_tiles + (head_idx / TILE_HEIGHT)) *
               p_tiles +
           p_tile;
}

FORCE_INLINE uint32_t states_tile_id(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t p_tiles,
    uint32_t p_tile,
    uint32_t n_tiles,
    uint32_t n_tile) {
    return (((((batch_idx * num_chunks) + chunk_idx) * num_heads + head_idx) * p_tiles + p_tile) * n_tiles) + n_tile;
}

void kernel_main() {
    constexpr uint32_t y_diag_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t c_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t a_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t scratch_a_cb_index = 8;
    constexpr uint32_t scratch_c_cb_index = 9;
    constexpr uint32_t scratch_y_cb_index = 10;
    constexpr uint32_t y_diag_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t states_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t c_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t y_diag_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t c_dtype = get_compile_time_arg_val(9);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(10);
    constexpr uint32_t states_dtype = get_compile_time_arg_val(11);

    const uint32_t y_diag_addr = get_arg_val<uint32_t>(0);
    const uint32_t states_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_rows = get_arg_val<uint32_t>(4);
    const uint32_t start_row = get_arg_val<uint32_t>(5);
    const uint32_t seq_len = get_arg_val<uint32_t>(6);
    const uint32_t num_chunks = get_arg_val<uint32_t>(7);
    const uint32_t chunk_size = get_arg_val<uint32_t>(8);
    const uint32_t num_heads = get_arg_val<uint32_t>(9);
    const uint32_t head_dim = get_arg_val<uint32_t>(10);
    const uint32_t state_size = get_arg_val<uint32_t>(11);

    constexpr auto y_diag_args = TensorAccessorArgs<12>();
    constexpr auto states_args = TensorAccessorArgs<y_diag_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<states_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto y_diag = TensorAccessor(y_diag_args, y_diag_addr, y_diag_page_size);
    const auto states = TensorAccessor(states_args, states_addr, states_page_size);
    const auto c = TensorAccessor(c_args, c_addr, c_page_size);
    const auto a = TensorAccessor(a_args, a_addr, a_page_size);

    const uint32_t t_tiles = (chunk_size + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t p_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t n_tiles = (state_size + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;

    cb_reserve_back(scratch_a_cb_index, 1);
    const uint32_t scratch_a_l1_addr = get_write_ptr(scratch_a_cb_index);
    cb_reserve_back(scratch_c_cb_index, 1);
    const uint32_t scratch_c_l1_addr = get_write_ptr(scratch_c_cb_index);
    cb_reserve_back(scratch_y_cb_index, 1);
    const uint32_t scratch_y_l1_addr = get_write_ptr(scratch_y_cb_index);

    for (uint32_t row_local = 0; row_local < num_rows; ++row_local) {
        const uint32_t global_row = start_row + row_local;
        const uint32_t batch_idx = global_row / seq_len;
        const uint32_t seq_idx = global_row % seq_len;
        const uint32_t chunk_idx = seq_idx / chunk_size;
        const uint32_t time_idx = seq_idx % chunk_size;
        const uint32_t time_tile = time_idx / TILE_HEIGHT;
        const uint32_t time_local = time_idx % TILE_HEIGHT;
        const uint32_t chunk_row = chunk_idx % TILE_HEIGHT;

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            const uint32_t a_tile =
                a_tile_id_fixed(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, time_tile);
            noc_async_read_tile(a_tile, a, scratch_a_l1_addr);
            noc_async_read_barrier();
            fill_scalar_tile(
                scratch_a_l1_addr, fast_exp(read_tile_scalar(scratch_a_l1_addr, a_dtype, chunk_row, time_local)));

            if (n_tiles == 1) {
                const uint32_t c_tile = c_tile_id(batch_idx, num_chunks, t_tiles, n_tiles, chunk_idx, time_tile, 0);
                noc_async_read_tile(c_tile, c, scratch_c_l1_addr);
                noc_async_read_barrier();
            }

            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                cb_reserve_back(a_cb_index, 1);
                uint32_t a_l1_addr = get_write_ptr(a_cb_index);
                copy_float_tile(a_l1_addr, scratch_a_l1_addr);
                cb_push_back(a_cb_index, 1);

                const uint32_t y_tile = y_diag_tile_id_fixed(
                    batch_idx, num_chunks, chunk_idx, chunk_size, time_idx, h_tiles, head_idx, p_tiles, p_tile);
                noc_async_read_tile(y_tile, y_diag, scratch_y_l1_addr);
                noc_async_read_barrier();
                cb_reserve_back(y_diag_cb_index, 1);
                uint32_t y_l1_addr = get_write_ptr(y_diag_cb_index);
                const uint32_t valid_p = ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
                fill_float_row_tile(
                    y_l1_addr,
                    scratch_y_l1_addr,
                    y_diag_dtype,
                    head_idx % TILE_HEIGHT,
                    valid_p == 0 ? TILE_WIDTH : valid_p);
                cb_push_back(y_diag_cb_index, 1);

                for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    cb_reserve_back(c_cb_index, 1);
                    uint32_t c_l1_addr = get_write_ptr(c_cb_index);
                    if (n_tiles == 1) {
                        fill_c_row_tile(c_l1_addr, scratch_c_l1_addr, c_dtype, states_dtype, time_local, state_size);
                    } else {
                        const uint32_t c_tile =
                            c_tile_id(batch_idx, num_chunks, t_tiles, n_tiles, chunk_idx, time_tile, n_tile);
                        noc_async_read_tile(c_tile, c, scratch_c_l1_addr);
                        noc_async_read_barrier();
                        const uint32_t valid_n =
                            ((n_tile + 1) * TILE_WIDTH <= state_size) ? TILE_WIDTH : (state_size % TILE_WIDTH);
                        fill_c_row_tile(
                            c_l1_addr,
                            scratch_c_l1_addr,
                            c_dtype,
                            states_dtype,
                            time_local,
                            valid_n == 0 ? TILE_WIDTH : valid_n);
                    }
                    cb_push_back(c_cb_index, 1);

                    const uint32_t state_tile = states_tile_id(
                        batch_idx, num_chunks, chunk_idx, num_heads, head_idx, p_tiles, p_tile, n_tiles, n_tile);
                    cb_reserve_back(states_cb_index, 1);
                    noc_async_read_tile(state_tile, states, get_write_ptr(states_cb_index));
                    noc_async_read_barrier();
                    cb_push_back(states_cb_index, 1);
                }
            }
        }
    }
}
