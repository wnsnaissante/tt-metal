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

FORCE_INLINE float read_tile_scalar(uint32_t l1_addr, uint32_t dtype, uint32_t row, uint32_t col) {
    const uint32_t idx = get_tilized_idx(row, col);
    if (dtype == DTYPE_FLOAT32) {
        const volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
        return bits_to_float(ptr[idx]);
    }
    const volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    return bfloat16_to_float(ptr[idx]);
}

FORCE_INLINE void write_fp32_tile_scalar(uint32_t l1_addr, uint32_t row, uint32_t col, float value) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    ptr[get_tilized_idx(row, col)] = float_to_bits(value);
}

FORCE_INLINE void zero_fp32_tile(uint32_t l1_addr) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < TILE_HEIGHT * TILE_WIDTH; ++i) {
        ptr[i] = 0;
    }
}

FORCE_INLINE void zero_fp32_tile_once(uint32_t l1_addr, uint32_t& initialized_addr0, uint32_t& initialized_addr1) {
    if (l1_addr == initialized_addr0 || l1_addr == initialized_addr1) {
        return;
    }
    zero_fp32_tile(l1_addr);
    if (initialized_addr0 == 0) {
        initialized_addr0 = l1_addr;
    } else {
        initialized_addr1 = l1_addr;
    }
}

FORCE_INLINE uint32_t a_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t c_tiles,
    uint32_t chunk_idx,
    uint32_t t_tiles,
    uint32_t t_tile) {
    return (((batch_idx * num_heads + head_idx) * c_tiles + (chunk_idx / TILE_HEIGHT)) * t_tiles) + t_tile;
}

FORCE_INLINE uint32_t
btn_tile_id(uint32_t batch_idx, uint32_t num_chunks, uint32_t chunk_idx, uint32_t t_tiles, uint32_t t_tile) {
    return (((batch_idx * num_chunks + chunk_idx) * t_tiles) + t_tile);
}

FORCE_INLINE uint32_t x_y_tile_id(
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

void kernel_main() {
    constexpr uint32_t x_src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t a_src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t b_src_cb = get_compile_time_arg_val(2);
    constexpr uint32_t c_src_cb = get_compile_time_arg_val(3);
    constexpr uint32_t decay_cb = get_compile_time_arg_val(4);
    constexpr uint32_t x_col_cb = get_compile_time_arg_val(5);
    constexpr uint32_t b_row_cb = get_compile_time_arg_val(6);
    constexpr uint32_t c_row_cb = get_compile_time_arg_val(7);
    constexpr uint32_t zero_state_cb = get_compile_time_arg_val(8);
    constexpr uint32_t reduce_scaler_cb = get_compile_time_arg_val(9);
    constexpr uint32_t x_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t b_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t c_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t fp32_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t x_dtype = get_compile_time_arg_val(15);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(16);
    constexpr uint32_t b_dtype = get_compile_time_arg_val(17);
    constexpr uint32_t c_dtype = get_compile_time_arg_val(18);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_arg_val<uint32_t>(3);
    const uint32_t batch_size = get_arg_val<uint32_t>(4);
    const uint32_t num_chunks = get_arg_val<uint32_t>(5);
    const uint32_t chunk_size = get_arg_val<uint32_t>(6);
    const uint32_t num_heads = get_arg_val<uint32_t>(7);
    const uint32_t head_dim = get_arg_val<uint32_t>(8);
    const uint32_t state_size = get_arg_val<uint32_t>(9);
    const uint32_t p_tiles = get_arg_val<uint32_t>(10);
    const uint32_t t_tiles = get_arg_val<uint32_t>(11);
    const uint32_t scan_start = get_arg_val<uint32_t>(12);
    const uint32_t scan_count = get_arg_val<uint32_t>(13);

    constexpr auto x_args = TensorAccessorArgs<19>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, x_addr, x_page_size);
    const auto a = TensorAccessor(a_args, a_addr, a_page_size);
    const auto b = TensorAccessor(b_args, b_addr, b_page_size);
    const auto c = TensorAccessor(c_args, c_addr, c_page_size);

    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;

#ifdef MAMBA_CHUNK_SCAN_MERGE_P_TILES
    cb_reserve_back(x_src_cb, 2);
#else
    cb_reserve_back(x_src_cb, 1);
#endif
    const uint32_t x_l1 = get_write_ptr(x_src_cb);
    cb_reserve_back(a_src_cb, num_heads);
    const uint32_t a_l1 = get_write_ptr(a_src_cb);
    cb_reserve_back(b_src_cb, 1);
    const uint32_t b_l1 = get_write_ptr(b_src_cb);
    cb_reserve_back(c_src_cb, 1);
    const uint32_t c_l1 = get_write_ptr(c_src_cb);
    cb_reserve_back(reduce_scaler_cb, 1);
    const uint32_t reduce_scaler_l1 = get_write_ptr(reduce_scaler_cb);
    volatile tt_l1_ptr uint32_t* reduce_scaler_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_scaler_l1);
    for (uint32_t i = 0; i < TILE_HEIGHT * TILE_WIDTH; ++i) {
        reduce_scaler_ptr[i] = float_to_bits(1.0f);
    }
    cb_push_back(reduce_scaler_cb, 1);

    uint32_t zero_state_initialized_addr0 = 0;
    uint32_t zero_state_initialized_addr1 = 0;
    uint32_t decay_initialized_addr0 = 0;
    uint32_t decay_initialized_addr1 = 0;
    uint32_t x_col_initialized_addr0 = 0;
    uint32_t x_col_initialized_addr1 = 0;
    uint32_t b_row_initialized_addr0 = 0;
    uint32_t b_row_initialized_addr1 = 0;
    uint32_t c_row_initialized_addr0 = 0;
    uint32_t c_row_initialized_addr1 = 0;

#ifdef MAMBA_CHUNK_SCAN_MERGE_P_TILES
    for (uint32_t local = 0; local < scan_count; ++local) {
        const uint32_t unit = scan_start + local;
        const uint32_t chunk_idx = unit % num_chunks;
        const uint32_t batch_idx = unit / num_chunks;

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                cb_reserve_back(zero_state_cb, 1);
                const uint32_t zero_l1 = get_write_ptr(zero_state_cb);
                zero_fp32_tile_once(zero_l1, zero_state_initialized_addr0, zero_state_initialized_addr1);
                cb_push_back(zero_state_cb, 1);
            }
        }

        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), b, b_l1);
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), c, c_l1);
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                noc_async_read_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, t_tile),
                    a,
                    a_l1 + head_idx * a_page_size);
            }
            noc_async_read_barrier();

            for (uint32_t t_local = 0; t_local < TILE_WIDTH; ++t_local) {
                const uint32_t t = t_tile * TILE_WIDTH + t_local;
                if (t >= chunk_size) {
                    break;
                }

                for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                    noc_async_read_tile(
                        x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile),
                        x,
                        x_l1 + p_tile * x_page_size);
                }
                noc_async_read_barrier();

                cb_reserve_back(b_row_cb, 1);
                const uint32_t b_row_l1 = get_write_ptr(b_row_cb);
                zero_fp32_tile_once(b_row_l1, b_row_initialized_addr0, b_row_initialized_addr1);
                for (uint32_t n = 0; n < state_size; ++n) {
                    write_fp32_tile_scalar(b_row_l1, 0, n, read_tile_scalar(b_l1, b_dtype, t_local, n));
                }
                cb_push_back(b_row_cb, 1);

                cb_reserve_back(c_row_cb, 1);
                const uint32_t c_row_l1 = get_write_ptr(c_row_cb);
                zero_fp32_tile_once(c_row_l1, c_row_initialized_addr0, c_row_initialized_addr1);
                for (uint32_t n = 0; n < state_size; ++n) {
                    write_fp32_tile_scalar(c_row_l1, 0, n, read_tile_scalar(c_l1, c_dtype, t_local, n));
                }
                cb_push_back(c_row_cb, 1);

                for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    cb_reserve_back(decay_cb, 1);
                    const uint32_t decay_l1 = get_write_ptr(decay_cb);
                    zero_fp32_tile_once(decay_l1, decay_initialized_addr0, decay_initialized_addr1);
                    const uint32_t a_head_l1 = a_l1 + head_idx * a_page_size;
                    const float decay =
                        fast_exp(read_tile_scalar(a_head_l1, a_dtype, chunk_idx % TILE_HEIGHT, t_local));
                    write_fp32_tile_scalar(decay_l1, 0, 0, decay);
                    cb_push_back(decay_cb, 1);

                    for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                        const uint32_t valid_p =
                            ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
                        const uint32_t use_valid_p = valid_p == 0 ? TILE_WIDTH : valid_p;

                        cb_reserve_back(x_col_cb, 1);
                        const uint32_t x_col_l1 = get_write_ptr(x_col_cb);
                        zero_fp32_tile_once(x_col_l1, x_col_initialized_addr0, x_col_initialized_addr1);
                        const uint32_t x_tile_l1 = x_l1 + p_tile * x_page_size;
                        for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
                            const float x_value = read_tile_scalar(x_tile_l1, x_dtype, head_idx % TILE_HEIGHT, p_local);
                            write_fp32_tile_scalar(x_col_l1, p_local, 0, x_value);
                        }
                        cb_push_back(x_col_cb, 1);
                    }
                }
            }
        }
    }
#else
    for (uint32_t local = 0; local < scan_count; ++local) {
        const uint32_t unit = scan_start + local;
        const uint32_t p_tile = unit % p_tiles;
        const uint32_t chunk_idx = (unit / p_tiles) % num_chunks;
        const uint32_t batch_idx = unit / (p_tiles * num_chunks);
        const uint32_t valid_p = ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
        const uint32_t use_valid_p = valid_p == 0 ? TILE_WIDTH : valid_p;

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            cb_reserve_back(zero_state_cb, 1);
            const uint32_t zero_l1 = get_write_ptr(zero_state_cb);
            zero_fp32_tile_once(zero_l1, zero_state_initialized_addr0, zero_state_initialized_addr1);
            cb_push_back(zero_state_cb, 1);
        }

        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), b, b_l1);
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), c, c_l1);
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                noc_async_read_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, t_tile),
                    a,
                    a_l1 + head_idx * a_page_size);
            }
            noc_async_read_barrier();

            for (uint32_t t_local = 0; t_local < TILE_WIDTH; ++t_local) {
                const uint32_t t = t_tile * TILE_WIDTH + t_local;
                if (t >= chunk_size) {
                    break;
                }

                noc_async_read_tile(
                    x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile), x, x_l1);
                noc_async_read_barrier();

                cb_reserve_back(b_row_cb, 1);
                const uint32_t b_row_l1 = get_write_ptr(b_row_cb);
                zero_fp32_tile_once(b_row_l1, b_row_initialized_addr0, b_row_initialized_addr1);
                for (uint32_t n = 0; n < state_size; ++n) {
                    write_fp32_tile_scalar(b_row_l1, 0, n, read_tile_scalar(b_l1, b_dtype, t_local, n));
                }
                cb_push_back(b_row_cb, 1);

                cb_reserve_back(c_row_cb, 1);
                const uint32_t c_row_l1 = get_write_ptr(c_row_cb);
                zero_fp32_tile_once(c_row_l1, c_row_initialized_addr0, c_row_initialized_addr1);
                for (uint32_t n = 0; n < state_size; ++n) {
                    write_fp32_tile_scalar(c_row_l1, 0, n, read_tile_scalar(c_l1, c_dtype, t_local, n));
                }
                cb_push_back(c_row_cb, 1);

                for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                    cb_reserve_back(decay_cb, 1);
                    const uint32_t decay_l1 = get_write_ptr(decay_cb);
                    zero_fp32_tile_once(decay_l1, decay_initialized_addr0, decay_initialized_addr1);
                    const uint32_t a_head_l1 = a_l1 + head_idx * a_page_size;
                    const float decay =
                        fast_exp(read_tile_scalar(a_head_l1, a_dtype, chunk_idx % TILE_HEIGHT, t_local));
                    write_fp32_tile_scalar(decay_l1, 0, 0, decay);
                    cb_push_back(decay_cb, 1);

                    cb_reserve_back(x_col_cb, 1);
                    const uint32_t x_col_l1 = get_write_ptr(x_col_cb);
                    zero_fp32_tile_once(x_col_l1, x_col_initialized_addr0, x_col_initialized_addr1);
                    for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
                        const float x_value = read_tile_scalar(x_l1, x_dtype, head_idx % TILE_HEIGHT, p_local);
                        write_fp32_tile_scalar(x_col_l1, p_local, 0, x_value);
                    }
                    cb_push_back(x_col_cb, 1);
                }
            }
        }
    }
#endif

    volatile uint32_t sink = batch_size + state_size + fp32_page_size;
    (void)sink;
}
