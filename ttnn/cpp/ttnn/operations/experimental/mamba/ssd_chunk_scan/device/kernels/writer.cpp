// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if 0
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

FORCE_INLINE float bfloat16_to_float(uint16_t bf16) {
    return bits_to_float(static_cast<uint32_t>(bf16) << 16);
}

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

FORCE_INLINE float read_fp32_tile_scalar(uint32_t l1_addr, uint32_t row, uint32_t col) {
    const volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    return bits_to_float(ptr[get_tilized_idx(row, col)]);
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

FORCE_INLINE uint32_t btn_tile_id(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t t_tiles,
    uint32_t t_tile) {
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

FORCE_INLINE uint32_t states_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t p_tiles,
    uint32_t p_tile) {
    return (((((batch_idx * num_heads) + head_idx) * num_chunks + chunk_idx) * p_tiles) + p_tile);
}

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t a_cb = get_compile_time_arg_val(1);
    constexpr uint32_t b_cb = get_compile_time_arg_val(2);
    constexpr uint32_t c_cb = get_compile_time_arg_val(3);
    constexpr uint32_t y_cb = get_compile_time_arg_val(4);
    constexpr uint32_t state_cb = get_compile_time_arg_val(5);
    constexpr uint32_t a_out_cb = get_compile_time_arg_val(6);
    constexpr uint32_t x_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t b_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t c_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t fp32_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t x_dtype = get_compile_time_arg_val(12);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(13);
    constexpr uint32_t b_dtype = get_compile_time_arg_val(14);
    constexpr uint32_t c_dtype = get_compile_time_arg_val(15);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t a_addr = get_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_arg_val<uint32_t>(3);
    const uint32_t y_out_addr = get_arg_val<uint32_t>(4);
    const uint32_t states_out_addr = get_arg_val<uint32_t>(5);
    const uint32_t a_out_addr = get_arg_val<uint32_t>(6);
    const uint32_t batch_size = get_arg_val<uint32_t>(7);
    const uint32_t num_chunks = get_arg_val<uint32_t>(8);
    const uint32_t chunk_size = get_arg_val<uint32_t>(9);
    const uint32_t num_heads = get_arg_val<uint32_t>(10);
    const uint32_t head_dim = get_arg_val<uint32_t>(11);
    const uint32_t state_size = get_arg_val<uint32_t>(12);
    const uint32_t p_tiles = get_arg_val<uint32_t>(13);
    const uint32_t t_tiles = get_arg_val<uint32_t>(14);
    const uint32_t cumsum_start = get_arg_val<uint32_t>(15);
    const uint32_t cumsum_count = get_arg_val<uint32_t>(16);
    const uint32_t scan_start = get_arg_val<uint32_t>(17);
    const uint32_t scan_count = get_arg_val<uint32_t>(18);

    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto states_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto a_out_args = TensorAccessorArgs<states_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, x_addr, x_page_size);
    const auto a = TensorAccessor(a_args, a_addr, a_page_size);
    const auto b = TensorAccessor(b_args, b_addr, b_page_size);
    const auto c = TensorAccessor(c_args, c_addr, c_page_size);
    const auto y_out = TensorAccessor(y_args, y_out_addr, fp32_page_size);
    const auto states_out = TensorAccessor(states_args, states_out_addr, fp32_page_size);
    const auto a_out = TensorAccessor(a_out_args, a_out_addr, fp32_page_size);

    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;

    cb_reserve_back(x_cb, 1);
    const uint32_t x_l1 = get_write_ptr(x_cb);
    cb_reserve_back(a_cb, 1);
    const uint32_t a_l1 = get_write_ptr(a_cb);
    cb_reserve_back(b_cb, 1);
    const uint32_t b_l1 = get_write_ptr(b_cb);
    cb_reserve_back(c_cb, 1);
    const uint32_t c_l1 = get_write_ptr(c_cb);
    cb_reserve_back(y_cb, 1);
    const uint32_t y_l1 = get_write_ptr(y_cb);
    cb_reserve_back(a_out_cb, 1);
    const uint32_t a_out_l1 = get_write_ptr(a_out_cb);
    cb_reserve_back(state_cb, num_heads);
    const uint32_t state_l1_base = get_write_ptr(state_cb);

    for (uint32_t local = 0; local < cumsum_count; ++local) {
        const uint32_t unit = cumsum_start + local;
        const uint32_t t_tile = unit % t_tiles;
        const uint32_t head_idx = (unit / t_tiles) % num_heads;
        const uint32_t batch_idx = unit / (t_tiles * num_heads);
        zero_fp32_tile(a_out_l1);

        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            float running = 0.0f;
            for (uint32_t src_tile = 0; src_tile <= t_tile; ++src_tile) {
                noc_async_read_tile(a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, src_tile), a, a_l1);
                noc_async_read_barrier();
                for (uint32_t t_local = 0; t_local < TILE_WIDTH; ++t_local) {
                    const uint32_t t = src_tile * TILE_WIDTH + t_local;
                    if (t >= chunk_size) {
                        break;
                    }
                    running += read_tile_scalar(a_l1, a_dtype, chunk_idx % TILE_HEIGHT, t_local);
                    if (src_tile == t_tile) {
                        write_fp32_tile_scalar(a_out_l1, chunk_idx % TILE_HEIGHT, t_local, running);
                    }
                }
            }
        }
        noc_async_write_tile(a_tile_id(batch_idx, num_heads, head_idx, c_tiles, 0, t_tiles, t_tile), a_out, a_out_l1);
        noc_async_write_barrier();
    }

    for (uint32_t local = 0; local < scan_count; ++local) {
        const uint32_t unit = scan_start + local;
        const uint32_t p_tile = unit % p_tiles;
        const uint32_t chunk_idx = (unit / p_tiles) % num_chunks;
        const uint32_t batch_idx = unit / (p_tiles * num_chunks);
        const uint32_t valid_p =
            ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
        const uint32_t use_valid_p = valid_p == 0 ? TILE_WIDTH : valid_p;

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            zero_fp32_tile(state_l1_base + head_idx * fp32_page_size);
        }

        for (uint32_t t = 0; t < chunk_size; ++t) {
            const uint32_t t_tile = t / TILE_WIDTH;
            const uint32_t t_local = t % TILE_WIDTH;
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), b, b_l1);
            noc_async_read_tile(btn_tile_id(batch_idx, num_chunks, chunk_idx, t_tiles, t_tile), c, c_l1);
            noc_async_read_tile(
                x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile), x, x_l1);
            noc_async_read_barrier();

            zero_fp32_tile(y_l1);
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                noc_async_read_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, t_tile), a, a_l1);
                noc_async_read_barrier();
                const float decay = fast_exp(read_tile_scalar(a_l1, a_dtype, chunk_idx % TILE_HEIGHT, t_local));
                const uint32_t state_l1 = state_l1_base + head_idx * fp32_page_size;

                for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
                    const float x_value = read_tile_scalar(x_l1, x_dtype, head_idx % TILE_HEIGHT, p_local);
                    float y_value = 0.0f;
                    for (uint32_t n = 0; n < state_size; ++n) {
                        const float b_value = read_tile_scalar(b_l1, b_dtype, t_local, n);
                        const float c_value = read_tile_scalar(c_l1, c_dtype, t_local, n);
                        const float prev = read_fp32_tile_scalar(state_l1, p_local, n);
                        const float next = decay * prev + x_value * b_value;
                        write_fp32_tile_scalar(state_l1, p_local, n, next);
                        y_value += next * c_value;
                    }
                    write_fp32_tile_scalar(y_l1, head_idx % TILE_HEIGHT, p_local, y_value);
                }
            }
            noc_async_write_tile(
                x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile),
                y_out,
                y_l1);
            noc_async_write_barrier();
        }

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            noc_async_write_tile(
                states_tile_id(batch_idx, num_heads, head_idx, num_chunks, chunk_idx, p_tiles, p_tile),
                states_out,
                state_l1_base + head_idx * fp32_page_size);
            noc_async_write_barrier();
        }
    }

    volatile uint32_t sink = batch_size;
    (void)sink;
}
#endif

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

FORCE_INLINE float read_tile_scalar(uint32_t l1_addr, uint32_t dtype, uint32_t row, uint32_t col) {
    const uint32_t idx = get_tilized_idx(row, col);
    if (dtype == DTYPE_FLOAT32) {
        const volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
        return bits_to_float(ptr[idx]);
    }
    const volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    return bfloat16_to_float(ptr[idx]);
}

FORCE_INLINE float read_fp32_tile_scalar(uint32_t l1_addr, uint32_t row, uint32_t col) {
    const volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    return bits_to_float(ptr[get_tilized_idx(row, col)]);
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

FORCE_INLINE uint32_t states_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t p_tiles,
    uint32_t p_tile) {
    return (((((batch_idx * num_heads) + head_idx) * num_chunks + chunk_idx) * p_tiles) + p_tile);
}

void kernel_main() {
    constexpr uint32_t a_src_cb = get_compile_time_arg_val(0);
    constexpr uint32_t y_vec_cb = get_compile_time_arg_val(1);
    constexpr uint32_t final_state_cb = get_compile_time_arg_val(2);
    constexpr uint32_t y_tile_cb = get_compile_time_arg_val(3);
    constexpr uint32_t a_out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t fp32_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(7);

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t states_out_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_out_addr = get_arg_val<uint32_t>(3);
    const uint32_t batch_size = get_arg_val<uint32_t>(4);
    const uint32_t num_chunks = get_arg_val<uint32_t>(5);
    const uint32_t chunk_size = get_arg_val<uint32_t>(6);
    const uint32_t num_heads = get_arg_val<uint32_t>(7);
    const uint32_t head_dim = get_arg_val<uint32_t>(8);
    const uint32_t state_size = get_arg_val<uint32_t>(9);
    const uint32_t p_tiles = get_arg_val<uint32_t>(10);
    const uint32_t t_tiles = get_arg_val<uint32_t>(11);
    const uint32_t cumsum_start = get_arg_val<uint32_t>(12);
    const uint32_t cumsum_count = get_arg_val<uint32_t>(13);
    const uint32_t scan_start = get_arg_val<uint32_t>(14);
    const uint32_t scan_count = get_arg_val<uint32_t>(15);

    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto y_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto states_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto a_out_args = TensorAccessorArgs<states_args.next_compile_time_args_offset()>();

    const auto a = TensorAccessor(a_args, a_addr, a_page_size);
    const auto y_out = TensorAccessor(y_args, y_out_addr, fp32_page_size);
    const auto states_out = TensorAccessor(states_args, states_out_addr, fp32_page_size);
    const auto a_out = TensorAccessor(a_out_args, a_out_addr, fp32_page_size);

    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;

    cb_reserve_back(a_src_cb, 1);
    const uint32_t a_l1 = get_write_ptr(a_src_cb);
    cb_reserve_back(a_out_cb, 1);
    const uint32_t a_out_l1 = get_write_ptr(a_out_cb);
#ifdef MAMBA_CHUNK_SCAN_MERGE_P_TILES
    cb_reserve_back(y_tile_cb, p_tiles);
#else
    cb_reserve_back(y_tile_cb, 1);
#endif
    const uint32_t y_l1 = get_write_ptr(y_tile_cb);
    bool y_tile_initialized = false;

    for (uint32_t local = 0; local < cumsum_count; ++local) {
        const uint32_t unit = cumsum_start + local;
#ifdef MAMBA_CHUNK_SCAN_CUMSUM_V2
        const uint32_t head_idx = unit % num_heads;
        const uint32_t batch_idx = unit / num_heads;

        for (uint32_t c_tile = 0; c_tile < c_tiles; ++c_tile) {
            float running[TILE_HEIGHT];
            for (uint32_t c_local = 0; c_local < TILE_HEIGHT; ++c_local) {
                running[c_local] = 0.0f;
            }

            for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
                zero_fp32_tile(a_out_l1);
                const uint32_t chunk_base = c_tile * TILE_HEIGHT;
                noc_async_read_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_base, t_tiles, t_tile), a, a_l1);
                noc_async_read_barrier();

                for (uint32_t c_local = 0; c_local < TILE_HEIGHT; ++c_local) {
                    const uint32_t chunk_idx = chunk_base + c_local;
                    if (chunk_idx >= num_chunks) {
                        break;
                    }
                    for (uint32_t t_local = 0; t_local < TILE_WIDTH; ++t_local) {
                        const uint32_t t = t_tile * TILE_WIDTH + t_local;
                        if (t >= chunk_size) {
                            break;
                        }
                        running[c_local] += read_tile_scalar(a_l1, a_dtype, c_local, t_local);
                        write_fp32_tile_scalar(a_out_l1, c_local, t_local, running[c_local]);
                    }
                }
                noc_async_write_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_base, t_tiles, t_tile), a_out, a_out_l1);
                noc_async_write_barrier();
            }
        }
#else
        const uint32_t t_tile = unit % t_tiles;
        const uint32_t head_idx = (unit / t_tiles) % num_heads;
        const uint32_t batch_idx = unit / (t_tiles * num_heads);
        zero_fp32_tile(a_out_l1);

        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            float running = 0.0f;
            for (uint32_t src_tile = 0; src_tile <= t_tile; ++src_tile) {
                noc_async_read_tile(
                    a_tile_id(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, src_tile), a, a_l1);
                noc_async_read_barrier();
                for (uint32_t t_local = 0; t_local < TILE_WIDTH; ++t_local) {
                    const uint32_t t = src_tile * TILE_WIDTH + t_local;
                    if (t >= chunk_size) {
                        break;
                    }
                    running += read_tile_scalar(a_l1, a_dtype, chunk_idx % TILE_HEIGHT, t_local);
                    if (src_tile == t_tile) {
                        write_fp32_tile_scalar(a_out_l1, chunk_idx % TILE_HEIGHT, t_local, running);
                    }
                }
            }
        }
        noc_async_write_tile(a_tile_id(batch_idx, num_heads, head_idx, c_tiles, 0, t_tiles, t_tile), a_out, a_out_l1);
        noc_async_write_barrier();
#endif
    }

#ifdef MAMBA_CHUNK_SCAN_MERGE_P_TILES
    for (uint32_t local = 0; local < scan_count; ++local) {
        const uint32_t unit = scan_start + local;
        const uint32_t chunk_idx = unit % num_chunks;
        const uint32_t batch_idx = unit / num_chunks;

        for (uint32_t t = 0; t < chunk_size; ++t) {
            if (!y_tile_initialized) {
                for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                    zero_fp32_tile(y_l1 + p_tile * fp32_page_size);
                }
                y_tile_initialized = true;
            }
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                    const uint32_t valid_p =
                        ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
                    const uint32_t use_valid_p = valid_p == 0 ? TILE_WIDTH : valid_p;

                    cb_wait_front(y_vec_cb, 1);
                    const uint32_t y_vec_l1 = get_read_ptr(y_vec_cb);
                    const uint32_t y_tile_l1 = y_l1 + p_tile * fp32_page_size;
                    for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
                        write_fp32_tile_scalar(
                            y_tile_l1, head_idx % TILE_HEIGHT, p_local, read_fp32_tile_scalar(y_vec_l1, p_local, 0));
                    }
                    cb_pop_front(y_vec_cb, 1);
                }
            }
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                noc_async_write_tile(
                    x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile),
                    y_out,
                    y_l1 + p_tile * fp32_page_size);
                noc_async_write_barrier();
            }
        }

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                cb_wait_front(final_state_cb, 1);
                noc_async_write_tile(
                    states_tile_id(batch_idx, num_heads, head_idx, num_chunks, chunk_idx, p_tiles, p_tile),
                    states_out,
                    get_read_ptr(final_state_cb));
                noc_async_write_barrier();
                cb_pop_front(final_state_cb, 1);
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

        for (uint32_t t = 0; t < chunk_size; ++t) {
            if (!y_tile_initialized) {
                zero_fp32_tile(y_l1);
                y_tile_initialized = true;
            }
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                cb_wait_front(y_vec_cb, 1);
                const uint32_t y_vec_l1 = get_read_ptr(y_vec_cb);
                for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
                    write_fp32_tile_scalar(
                        y_l1, head_idx % TILE_HEIGHT, p_local, read_fp32_tile_scalar(y_vec_l1, p_local, 0));
                }
                cb_pop_front(y_vec_cb, 1);
            }
            noc_async_write_tile(
                x_y_tile_id(batch_idx, num_chunks, chunk_idx, chunk_size, t, h_tiles, 0, p_tiles, p_tile), y_out, y_l1);
            noc_async_write_barrier();
        }

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            cb_wait_front(final_state_cb, 1);
            noc_async_write_tile(
                states_tile_id(batch_idx, num_heads, head_idx, num_chunks, chunk_idx, p_tiles, p_tile),
                states_out,
                get_read_ptr(final_state_cb));
            noc_async_write_barrier();
            cb_pop_front(final_state_cb, 1);
        }
    }
#endif

    volatile uint32_t sink = batch_size + state_size + fp32_page_size;
    (void)sink;
}
