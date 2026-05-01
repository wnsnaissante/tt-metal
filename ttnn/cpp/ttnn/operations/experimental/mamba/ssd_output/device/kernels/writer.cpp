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

FORCE_INLINE float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

FORCE_INLINE float fast_exp(float x) {
    constexpr float log2e = 1.4426950408889634f;
    const float y = clampf(x * log2e, -126.0f, 126.0f);
    const int32_t exponent = static_cast<int32_t>(y);
    const float frac = y - static_cast<float>(exponent);
    const float two_to_frac = 1.0f + frac * (0.696065642f + frac * (0.224494337f + frac * 0.079440238f));
    const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    return two_to_frac * bits_to_float(bits);
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

__attribute__((noinline)) void accumulate_p_tile(
    volatile tt_l1_ptr uint32_t* out_ptr,
    const InterleavedAddrGen<true>& y_diag,
    const InterleavedAddrGen<true>& states,
    const InterleavedAddrGen<true>& c,
    uint32_t y_diag_cb_index,
    uint32_t states_cb_index,
    uint32_t c_cb_index,
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t chunk_size,
    uint32_t time_idx,
    uint32_t h_tiles,
    uint32_t head_idx,
    uint32_t p_tiles,
    uint32_t p_tile,
    uint32_t n_tiles,
    uint32_t state_size,
    uint32_t time_tile,
    uint32_t time_local,
    uint32_t num_heads,
    uint32_t y_diag_dtype,
    uint32_t states_dtype,
    uint32_t c_dtype,
    uint32_t head_dim,
    float decay) {
    const uint32_t y_tile = y_diag_tile_id_fixed(
        batch_idx, num_chunks, chunk_idx, chunk_size, time_idx, h_tiles, head_idx, p_tiles, p_tile);
    cb_reserve_back(y_diag_cb_index, 1);
    uint32_t y_l1_write_addr = get_write_ptr(y_diag_cb_index);
    noc_async_read_tile(y_tile, y_diag, y_l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(y_diag_cb_index, 1);
    cb_wait_front(y_diag_cb_index, 1);

    const uint32_t valid_p = ((p_tile + 1) * TILE_WIDTH <= head_dim) ? TILE_WIDTH : (head_dim % TILE_WIDTH);
    const uint32_t use_valid_p = valid_p == 0 ? TILE_WIDTH : valid_p;
    for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
        const uint32_t out_col = head_idx * head_dim + p_tile * TILE_WIDTH + p_local;
        out_ptr[out_col] = 0;
    }

    for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        const uint32_t c_tile = c_tile_id(
            batch_idx, num_chunks, (chunk_size + TILE_HEIGHT - 1) / TILE_HEIGHT, n_tiles, chunk_idx, time_tile, n_tile);
        const uint32_t state_tile =
            states_tile_id(batch_idx, num_chunks, chunk_idx, num_heads, head_idx, p_tiles, p_tile, n_tiles, n_tile);

        cb_reserve_back(c_cb_index, 1);
        uint32_t c_l1_write_addr = get_write_ptr(c_cb_index);
        noc_async_read_tile(c_tile, c, c_l1_write_addr);

        cb_reserve_back(states_cb_index, 1);
        uint32_t states_l1_write_addr = get_write_ptr(states_cb_index);
        noc_async_read_tile(state_tile, states, states_l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(c_cb_index, 1);
        cb_push_back(states_cb_index, 1);
        cb_wait_front(c_cb_index, 1);
        cb_wait_front(states_cb_index, 1);

        const uint32_t valid_n = ((n_tile + 1) * TILE_WIDTH <= state_size) ? TILE_WIDTH : (state_size % TILE_WIDTH);
        const uint32_t use_valid_n = valid_n == 0 ? TILE_WIDTH : valid_n;

        for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
            const uint32_t out_col = head_idx * head_dim + p_tile * TILE_WIDTH + p_local;
            float dot = bits_to_float(out_ptr[out_col]);
            for (uint32_t n_local = 0; n_local < use_valid_n; ++n_local) {
                const float state_val = read_tile_scalar(states_l1_write_addr, states_dtype, p_local, n_local);
                const float c_val = read_tile_scalar(c_l1_write_addr, c_dtype, time_local, n_local);
                dot += state_val * c_val;
            }
            out_ptr[out_col] = float_to_bits(dot);
        }

        cb_pop_front(c_cb_index, 1);
        cb_pop_front(states_cb_index, 1);
    }

    for (uint32_t p_local = 0; p_local < use_valid_p; ++p_local) {
        const float y_diag_val = read_tile_scalar(y_l1_write_addr, y_diag_dtype, head_idx % TILE_HEIGHT, p_local);
        const uint32_t out_col = head_idx * head_dim + p_tile * TILE_WIDTH + p_local;
        out_ptr[out_col] = float_to_bits(y_diag_val + decay * bits_to_float(out_ptr[out_col]));
    }
    cb_pop_front(y_diag_cb_index, 1);
}

void kernel_main() {
    constexpr uint32_t y_diag_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t c_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t a_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t y_diag_dtype = get_compile_time_arg_val(5);
    constexpr uint32_t states_dtype = get_compile_time_arg_val(6);
    constexpr uint32_t c_dtype = get_compile_time_arg_val(7);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(8);
    constexpr uint32_t y_diag_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t states_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t c_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(13);

    const uint32_t y_diag_addr = get_arg_val<uint32_t>(0);
    const uint32_t states_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_addr = get_arg_val<uint32_t>(3);
    const uint32_t output_addr = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t start_row = get_arg_val<uint32_t>(6);
    const uint32_t seq_len = get_arg_val<uint32_t>(7);
    const uint32_t num_chunks = get_arg_val<uint32_t>(8);
    const uint32_t chunk_size = get_arg_val<uint32_t>(9);
    const uint32_t num_heads = get_arg_val<uint32_t>(10);
    const uint32_t head_dim = get_arg_val<uint32_t>(11);
    const uint32_t state_size = get_arg_val<uint32_t>(12);
    const InterleavedAddrGen<true> y_diag = {
        .bank_base_address = y_diag_addr,
        .page_size = y_diag_page_size,
    };
    const InterleavedAddrGen<true> states = {
        .bank_base_address = states_addr,
        .page_size = states_page_size,
    };
    const InterleavedAddrGen<true> c = {
        .bank_base_address = c_addr,
        .page_size = c_page_size,
    };
    const InterleavedAddrGen<true> a = {
        .bank_base_address = a_addr,
        .page_size = a_page_size,
    };
    const InterleavedAddrGen<true> output = {
        .bank_base_address = output_addr,
        .page_size = out_page_size,
    };

    const uint32_t t_tiles = (chunk_size + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t p_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t n_tiles = (state_size + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t c_tiles = (num_chunks + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t row_local = 0; row_local < num_rows; ++row_local) {
        const uint32_t global_row = start_row + row_local;
        const uint32_t batch_idx = global_row / seq_len;
        const uint32_t seq_idx = global_row % seq_len;
        const uint32_t chunk_idx = seq_idx / chunk_size;
        const uint32_t time_idx = seq_idx % chunk_size;
        const uint32_t time_tile = time_idx / TILE_HEIGHT;
        const uint32_t time_local = time_idx % TILE_HEIGHT;

        cb_reserve_back(out_cb_index, 1);
        volatile tt_l1_ptr uint32_t* out_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(out_cb_index));

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            const uint32_t a_tile =
                a_tile_id_fixed(batch_idx, num_heads, head_idx, c_tiles, chunk_idx, t_tiles, time_tile);
            cb_reserve_back(a_cb_index, 1);
            uint32_t a_l1_write_addr = get_write_ptr(a_cb_index);
            noc_async_read_tile(a_tile, a, a_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(a_cb_index, 1);
            cb_wait_front(a_cb_index, 1);
            const float decay =
                fast_exp(read_tile_scalar(a_l1_write_addr, a_dtype, chunk_idx % TILE_HEIGHT, time_local));

            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                accumulate_p_tile(
                    out_ptr,
                    y_diag,
                    states,
                    c,
                    y_diag_cb_index,
                    states_cb_index,
                    c_cb_index,
                    batch_idx,
                    num_chunks,
                    chunk_idx,
                    chunk_size,
                    time_idx,
                    h_tiles,
                    head_idx,
                    p_tiles,
                    p_tile,
                    n_tiles,
                    state_size,
                    time_tile,
                    time_local,
                    num_heads,
                    y_diag_dtype,
                    states_dtype,
                    c_dtype,
                    head_dim,
                    decay);
            }
            cb_pop_front(a_cb_index, 1);
        }

        cb_push_back(out_cb_index, 1);
        cb_wait_front(out_cb_index, 1);
        noc_async_write(get_read_ptr(out_cb_index), get_noc_addr(global_row, output), out_page_size);
        noc_async_write_barrier();
        cb_pop_front(out_cb_index, 1);
    }
}
