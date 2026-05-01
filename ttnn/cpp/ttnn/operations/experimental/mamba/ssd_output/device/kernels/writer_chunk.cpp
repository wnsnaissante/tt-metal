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
    for (uint32_t i = 0; i < TILE_HEIGHT * TILE_WIDTH; ++i) {
        ptr[i] = 0;
    }
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

void kernel_main() {
    constexpr uint32_t off_tile_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t out_tile_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_y_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t scratch_x_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t scratch_a_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t scratch_d_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t y_diag_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t x_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t a_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t d_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t fp32_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t y_diag_dtype = get_compile_time_arg_val(12);
    constexpr uint32_t x_dtype = get_compile_time_arg_val(13);
    constexpr uint32_t a_dtype = get_compile_time_arg_val(14);
    constexpr uint32_t d_dtype = get_compile_time_arg_val(15);
    constexpr bool d_values_in_rows = get_compile_time_arg_val(16) != 0;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_diag_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t unit_start = get_arg_val<uint32_t>(5);
    const uint32_t unit_count = get_arg_val<uint32_t>(6);
    const uint32_t batch_size = get_arg_val<uint32_t>(7);
    const uint32_t num_chunks = get_arg_val<uint32_t>(8);
    const uint32_t chunk_size = get_arg_val<uint32_t>(9);
    const uint32_t seq_len = get_arg_val<uint32_t>(10);
    const uint32_t num_heads = get_arg_val<uint32_t>(11);
    const uint32_t head_dim = get_arg_val<uint32_t>(12);

    (void)batch_size;
    (void)scratch_a_cb_index;
    (void)a_page_size;
    (void)a_dtype;
    (void)a_addr;
    constexpr auto output_args = TensorAccessorArgs<17>();
    constexpr auto y_diag_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<y_diag_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto output = TensorAccessor(output_args, output_addr, out_page_size);
    const auto y_diag = TensorAccessor(y_diag_args, y_diag_addr, y_diag_page_size);
    const auto x = TensorAccessor(x_args, x_addr, x_page_size);
    const auto d = TensorAccessor(d_args, d_addr, d_page_size);

    const uint32_t p_tiles = (head_dim + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t t_tiles = (chunk_size + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t h_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t state_tiles = num_heads * p_tiles;
    const uint32_t seq_tiles = (seq_len + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t hidden_tiles = (num_heads * head_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    cb_reserve_back(scratch_y_cb_index, 1);
    const uint32_t scratch_y_l1_addr = get_write_ptr(scratch_y_cb_index);
    cb_reserve_back(scratch_x_cb_index, 1);
    const uint32_t scratch_x_l1_addr = get_write_ptr(scratch_x_cb_index);
    cb_reserve_back(scratch_d_cb_index, 1);
    const uint32_t scratch_d_l1_addr = get_write_ptr(scratch_d_cb_index);

    noc_async_read_tile(0, d, scratch_d_l1_addr);
    noc_async_read_barrier();

    for (uint32_t unit_local = 0; unit_local < unit_count; ++unit_local) {
        const uint32_t unit = unit_start + unit_local;
        const uint32_t batch_idx = unit / num_chunks;
        const uint32_t chunk_idx = unit % num_chunks;
        const uint32_t chunk_seq_start = chunk_idx * chunk_size;
        if (chunk_seq_start >= seq_len) {
            continue;
        }
        const uint32_t remaining = seq_len - chunk_seq_start;
        const uint32_t valid_t = remaining < chunk_size ? remaining : chunk_size;

        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            cb_wait_front(off_tile_cb_index, state_tiles);
            const uint32_t group_time_start = t_tile * TILE_HEIGHT;
            if (group_time_start >= valid_t) {
                cb_pop_front(off_tile_cb_index, state_tiles);
                continue;
            }
            const uint32_t group_valid_t =
                (valid_t - group_time_start) < TILE_HEIGHT ? (valid_t - group_time_start) : TILE_HEIGHT;

            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                uint32_t segment_start = 0;
                while (segment_start < group_valid_t) {
                    const uint32_t global_seq = chunk_seq_start + group_time_start + segment_start;
                    const uint32_t output_seq_tile = global_seq / TILE_HEIGHT;
                    const uint32_t output_row_start = global_seq % TILE_HEIGHT;
                    const uint32_t segment_len = (group_valid_t - segment_start) < (TILE_HEIGHT - output_row_start)
                                                     ? (group_valid_t - segment_start)
                                                     : (TILE_HEIGHT - output_row_start);

                    cb_reserve_back(out_tile_cb_index, num_heads);
                    const uint32_t out_l1_addr = get_write_ptr(out_tile_cb_index);
                    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                        const uint32_t hidden_tile = head_idx * p_tiles + p_tile;
                        const uint32_t output_tile_id =
                            (batch_idx * seq_tiles + output_seq_tile) * hidden_tiles + hidden_tile;
                        const uint32_t head_out_l1_addr = out_l1_addr + head_idx * out_page_size;
                        if (output_row_start == 0) {
                            zero_float_tile(head_out_l1_addr);
                        } else {
                            noc_async_read_page(output_tile_id, output, head_out_l1_addr);
                        }
                    }
                    if (output_row_start != 0) {
                        noc_async_read_barrier();
                    }

                    for (uint32_t s = 0; s < segment_len; ++s) {
                        const uint32_t time_idx = group_time_start + segment_start + s;
                        const uint32_t output_row = output_row_start + s;
                        const uint32_t y_tile = y_diag_tile_id_fixed(
                            batch_idx, num_chunks, chunk_idx, chunk_size, time_idx, h_tiles, 0, p_tiles, p_tile);
                        noc_async_read_tile(y_tile, y_diag, scratch_y_l1_addr);
                        const uint32_t x_tile = y_diag_tile_id_fixed(
                            batch_idx, num_chunks, chunk_idx, chunk_size, time_idx, h_tiles, 0, p_tiles, p_tile);
                        noc_async_read_tile(x_tile, x, scratch_x_l1_addr);
                        noc_async_read_barrier();

                        const uint32_t t_local = time_idx % TILE_HEIGHT;
                        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                            const uint32_t hidden_tile = head_idx * p_tiles + p_tile;
                            const uint32_t off_l1_addr = get_read_ptr(off_tile_cb_index) + hidden_tile * fp32_page_size;
                            const volatile tt_l1_ptr uint32_t* off_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off_l1_addr);
                            volatile tt_l1_ptr uint32_t* out_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1_addr + head_idx * out_page_size);
                            const uint32_t d_row = d_values_in_rows ? (head_idx % TILE_HEIGHT) : 0;
                            const uint32_t d_col = d_values_in_rows ? 0 : (head_idx % TILE_WIDTH);
                            const float d_value = read_tile_scalar(scratch_d_l1_addr, d_dtype, d_row, d_col);
                            for (uint32_t p_local = 0; p_local < TILE_WIDTH; ++p_local) {
                                const float y_value =
                                    read_tile_scalar(scratch_y_l1_addr, y_diag_dtype, head_idx % TILE_HEIGHT, p_local);
                                const float x_value =
                                    read_tile_scalar(scratch_x_l1_addr, x_dtype, head_idx % TILE_HEIGHT, p_local);
                                const float off_value = bits_to_float(off_ptr[get_tilized_idx(p_local, t_local)]);
                                out_ptr[get_tilized_idx(output_row, p_local)] =
                                    float_to_bits(y_value + off_value + d_value * x_value);
                            }
                        }
                    }

                    cb_push_back(out_tile_cb_index, num_heads);
                    cb_wait_front(out_tile_cb_index, num_heads);
                    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                        const uint32_t hidden_tile = head_idx * p_tiles + p_tile;
                        const uint32_t output_tile_id =
                            (batch_idx * seq_tiles + output_seq_tile) * hidden_tiles + hidden_tile;
                        noc_async_write_page(
                            output_tile_id, output, get_read_ptr(out_tile_cb_index) + head_idx * out_page_size);
                    }
                    noc_async_write_barrier();
                    cb_pop_front(out_tile_cb_index, num_heads);

                    segment_start += segment_len;
                }
            }

            cb_pop_front(off_tile_cb_index, state_tiles);
        }
    }
}
