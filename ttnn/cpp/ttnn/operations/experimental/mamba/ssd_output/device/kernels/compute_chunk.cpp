// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"

constexpr uint32_t LOCAL_TILE_WIDTH = 32;

FORCE_INLINE void transpose_tile(uint32_t cb_in, uint32_t cb_out) {
    transpose_wh_init(cb_in, cb_out);
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    transpose_wh_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}

FORCE_INLINE void copy_tile_to_cb_keep_input(uint32_t cb_in, uint32_t cb_out) {
    copy_tile_to_dst_init_short(cb_in);
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}

FORCE_INLINE void mul_inplace_pop_b(uint32_t cb_inout, uint32_t cb_b) {
    reconfig_data_format(cb_inout, cb_b);
    pack_reconfig_data_format(cb_inout);
    mul_tiles_init(cb_inout, cb_b);

    cb_wait_front(cb_inout, 1);
    cb_wait_front(cb_b, 1);
    tile_regs_acquire();
    mul_tiles(cb_inout, cb_b, 0, 0, 0);
    cb_pop_front(cb_inout, 1);
    cb_reserve_back(cb_inout, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inout);
    tile_regs_release();
    cb_push_back(cb_inout, 1);
    cb_pop_front(cb_b, 1);
}

void kernel_main() {
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t c_raw_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t c_t_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t off_tile_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t decay_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t c_scaled_cb_index = get_compile_time_arg_val(5);

    const uint32_t unit_count = get_arg_val<uint32_t>(0);
    const uint32_t chunk_size = get_arg_val<uint32_t>(1);
    const uint32_t num_heads = get_arg_val<uint32_t>(2);
    const uint32_t head_dim = get_arg_val<uint32_t>(3);

    const uint32_t p_tiles = (head_dim + LOCAL_TILE_WIDTH - 1) / LOCAL_TILE_WIDTH;
    const uint32_t t_tiles = (chunk_size + LOCAL_TILE_WIDTH - 1) / LOCAL_TILE_WIDTH;
    const uint32_t state_tiles = num_heads * p_tiles;

    for (uint32_t unit_local = 0; unit_local < unit_count; ++unit_local) {
        cb_wait_front(states_cb_index, state_tiles);
        for (uint32_t t_tile = 0; t_tile < t_tiles; ++t_tile) {
            transpose_tile(c_raw_cb_index, c_t_cb_index);
            cb_wait_front(c_t_cb_index, 1);

            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                // (state @ C^T) * decay == state @ (C^T * decay).  Scaling C once per head
                // removes the matmul-result CB round-trip and reuses the scaled tile for both P tiles.
                copy_tile_to_cb_keep_input(c_t_cb_index, c_scaled_cb_index);
                mul_inplace_pop_b(c_scaled_cb_index, decay_cb_index);
                cb_wait_front(c_scaled_cb_index, 1);

                for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                    const uint32_t state_idx = head_idx * p_tiles + p_tile;
                    mm_init(states_cb_index, c_scaled_cb_index, off_tile_cb_index);
                    cb_reserve_back(off_tile_cb_index, 1);
                    acquire_dst();
                    matmul_tiles(states_cb_index, c_scaled_cb_index, state_idx, 0, 0);
                    pack_tile(0, off_tile_cb_index);
                    release_dst();
                    cb_push_back(off_tile_cb_index, 1);
                }

                cb_pop_front(c_scaled_cb_index, 1);
            }

            cb_pop_front(c_t_cb_index, 1);
        }
        cb_pop_front(states_cb_index, state_tiles);
    }
}
