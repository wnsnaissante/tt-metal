// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
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

FORCE_INLINE void mul_tiles_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    mul_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}

FORCE_INLINE void add_tiles_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    add_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}

void kernel_main() {
    constexpr uint32_t y_diag_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t c_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t a_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t states_t_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t decay_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t scaled_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t dot_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t out_tile_cb_index = get_compile_time_arg_val(8);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_heads = get_arg_val<uint32_t>(1);
    const uint32_t head_dim = get_arg_val<uint32_t>(2);
    const uint32_t state_size = get_arg_val<uint32_t>(3);
    const uint32_t p_tiles = (head_dim + LOCAL_TILE_WIDTH - 1) / LOCAL_TILE_WIDTH;
    const uint32_t n_tiles = (state_size + LOCAL_TILE_WIDTH - 1) / LOCAL_TILE_WIDTH;

    binary_op_init_common(y_diag_cb_index, scaled_cb_index, out_tile_cb_index);
    pack_reconfig_data_format(out_tile_cb_index);

    for (uint32_t row_local = 0; row_local < num_rows; ++row_local) {
        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    transpose_tile(states_cb_index, states_t_cb_index);
                }

                mm_init(c_cb_index, states_t_cb_index, dot_cb_index);
                acquire_dst();
                for (uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
                    cb_wait_front(c_cb_index, 1);
                    cb_wait_front(states_t_cb_index, 1);
                    matmul_tiles(c_cb_index, states_t_cb_index, 0, 0, 0);
                    cb_pop_front(c_cb_index, 1);
                    cb_pop_front(states_t_cb_index, 1);
                }
                cb_reserve_back(dot_cb_index, 1);
                pack_tile(0, dot_cb_index);
                cb_push_back(dot_cb_index, 1);
                release_dst();

                mul_tiles_to_cb(a_cb_index, dot_cb_index, scaled_cb_index);
                add_tiles_to_cb(y_diag_cb_index, scaled_cb_index, out_tile_cb_index);
            }
        }
    }
}
