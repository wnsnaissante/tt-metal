// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

ALWI void copy_block(uint32_t src_cb, uint32_t dst_cb, uint32_t width_tiles) {
    cb_wait_front(src_cb, width_tiles);
    cb_reserve_back(dst_cb, width_tiles);
    copy_tile_to_dst_init_short(src_cb);
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(src_cb, tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dst_cb);
        tile_regs_release();
    }
    cb_push_back(dst_cb, width_tiles);
    cb_pop_front(src_cb, width_tiles);
}

ALWI void mul_block_into_cb(
    uint32_t act_cb, uint32_t weight_cb, uint32_t dst_cb, uint32_t width_tiles, uint32_t weight_tile_offset) {
    cb_wait_front(act_cb, width_tiles);
    cb_reserve_back(dst_cb, width_tiles);
    mul_tiles_init(act_cb, weight_cb);
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        mul_tiles(act_cb, weight_cb, tile_idx, weight_tile_offset + tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dst_cb);
        tile_regs_release();
    }
    cb_push_back(dst_cb, width_tiles);
    cb_pop_front(act_cb, width_tiles);
}

ALWI void add_blocks(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t out_cb, uint32_t width_tiles) {
    cb_wait_front(lhs_cb, width_tiles);
    cb_wait_front(rhs_cb, width_tiles);
    cb_reserve_back(out_cb, width_tiles);
    add_tiles_init(lhs_cb, rhs_cb);
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        add_tiles(lhs_cb, rhs_cb, tile_idx, tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, width_tiles);
    cb_pop_front(lhs_cb, width_tiles);
    cb_pop_front(rhs_cb, width_tiles);
}

ALWI void add_blocks_rhs_persistent(uint32_t lhs_cb, uint32_t rhs_cb, uint32_t out_cb, uint32_t width_tiles) {
    cb_wait_front(lhs_cb, width_tiles);
    cb_wait_front(rhs_cb, width_tiles);
    cb_reserve_back(out_cb, width_tiles);
    add_tiles_init(lhs_cb, rhs_cb);
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        add_tiles(lhs_cb, rhs_cb, tile_idx, tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, width_tiles);
    cb_pop_front(lhs_cb, width_tiles);
}

ALWI void silu_block(uint32_t in_cb, uint32_t out_cb, uint32_t width_tiles) {
    cb_wait_front(in_cb, width_tiles);
    cb_reserve_back(out_cb, width_tiles);
    copy_tile_to_dst_init_short(in_cb);
    silu_tile_init();
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(in_cb, tile_idx, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, width_tiles);
    cb_pop_front(in_cb, width_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t width_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_height = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act_rm = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act_tiled = get_compile_time_arg_val(4);
    constexpr uint32_t cb_weight_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_weight_tiled = get_compile_time_arg_val(6);
    constexpr uint32_t cb_bias_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bias_tiled = get_compile_time_arg_val(8);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_tiled = get_compile_time_arg_val(10);
    constexpr uint32_t cb_temp_sum = get_compile_time_arg_val(11);
    constexpr uint32_t cb_final_out = get_compile_time_arg_val(12);
    constexpr uint32_t has_bias = get_compile_time_arg_val(13);
    constexpr uint32_t silu_activation = get_compile_time_arg_val(14);
    constexpr uint32_t debug_version = get_compile_time_arg_val(15);
    (void)debug_version;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_weight_rm, cb_weight_tiled);

    for (uint32_t k = 0; k < kernel_size; ++k) {
        compute_kernel_lib::tilize<width_tiles, cb_weight_rm, cb_weight_tiled>(1, block_height);
    }
    if constexpr (has_bias) {
        compute_kernel_lib::tilize<width_tiles, cb_bias_rm, cb_bias_tiled>(1, block_height);
    }
    for (uint32_t block = 0; block < num_blocks; ++block) {
        uint32_t current_accum_cb = cb_out_tiled;
        uint32_t next_accum_cb = cb_temp_sum;

        for (uint32_t k = 0; k < kernel_size; ++k) {
            compute_kernel_lib::tilize<width_tiles, cb_act_rm, cb_act_tiled>(1, block_height);
            if (k == 0) {
                mul_block_into_cb(cb_act_tiled, cb_weight_tiled, current_accum_cb, width_tiles, k * width_tiles);
            } else {
                mul_block_into_cb(cb_act_tiled, cb_weight_tiled, cb_partial, width_tiles, k * width_tiles);
                add_blocks(cb_partial, current_accum_cb, next_accum_cb, width_tiles);
                const uint32_t previous_accum_cb = current_accum_cb;
                current_accum_cb = next_accum_cb;
                next_accum_cb = previous_accum_cb;
            }
        }

        if constexpr (has_bias) {
            add_blocks_rhs_persistent(current_accum_cb, cb_bias_tiled, next_accum_cb, width_tiles);
            const uint32_t previous_accum_cb = current_accum_cb;
            current_accum_cb = next_accum_cb;
            next_accum_cb = previous_accum_cb;
        }

        if constexpr (silu_activation) {
            silu_block(current_accum_cb, next_accum_cb, width_tiles);
            const uint32_t previous_accum_cb = current_accum_cb;
            current_accum_cb = next_accum_cb;
            next_accum_cb = previous_accum_cb;
        }

        copy_block(current_accum_cb, cb_final_out, width_tiles);
    }
}
