// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

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

ALWI void mul_block_into_partial(
    uint32_t act_cb, uint32_t weight_cb, uint32_t partial_cb, uint32_t width_tiles, uint32_t weight_tile_offset) {
    cb_wait_front(act_cb, width_tiles);
    cb_reserve_back(partial_cb, width_tiles);
    mul_tiles_init(act_cb, weight_cb);
    for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
        tile_regs_acquire();
        mul_tiles(act_cb, weight_cb, tile_idx, weight_tile_offset + tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, partial_cb);
        tile_regs_release();
    }
    cb_push_back(partial_cb, width_tiles);
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
    constexpr uint32_t cb_out_rm0 = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_rm1 = get_compile_time_arg_val(13);
    constexpr uint32_t has_bias = get_compile_time_arg_val(14);
    constexpr uint32_t silu_activation = get_compile_time_arg_val(15);
    constexpr uint32_t debug_version = get_compile_time_arg_val(16);

    DPRINT << "dwconv1d compute entry v=" << debug_version << ENDL();

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    DPRINT << "dwconv1d compute start v=" << debug_version << " num_blocks=" << num_blocks << ENDL();
    compute_kernel_hw_startup(cb_weight_rm, cb_weight_tiled);
    DPRINT << "dwconv1d compute hw startup done v=" << debug_version << ENDL();
    compute_kernel_lib::untilize_init<width_tiles, cb_out_tiled, cb_out_rm0>();

    for (uint32_t k = 0; k < kernel_size; ++k) {
        compute_kernel_lib::tilize<width_tiles, cb_weight_rm, cb_weight_tiled>(1, block_height);
    }
    DPRINT << "dwconv1d compute weight tiled" << ENDL();
    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t k = 0; k < kernel_size; ++k) {
            compute_kernel_lib::tilize<width_tiles, cb_act_rm, cb_act_tiled>(1, block_height);
            mul_block_into_partial(cb_act_tiled, cb_weight_tiled, cb_partial, width_tiles, k * width_tiles);
            if (k == 0) {
                copy_block(cb_partial, cb_out_tiled, width_tiles);
            } else {
                add_blocks(cb_partial, cb_out_tiled, cb_temp_sum, width_tiles);
                copy_block(cb_temp_sum, cb_out_tiled, width_tiles);
            }
        }

        if constexpr (has_bias) {
            compute_kernel_lib::tilize<width_tiles, cb_bias_rm, cb_bias_tiled>(1, block_height);
            add_blocks(cb_bias_tiled, cb_out_tiled, cb_temp_sum, width_tiles);
            copy_block(cb_temp_sum, cb_out_tiled, width_tiles);
        }

        if constexpr (silu_activation) {
            silu_block(cb_out_tiled, cb_temp_sum, width_tiles);
            copy_block(cb_temp_sum, cb_out_tiled, width_tiles);
        }

        if (block % 2 == 0) {
            compute_kernel_lib::untilize<
                width_tiles,
                cb_out_tiled,
                cb_out_rm0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        } else {
            compute_kernel_lib::untilize<
                width_tiles,
                cb_out_tiled,
                cb_out_rm1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }

        if (block == 0) {
            DPRINT << "dwconv1d compute first block done" << ENDL();
        }
    }

    compute_kernel_lib::untilize_uninit<width_tiles, cb_out_tiled, cb_out_rm0>();
    DPRINT << "dwconv1d compute done" << ENDL();
}
