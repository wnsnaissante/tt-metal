// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t ONE_TILE = 1;

template <bool pop_input>
FORCE_INLINE void copy_tile_to_cb(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    copy_tile_to_dst_init_short(cb_in);

    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    if constexpr (pop_input) {
        cb_pop_front(cb_in, ONE_TILE);
    }
}

FORCE_INLINE void mul_bcast_scalar_to_cb(uint32_t cb_a, uint32_t cb_scalar, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_scalar);
    pack_reconfig_data_format(cb_out);
    mul_tiles_bcast_scalar_init_short(cb_a, cb_scalar);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_scalar, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_a, cb_scalar, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_scalar, ONE_TILE);
}

template <bool pop_scalar>
FORCE_INLINE void mul_bcast_scalar_to_cb_reuse(uint32_t cb_a, uint32_t cb_scalar, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_scalar);
    pack_reconfig_data_format(cb_out);
    mul_tiles_bcast_scalar_init_short(cb_a, cb_scalar);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_scalar, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_scalar(cb_a, cb_scalar, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    if constexpr (pop_scalar) {
        cb_pop_front(cb_scalar, ONE_TILE);
    }
}

FORCE_INLINE void add_tiles_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    add_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

FORCE_INLINE void add_tiles_to_two_cbs(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out0, uint32_t cb_out1) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out0);
    add_tiles_init(cb_a, cb_b);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out0, ONE_TILE);
    cb_reserve_back(cb_out1, ONE_TILE);

    tile_regs_acquire();
    add_tiles(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out0);
    pack_tile(0, cb_out1);
    tile_regs_release();

    cb_push_back(cb_out0, ONE_TILE);
    cb_push_back(cb_out1, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

FORCE_INLINE void unary_bcast_col_to_cb(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);
    unary_bcast_init<BroadcastType::COL>(cb_in, cb_out);

    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    unary_bcast<BroadcastType::COL>(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

FORCE_INLINE void mul_bcast_rows_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_bcast_rows_init_short(cb_a, cb_b);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_rows(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    cb_pop_front(cb_b, ONE_TILE);
}

template <bool pop_b>
FORCE_INLINE void mul_bcast_rows_to_cb(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
    reconfig_data_format(cb_a, cb_b);
    pack_reconfig_data_format(cb_out);
    mul_bcast_rows_init_short(cb_a, cb_b);

    cb_wait_front(cb_a, ONE_TILE);
    cb_wait_front(cb_b, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_rows(cb_a, cb_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_a, ONE_TILE);
    if constexpr (pop_b) {
        cb_pop_front(cb_b, ONE_TILE);
    }
}

FORCE_INLINE void reduce_rows_to_cb(uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_out) {
    reconfig_data_format(cb_in, cb_scaler);
    pack_reconfig_data_format(cb_out);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_out);

    cb_wait_front(cb_in, ONE_TILE);
    cb_reserve_back(cb_out, ONE_TILE);

    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    reduce_uninit();
    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

FORCE_INLINE void emit_y_reduce_cb_path(
    uint32_t state_next_cb, uint32_t c_row_cb, uint32_t scratch_cb, uint32_t reduce_scaler_cb, uint32_t y_vec_cb) {
    mul_bcast_rows_to_cb<false>(state_next_cb, c_row_cb, scratch_cb);
    reduce_rows_to_cb(scratch_cb, reduce_scaler_cb, y_vec_cb);
}

#ifdef MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE
FORCE_INLINE void emit_y_sfpu_path(uint32_t state_next_cb, uint32_t c_row_cb, uint32_t y_vec_cb) {
    reconfig_data_format(state_next_cb, c_row_cb);
    pack_reconfig_data_format(y_vec_cb);
    mul_bcast_rows_init_short(state_next_cb, c_row_cb);

    cb_wait_front(state_next_cb, ONE_TILE);
    cb_wait_front(c_row_cb, ONE_TILE);
    cb_reserve_back(y_vec_cb, ONE_TILE);

    tile_regs_acquire();
    mul_tiles_bcast_rows(state_next_cb, c_row_cb, 0, 0, 0);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
    PACK((llk_pack_reduce_mask_config<false, ReduceDim::REDUCE_ROW>()));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, y_vec_cb);
    PACK((llk_pack_reduce_mask_clear()));
    tile_regs_release();

    cb_push_back(y_vec_cb, ONE_TILE);
    cb_pop_front(state_next_cb, ONE_TILE);
}
#endif

FORCE_INLINE void emit_y_to_cb(
    uint32_t state_next_cb, uint32_t c_row_cb, uint32_t scratch_cb, uint32_t reduce_scaler_cb, uint32_t y_vec_cb) {
#ifdef MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE
    emit_y_sfpu_path(state_next_cb, c_row_cb, y_vec_cb);
#else
    emit_y_reduce_cb_path(state_next_cb, c_row_cb, scratch_cb, reduce_scaler_cb, y_vec_cb);
#endif
}

void kernel_main() {
    constexpr uint32_t decay_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_col_cb = get_compile_time_arg_val(1);
    constexpr uint32_t b_row_cb = get_compile_time_arg_val(2);
    constexpr uint32_t c_row_cb = get_compile_time_arg_val(3);
    constexpr uint32_t zero_state_cb = get_compile_time_arg_val(4);
    constexpr uint32_t state_acc_cb = get_compile_time_arg_val(5);
    constexpr uint32_t state_scaled_cb = get_compile_time_arg_val(6);
    constexpr uint32_t outer_cb = get_compile_time_arg_val(7);
    constexpr uint32_t state_next_cb = get_compile_time_arg_val(8);
    constexpr uint32_t y_vec_cb = get_compile_time_arg_val(9);
    constexpr uint32_t final_state_cb = get_compile_time_arg_val(10);
    constexpr uint32_t x_full_cb = get_compile_time_arg_val(11);
    constexpr uint32_t reduce_scaler_cb = get_compile_time_arg_val(12);

    const uint32_t scan_count = get_arg_val<uint32_t>(0);
    const uint32_t chunk_size = get_arg_val<uint32_t>(1);
    const uint32_t num_heads = get_arg_val<uint32_t>(2);
    const uint32_t p_tiles = get_arg_val<uint32_t>(3);

    binary_op_init_common(state_acc_cb, decay_cb, state_next_cb);
    pack_reconfig_data_format(state_next_cb);
#ifndef MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE
    cb_wait_front(reduce_scaler_cb, ONE_TILE);
#endif

#ifdef MAMBA_CHUNK_SCAN_MERGE_P_TILES
    for (uint32_t local = 0; local < scan_count; ++local) {
        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                copy_tile_to_cb<true>(zero_state_cb, state_acc_cb);
            }
        }

        for (uint32_t t = 0; t < chunk_size; ++t) {
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                    if (p_tile + 1 == p_tiles) {
                        mul_bcast_scalar_to_cb_reuse<true>(state_acc_cb, decay_cb, state_scaled_cb);
                    } else {
                        mul_bcast_scalar_to_cb_reuse<false>(state_acc_cb, decay_cb, state_scaled_cb);
                    }
                    unary_bcast_col_to_cb(x_col_cb, x_full_cb);
                    mul_bcast_rows_to_cb<false>(x_full_cb, b_row_cb, outer_cb);
                    add_tiles_to_two_cbs(state_scaled_cb, outer_cb, state_next_cb, state_acc_cb);
                    emit_y_to_cb(state_next_cb, c_row_cb, outer_cb, reduce_scaler_cb, y_vec_cb);
                }
            }
            cb_pop_front(b_row_cb, ONE_TILE);
            cb_pop_front(c_row_cb, ONE_TILE);
        }

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (uint32_t p_tile = 0; p_tile < p_tiles; ++p_tile) {
                copy_tile_to_cb<true>(state_acc_cb, final_state_cb);
            }
        }
    }
#else
    for (uint32_t local = 0; local < scan_count; ++local) {
        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            copy_tile_to_cb<true>(zero_state_cb, state_acc_cb);
        }

        for (uint32_t t = 0; t < chunk_size; ++t) {
            for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
                mul_bcast_scalar_to_cb(state_acc_cb, decay_cb, state_scaled_cb);
                unary_bcast_col_to_cb(x_col_cb, x_full_cb);
                mul_bcast_rows_to_cb<false>(x_full_cb, b_row_cb, outer_cb);
                add_tiles_to_two_cbs(state_scaled_cb, outer_cb, state_next_cb, state_acc_cb);
                emit_y_to_cb(state_next_cb, c_row_cb, outer_cb, reduce_scaler_cb, y_vec_cb);
            }
            cb_pop_front(b_row_cb, ONE_TILE);
            cb_pop_front(c_row_cb, ONE_TILE);
        }

        for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            copy_tile_to_cb<true>(state_acc_cb, final_state_cb);
        }
    }
#endif

    volatile uint32_t sink = scan_count + chunk_size + num_heads + final_state_cb;
    (void)sink;
}
