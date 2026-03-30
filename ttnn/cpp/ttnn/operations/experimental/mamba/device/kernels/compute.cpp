// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t TILE_HW = 32 * 32;

FORCE_INLINE void mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
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

FORCE_INLINE void sum(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
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

FORCE_INLINE void exp(uint32_t cb_in, uint32_t cb_out) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    copy_tile_to_dst_init_short(cb_in);
    exp_tile_init();

    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}

FORCE_INLINE void copy(uint32_t cb_in, uint32_t cb_out, uint32_t num_input_units = 1) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(cb_out);

    copy_tile_to_dst_init_short(cb_in);

    cb_wait_front(cb_in, num_input_units);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_push_back(cb_out, 1);
}

template <bool exp_a>
FORCE_INLINE void compute_recurrence_tile(
    uint32_t cb_a,
    uint32_t cb_x,
    uint32_t cb_h_prev,
    uint32_t cb_exp_a,
    uint32_t cb_ah,
    uint32_t cb_h,
    uint32_t cb_states_out,
    uint32_t cb_h_acc,
    uint32_t cb_final_state,
    bool emit_final_state) {
    copy(cb_h_acc, cb_h_prev);
    copy(cb_h_acc, cb_states_out);
    cb_wait_front(cb_h_acc, 1);
    cb_pop_front(cb_h_acc, 1);

    if constexpr (exp_a) {
        exp(cb_a, cb_exp_a);
        mul(cb_exp_a, cb_h_prev, cb_ah);
    } else {
        mul(cb_a, cb_h_prev, cb_ah);
    }
    sum(cb_ah, cb_x, cb_h);
    if (emit_final_state) {
        copy(cb_h, cb_final_state);
    }
    copy(cb_h, cb_h_acc);
    cb_pop_front(cb_h, 1);
}

void kernel_main() {
    // Compile-time args:
    // 0: CB index for states_bhcpn
    // 1: CB index for initial_states
    // 2: CB index for A_end_bhc
    // 3: CB index for states_out
    // 4: CB index for final_state
    // 5: CB index for h_prev
    // 6: CB index for ah
    // 7: CB index for h
    // 8: CB index for exp(a)
    // 9: CB index for hidden-state accumulator
    // 10: recurrence kernel ABI version (non-zero enables fused recurrence path)
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t initial_states_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t a_end_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t states_out_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t final_state_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t h_prev_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t ah_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t h_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t exp_a_cb_index = get_compile_time_arg_val(8);
    constexpr uint32_t h_acc_cb_index = get_compile_time_arg_val(9);
    constexpr bool enable_recurrence = get_compile_time_arg_val(10) != 0;

    // Runtime args:
    // 0-3: chunk count, state dimensions, hidden tiles assigned to this core
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);
    const uint32_t head_dim = get_arg_val<uint32_t>(1);
    const uint32_t state_size = get_arg_val<uint32_t>(2);
    const uint32_t hidden_tile_count = get_arg_val<uint32_t>(3);
    const uint32_t head_tiles = (head_dim + 31) / 32;
    const uint32_t state_tiles = (state_size + 31) / 32;
    const uint32_t tiles_per_bh = head_tiles * state_tiles;

    if constexpr (enable_recurrence) {
        binary_op_init_common(a_end_cb_index, states_cb_index, states_out_cb_index);

        for (uint32_t hidden_tile = 0; hidden_tile < hidden_tile_count; ++hidden_tile) {
            copy(initial_states_cb_index, h_acc_cb_index);
            cb_wait_front(initial_states_cb_index, 1);
            cb_pop_front(initial_states_cb_index, 1);
        }

        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const bool emit_final_state = chunk_idx + 1 == num_chunks;
            for (uint32_t hidden_tile = 0; hidden_tile < hidden_tile_count; ++hidden_tile) {
                compute_recurrence_tile<true>(
                    a_end_cb_index,
                    states_cb_index,
                    h_prev_cb_index,
                    exp_a_cb_index,
                    ah_cb_index,
                    h_cb_index,
                    states_out_cb_index,
                    h_acc_cb_index,
                    final_state_cb_index,
                    emit_final_state);
            }
        }
    }

    volatile uint32_t sink = head_dim + state_size + states_cb_index + initial_states_cb_index + a_end_cb_index +
                             states_out_cb_index + final_state_cb_index + h_prev_cb_index + ah_cb_index + h_cb_index +
                             exp_a_cb_index + h_acc_cb_index + head_tiles + state_tiles + tiles_per_bh +
                             hidden_tile_count + num_chunks;
    (void)sink;
}
