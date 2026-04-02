// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t block_height = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_size = get_compile_time_arg_val(1);
    constexpr uint32_t channels = get_compile_time_arg_val(2);
    constexpr uint32_t padded_sequence_length = get_compile_time_arg_val(3);
    constexpr uint32_t sequence_length = get_compile_time_arg_val(4);
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(5);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(6);
    constexpr uint32_t cb_act_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_weight_rm = get_compile_time_arg_val(8);
    constexpr uint32_t cb_bias_rm = get_compile_time_arg_val(9);
    constexpr uint32_t has_bias = get_compile_time_arg_val(10);
    constexpr uint32_t debug_version = get_compile_time_arg_val(11);
    constexpr auto input_args = TensorAccessorArgs<12>();
    constexpr auto weight_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    DPRINT << "dwconv1d reader entry v=" << debug_version << ENDL();

    const uint32_t input_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_dram_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_block = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(4);

    DPRINT << "dwconv1d reader start v=" << debug_version << " blocks=" << num_blocks << " start_block=" << start_block
           << ENDL();

    const auto input_accessor = TensorAccessor(input_args, input_dram_addr, stick_nbytes);
    const auto weight_accessor = TensorAccessor(weight_args, weight_dram_addr, stick_nbytes);
    const auto bias_accessor = TensorAccessor(bias_args, bias_dram_addr, stick_nbytes);
    DPRINT << "dwconv1d reader accessors ready v=" << debug_version << ENDL();

    cb_reserve_back(cb_weight_rm, kernel_size * block_height);
    uint32_t weight_write_addr = get_write_ptr(cb_weight_rm);
    for (uint32_t k = 0; k < kernel_size; ++k) {
        for (uint32_t row = 0; row < block_height; ++row) {
            noc_async_read_page(k, weight_accessor, weight_write_addr + (k * block_height + row) * stick_nbytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_weight_rm, kernel_size * block_height);
    DPRINT << "dwconv1d reader weight ready" << ENDL();

    if constexpr (has_bias) {
        cb_reserve_back(cb_bias_rm, block_height);
        uint32_t bias_write_addr = get_write_ptr(cb_bias_rm);
        for (uint32_t row = 0; row < block_height; ++row) {
            noc_async_read_page(0, bias_accessor, bias_write_addr + row * stick_nbytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_bias_rm, block_height);
    }

    for (uint32_t local_block = 0; local_block < num_blocks; ++local_block) {
        const uint32_t block = start_block + local_block;
        const uint32_t batch_idx = block / blocks_per_batch;
        const uint32_t block_in_batch = block % blocks_per_batch;
        const uint32_t start_pos = block_in_batch * block_height;
        const uint32_t remaining_rows = sequence_length > start_pos ? (sequence_length - start_pos) : 0;
        const uint32_t valid_rows = remaining_rows < block_height ? remaining_rows : block_height;

        cb_reserve_back(cb_act_rm, kernel_size * block_height);
        uint32_t act_write_addr = get_write_ptr(cb_act_rm);
        for (uint32_t k = 0; k < kernel_size; ++k) {
            for (uint32_t row = 0; row < valid_rows; ++row) {
                const uint32_t input_page = batch_idx * padded_sequence_length + start_pos + row + k;
                noc_async_read_page(
                    input_page, input_accessor, act_write_addr + (k * block_height + row) * stick_nbytes);
            }
        }
        noc_async_read_barrier();

        if (valid_rows < block_height) {
            volatile tt_l1_ptr uint16_t* act_block_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(act_write_addr);
            for (uint32_t k = 0; k < kernel_size; ++k) {
                for (uint32_t row = valid_rows; row < block_height; ++row) {
                    auto* row_ptr = act_block_ptr + (k * block_height + row) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        row_ptr[c] = 0;
                    }
                }
            }
        }
        cb_push_back(cb_act_rm, kernel_size * block_height);

        if (local_block == 0) {
            DPRINT << "dwconv1d reader first block pushed block=" << block << ENDL();
        }
    }

    DPRINT << "dwconv1d reader done" << ENDL();
}
