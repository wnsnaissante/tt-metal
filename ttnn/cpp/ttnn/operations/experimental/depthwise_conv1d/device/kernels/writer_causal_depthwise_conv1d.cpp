// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t block_height = get_compile_time_arg_val(0);
    constexpr uint32_t sequence_length = get_compile_time_arg_val(1);
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(2);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out_rm0 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_rm1 = get_compile_time_arg_val(6);
    constexpr uint32_t debug_version = get_compile_time_arg_val(7);
    constexpr auto output_args = TensorAccessorArgs<8>();

    DPRINT << "dwconv1d writer entry v=" << debug_version << ENDL();

    const uint32_t output_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const auto output_accessor = TensorAccessor(output_args, output_dram_addr, stick_nbytes);

    DPRINT << "dwconv1d writer start v=" << debug_version << " blocks=" << num_blocks << " start_block=" << start_block
           << ENDL();

    for (uint32_t local_block = 0; local_block < num_blocks; ++local_block) {
        const uint32_t block = start_block + local_block;
        const uint32_t batch_idx = block / blocks_per_batch;
        const uint32_t block_in_batch = block % blocks_per_batch;
        const uint32_t start_pos = block_in_batch * block_height;
        const uint32_t valid_rows =
            sequence_length > start_pos
                ? ((sequence_length - start_pos) < block_height ? (sequence_length - start_pos) : block_height)
                : 0;

        const uint32_t out_cb = local_block % 2 == 0 ? cb_out_rm0 : cb_out_rm1;
        cb_wait_front(out_cb, width_tiles);
        const uint32_t out_l1_addr = get_read_ptr(out_cb);

        for (uint32_t row = 0; row < valid_rows; ++row) {
            const uint32_t output_page = batch_idx * sequence_length + start_pos + row;
            noc_async_write_page(output_page, output_accessor, out_l1_addr + row * stick_nbytes);
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, width_tiles);

        if (local_block == 0) {
            DPRINT << "dwconv1d writer first block done block=" << block << ENDL();
        }
    }

    DPRINT << "dwconv1d writer done" << ENDL();
}
