// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t blocks_per_batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_nbytes = get_compile_time_arg_val(1);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cb_final_out = get_compile_time_arg_val(3);
    constexpr uint32_t debug_version = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();
    (void)debug_version;

    const uint32_t output_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const auto output_accessor = TensorAccessor(output_args, output_dram_addr, tile_nbytes);

    for (uint32_t local_block = 0; local_block < num_blocks; ++local_block) {
        const uint32_t block = start_block + local_block;
        const uint32_t batch_idx = block / blocks_per_batch;
        const uint32_t block_in_batch = block % blocks_per_batch;

        cb_wait_front(cb_final_out, width_tiles);
        const uint32_t out_l1_addr = get_read_ptr(cb_final_out);

        const uint32_t output_tile_base = (batch_idx * blocks_per_batch + block_in_batch) * width_tiles;
        for (uint32_t tile_idx = 0; tile_idx < width_tiles; ++tile_idx) {
            noc_async_write_page(output_tile_base + tile_idx, output_accessor, out_l1_addr + tile_idx * tile_nbytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_final_out, width_tiles);
    }
}
