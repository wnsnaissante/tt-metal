// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"

namespace {

FORCE_INLINE uint32_t output_page_id(
    uint32_t batch_idx,
    uint32_t sequence_tile_idx,
    uint32_t feature_tile_idx,
    uint32_t sequence_tiles,
    uint32_t feature_tiles) {
    return batch_idx * sequence_tiles * feature_tiles + sequence_tile_idx * feature_tiles + feature_tile_idx;
}

FORCE_INLINE void decode_tile_id(
    uint32_t tile_id,
    uint32_t sequence_tiles,
    uint32_t feature_tiles,
    uint32_t& batch_idx,
    uint32_t& sequence_tile_idx,
    uint32_t& feature_tile_idx) {
    const uint32_t tiles_per_batch = sequence_tiles * feature_tiles;
    const uint32_t tile_in_batch = tile_id % tiles_per_batch;
    batch_idx = tile_id / tiles_per_batch;
    sequence_tile_idx = tile_in_batch / feature_tiles;
    feature_tile_idx = tile_in_batch % feature_tiles;
}

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t sequence_tiles = get_arg_val<uint32_t>(3);
    const uint32_t feature_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_out);

    for (uint32_t local_tile = 0; local_tile < num_tiles; ++local_tile) {
        uint32_t batch_idx = 0;
        uint32_t sequence_tile_idx = 0;
        uint32_t feature_tile_idx = 0;
        decode_tile_id(
            start_tile + local_tile, sequence_tiles, feature_tiles, batch_idx, sequence_tile_idx, feature_tile_idx);

        cb.wait_front(1);
        noc.async_write(
            cb,
            dst_accessor,
            page_bytes,
            {},
            {.page_id = output_page_id(batch_idx, sequence_tile_idx, feature_tile_idx, sequence_tiles, feature_tiles)});
        noc.async_writes_flushed();
        cb.pop_front(1);
    }
    noc.async_write_barrier();
}
