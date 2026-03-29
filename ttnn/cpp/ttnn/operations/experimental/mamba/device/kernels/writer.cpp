// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;

FORCE_INLINE uint32_t states_out_tile_id(
    uint32_t batch_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t head_tiles,
    uint32_t head_tile_idx,
    uint32_t state_tiles,
    uint32_t state_tile_idx) {
    return (((((batch_idx * num_chunks) + chunk_idx) * num_heads + head_idx) * head_tiles + head_tile_idx) *
            state_tiles) +
           state_tile_idx;
}

FORCE_INLINE uint32_t final_state_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t head_tiles,
    uint32_t head_tile_idx,
    uint32_t state_tiles,
    uint32_t state_tile_idx) {
    return ((((batch_idx * num_heads) + head_idx) * head_tiles + head_tile_idx) * state_tiles) + state_tile_idx;
}

struct HiddenTileCoord {
    uint32_t batch_idx;
    uint32_t head_idx;
    uint32_t head_tile_idx;
    uint32_t state_tile_idx;
};

FORCE_INLINE HiddenTileCoord
decode_hidden_tile(uint32_t global_hidden_tile, uint32_t num_heads, uint32_t state_tiles, uint32_t tiles_per_bh) {
    const uint32_t bh_index = global_hidden_tile / tiles_per_bh;
    const uint32_t tile_local = global_hidden_tile % tiles_per_bh;
    return HiddenTileCoord{
        .batch_idx = bh_index / num_heads,
        .head_idx = bh_index % num_heads,
        .head_tile_idx = tile_local / state_tiles,
        .state_tile_idx = tile_local % state_tiles,
    };
}

void kernel_main() {
    constexpr uint32_t states_out_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t final_state_cb_index = get_compile_time_arg_val(1);

    const uint32_t states_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t final_state_addr = get_arg_val<uint32_t>(1);
    const uint32_t batch_size = get_arg_val<uint32_t>(2);
    const uint32_t num_heads = get_arg_val<uint32_t>(3);
    const uint32_t num_chunks = get_arg_val<uint32_t>(4);
    const uint32_t head_dim = get_arg_val<uint32_t>(5);
    const uint32_t state_size = get_arg_val<uint32_t>(6);
    const uint32_t hidden_tile_start = get_arg_val<uint32_t>(7);
    const uint32_t hidden_tile_count = get_arg_val<uint32_t>(8);

    const uint32_t tile_size_bytes = get_tile_size(states_out_cb_index);
    constexpr auto states_out_args = TensorAccessorArgs<2>();
    const auto states_out = TensorAccessor(states_out_args, states_out_addr, tile_size_bytes);
    constexpr auto final_state_args = TensorAccessorArgs<states_out_args.next_compile_time_args_offset()>();
    const auto final_state = TensorAccessor(final_state_args, final_state_addr, tile_size_bytes);

    const uint32_t head_tiles = (head_dim + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t state_tiles = (state_size + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t tiles_per_bh = head_tiles * state_tiles;

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        for (uint32_t hidden_tile_local = 0; hidden_tile_local < hidden_tile_count; ++hidden_tile_local) {
            const auto coord =
                decode_hidden_tile(hidden_tile_start + hidden_tile_local, num_heads, state_tiles, tiles_per_bh);
            const uint32_t tile_id = states_out_tile_id(
                coord.batch_idx,
                num_chunks,
                chunk_idx,
                num_heads,
                coord.head_idx,
                head_tiles,
                coord.head_tile_idx,
                state_tiles,
                coord.state_tile_idx);

            cb_wait_front(states_out_cb_index, 1);
            const uint32_t l1_read_addr = get_read_ptr(states_out_cb_index);
            noc_async_write_tile(tile_id, states_out, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(states_out_cb_index, 1);
        }
    }

    for (uint32_t hidden_tile_local = 0; hidden_tile_local < hidden_tile_count; ++hidden_tile_local) {
        const auto coord =
            decode_hidden_tile(hidden_tile_start + hidden_tile_local, num_heads, state_tiles, tiles_per_bh);
        const uint32_t tile_id = final_state_tile_id(
            coord.batch_idx,
            num_heads,
            coord.head_idx,
            head_tiles,
            coord.head_tile_idx,
            state_tiles,
            coord.state_tile_idx);

        cb_wait_front(final_state_cb_index, 1);
        const uint32_t l1_read_addr = get_read_ptr(final_state_cb_index);
        noc_async_write_tile(tile_id, final_state, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(final_state_cb_index, 1);
    }
}
