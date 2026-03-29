// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    h = h % TILE_HEIGHT;
    w = w % TILE_WIDTH;
    uint32_t idx = 0;
    if (w >= FACE_WIDTH) {
        w -= FACE_WIDTH;
        idx += FACE_HEIGHT * FACE_WIDTH;
    }
    if (h >= FACE_WIDTH) {
        h -= FACE_WIDTH;
        idx += FACE_HEIGHT * TILE_WIDTH;
    }
    idx += h * FACE_WIDTH + w;
    return idx;
}

FORCE_INLINE uint32_t states_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t num_chunks,
    uint32_t chunk_idx,
    uint32_t head_tiles,
    uint32_t head_tile_idx,
    uint32_t state_tiles,
    uint32_t state_tile_idx) {
    return (((((batch_idx * num_heads) + head_idx) * num_chunks + chunk_idx) * head_tiles + head_tile_idx) *
            state_tiles) +
           state_tile_idx;
}

FORCE_INLINE uint32_t initial_states_tile_id(
    uint32_t batch_idx,
    uint32_t num_heads,
    uint32_t head_idx,
    uint32_t head_tiles,
    uint32_t head_tile_idx,
    uint32_t state_tiles,
    uint32_t state_tile_idx) {
    return ((((batch_idx * num_heads) + head_idx) * head_tiles + head_tile_idx) * state_tiles) + state_tile_idx;
}

FORCE_INLINE uint32_t a_end_tile_id(
    uint32_t batch_idx,
    uint32_t num_head_tiles,
    uint32_t head_tile_idx,
    uint32_t num_chunk_tiles,
    uint32_t chunk_tile_idx) {
    return (((batch_idx * num_head_tiles) + head_tile_idx) * num_chunk_tiles) + chunk_tile_idx;
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

FORCE_INLINE void fill_broadcast_tile_from_scalar(uint32_t cb_id, uint32_t scalar_bits) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t h = 0; h < TILE_HEIGHT; ++h) {
        for (uint32_t w = 0; w < TILE_WIDTH; ++w) {
            ptr[get_tilized_idx(h, w)] = scalar_bits;
        }
    }
    cb_push_back(cb_id, 1);
}

void kernel_main() {
    constexpr uint32_t states_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t initial_states_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t a_end_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t a_end_scratch_cb_index = get_compile_time_arg_val(3);

    const uint32_t states_addr = get_arg_val<uint32_t>(0);
    const uint32_t initial_states_addr = get_arg_val<uint32_t>(1);
    const uint32_t a_end_addr = get_arg_val<uint32_t>(2);
    const uint32_t batch_size = get_arg_val<uint32_t>(3);
    const uint32_t num_heads = get_arg_val<uint32_t>(4);
    const uint32_t num_chunks = get_arg_val<uint32_t>(5);
    const uint32_t head_dim = get_arg_val<uint32_t>(6);
    const uint32_t state_size = get_arg_val<uint32_t>(7);
    const uint32_t hidden_tile_start = get_arg_val<uint32_t>(8);
    const uint32_t hidden_tile_count = get_arg_val<uint32_t>(9);

    const uint32_t tile_size_bytes = get_tile_size(states_cb_index);
    constexpr auto states_args = TensorAccessorArgs<4>();
    const auto states = TensorAccessor(states_args, states_addr, tile_size_bytes);
    constexpr auto initial_states_args = TensorAccessorArgs<states_args.next_compile_time_args_offset()>();
    const auto initial_states = TensorAccessor(initial_states_args, initial_states_addr, tile_size_bytes);
    constexpr auto a_end_args = TensorAccessorArgs<initial_states_args.next_compile_time_args_offset()>();
    const auto a_end = TensorAccessor(a_end_args, a_end_addr, tile_size_bytes);

    const uint32_t head_tiles = (head_dim + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t state_tiles = (state_size + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t tiles_per_bh = head_tiles * state_tiles;
    const uint32_t a_end_head_tiles = (num_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t a_end_chunk_tiles = (num_chunks + TILE_WIDTH - 1) / TILE_WIDTH;

    for (uint32_t hidden_tile_local = 0; hidden_tile_local < hidden_tile_count; ++hidden_tile_local) {
        const auto coord =
            decode_hidden_tile(hidden_tile_start + hidden_tile_local, num_heads, state_tiles, tiles_per_bh);
        const uint32_t tile_id = initial_states_tile_id(
            coord.batch_idx,
            num_heads,
            coord.head_idx,
            head_tiles,
            coord.head_tile_idx,
            state_tiles,
            coord.state_tile_idx);
        cb_reserve_back(initial_states_cb_index, 1);
        const uint32_t l1_write_addr = get_write_ptr(initial_states_cb_index);
        noc_async_read_tile(tile_id, initial_states, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(initial_states_cb_index, 1);
    }

    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const uint32_t chunk_tile_idx = chunk_idx / TILE_WIDTH;
        const uint32_t chunk_in_tile = chunk_idx % TILE_WIDTH;

        for (uint32_t hidden_tile_local = 0; hidden_tile_local < hidden_tile_count; ++hidden_tile_local) {
            const auto coord =
                decode_hidden_tile(hidden_tile_start + hidden_tile_local, num_heads, state_tiles, tiles_per_bh);
            const uint32_t a_tile_id = a_end_tile_id(
                coord.batch_idx, a_end_head_tiles, coord.head_idx / TILE_HEIGHT, a_end_chunk_tiles, chunk_tile_idx);

            cb_reserve_back(a_end_scratch_cb_index, 1);
            const uint32_t a_scratch_addr = get_write_ptr(a_end_scratch_cb_index);
            noc_async_read_tile(a_tile_id, a_end, a_scratch_addr);
            noc_async_read_barrier();

            const volatile tt_l1_ptr uint32_t* a_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_scratch_addr);
            const uint32_t a_scalar = a_ptr[get_tilized_idx(coord.head_idx % TILE_HEIGHT, chunk_in_tile)];

            const uint32_t tile_id = states_tile_id(
                coord.batch_idx,
                num_heads,
                coord.head_idx,
                num_chunks,
                chunk_idx,
                head_tiles,
                coord.head_tile_idx,
                state_tiles,
                coord.state_tile_idx);
            cb_reserve_back(states_cb_index, 1);
            const uint32_t states_l1_write_addr = get_write_ptr(states_cb_index);
            noc_async_read_tile(tile_id, states, states_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(states_cb_index, 1);
            fill_broadcast_tile_from_scalar(a_end_cb_index, a_scalar);

            cb_push_back(a_end_scratch_cb_index, 1);
            cb_pop_front(a_end_scratch_cb_index, 1);
        }
    }
}
