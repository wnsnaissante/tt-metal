// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDRecurrenceParams {
    tt::tt_metal::MemoryConfig memory_config = ttnn::DRAM_MEMORY_CONFIG;
    bool has_core_grid = false;
    uint32_t core_grid_x = 0;
    uint32_t core_grid_y = 0;
};

struct MambaSSDRecurrenceInputs {
    Tensor states_bhcpn;
    Tensor initial_states;
    Tensor a_end_bhc;
};

struct MambaSSDRecurrenceKernelConfig {
    uint32_t batch_size = 0;
    uint32_t num_heads = 0;
    uint32_t num_chunks = 0;
    uint32_t head_dim = 0;
    uint32_t state_size = 0;
    uint32_t hidden_dim = 0;
    uint32_t bh = 0;
    uint32_t state_plane = 0;
    uint32_t segment_hidden_dim_limit = 0;
    uint32_t num_hidden_segments = 0;
    uint32_t chunk_size = 0;
    uint32_t num_chunk_segments = 0;
};

}  // namespace ttnn::experimental::prim
