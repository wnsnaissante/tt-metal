// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDChunkScanParams {
    tt::tt_metal::MemoryConfig memory_config = ttnn::DRAM_MEMORY_CONFIG;
    bool has_core_grid = false;
    uint32_t core_grid_x = 0;
    uint32_t core_grid_y = 0;
};

struct MambaSSDChunkScanInputs {
    Tensor x_blk_bcthp;
    Tensor a_blk_bhct;
    Tensor b_blk_bctn;
    Tensor c_blk_bctn;
};

struct MambaSSDChunkScanKernelConfig {
    uint32_t batch_size = 0;
    uint32_t num_chunks = 0;
    uint32_t chunk_size = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t state_size = 0;
    uint32_t p_tiles = 0;
    uint32_t t_tiles = 0;
};

}  // namespace ttnn::experimental::prim
