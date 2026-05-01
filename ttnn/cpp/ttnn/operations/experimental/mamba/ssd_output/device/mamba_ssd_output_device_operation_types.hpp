// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDOutputParams {
    tt::tt_metal::MemoryConfig memory_config = ttnn::DRAM_MEMORY_CONFIG;
    bool has_core_grid = false;
    uint32_t core_grid_x = 0;
    uint32_t core_grid_y = 0;
    uint32_t seq_len = 0;
    uint32_t pad_size = 0;
};

struct MambaSSDOutputInputs {
    Tensor y_diag_bcthp;
    Tensor states_out_bchpn;
    Tensor c_blk_bctn;
    Tensor a_cumsum_bhct;
    Tensor x_orig_blk_bcthp;
    Tensor d_h;
};

struct MambaSSDOutputKernelConfig {
    uint32_t batch_size = 0;
    uint32_t num_chunks = 0;
    uint32_t chunk_size = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t state_size = 0;
    uint32_t seq_len = 0;
    uint32_t pad_size = 0;
    uint32_t hidden_dim = 0;
};

}  // namespace ttnn::experimental::prim
