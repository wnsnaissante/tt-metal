// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid);

}  // namespace ttnn::experimental
