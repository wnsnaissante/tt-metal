// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

Tensor mamba_ssd_output(
    const Tensor& y_diag_bcthp,
    const Tensor& states_out_bchpn,
    const Tensor& c_blk_bctn,
    const Tensor& a_cumsum_bhct,
    const Tensor& x_orig_blk_bcthp,
    const Tensor& d_h,
    uint32_t seq_len,
    uint32_t pad_size,
    std::optional<CoreGrid> core_grid = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

Tensor mamba_ssd_output(
    const Tensor& y_diag_bcthp,
    const Tensor& states_out_bchpn,
    const Tensor& c_blk_bctn,
    const Tensor& a_cumsum_bhct,
    const Tensor& x_orig_blk_bcthp,
    const Tensor& d_h,
    uint32_t seq_len,
    uint32_t pad_size,
    std::optional<CoreGrid> core_grid);

}  // namespace ttnn::experimental
