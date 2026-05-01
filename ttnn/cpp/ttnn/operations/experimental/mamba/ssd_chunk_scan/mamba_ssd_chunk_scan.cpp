// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_chunk_scan.hpp"

#include "device/mamba_ssd_chunk_scan_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::mamba_ssd_chunk_scan(
        x_blk_bcthp, a_blk_bhct, b_blk_bctn, c_blk_bctn, std::move(core_grid), memory_config);
}

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid) {
    return mamba_ssd_chunk_scan(x_blk_bcthp, a_blk_bhct, b_blk_bctn, c_blk_bctn, std::move(core_grid), std::nullopt);
}

}  // namespace ttnn::experimental
