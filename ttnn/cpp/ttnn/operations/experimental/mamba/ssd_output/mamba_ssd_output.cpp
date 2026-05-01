// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_output.hpp"

#include "device/mamba_ssd_output_device_operation.hpp"

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
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::mamba_ssd_output(
        y_diag_bcthp,
        states_out_bchpn,
        c_blk_bctn,
        a_cumsum_bhct,
        x_orig_blk_bcthp,
        d_h,
        seq_len,
        pad_size,
        std::move(core_grid),
        memory_config);
}

Tensor mamba_ssd_output(
    const Tensor& y_diag_bcthp,
    const Tensor& states_out_bchpn,
    const Tensor& c_blk_bctn,
    const Tensor& a_cumsum_bhct,
    const Tensor& x_orig_blk_bcthp,
    const Tensor& d_h,
    uint32_t seq_len,
    uint32_t pad_size,
    std::optional<CoreGrid> core_grid) {
    return mamba_ssd_output(
        y_diag_bcthp,
        states_out_bchpn,
        c_blk_bctn,
        a_cumsum_bhct,
        x_orig_blk_bcthp,
        d_h,
        seq_len,
        pad_size,
        std::move(core_grid),
        std::nullopt);
}

}  // namespace ttnn::experimental
