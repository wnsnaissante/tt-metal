// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_output_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/mamba/ssd_output/mamba_ssd_output.hpp"

namespace ttnn::operations::experimental::mamba::detail {

void bind_mamba_ssd_output(nb::module_& mod) {
    auto mamba_ssd_output_fn = static_cast<ttnn::Tensor (*)(
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        uint32_t,
        uint32_t,
        std::optional<ttnn::CoreGrid>,
        const std::optional<ttnn::MemoryConfig>&)>(&ttnn::experimental::mamba_ssd_output);
    ttnn::bind_function<"mamba_ssd_output", "ttnn.experimental.">(
        mod,
        R"doc(
        Skeleton shell for the fused Mamba SSD output op.

        Args:
            y_diag_bcthp (ttnn.Tensor): Diagonal output contribution tensor.
            states_out_bchpn (ttnn.Tensor): Recurrence output states tensor.
            c_blk_bctn (ttnn.Tensor): Canonicalized C tensor.
            a_cumsum_bhct (ttnn.Tensor): Canonicalized cumulative A tensor.
            x_orig_blk_bcthp (ttnn.Tensor): Original X tensor in chunked form.
            d_h (ttnn.Tensor): D residual tensor.
            seq_len (int): Unpadded sequence length.
            pad_size (int): Tail padding removed after flattening.

        Returns:
            ttnn.Tensor: Placeholder output tensor shaped [B, seq_len, H * P].
        )doc",
        mamba_ssd_output_fn,
        nb::arg("y_diag_bcthp"),
        nb::arg("states_out_bchpn"),
        nb::arg("c_blk_bctn"),
        nb::arg("a_cumsum_bhct"),
        nb::arg("x_orig_blk_bcthp"),
        nb::arg("d_h"),
        nb::kw_only(),
        nb::arg("seq_len"),
        nb::arg("pad_size"),
        nb::arg("core_grid") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::mamba::detail
