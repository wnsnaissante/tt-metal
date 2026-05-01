// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_chunk_scan_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/mamba/ssd_chunk_scan/mamba_ssd_chunk_scan.hpp"

namespace ttnn::operations::experimental::mamba::detail {

void bind_mamba_ssd_chunk_scan(nb::module_& mod) {
    auto fn = static_cast<std::vector<ttnn::Tensor> (*)(
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        std::optional<ttnn::CoreGrid>,
        const std::optional<ttnn::MemoryConfig>&)>(&ttnn::experimental::mamba_ssd_chunk_scan);
    ttnn::bind_function<"mamba_ssd_chunk_scan", "ttnn.experimental.">(
        mod,
        R"doc(
        Fused Mamba SSD chunk scan for the Netmamba2 inference shape.

        Returns [y_diag_bcthp, states_bhcpn, a_cumsum_bhct].
        )doc",
        fn,
        nb::arg("x_blk_bcthp"),
        nb::arg("a_blk_bhct"),
        nb::arg("b_blk_bctn"),
        nb::arg("c_blk_bctn"),
        nb::kw_only(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::mamba::detail
