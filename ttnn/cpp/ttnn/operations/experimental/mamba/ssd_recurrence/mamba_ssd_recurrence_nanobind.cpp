// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_recurrence_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/mamba/ssd_recurrence/mamba_ssd_recurrence.hpp"

namespace ttnn::operations::experimental::mamba::detail {

void bind_mamba_ssd_recurrence(nb::module_& mod) {
    auto mamba_ssd_recurrence_fn = static_cast<std::vector<ttnn::Tensor> (*)(
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        const ttnn::Tensor&,
        std::optional<ttnn::CoreGrid>,
        const std::optional<ttnn::MemoryConfig>&)>(&ttnn::experimental::mamba_ssd_recurrence);
    ttnn::bind_function<"mamba_ssd_recurrence", "ttnn.experimental.">(
        mod,
        R"doc(
        Placeholder shell for the fused Mamba SSD recurrence op.

        Args:
            states_bhcpn (ttnn.Tensor): Input state tensor.
            initial_states (ttnn.Tensor): Initial recurrence state tensor.
            a_end_bhc (ttnn.Tensor): Recurrence coefficient tensor.

        Returns:
            List[ttnn.Tensor]: Placeholder output list in the order [states_bhcpn_out, final_state].
        )doc",
        mamba_ssd_recurrence_fn,
        nb::arg("states_bhcpn"),
        nb::arg("initial_states"),
        nb::arg("a_end_bhc"),
        nb::kw_only(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::mamba::detail
