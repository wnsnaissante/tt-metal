// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "depthwise_conv1d.hpp"

namespace ttnn::operations::experimental::depthwise_conv1d::detail {

void bind_depthwise_conv1d(nb::module_& mod) {
    ttnn::bind_function<"depthwise_conv1d", "ttnn.experimental.">(
        mod,
        R"doc(
        Performs causal depthwise Conv1d.

        Stateless mode accepts tensors shaped [B, 1, L, C] or [B, L, 1, C] and applies
        left zero padding to guarantee causal-conv1d semantics.

        Stateful mode accepts [x, conv_state, weight, bias, features, kernel_size] and
        updates conv_state while preserving the same causal-conv1d semantics for cached flows.
        )doc",
        ttnn::overload_t(
            static_cast<ttnn::Tensor (*)(
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                uint32_t,
                bool,
                const std::optional<ttnn::Tensor>&,
                bool,
                const std::optional<ttnn::MemoryConfig>&)>(&ttnn::experimental::depthwise_conv1d),
            nb::arg("input_tensor"),
            nb::arg("weight_tensor"),
            nb::arg("kernel_size"),
            nb::kw_only(),
            nb::arg("causal") = true,
            nb::arg("bias") = nb::none(),
            nb::arg("silu_activation") = false,
            nb::arg("memory_config") = nb::none()),
        ttnn::overload_t(
            static_cast<std::vector<ttnn::Tensor> (*)(
                const ttnn::Tensor&,
                const std::optional<ttnn::Tensor>&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                uint32_t,
                uint32_t,
                bool)>(&ttnn::experimental::depthwise_conv1d),
            nb::arg("input_tensor"),
            nb::arg("conv_state") = nb::none(),
            nb::arg("weight_tensor"),
            nb::arg("bias_tensor"),
            nb::arg("features"),
            nb::arg("kernel_size"),
            nb::kw_only(),
            nb::arg("silu_activation") = false));
}

}  // namespace ttnn::operations::experimental::depthwise_conv1d::detail
