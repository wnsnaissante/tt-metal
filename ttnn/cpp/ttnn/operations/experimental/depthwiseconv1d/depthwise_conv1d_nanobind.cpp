// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_nanobind.hpp"

#include <optional>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/depthwiseconv1d/depthwise_conv1d.hpp"

namespace ttnn::operations::experimental::depthwiseconv1d::detail {

void bind_depthwise_conv1d(nb::module_& mod) {
    ttnn::bind_function<"depthwise_conv1d", "ttnn.experimental.">(
        mod,
        R"doc(
        Executes causal depthwise conv1d with cached state update.

        Args:
            x (ttnn.Tensor): Input tensor of shape [B, L, F].
            conv_state (Optional[ttnn.Tensor]): Cached state tensor of shape [B, F, K-1].
            weight (ttnn.Tensor): Depthwise conv1d weight tensor.
            bias (ttnn.Tensor): Depthwise conv1d bias tensor.
            features (int): Active feature count.
            kernel_size (int): Depthwise convolution kernel size.

        Returns:
            List[ttnn.Tensor]: [output, new_conv_state]
        )doc",
        &ttnn::experimental::depthwise_conv1d,
        nb::arg("x"),
        nb::arg("conv_state") = nb::none(),
        nb::arg("weight"),
        nb::arg("bias"),
        nb::arg("features"),
        nb::arg("kernel_size"));
}

}  // namespace ttnn::operations::experimental::depthwiseconv1d::detail
