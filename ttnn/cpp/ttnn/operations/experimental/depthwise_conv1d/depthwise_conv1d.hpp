// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

ttnn::Tensor depthwise_conv1d(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t kernel_size,
    bool causal = true,
    const std::optional<Tensor>& bias = std::nullopt,
    bool silu_activation = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

std::vector<Tensor> depthwise_conv1d(
    const Tensor& x,
    const std::optional<Tensor>& conv_state,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t features,
    uint32_t kernel_size,
    bool silu_activation = false);

}  // namespace ttnn::experimental
