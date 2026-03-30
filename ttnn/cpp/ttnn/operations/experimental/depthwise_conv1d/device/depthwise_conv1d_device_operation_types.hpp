// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::depthwise_conv1d {

struct DepthwiseConv1dParams {
    uint32_t kernel_size = 0;
    uint32_t channels = 0;
    uint32_t sequence_length = 0;
    bool has_bias = false;
    bool silu_activation = false;
};

}  // namespace ttnn::operations::experimental::depthwise_conv1d
