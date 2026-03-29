// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "depthwise_conv1d_forward_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DepthwiseConv1dForwardParams {
    MemoryConfig output_memory_config = ttnn::DRAM_MEMORY_CONFIG;
    uint32_t features = 0;
    uint32_t kernel_size = 0;
};

struct DepthwiseConv1dForwardInputs {
    const Tensor& x_blf;
    const Tensor& conv_state_bfk;
    const Tensor& weight_1fk;
    const Tensor& bias_11f;
};

struct DepthwiseConv1dForwardDeviceOperation {
    using operation_attributes_t = DepthwiseConv1dForwardParams;
    using tensor_args_t = DepthwiseConv1dForwardInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DepthwiseConv1dForwardProgramFactory>;
    using shared_variables_t = DepthwiseConv1dForwardProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor depthwise_conv1d_forward_custom(
    const Tensor& x_blf,
    const Tensor& conv_state_bfk,
    const Tensor& weight_1fk,
    const Tensor& bias_11f,
    uint32_t features,
    uint32_t kernel_size);

}  // namespace ttnn::experimental::prim
