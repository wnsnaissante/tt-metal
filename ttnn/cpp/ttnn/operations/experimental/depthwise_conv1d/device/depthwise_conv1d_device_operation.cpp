// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::depthwise_conv1d {

void DepthwiseConv1dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;

    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(weight.storage_type() == StorageType::DEVICE, "Weight must be on device");
    TT_FATAL(input.device() == weight.device(), "Input and weight must be on the same device");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input must be ROW_MAJOR layout");
    TT_FATAL(weight.layout() == Layout::ROW_MAJOR, "Weight must be ROW_MAJOR layout");

    const auto& input_shape = input.logical_shape();
    const auto& weight_shape = weight.logical_shape();

    TT_FATAL(input_shape.rank() == 4, "Input must be rank 4: [B, 1, L_padded, C]");
    TT_FATAL(input_shape[1] == 1, "Input dim 1 must be 1");
    TT_FATAL(
        input_shape[2] == attrs.sequence_length + attrs.kernel_size - 1,
        "Input L_padded must be sequence_length + kernel_size - 1");
    TT_FATAL(input_shape[3] == attrs.channels, "Input channels must match attrs.channels");

    TT_FATAL(weight_shape.rank() == 4, "Weight must be rank 4: [1, 1, K, C]");
    TT_FATAL(weight_shape[0] == 1 && weight_shape[1] == 1, "Weight dims 0,1 must be 1");
    TT_FATAL(weight_shape[2] == attrs.kernel_size, "Weight dim 2 must match kernel_size");
    TT_FATAL(weight_shape[3] == attrs.channels, "Weight dim 3 must match channels");

    if (tensor_args.bias_tensor.has_value()) {
        const auto& bias = *tensor_args.bias_tensor;
        TT_FATAL(bias.storage_type() == StorageType::DEVICE, "Bias must be on device");
        TT_FATAL(bias.logical_shape().volume() == attrs.channels, "Bias volume must match channels");
    }

    TT_FATAL(attrs.kernel_size > 0 && attrs.kernel_size <= 16, "kernel_size must be in [1, 16]");
    TT_FATAL(attrs.channels > 0, "channels must be > 0");
    TT_FATAL(attrs.sequence_length > 0, "sequence_length must be > 0");

    TT_FATAL(attrs.channels % 32 == 0, "channels must be a multiple of 32 for tile-width alignment");
}

DepthwiseConv1dDeviceOperation::spec_return_value_t DepthwiseConv1dDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const uint32_t batch = input.logical_shape()[0];

    ttnn::Shape output_shape({batch, 1, attrs.sequence_length, attrs.channels});
    const uint32_t padded_sequence_length = ((attrs.sequence_length + 31) / 32) * 32;
    ttnn::Shape padded_output_shape({batch, 1, padded_sequence_length, attrs.channels});
    auto output_mem_config = input.memory_config();

    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input.dtype(), PageConfig(Layout::TILE), output_mem_config, output_shape, padded_output_shape));
}

DepthwiseConv1dDeviceOperation::tensor_return_value_t DepthwiseConv1dDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(attrs, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::experimental::depthwise_conv1d

namespace ttnn::prim {

Tensor depthwise_conv1d_forward(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t kernel_size,
    uint32_t channels,
    uint32_t sequence_length,
    const std::optional<Tensor>& bias,
    bool silu_activation) {
    using Op = ttnn::operations::experimental::depthwise_conv1d::DepthwiseConv1dDeviceOperation;

    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .kernel_size = kernel_size,
            .channels = channels,
            .sequence_length = sequence_length,
            .has_bias = bias.has_value(),
            .silu_activation = silu_activation},
        Op::tensor_args_t{.input_tensor = input_tensor, .weight_tensor = weight_tensor, .bias_tensor = bias});
}

}  // namespace ttnn::prim
