// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_forward_device_operation.hpp"

#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

void DepthwiseConv1dForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x_blf;
    const auto& conv_state = tensor_args.conv_state_bfk;
    const auto& weight = tensor_args.weight_1fk;
    const auto& bias = tensor_args.bias_11f;

    TT_FATAL(x.device() != nullptr, "x_blf must be on device");
    TT_FATAL(conv_state.device() != nullptr, "conv_state_bfk must be on device");
    TT_FATAL(weight.device() != nullptr, "weight_1fk must be on device");
    TT_FATAL(bias.device() != nullptr, "bias_11f must be on device");
    TT_FATAL(
        x.device() == conv_state.device() && x.device() == weight.device() && x.device() == bias.device(),
        "all tensors must share a device");

    TT_FATAL(x.layout() == Layout::TILE, "x_blf must be tiled");
    TT_FATAL(conv_state.layout() == Layout::TILE, "conv_state_bfk must be tiled");
    TT_FATAL(weight.layout() == Layout::TILE, "weight_1fk must be tiled");
    TT_FATAL(bias.layout() == Layout::TILE, "bias_11f must be tiled");
    TT_FATAL(
        x.dtype() == DataType::BFLOAT16 && conv_state.dtype() == DataType::BFLOAT16 &&
            weight.dtype() == DataType::BFLOAT16 && bias.dtype() == DataType::BFLOAT16,
        "depthwise_conv1d forward custom op currently supports bfloat16 only");

    const auto& x_shape = x.logical_shape();
    const auto& state_shape = conv_state.logical_shape();
    const auto& weight_shape = weight.logical_shape();
    const auto& bias_shape = bias.logical_shape();

    TT_FATAL(x_shape.rank() == 3, "x_blf must have shape [B, L_padded, F]");
    TT_FATAL(state_shape.rank() == 3, "conv_state_bfk must have shape [B, F, K-1]");
    TT_FATAL(weight_shape.rank() == 3, "weight_1fk must have shape [1, F, K]");
    TT_FATAL(bias_shape.rank() == 3, "bias_11f must have shape [1, 1, F]");
    TT_FATAL(x_shape[2] == args.features, "x_blf feature dimension mismatch");
    TT_FATAL(x_shape[1] >= args.kernel_size, "x_blf sequence dimension must be >= kernel_size");
    TT_FATAL(
        state_shape[0] == x_shape[0] && state_shape[1] == args.features && state_shape[2] == args.kernel_size - 1,
        "conv_state_bfk shape mismatch");
    TT_FATAL(
        weight_shape[0] == 1 && weight_shape[1] == args.features && weight_shape[2] == args.kernel_size,
        "weight_1fk shape mismatch");
    TT_FATAL(bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == args.features, "bias_11f shape mismatch");
    TT_FATAL(args.kernel_size >= 2 && args.kernel_size <= 4, "forward custom op currently supports 2 <= k <= 4");
    TT_FATAL(
        args.output_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM,
        "forward custom op expects DRAM outputs");
}

void DepthwiseConv1dForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

TensorSpec DepthwiseConv1dForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x_shape = tensor_args.x_blf.logical_shape();
    const uint32_t output_sequence_length = x_shape[1] - (args.kernel_size - 1);
    return TensorSpec(
        ttnn::Shape({x_shape[0], 1, output_sequence_length, args.features}),
        tt::tt_metal::TensorLayout(
            tensor_args.x_blf.dtype(), tt::tt_metal::PageConfig(Layout::TILE), args.output_memory_config));
}

Tensor DepthwiseConv1dForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.x_blf.device());
}

ttsl::hash::hash_t DepthwiseConv1dForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<DepthwiseConv1dForwardDeviceOperation>(
        args.output_memory_config,
        args.features,
        args.kernel_size,
        tensor_args.x_blf.dtype(),
        tensor_args.x_blf.memory_config(),
        tensor_args.x_blf.padded_shape(),
        tensor_args.conv_state_bfk.padded_shape(),
        tensor_args.weight_1fk.padded_shape(),
        tensor_args.bias_11f.padded_shape());
}

Tensor depthwise_conv1d_forward_custom(
    const Tensor& x_blf,
    const Tensor& conv_state_bfk,
    const Tensor& weight_1fk,
    const Tensor& bias_11f,
    uint32_t features,
    uint32_t kernel_size) {
    using OperationType = DepthwiseConv1dForwardDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .output_memory_config = ttnn::DRAM_MEMORY_CONFIG, .features = features, .kernel_size = kernel_size};
    auto tensor_args = OperationType::tensor_args_t{
        .x_blf = x_blf, .conv_state_bfk = conv_state_bfk, .weight_1fk = weight_1fk, .bias_11f = bias_11f};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim
