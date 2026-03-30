// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d.hpp"

#include <array>
#include <functional>
#include <vector>

#include "device/depthwise_conv1d_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"

namespace ttnn::experimental {

namespace {

Tensor ensure_tensor_on_input_device(
    const Tensor& tensor, const Tensor& input_tensor, const MemoryConfig& memory_config, std::string_view tensor_name) {
    TT_FATAL(input_tensor.device() != nullptr, "depthwise_conv1d expects input_tensor to be on device");

    if (tensor.device() == nullptr) {
        return tensor.to_device(input_tensor.device(), memory_config);
    }

    TT_FATAL(
        tensor.device() == input_tensor.device(),
        "depthwise_conv1d expects {} to be on the same device as input_tensor",
        tensor_name);
    return tensor;
}

std::optional<Tensor> ensure_optional_tensor_on_input_device(
    const std::optional<Tensor>& tensor,
    const Tensor& input_tensor,
    const MemoryConfig& memory_config,
    std::string_view tensor_name) {
    if (!tensor.has_value()) {
        return std::nullopt;
    }
    return ensure_tensor_on_input_device(*tensor, input_tensor, memory_config, tensor_name);
}

Tensor normalize_input(const Tensor& input_tensor, bool& restore_hw_layout) {
    const auto& shape = input_tensor.logical_shape();
    TT_FATAL(shape.rank() == 4, "depthwise_conv1d expects a rank-4 input tensor");
    TT_FATAL(shape[3] > 0, "depthwise_conv1d expects a non-zero channel dimension");
    TT_FATAL(
        shape[1] == 1 || shape[2] == 1,
        "depthwise_conv1d expects input shape [B, 1, L, C] or [B, L, 1, C], got {}",
        shape);

    restore_hw_layout = shape[1] != 1;
    if (!restore_hw_layout) {
        return input_tensor;
    }

    return ttnn::reshape(input_tensor, ttnn::Shape({shape[0], 1, shape[1], shape[3]}));
}

Tensor normalize_weight(const Tensor& weight_tensor, uint32_t channels, uint32_t kernel_size) {
    const auto& shape = weight_tensor.logical_shape();
    TT_FATAL(shape.rank() == 3 || shape.rank() == 4, "depthwise_conv1d expects rank-3 or rank-4 weights");

    if (shape.rank() == 3) {
        TT_FATAL(
            shape[0] == channels && shape[1] == 1 && shape[2] == kernel_size,
            "rank-3 weights must have shape [C, 1, K], got {}",
            shape);
        auto reshaped = ttnn::reshape(weight_tensor, ttnn::Shape({1, 1, channels, kernel_size}));
        return ttnn::transpose(reshaped, 2, 3);
    }

    TT_FATAL(
        shape[0] == 1 && shape[1] == 1, "rank-4 weights must have shape [1, 1, C, K] or [1, 1, K, C], got {}", shape);
    if (shape[2] == kernel_size && shape[3] == channels) {
        return weight_tensor;
    }
    TT_FATAL(
        shape[2] == channels && shape[3] == kernel_size,
        "rank-4 weights must have shape [1, 1, C, K] or [1, 1, K, C], got {}",
        shape);
    return ttnn::transpose(weight_tensor, 2, 3);
}

Tensor normalize_bias(const Tensor& bias_tensor, uint32_t channels) {
    TT_FATAL(
        bias_tensor.logical_shape().volume() == channels,
        "bias volume must match the channel count, expected {} but got shape {}",
        channels,
        bias_tensor.logical_shape());
    return ttnn::reshape(bias_tensor, ttnn::Shape({1, 1, 1, channels}));
}

Tensor maybe_to_interleaved(const Tensor& tensor, const MemoryConfig& working_memory_config) {
    return tensor.is_sharded() ? ttnn::sharded_to_interleaved(tensor, working_memory_config) : tensor;
}

Tensor canonicalize_sequence_input(const Tensor& input_tensor, const MemoryConfig& memory_config, uint32_t features) {
    TT_FATAL(input_tensor.logical_shape().rank() == 3, "stateful depthwise_conv1d expects input shape [B, L, C]");
    TT_FATAL(
        input_tensor.logical_shape()[2] == features,
        "stateful depthwise_conv1d expected {} features but got input shape {}",
        features,
        input_tensor.logical_shape());

    auto input_rm = input_tensor.layout() == Layout::ROW_MAJOR
                        ? input_tensor
                        : ttnn::to_layout(input_tensor, Layout::ROW_MAJOR, std::nullopt, memory_config);
    return ttnn::reshape(
        maybe_to_interleaved(input_rm, memory_config),
        ttnn::Shape({input_rm.logical_shape()[0], 1, input_rm.logical_shape()[1], input_rm.logical_shape()[2]}),
        memory_config);
}

Tensor canonicalize_state_input(
    const std::optional<Tensor>& conv_state,
    const Tensor& reference_input,
    uint32_t features,
    uint32_t cache_len,
    const MemoryConfig& memory_config) {
    const uint32_t batch_size = reference_input.logical_shape()[0];

    if (!conv_state.has_value()) {
        return ttnn::zeros(
            ttnn::Shape({batch_size, 1, cache_len, features}),
            reference_input.dtype(),
            Layout::ROW_MAJOR,
            std::ref(*reference_input.device()),
            memory_config);
    }

    TT_FATAL(conv_state->logical_shape().rank() == 3, "conv_state must have shape [B, C, K-1]");
    TT_FATAL(
        conv_state->logical_shape()[0] == batch_size && conv_state->logical_shape()[1] == features &&
            conv_state->logical_shape()[2] == cache_len,
        "conv_state must have shape [{}, {}, {}], got {}",
        batch_size,
        features,
        cache_len,
        conv_state->logical_shape());

    auto state_rm = conv_state->layout() == Layout::ROW_MAJOR
                        ? *conv_state
                        : ttnn::to_layout(*conv_state, Layout::ROW_MAJOR, std::nullopt, memory_config);
    auto state_interleaved = maybe_to_interleaved(state_rm, memory_config);
    auto state_seq_first = ttnn::transpose(state_interleaved, 1, 2, memory_config);
    return ttnn::reshape(
        state_seq_first,
        ttnn::Shape(
            {state_seq_first.logical_shape()[0],
             1,
             state_seq_first.logical_shape()[1],
             state_seq_first.logical_shape()[2]}),
        memory_config);
}

}  // namespace

ttnn::Tensor depthwise_conv1d(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t kernel_size,
    bool causal,
    const std::optional<Tensor>& bias,
    bool silu_activation,
    const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(kernel_size > 0, "kernel_size must be greater than zero");
    TT_FATAL(causal, "depthwise_conv1d currently supports only causal=True");
    TT_FATAL(input_tensor.device() != nullptr, "depthwise_conv1d expects input_tensor to be on device");

    const auto requested_memory_config = memory_config.value_or(input_tensor.memory_config());
    const MemoryConfig working_memory_config =
        (input_tensor.is_sharded() || weight_tensor.is_sharded() || requested_memory_config.is_sharded())
            ? ttnn::DRAM_MEMORY_CONFIG
            : requested_memory_config;

    auto device_weight_tensor =
        ensure_tensor_on_input_device(weight_tensor, input_tensor, working_memory_config, "weight_tensor");
    auto device_bias_tensor = ensure_optional_tensor_on_input_device(bias, input_tensor, working_memory_config, "bias");

    bool restore_hw_layout = false;
    auto canonical_input =
        normalize_input(maybe_to_interleaved(input_tensor, working_memory_config), restore_hw_layout);
    const auto& canonical_shape = canonical_input.logical_shape();
    const uint32_t batch_size = canonical_shape[0];
    const uint32_t sequence_length = canonical_shape[2];
    const uint32_t channels = canonical_shape[3];

    auto canonical_weight =
        normalize_weight(maybe_to_interleaved(device_weight_tensor, working_memory_config), channels, kernel_size);

    std::optional<Tensor> canonical_bias = std::nullopt;
    if (device_bias_tensor.has_value()) {
        canonical_bias = normalize_bias(maybe_to_interleaved(*device_bias_tensor, working_memory_config), channels);
    }

    TT_FATAL(
        channels % 32 == 0,
        "depthwise_conv1d device operation requires channels to be a multiple of 32, got {}",
        channels);

    // Prepare padded input: [B, 1, L + K - 1, C]
    auto padded_input = canonical_input;
    if (kernel_size > 1) {
        auto zero_prefix = ttnn::zeros(
            ttnn::Shape({batch_size, 1, kernel_size - 1, channels}),
            canonical_input.dtype(),
            canonical_input.layout(),
            std::ref(*canonical_input.device()),
            working_memory_config);
        padded_input = ttnn::concat({zero_prefix, canonical_input}, 2, working_memory_config);
    }

    auto output = ttnn::prim::depthwise_conv1d_forward(
        padded_input, canonical_weight, kernel_size, channels, sequence_length, canonical_bias, silu_activation);

    if (restore_hw_layout) {
        const auto& original_shape = input_tensor.logical_shape();
        output = ttnn::reshape(
            output, ttnn::Shape({original_shape[0], original_shape[1], original_shape[2], original_shape[3]}));
    }

    if (output.memory_config() != requested_memory_config) {
        output = ttnn::to_memory_config(output, requested_memory_config);
    }

    return output;
}

std::vector<Tensor> depthwise_conv1d(
    const Tensor& input_tensor,
    const std::optional<Tensor>& conv_state,
    const Tensor& weight_tensor,
    const Tensor& bias_tensor,
    uint32_t features,
    uint32_t kernel_size) {
    TT_FATAL(kernel_size > 0, "kernel_size must be greater than zero");
    TT_FATAL(input_tensor.device() != nullptr, "stateful depthwise_conv1d expects input_tensor to be on device");

    const uint32_t cache_len = kernel_size - 1;
    const MemoryConfig working_memory_config = ttnn::DRAM_MEMORY_CONFIG;

    auto device_conv_state =
        ensure_optional_tensor_on_input_device(conv_state, input_tensor, working_memory_config, "conv_state");
    auto device_weight_tensor =
        ensure_tensor_on_input_device(weight_tensor, input_tensor, working_memory_config, "weight_tensor");
    auto device_bias_tensor =
        ensure_tensor_on_input_device(bias_tensor, input_tensor, working_memory_config, "bias_tensor");

    auto canonical_input = canonicalize_sequence_input(input_tensor, working_memory_config, features);
    auto canonical_state =
        canonicalize_state_input(device_conv_state, canonical_input, features, cache_len, working_memory_config);
    auto concatenated_input = ttnn::concat({canonical_state, canonical_input}, 2, working_memory_config);

    auto output = depthwise_conv1d(
        concatenated_input, device_weight_tensor, kernel_size, true, device_bias_tensor, false, working_memory_config);

    const uint32_t batch_size = canonical_input.logical_shape()[0];
    const uint32_t sequence_length = canonical_input.logical_shape()[2];
    const std::array<uint32_t, 4> slice_step = {1, 1, 1, 1};
    output = ttnn::slice(
        output,
        std::array<uint32_t, 4>{0, 0, cache_len, 0},
        std::array<uint32_t, 4>{batch_size, 1, cache_len + sequence_length, features},
        slice_step,
        working_memory_config);
    output = ttnn::reshape(output, ttnn::Shape({batch_size, sequence_length, features}), working_memory_config);

    auto new_conv_state =
        cache_len == 0 ? ttnn::reshape(canonical_state, ttnn::Shape({batch_size, features, 0}), working_memory_config)
                       : ttnn::slice(
                             concatenated_input,
                             std::array<uint32_t, 4>{0, 0, sequence_length, 0},
                             std::array<uint32_t, 4>{batch_size, 1, sequence_length + cache_len, features},
                             slice_step,
                             working_memory_config);
    if (cache_len > 0) {
        new_conv_state =
            ttnn::reshape(new_conv_state, ttnn::Shape({batch_size, cache_len, features}), working_memory_config);
        new_conv_state = ttnn::transpose(new_conv_state, 1, 2, working_memory_config);
    }

    if (input_tensor.layout() == Layout::TILE) {
        output = ttnn::to_layout(output, Layout::TILE, std::nullopt, working_memory_config);
        new_conv_state = ttnn::to_layout(new_conv_state, Layout::TILE, std::nullopt, working_memory_config);
    }

    return {output, new_conv_state};
}

}  // namespace ttnn::experimental
