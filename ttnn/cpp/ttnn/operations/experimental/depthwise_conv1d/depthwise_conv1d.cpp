// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d.hpp"

#include <array>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <tt-metalium/constants.hpp>

#include "device/depthwise_conv1d_device_operation.hpp"
#include "ttnn/operations/conv/conv1d/conv1d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::experimental {

namespace {

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

bool is_depthwise_conv1d_forward_custom_enabled() {
    if (const char* value = std::getenv("TTNN_DEPTHWISE_CONV1D_FORWARD_CUSTOM")) {
        return value[0] != '0';
    }
    return true;
}

bool is_depthwise_conv1d_forward_custom_shape_supported(
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    uint32_t batch_size,
    uint32_t features,
    uint32_t kernel_size) {
    return kernel_size >= 2 && kernel_size <= 4 && batch_size == 32 && features == 544 &&
           x_prepared.layout() == ttnn::TILE_LAYOUT && conv_state_tensor.layout() == ttnn::TILE_LAYOUT &&
           x_prepared.dtype() == ttnn::DataType::BFLOAT16 && conv_state_tensor.dtype() == ttnn::DataType::BFLOAT16 &&
           !x_prepared.is_sharded() && !conv_state_tensor.is_sharded();
}

Tensor cast_to_bfloat16_if_needed(const Tensor& tensor) {
    Tensor result = tensor;
    if (result.dtype() == ttnn::DataType::BFLOAT16) {
        return result;
    }
    if (is_cpu_tensor(result)) {
        return ttnn::to_dtype(result, ttnn::DataType::BFLOAT16);
    }
    if (result.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        result = ttnn::to_layout(result, ttnn::TILE_LAYOUT);
    }
    return ttnn::typecast(result, ttnn::DataType::BFLOAT16, result.memory_config());
}

Tensor cast_to_bfloat16_preserve_layout_if_needed(const Tensor& tensor) {
    if (tensor.dtype() == ttnn::DataType::BFLOAT16) {
        return tensor;
    }
    if (is_cpu_tensor(tensor)) {
        return ttnn::to_dtype(tensor, ttnn::DataType::BFLOAT16);
    }
    return ttnn::typecast(tensor, ttnn::DataType::BFLOAT16, tensor.memory_config());
}

struct PreparedTensorCacheKey {
    std::uint64_t tensor_id;
    const void* device;
    uint32_t features;
    uint32_t kernel_size;

    bool operator==(const PreparedTensorCacheKey& other) const = default;
};

struct PreparedTensorCacheKeyHash {
    std::size_t operator()(const PreparedTensorCacheKey& key) const {
        std::size_t hash = std::hash<std::uint64_t>{}(key.tensor_id);
        hash ^= std::hash<const void*>{}(key.device) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.features) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.kernel_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

using PreparedTensorCache = std::unordered_map<PreparedTensorCacheKey, Tensor, PreparedTensorCacheKeyHash>;

struct PreparedFallbackConvCacheKey {
    std::uint64_t weight_tensor_id;
    std::uint64_t bias_tensor_id;
    const void* device;
    uint32_t batch_size;
    uint32_t input_width;
    uint32_t features;
    uint32_t kernel_size;
    DataType input_dtype;
    Layout input_layout;
    BufferType input_buffer_type;
    TensorMemoryLayout input_memory_layout;

    bool operator==(const PreparedFallbackConvCacheKey& other) const = default;
};

struct PreparedFallbackConvCacheKeyHash {
    std::size_t operator()(const PreparedFallbackConvCacheKey& key) const {
        std::size_t hash = std::hash<std::uint64_t>{}(key.weight_tensor_id);
        hash ^= std::hash<std::uint64_t>{}(key.bias_tensor_id) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<const void*>{}(key.device) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.batch_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.input_width) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.features) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint32_t>{}(key.kernel_size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(static_cast<int>(key.input_dtype)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(static_cast<int>(key.input_layout)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(static_cast<int>(key.input_buffer_type)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(static_cast<int>(key.input_memory_layout)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct PreparedFallbackConvTensors {
    Tensor weight;
    Tensor bias;
};

using PreparedFallbackConvCache =
    std::unordered_map<PreparedFallbackConvCacheKey, PreparedFallbackConvTensors, PreparedFallbackConvCacheKeyHash>;

PreparedTensorCache& depthwise_weight_cache() {
    static PreparedTensorCache cache;
    return cache;
}

PreparedTensorCache& depthwise_bias_cache() {
    static PreparedTensorCache cache;
    return cache;
}

PreparedTensorCache& depthwise_causal_weight_cache() {
    static PreparedTensorCache cache;
    return cache;
}

PreparedTensorCache& depthwise_causal_bias_cache() {
    static PreparedTensorCache cache;
    return cache;
}

std::mutex& depthwise_prepare_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

PreparedFallbackConvCache& depthwise_fallback_conv_cache() {
    static PreparedFallbackConvCache cache;
    return cache;
}

PreparedFallbackConvCacheKey create_prepared_fallback_conv_cache_key(
    const Tensor& weight, const Tensor& bias, const Tensor& x_padded, uint32_t features, uint32_t kernel_size) {
    const auto& memory_config = x_padded.memory_config();
    return PreparedFallbackConvCacheKey{
        .weight_tensor_id = weight.tensor_id,
        .bias_tensor_id = bias.tensor_id,
        .device = x_padded.device(),
        .batch_size = static_cast<uint32_t>(x_padded.logical_shape()[0]),
        .input_width = static_cast<uint32_t>(x_padded.logical_shape()[1]),
        .features = features,
        .kernel_size = kernel_size,
        .input_dtype = x_padded.dtype(),
        .input_layout = x_padded.layout(),
        .input_buffer_type = memory_config.buffer_type(),
        .input_memory_layout = memory_config.memory_layout()};
}

PreparedFallbackConvTensors prepare_fallback_conv_tensors_uncached(
    const Tensor& weight, const Tensor& bias, const Tensor& x_padded, uint32_t features, uint32_t kernel_size) {
    using namespace ttnn::operations::conv::conv2d;

    const auto batch_size = static_cast<uint32_t>(x_padded.logical_shape()[0]);
    const auto input_width = static_cast<uint32_t>(x_padded.logical_shape()[1]);
    const auto slice_config = std::optional<const ttnn::prim::Conv2dSliceConfig>(
        ttnn::prim::Conv2dSliceConfig{.slice_type = ttnn::prim::Conv2dSliceConfig::SliceType::L1_FULL});
    ttnn::prim::Conv2dConfig conv_config;
    conv_config.weights_dtype = weight.dtype();

    auto prepared_weight = prepare_conv_weights(
        weight,
        x_padded.memory_config(),
        x_padded.layout(),
        "OIHW",
        features,
        features,
        batch_size,
        1,
        input_width,
        std::array<uint32_t, 2>{1, kernel_size},
        std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2>{1, 1},
        true,
        features,
        x_padded.device(),
        x_padded.dtype(),
        std::nullopt,
        conv_config,
        std::nullopt,
        slice_config);

    conv_config.weights_dtype = prepared_weight.dtype();

    auto prepared_bias = prepare_conv_bias(
        bias,
        x_padded.memory_config(),
        x_padded.layout(),
        features,
        features,
        batch_size,
        1,
        input_width,
        std::array<uint32_t, 2>{1, kernel_size},
        std::array<uint32_t, 2>{1, 1},
        std::array<uint32_t, 2>{0, 0},
        std::array<uint32_t, 2>{1, 1},
        features,
        x_padded.device(),
        x_padded.dtype(),
        std::nullopt,
        conv_config,
        std::nullopt,
        slice_config);

    return PreparedFallbackConvTensors{.weight = std::move(prepared_weight), .bias = std::move(prepared_bias)};
}

PreparedFallbackConvTensors prepare_fallback_conv_tensors(
    const Tensor& weight, const Tensor& bias, const Tensor& x_padded, uint32_t features, uint32_t kernel_size) {
    const auto cache_key = create_prepared_fallback_conv_cache_key(weight, bias, x_padded, features, kernel_size);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        auto& cache = depthwise_fallback_conv_cache();
        if (auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    auto prepared = prepare_fallback_conv_tensors_uncached(weight, bias, x_padded, features, kernel_size);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        depthwise_fallback_conv_cache()[cache_key] = prepared;
    }
    return prepared;
}

Tensor prepare_depthwise_weight_uncached(
    const Tensor& weight,
    const Tensor& reference_tensor,
    uint32_t features,
    uint32_t kernel_size,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    auto weight_tensor = weight;
    if (weight_tensor.storage_type() != StorageType::DEVICE) {
        weight_tensor = cast_to_bfloat16_if_needed(weight_tensor);
        weight_tensor = ttnn::to_device(weight_tensor, reference_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }
    weight_tensor = cast_to_bfloat16_if_needed(weight_tensor);
    if (weight_tensor.layout() != ttnn::TILE_LAYOUT) {
        weight_tensor = ttnn::to_layout(weight_tensor, ttnn::TILE_LAYOUT);
    }
    const auto& prepared_shape = weight_tensor.logical_shape();
    Tensor weight_prepared = weight_tensor;
    if (prepared_shape.rank() == 4) {
        weight_prepared =
            ttnn::reshape(weight_tensor, ttnn::Shape({prepared_shape[0], prepared_shape[1], prepared_shape[3]}), mem);
    } else if (prepared_shape.rank() != 3) {
        TT_FATAL(false, "weight must have rank 3 or 4 for depthwise conv1d");
    }
    if (weight_prepared.logical_shape()[0] != features || weight_prepared.logical_shape()[2] != kernel_size) {
        weight_prepared = ttnn::slice(
            weight_prepared,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{features, weight_prepared.logical_shape()[1], kernel_size},
            step,
            mem);
    }
    if (weight_prepared.logical_shape()[1] != 1) {
        weight_prepared = ttnn::slice(
            weight_prepared,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{features, 1, kernel_size},
            step,
            mem);
    }
    return ttnn::reshape(weight_prepared, ttnn::Shape({1, features, kernel_size}), mem);
}

Tensor prepare_depthwise_weight(
    const Tensor& weight,
    const Tensor& reference_tensor,
    uint32_t features,
    uint32_t kernel_size,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    PreparedTensorCacheKey cache_key{
        .tensor_id = weight.tensor_id,
        .device = reference_tensor.device(),
        .features = features,
        .kernel_size = kernel_size};
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        auto& cache = depthwise_weight_cache();
        if (auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    auto prepared = prepare_depthwise_weight_uncached(weight, reference_tensor, features, kernel_size, step, mem);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        depthwise_weight_cache()[cache_key] = prepared;
    }
    return prepared;
}

Tensor prepare_depthwise_bias_uncached(
    const Tensor& bias,
    const Tensor& reference_tensor,
    uint32_t features,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    auto bias_tensor = bias;
    Tensor bias_prepared = bias_tensor;
    const auto& bias_shape = bias_prepared.logical_shape();
    if (bias_shape.rank() == 4) {
        bias_prepared = ttnn::reshape(bias_prepared, ttnn::Shape({bias_shape[0], bias_shape[1], bias_shape[3]}), mem);
    }
    if (bias_prepared.logical_shape()[2] != features) {
        bias_prepared = ttnn::slice(
            bias_prepared,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{1, 1, features},
            step,
            mem);
    }
    bias_prepared = cast_to_bfloat16_if_needed(bias_prepared);
    if (bias_prepared.storage_type() != StorageType::DEVICE) {
        bias_prepared = ttnn::to_device(bias_prepared, reference_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }
    if (bias_prepared.layout() != ttnn::TILE_LAYOUT) {
        bias_prepared = ttnn::to_layout(bias_prepared, ttnn::TILE_LAYOUT);
    }
    return bias_prepared;
}

Tensor prepare_depthwise_bias(
    const Tensor& bias,
    const Tensor& reference_tensor,
    uint32_t features,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    PreparedTensorCacheKey cache_key{
        .tensor_id = bias.tensor_id, .device = reference_tensor.device(), .features = features, .kernel_size = 0};
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        auto& cache = depthwise_bias_cache();
        if (auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    auto prepared = prepare_depthwise_bias_uncached(bias, reference_tensor, features, step, mem);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        depthwise_bias_cache()[cache_key] = prepared;
    }
    return prepared;
}

Tensor prepare_depthwise_weight_causal_uncached(
    const Tensor& weight, const Tensor& reference_tensor, uint32_t features, uint32_t kernel_size) {
    auto weight_tensor = weight;
    if (weight_tensor.storage_type() != StorageType::DEVICE) {
        weight_tensor = cast_to_bfloat16_preserve_layout_if_needed(weight_tensor);
        weight_tensor = weight_tensor.to_device(reference_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }
    weight_tensor = cast_to_bfloat16_preserve_layout_if_needed(weight_tensor);
    weight_tensor = maybe_to_interleaved(weight_tensor, ttnn::DRAM_MEMORY_CONFIG);
    auto weight_causal = normalize_weight(weight_tensor, features, kernel_size);
    if (weight_causal.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        weight_causal = ttnn::to_layout(weight_causal, ttnn::ROW_MAJOR_LAYOUT);
    }
    weight_causal = cast_to_bfloat16_preserve_layout_if_needed(weight_causal);
    return weight_causal;
}

[[maybe_unused]] Tensor prepare_depthwise_weight_causal(
    const Tensor& weight, const Tensor& reference_tensor, uint32_t features, uint32_t kernel_size) {
    PreparedTensorCacheKey cache_key{
        .tensor_id = weight.tensor_id,
        .device = reference_tensor.device(),
        .features = features,
        .kernel_size = kernel_size};
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        auto& cache = depthwise_causal_weight_cache();
        if (auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }
    auto prepared = prepare_depthwise_weight_causal_uncached(weight, reference_tensor, features, kernel_size);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        depthwise_causal_weight_cache()[cache_key] = prepared;
    }
    return prepared;
}

Tensor prepare_depthwise_bias_causal_uncached(const Tensor& bias, const Tensor& reference_tensor, uint32_t features) {
    auto bias_tensor = bias;
    if (bias_tensor.storage_type() != StorageType::DEVICE) {
        bias_tensor = cast_to_bfloat16_preserve_layout_if_needed(bias_tensor);
        bias_tensor = bias_tensor.to_device(reference_tensor.device(), ttnn::DRAM_MEMORY_CONFIG);
    }
    bias_tensor = cast_to_bfloat16_preserve_layout_if_needed(bias_tensor);
    bias_tensor = maybe_to_interleaved(bias_tensor, ttnn::DRAM_MEMORY_CONFIG);
    auto bias_causal = normalize_bias(bias_tensor, features);
    if (bias_causal.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        bias_causal = ttnn::to_layout(bias_causal, ttnn::ROW_MAJOR_LAYOUT);
    }
    bias_causal = cast_to_bfloat16_preserve_layout_if_needed(bias_causal);
    return bias_causal;
}

[[maybe_unused]] Tensor prepare_depthwise_bias_causal(
    const Tensor& bias, const Tensor& reference_tensor, uint32_t features) {
    PreparedTensorCacheKey cache_key{
        .tensor_id = bias.tensor_id, .device = reference_tensor.device(), .features = features, .kernel_size = 0};
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        auto& cache = depthwise_causal_bias_cache();
        if (auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }
    auto prepared = prepare_depthwise_bias_causal_uncached(bias, reference_tensor, features);
    {
        std::lock_guard<std::mutex> lock(depthwise_prepare_cache_mutex());
        depthwise_causal_bias_cache()[cache_key] = prepared;
    }
    return prepared;
}

[[maybe_unused]] Tensor build_stateful_causal_padded_input(
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    uint32_t batch_size,
    uint32_t sequence_length,
    uint32_t features,
    uint32_t cache_len,
    const std::optional<MemoryConfig>& mem) {
    auto x_rm = x_prepared;
    if (x_rm.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        x_rm = ttnn::to_layout(x_rm, ttnn::ROW_MAJOR_LAYOUT);
    }

    if (cache_len == 0) {
        return ttnn::reshape(x_rm, ttnn::Shape({batch_size, 1, sequence_length, features}), mem);
    }

    auto conv_state_prefix = ttnn::transpose(conv_state_tensor, 1, 2, mem);
    if (conv_state_prefix.layout() != ttnn::ROW_MAJOR_LAYOUT) {
        conv_state_prefix = ttnn::to_layout(conv_state_prefix, ttnn::ROW_MAJOR_LAYOUT);
    }

    auto x_padded = ttnn::concat(std::vector<Tensor>{conv_state_prefix, x_rm}, 1, mem);
    return ttnn::reshape(x_padded, ttnn::Shape({batch_size, 1, sequence_length + cache_len, features}), mem);
}

Tensor compute_new_conv_state(
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    uint32_t batch_size,
    uint32_t features,
    uint32_t cache_len,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    if (cache_len == 0) {
        return conv_state_tensor;
    }

    const uint32_t sequence_length = x_prepared.logical_shape()[1];
    if (sequence_length >= cache_len) {
        auto x_tail = ttnn::slice(
            x_prepared,
            ttnn::SmallVector<uint32_t>{0, sequence_length - cache_len, 0},
            ttnn::SmallVector<uint32_t>{batch_size, sequence_length, features},
            step,
            mem);
        return ttnn::transpose(x_tail, 1, 2, mem);
    }

    auto x_t = ttnn::transpose(x_prepared, 1, 2, mem);
    auto state_tail = ttnn::slice(
        conv_state_tensor,
        ttnn::SmallVector<uint32_t>{0, 0, sequence_length},
        ttnn::SmallVector<uint32_t>{batch_size, features, cache_len},
        step,
        mem);
    return ttnn::concat(std::vector<Tensor>{state_tail, x_t}, 2, mem);
}

std::vector<Tensor> depthwise_conv1d_update_path(
    const Tensor& x,
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t batch_size,
    uint32_t features,
    uint32_t kernel_size,
    uint32_t cache_len,
    bool silu_activation,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    auto x_token = ttnn::transpose(x_prepared, 1, 2, mem);
    Tensor new_conv_state;
    Tensor conv_window;
    if (cache_len == 0) {
        new_conv_state = conv_state_tensor;
        conv_window = x_token;
    } else {
        if (cache_len > 1) {
            auto trimmed_state = ttnn::slice(
                conv_state_tensor,
                ttnn::SmallVector<uint32_t>{0, 0, 1},
                ttnn::SmallVector<uint32_t>{batch_size, features, cache_len},
                step,
                mem);
            new_conv_state = ttnn::concat(std::vector<Tensor>{trimmed_state, x_token}, 2, mem);
        } else {
            new_conv_state = x_token;
        }
        conv_window = ttnn::concat(std::vector<Tensor>{conv_state_tensor, x_token}, 2, mem);
    }

    auto weight_prepared = prepare_depthwise_weight(weight, x, features, kernel_size, step, mem);
    auto bias_prepared = prepare_depthwise_bias(bias, x, features, step, mem);

    auto weighted = ttnn::multiply(conv_window, weight_prepared, std::nullopt, mem);
    auto output = ttnn::sum(weighted, std::variant<int, int64_t, SmallVector<int>>(2), true, mem);
    output = ttnn::transpose(output, 1, 2, mem);
    output = ttnn::add(output, bias_prepared, std::nullopt, mem);
    if (silu_activation) {
        output = ttnn::silu(output, mem);
    }

    return {output, new_conv_state};
}

std::vector<Tensor> depthwise_conv1d_forward_path(
    const Tensor& x,
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t batch_size,
    uint32_t features,
    uint32_t kernel_size,
    uint32_t cache_len,
    bool silu_activation,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    auto new_conv_state =
        compute_new_conv_state(x_prepared, conv_state_tensor, batch_size, features, cache_len, step, mem);

    const bool supported_custom_shape = is_depthwise_conv1d_forward_custom_shape_supported(
        x_prepared, conv_state_tensor, batch_size, features, kernel_size);
    const bool force_disable_custom = !is_depthwise_conv1d_forward_custom_enabled();

    if (supported_custom_shape && !force_disable_custom) {
        auto weight_causal = prepare_depthwise_weight_causal(weight, x, features, kernel_size);
        auto bias_causal = prepare_depthwise_bias_causal(bias, x, features);
        auto padded_input = build_stateful_causal_padded_input(
            x_prepared, conv_state_tensor, batch_size, x.logical_shape()[1], features, cache_len, mem);
        auto output = ttnn::prim::depthwise_conv1d_forward(
            padded_input, weight_causal, kernel_size, features, x.logical_shape()[1], bias_causal, silu_activation);
        output = ttnn::reshape(output, ttnn::Shape({batch_size, x.logical_shape()[1], features}), mem);
        if (output.layout() != ttnn::TILE_LAYOUT) {
            output = ttnn::to_layout(output, ttnn::TILE_LAYOUT);
        }
        return {output, new_conv_state};
    }

    auto conv_state_t = ttnn::transpose(conv_state_tensor, 1, 2, mem);
    if (conv_state_t.layout() != ttnn::TILE_LAYOUT) {
        conv_state_t = ttnn::to_layout(conv_state_t, ttnn::TILE_LAYOUT);
    }
    auto x_padded = ttnn::concat(std::vector<Tensor>{conv_state_t, x_prepared}, 1, mem);
    auto prepared_conv_tensors = prepare_fallback_conv_tensors(weight, bias, x_padded, features, kernel_size);
    auto output = std::get<Tensor>(ttnn::conv1d(
        x_padded,
        prepared_conv_tensors.weight,
        x.device(),
        features,
        features,
        x_padded.logical_shape()[0],
        x_padded.logical_shape()[1],
        kernel_size,
        1,
        std::array<uint32_t, 2>{0, 0},
        1,
        features,
        std::nullopt,
        prepared_conv_tensors.bias,
        std::nullopt,
        std::nullopt,
        mem,
        false,
        false));
    output = ttnn::reshape(output, ttnn::Shape({batch_size, x.logical_shape()[1], features}), mem);
    if (output.layout() != ttnn::TILE_LAYOUT) {
        output = ttnn::to_layout(output, ttnn::TILE_LAYOUT);
    }
    if (silu_activation) {
        output = ttnn::silu(output, mem);
    }
    return {output, new_conv_state};
}

}  // namespace

std::vector<Tensor> depthwise_conv1d(
    const Tensor& x,
    const std::optional<Tensor>& conv_state,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t features,
    uint32_t kernel_size,
    bool silu_activation) {
    TT_FATAL(x.device() != nullptr, "x must be on device");
    TT_FATAL(x.logical_shape().rank() == 3, "x must have shape [B, L, F]");
    TT_FATAL(weight.layout() == ttnn::ROW_MAJOR_LAYOUT, "weight must be row-major");
    TT_FATAL(bias.layout() == ttnn::ROW_MAJOR_LAYOUT, "bias must be row-major");
    TT_FATAL(kernel_size >= 1, "kernel_size must be positive");
    TT_FATAL(features >= 1, "features must be positive");

    const auto& x_shape = x.logical_shape();
    const uint32_t batch_size = x_shape[0];
    const uint32_t sequence_length = x_shape[1];
    const uint32_t cache_len = kernel_size - 1;
    const auto mem = std::optional<MemoryConfig>(ttnn::DRAM_MEMORY_CONFIG);
    const auto step = ttnn::SmallVector<uint32_t>{1, 1, 1};

    TT_FATAL(x_shape[2] >= features, "x feature dimension must be >= features");

    Tensor conv_state_tensor;
    if (conv_state.has_value()) {
        conv_state_tensor = conv_state.value();
    } else {
        conv_state_tensor = ttnn::zeros(
            ttnn::Shape({batch_size, features, cache_len}),
            x.dtype(),
            ttnn::TILE_LAYOUT,
            *x.device(),
            ttnn::DRAM_MEMORY_CONFIG);
    }
    TT_FATAL(conv_state_tensor.device() != nullptr, "conv_state must be on device");
    TT_FATAL(x.device() == conv_state_tensor.device(), "x and conv_state must share a device");
    TT_FATAL(conv_state_tensor.logical_shape().rank() == 3, "conv_state must have shape [B, F, K-1]");

    const auto& state_shape = conv_state_tensor.logical_shape();
    TT_FATAL(state_shape[0] == batch_size, "conv_state batch must match x batch");
    TT_FATAL(state_shape[1] >= features, "conv_state feature dimension must be >= features");
    TT_FATAL(state_shape[2] >= cache_len, "conv_state cache dimension must be >= kernel_size - 1");

    auto x_prepared = x;
    if (x_prepared.layout() != ttnn::TILE_LAYOUT) {
        x_prepared = ttnn::to_layout(x_prepared, ttnn::TILE_LAYOUT);
    }
    if (x_prepared.logical_shape()[2] != features) {
        x_prepared = ttnn::slice(
            x_prepared,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{batch_size, sequence_length, features},
            step,
            mem);
    }
    if (conv_state_tensor.layout() != ttnn::TILE_LAYOUT) {
        conv_state_tensor = ttnn::to_layout(conv_state_tensor, ttnn::TILE_LAYOUT);
    }
    if (conv_state_tensor.logical_shape()[1] != features || conv_state_tensor.logical_shape()[2] != cache_len) {
        conv_state_tensor = ttnn::slice(
            conv_state_tensor,
            ttnn::SmallVector<uint32_t>{0, 0, 0},
            ttnn::SmallVector<uint32_t>{batch_size, features, cache_len},
            step,
            mem);
    }

    if (sequence_length == 1) {
        return depthwise_conv1d_update_path(
            x,
            x_prepared,
            conv_state_tensor,
            weight,
            bias,
            batch_size,
            features,
            kernel_size,
            cache_len,
            silu_activation,
            step,
            mem);
    }

    return depthwise_conv1d_forward_path(
        x,
        x_prepared,
        conv_state_tensor,
        weight,
        bias,
        batch_size,
        features,
        kernel_size,
        cache_len,
        silu_activation,
        step,
        mem);
}

}  // namespace ttnn::experimental
