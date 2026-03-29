// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "device/depthwise_conv1d_forward_device_operation.hpp"

#include "ttnn/operations/conv/conv1d/conv1d.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn::experimental {

namespace {

bool is_depthwise_conv1d_forward_custom_enabled() {
    if (const char* value = std::getenv("TTNN_DEPTHWISE_CONV1D_FORWARD_CUSTOM")) {
        return value[0] != '0';
    }
    return false;
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

PreparedTensorCache& depthwise_weight_cache() {
    static PreparedTensorCache cache;
    return cache;
}

PreparedTensorCache& depthwise_bias_cache() {
    static PreparedTensorCache cache;
    return cache;
}

std::mutex& depthwise_prepare_cache_mutex() {
    static std::mutex mutex;
    return mutex;
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

struct PaddedSequenceWithState {
    Tensor x_padded_blf;
    Tensor new_conv_state_bfk;
};

PaddedSequenceWithState build_padded_sequence_and_state(
    const Tensor& x_prepared,
    const Tensor& conv_state_tensor,
    uint32_t batch_size,
    uint32_t features,
    uint32_t cache_len,
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    auto conv_state_t = ttnn::transpose(conv_state_tensor, 1, 2, mem);
    if (conv_state_t.layout() != ttnn::TILE_LAYOUT) {
        conv_state_t = ttnn::to_layout(conv_state_t, ttnn::TILE_LAYOUT);
    }

    auto x_padded = ttnn::concat(std::vector<Tensor>{conv_state_t, x_prepared}, 1, mem);
    auto new_conv_state = ttnn::slice(
        x_padded,
        ttnn::SmallVector<uint32_t>{0, x_padded.logical_shape()[1] - cache_len, 0},
        ttnn::SmallVector<uint32_t>{batch_size, x_padded.logical_shape()[1], features},
        step,
        mem);
    new_conv_state = ttnn::transpose(new_conv_state, 1, 2, mem);

    return {.x_padded_blf = x_padded, .new_conv_state_bfk = new_conv_state};
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
    const ttnn::SmallVector<uint32_t>& step,
    const std::optional<MemoryConfig>& mem) {
    const bool allow_forward_custom =
        is_depthwise_conv1d_forward_custom_enabled() && kernel_size >= 2 && kernel_size <= 4 && batch_size == 32 &&
        (x.logical_shape()[1] == 133 || x.logical_shape()[1] == 134) && features == 544 &&
        x_prepared.layout() == ttnn::TILE_LAYOUT && conv_state_tensor.layout() == ttnn::TILE_LAYOUT &&
        x_prepared.dtype() == ttnn::DataType::BFLOAT16 && conv_state_tensor.dtype() == ttnn::DataType::BFLOAT16 &&
        !x_prepared.is_sharded() && !conv_state_tensor.is_sharded();
    if (allow_forward_custom) {
        auto padded_sequence =
            build_padded_sequence_and_state(x_prepared, conv_state_tensor, batch_size, features, cache_len, step, mem);
        auto weight_prepared = prepare_depthwise_weight(weight, x, features, kernel_size, step, mem);
        auto bias_prepared = prepare_depthwise_bias(bias, x, features, step, mem);
        auto output = ttnn::experimental::prim::depthwise_conv1d_forward_custom(
            padded_sequence.x_padded_blf, conv_state_tensor, weight_prepared, bias_prepared, features, kernel_size);
        output = ttnn::reshape(output, ttnn::Shape({1, 1, batch_size * x.logical_shape()[1], features}), mem);
        return {output, padded_sequence.new_conv_state_bfk};
    }

    auto padded_sequence =
        build_padded_sequence_and_state(x_prepared, conv_state_tensor, batch_size, features, cache_len, step, mem);
    auto output = std::get<Tensor>(ttnn::conv1d(
        padded_sequence.x_padded_blf,
        weight,
        x.device(),
        features,
        features,
        padded_sequence.x_padded_blf.logical_shape()[0],
        padded_sequence.x_padded_blf.logical_shape()[1],
        kernel_size,
        1,
        std::array<uint32_t, 2>{0, 0},
        1,
        features,
        std::nullopt,
        bias,
        std::nullopt,
        std::nullopt,
        mem,
        false,
        false));
    return {output, padded_sequence.new_conv_state_bfk};
}

}  // namespace

std::vector<Tensor> depthwise_conv1d(
    const Tensor& x,
    const std::optional<Tensor>& conv_state,
    const Tensor& weight,
    const Tensor& bias,
    uint32_t features,
    uint32_t kernel_size) {
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
            x, x_prepared, conv_state_tensor, weight, bias, batch_size, features, kernel_size, cache_len, step, mem);
    }

    return depthwise_conv1d_forward_path(
        x, x_prepared, conv_state_tensor, weight, bias, batch_size, features, kernel_size, cache_len, step, mem);
}

}  // namespace ttnn::experimental
