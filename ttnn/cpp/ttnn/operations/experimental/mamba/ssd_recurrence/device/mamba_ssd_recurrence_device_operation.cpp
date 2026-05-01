// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_recurrence_device_operation.hpp"

#include <algorithm>
#include <functional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/experimental/ssm/prefix_scan/prefix_scan.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {
uint32_t select_hidden_dim_segment_size(
    const MambaSSDRecurrenceParams& operation_attributes, const ttnn::Tensor& reference_tensor) {
    constexpr uint32_t max_shard_width = 2048;
    const auto compute_grid = reference_tensor.device()->compute_with_storage_grid_size();
    const uint32_t max_cores = operation_attributes.has_core_grid
                                   ? operation_attributes.core_grid_x * operation_attributes.core_grid_y
                                   : static_cast<uint32_t>(compute_grid.x * compute_grid.y);
    const uint32_t num_cores =
        std::max<uint32_t>(1, std::min<uint32_t>(max_cores, static_cast<uint32_t>(compute_grid.x * compute_grid.y)));
    return num_cores * max_shard_width;
}

uint32_t align_segment_hidden_dim(uint32_t segment_hidden_dim_limit, uint32_t state_plane, uint32_t hidden_dim) {
    const uint32_t aligned_limit =
        std::max<uint32_t>(state_plane, (segment_hidden_dim_limit / state_plane) * state_plane);
    return std::min<uint32_t>(aligned_limit, hidden_dim);
}

constexpr uint32_t kPrefixScanChunkSize = tt::constants::TILE_HEIGHT;

std::optional<CoreGrid> get_core_grid(const MambaSSDRecurrenceParams& operation_attributes) {
    if (!operation_attributes.has_core_grid) {
        return std::nullopt;
    }
    return CoreGrid{operation_attributes.core_grid_x, operation_attributes.core_grid_y};
}

bool is_supported_dtype(const Tensor& tensor) {
    return tensor.dtype() == ttnn::DataType::BFLOAT16 || tensor.dtype() == ttnn::DataType::FLOAT32;
}

Tensor segsum_local(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    const auto& mem = memory_config;
    auto input_fp32 =
        input.dtype() == ttnn::DataType::FLOAT32 ? input : ttnn::typecast(input, ttnn::DataType::FLOAT32, mem);
    auto input_cumsum = ttnn::cumsum(input_fp32, -1, std::nullopt, false, std::nullopt, mem);
    auto input_segsum =
        ttnn::subtract(ttnn::unsqueeze(input_cumsum, -1), ttnn::unsqueeze(input_cumsum, -2), std::nullopt, mem);
    auto mask = ttnn::ones(
        input_segsum.logical_shape(), ttnn::DataType::FLOAT32, ttnn::TILE_LAYOUT, *input.device(), mem.value());
    mask = ttnn::tril(mask, 0, mem);
    auto neg_large = ttnn::full(
        input_segsum.logical_shape(), -1.0e30f, ttnn::DataType::FLOAT32, ttnn::TILE_LAYOUT, *input.device(), mem);
    return ttnn::where(mask, input_segsum, neg_large, mem);
}

}  // namespace

MambaSSDRecurrenceKernelConfig build_mamba_ssd_recurrence_kernel_config(
    const MambaSSDRecurrenceParams& operation_attributes, const MambaSSDRecurrenceInputs& tensor_args) {
    const auto& states_bhcpn = tensor_args.states_bhcpn;
    const auto& s = states_bhcpn.logical_shape();

    const uint32_t B = s[0];
    const uint32_t H = s[1];
    const uint32_t C = s[2];
    const uint32_t P = s[3];
    const uint32_t N = s[4];
    const uint32_t hidden_dim = B * H * P * N;
    const uint32_t bh = B * H;
    const uint32_t state_plane = P * N;
    const uint32_t segment_hidden_dim_limit = align_segment_hidden_dim(
        select_hidden_dim_segment_size(operation_attributes, states_bhcpn), state_plane, hidden_dim);

    return MambaSSDRecurrenceKernelConfig{
        .batch_size = B,
        .num_heads = H,
        .num_chunks = C,
        .head_dim = P,
        .state_size = N,
        .hidden_dim = hidden_dim,
        .bh = bh,
        .state_plane = state_plane,
        .segment_hidden_dim_limit = segment_hidden_dim_limit,
        .num_hidden_segments = tt::div_up(hidden_dim, segment_hidden_dim_limit),
        .chunk_size = kPrefixScanChunkSize,
        .num_chunk_segments = tt::div_up(C, kPrefixScanChunkSize),
    };
}

MambaSSDRecurrenceDeviceOperation::tensor_return_value_t mamba_ssd_recurrence_fallback(
    const MambaSSDRecurrenceParams& operation_attributes, const MambaSSDRecurrenceInputs& tensor_args) {
    const auto& states_bhcpn = tensor_args.states_bhcpn;
    const auto& initial_states = tensor_args.initial_states;
    const auto& a_end_bhc = tensor_args.a_end_bhc;
    const auto& mem = operation_attributes.memory_config;
    const auto& s = states_bhcpn.logical_shape();
    const uint32_t B = s[0];
    const uint32_t H = s[1];
    const uint32_t C = s[2];
    const uint32_t P = s[3];
    const uint32_t N = s[4];

    auto initial_states_fp32 = initial_states.dtype() == ttnn::DataType::FLOAT32
                                   ? initial_states
                                   : ttnn::typecast(initial_states, ttnn::DataType::FLOAT32, mem);
    auto states_fp32 = states_bhcpn.dtype() == ttnn::DataType::FLOAT32
                           ? states_bhcpn
                           : ttnn::typecast(states_bhcpn, ttnn::DataType::FLOAT32, mem);
    auto a_end_fp32 = a_end_bhc.dtype() == ttnn::DataType::FLOAT32
                          ? a_end_bhc
                          : ttnn::typecast(a_end_bhc, ttnn::DataType::FLOAT32, mem);

    auto states_with_init = ttnn::concat(std::vector<Tensor>{initial_states_fp32, states_fp32}, 2, mem);
    auto a_end_prefix =
        ttnn::zeros(ttnn::Shape({B, H, 1}), ttnn::DataType::FLOAT32, ttnn::TILE_LAYOUT, *states_bhcpn.device(), mem);
    auto a_end_padded = ttnn::concat(std::vector<Tensor>{a_end_prefix, a_end_fp32}, -1, mem);
    auto decay_chunk = ttnn::exp(segsum_local(a_end_padded, mem), false, mem);

    auto states_with_init_flat = ttnn::reshape(states_with_init, ttnn::Shape({B * H, C + 1, P * N}), mem);
    auto decay_chunk_flat = ttnn::reshape(decay_chunk, ttnn::Shape({B * H, C + 1, C + 1}), mem);

    auto new_states = ttnn::matmul(
        decay_chunk_flat,
        states_with_init_flat,
        false,
        false,
        mem,
        ttnn::DataType::FLOAT32,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        get_core_grid(operation_attributes));
    new_states = ttnn::reshape(new_states, ttnn::Shape({B, H, C + 1, P, N}), mem);

    const auto step = ttnn::SmallVector<uint32_t>{1, 1, 1, 1, 1};
    auto final_state = ttnn::slice(
        new_states,
        ttnn::SmallVector<uint32_t>{0, 0, C, 0, 0},
        ttnn::SmallVector<uint32_t>{B, H, C + 1, P, N},
        step,
        mem);
    final_state = ttnn::reshape(final_state, ttnn::Shape({B, H, P, N}), mem);

    auto states = ttnn::slice(
        new_states, ttnn::SmallVector<uint32_t>{0, 0, 0, 0, 0}, ttnn::SmallVector<uint32_t>{B, H, C, P, N}, step, mem);
    states = ttnn::permute(states, ttnn::SmallVector<int64_t>{0, 2, 1, 3, 4}, mem);

    return {states, final_state};
}

void MambaSSDRecurrenceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& states_bhcpn = tensor_args.states_bhcpn;
    const auto& initial_states = tensor_args.initial_states;
    const auto& a_end_bhc = tensor_args.a_end_bhc;

    const auto& s = states_bhcpn.logical_shape();
    const auto& i = initial_states.logical_shape();
    const auto& a = a_end_bhc.logical_shape();

    TT_FATAL(states_bhcpn.device() != nullptr, "states_bhcpn must be on device");
    TT_FATAL(initial_states.device() != nullptr, "initial_states must be on device");
    TT_FATAL(a_end_bhc.device() != nullptr, "a_end_bhc must be on device");
    TT_FATAL(states_bhcpn.device() == initial_states.device(), "states_bhcpn and initial_states must share a device");
    TT_FATAL(states_bhcpn.device() == a_end_bhc.device(), "states_bhcpn and a_end_bhc must share a device");
    TT_FATAL(is_supported_dtype(states_bhcpn), "states_bhcpn must be BF16/FLOAT32");
    TT_FATAL(is_supported_dtype(initial_states), "initial_states must be BF16/FLOAT32");
    TT_FATAL(is_supported_dtype(a_end_bhc), "a_end_bhc must be BF16/FLOAT32");
    TT_FATAL(states_bhcpn.layout() == ttnn::TILE_LAYOUT, "states_bhcpn must be TILE layout");
    TT_FATAL(initial_states.layout() == ttnn::TILE_LAYOUT, "initial_states must be TILE layout");
    TT_FATAL(a_end_bhc.layout() == ttnn::TILE_LAYOUT, "a_end_bhc must be TILE layout");

    TT_FATAL(s.rank() == 5, "states_bhcpn must be rank 5");
    TT_FATAL(i.rank() == 5, "initial_states must be rank 5");
    TT_FATAL(a.rank() == 3, "a_end_bhc must be rank 3");

    const uint32_t B = s[0];
    const uint32_t H = s[1];
    const uint32_t C = s[2];
    const uint32_t P = s[3];
    const uint32_t N = s[4];

    TT_FATAL(i[0] == B && i[1] == H && i[2] == 1 && i[3] == P && i[4] == N, "initial_states shape mismatch");
    TT_FATAL(a[0] == B && a[1] == H && a[2] == C, "a_end_bhc shape mismatch");
    TT_FATAL(!args.memory_config.is_sharded(), "mamba_ssd_recurrence output memory_config must be interleaved");
    TT_FATAL(!states_bhcpn.memory_config().is_sharded(), "states_bhcpn must be interleaved");
    TT_FATAL(!initial_states.memory_config().is_sharded(), "initial_states must be interleaved");
    TT_FATAL(!a_end_bhc.memory_config().is_sharded(), "a_end_bhc must be interleaved");
}

void MambaSSDRecurrenceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

MambaSSDRecurrenceDeviceOperation::spec_return_value_t MambaSSDRecurrenceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& s = tensor_args.states_bhcpn.logical_shape();
    const uint32_t B = s[0];
    const uint32_t H = s[1];
    const uint32_t C = s[2];
    const uint32_t P = s[3];
    const uint32_t N = s[4];

    auto states_spec = TensorSpec(
        ttnn::Shape({B, C, H, P, N}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.memory_config));

    auto final_state_spec = TensorSpec(
        ttnn::Shape({B, H, P, N}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.memory_config));

    return {states_spec, final_state_spec};
}

MambaSSDRecurrenceDeviceOperation::tensor_return_value_t MambaSSDRecurrenceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.states_bhcpn.device()),
        create_device_tensor(output_specs[1], tensor_args.states_bhcpn.device())};
}

ttsl::hash::hash_t MambaSSDRecurrenceDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<MambaSSDRecurrenceDeviceOperation>(
        args.memory_config,
        args.has_core_grid,
        args.core_grid_x,
        args.core_grid_y,
        tensor_args.states_bhcpn.dtype(),
        tensor_args.states_bhcpn.memory_config(),
        tensor_args.initial_states.dtype(),
        tensor_args.initial_states.memory_config(),
        tensor_args.a_end_bhc.dtype(),
        tensor_args.a_end_bhc.memory_config(),
        tensor_args.states_bhcpn.padded_shape(),
        tensor_args.initial_states.padded_shape(),
        tensor_args.a_end_bhc.padded_shape());
    return hash;
}

bool MambaSSDRecurrenceDeviceOperation::skip_launch(
    const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&) {
    return false;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> mamba_ssd_recurrence(
    const Tensor& states_bhcpn,
    const Tensor& initial_states,
    const Tensor& a_end_bhc,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::MambaSSDRecurrenceDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .has_core_grid = core_grid.has_value(),
        .core_grid_x = core_grid.has_value() ? core_grid->x : 0,
        .core_grid_y = core_grid.has_value() ? core_grid->y : 0};
    auto tensor_args = OperationType::tensor_args_t{
        .states_bhcpn = states_bhcpn, .initial_states = initial_states, .a_end_bhc = a_end_bhc};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
