// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_output_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

std::optional<CoreGrid> get_core_grid(const MambaSSDOutputParams& operation_attributes) {
    if (!operation_attributes.has_core_grid) {
        return std::nullopt;
    }
    return CoreGrid{operation_attributes.core_grid_x, operation_attributes.core_grid_y};
}

bool is_supported_dtype(const Tensor& tensor) {
    return tensor.dtype() == ttnn::DataType::BFLOAT16 || tensor.dtype() == ttnn::DataType::FLOAT32;
}

bool can_use_chunk_output(const MambaSSDOutputParams& args, const MambaSSDOutputInputs& tensor_args) {
    const auto config = build_mamba_ssd_output_kernel_config(args, tensor_args);
    return config.batch_size == 32 && config.num_chunks == 1 && config.chunk_size == 134 &&
           (config.seq_len == 134 || config.seq_len == 133) && config.num_heads == 8 && config.head_dim == 64 &&
           config.hidden_dim == 512 && config.state_size <= 32 && !args.memory_config.is_sharded();
}

}  // namespace

MambaSSDOutputKernelConfig build_mamba_ssd_output_kernel_config(
    const MambaSSDOutputParams& operation_attributes, const MambaSSDOutputInputs& tensor_args) {
    const auto& y_shape = tensor_args.y_diag_bcthp.logical_shape();
    return MambaSSDOutputKernelConfig{
        .batch_size = static_cast<uint32_t>(y_shape[0]),
        .num_chunks = static_cast<uint32_t>(y_shape[1]),
        .chunk_size = static_cast<uint32_t>(y_shape[2]),
        .num_heads = static_cast<uint32_t>(y_shape[3]),
        .head_dim = static_cast<uint32_t>(y_shape[4]),
        .state_size = static_cast<uint32_t>(tensor_args.states_out_bchpn.logical_shape()[4]),
        .seq_len = operation_attributes.seq_len,
        .pad_size = operation_attributes.pad_size,
        .hidden_dim = static_cast<uint32_t>(y_shape[3] * y_shape[4])};
}

Tensor mamba_ssd_output_fallback(
    const MambaSSDOutputParams& operation_attributes, const MambaSSDOutputInputs& tensor_args) {
    const auto& y_diag_bcthp = tensor_args.y_diag_bcthp;
    const auto& states_out_bchpn = tensor_args.states_out_bchpn;
    const auto& c_blk_bctn = tensor_args.c_blk_bctn;
    const auto& a_cumsum_bhct = tensor_args.a_cumsum_bhct;
    const auto& x_orig_blk_bcthp = tensor_args.x_orig_blk_bcthp;
    const auto& mem = operation_attributes.memory_config;
    const auto& y = y_diag_bcthp.logical_shape();

    const uint32_t B = y[0];
    const uint32_t C = y[1];
    const uint32_t T = y[2];
    const uint32_t H = y[3];
    const uint32_t P = y[4];
    const uint32_t N = states_out_bchpn.logical_shape()[4];

    auto c_out_flat = ttnn::reshape(c_blk_bctn, ttnn::Shape({B * C, T, N}), mem);
    c_out_flat = ttnn::permute(c_out_flat, ttnn::SmallVector<int64_t>{0, 2, 1}, mem);

    auto states_out_flat = ttnn::reshape(states_out_bchpn, ttnn::Shape({B * C, H * P, N}), mem);
    auto y_off_bcthp = ttnn::matmul(
        states_out_flat,
        c_out_flat,
        false,
        false,
        mem,
        ttnn::DataType::FLOAT32,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        get_core_grid(operation_attributes));
    y_off_bcthp = ttnn::reshape(y_off_bcthp, ttnn::Shape({B, C, H, P, T}), mem);
    y_off_bcthp = ttnn::permute(y_off_bcthp, ttnn::SmallVector<int64_t>{0, 1, 4, 2, 3}, mem);

    auto a_cumsum_bcht = ttnn::permute(a_cumsum_bhct, ttnn::SmallVector<int64_t>{0, 2, 1, 3}, mem);
    auto state_decay_out = ttnn::exp(a_cumsum_bcht, false, mem);
    state_decay_out = ttnn::permute(state_decay_out, ttnn::SmallVector<int64_t>{0, 1, 3, 2}, mem);
    state_decay_out = ttnn::unsqueeze(state_decay_out, -1);
    state_decay_out = ttnn::to_memory_config(state_decay_out, mem);
    y_off_bcthp = ttnn::multiply(y_off_bcthp, state_decay_out, std::nullopt, mem);

    auto y_bcthp = ttnn::add(y_diag_bcthp, y_off_bcthp, std::nullopt, mem);

    auto y_blhp = ttnn::reshape(y_bcthp, ttnn::Shape({B, C * T, H, P}), mem);
    if (operation_attributes.pad_size > 0) {
        const auto step = ttnn::SmallVector<uint32_t>{1, 1, 1, 1};
        y_blhp = ttnn::slice(
            y_blhp,
            ttnn::SmallVector<uint32_t>{0, 0, 0, 0},
            ttnn::SmallVector<uint32_t>{B, operation_attributes.seq_len, H, P},
            step,
            mem);
    }

    auto x_blhp = ttnn::reshape(x_orig_blk_bcthp, ttnn::Shape({B, C * T, H, P}), mem);
    if (operation_attributes.pad_size > 0) {
        const auto step = ttnn::SmallVector<uint32_t>{1, 1, 1, 1};
        x_blhp = ttnn::slice(
            x_blhp,
            ttnn::SmallVector<uint32_t>{0, 0, 0, 0},
            ttnn::SmallVector<uint32_t>{B, operation_attributes.seq_len, H, P},
            step,
            mem);
    }
    auto d_residual = tensor_args.d_h;
    if (d_residual.logical_shape().rank() != 4) {
        d_residual = ttnn::reshape(d_residual, ttnn::Shape({1, 1, H, 1}), mem);
    }
    auto residual = ttnn::multiply(d_residual, x_blhp, std::nullopt, mem);
    y_blhp = ttnn::add(y_blhp, residual, std::nullopt, mem);

    return ttnn::reshape(y_blhp, ttnn::Shape({B, operation_attributes.seq_len, H * P}), mem);
}

void MambaSSDOutputDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& y = tensor_args.y_diag_bcthp.logical_shape();
    const auto& s = tensor_args.states_out_bchpn.logical_shape();
    const auto& c = tensor_args.c_blk_bctn.logical_shape();
    const auto& a = tensor_args.a_cumsum_bhct.logical_shape();
    const auto& x = tensor_args.x_orig_blk_bcthp.logical_shape();

    TT_FATAL(tensor_args.y_diag_bcthp.device() != nullptr, "y_diag_bcthp must be on device");
    TT_FATAL(tensor_args.states_out_bchpn.device() != nullptr, "states_out_bchpn must be on device");
    TT_FATAL(tensor_args.c_blk_bctn.device() != nullptr, "c_blk_bctn must be on device");
    TT_FATAL(tensor_args.a_cumsum_bhct.device() != nullptr, "a_cumsum_bhct must be on device");
    TT_FATAL(tensor_args.x_orig_blk_bcthp.device() != nullptr, "x_orig_blk_bcthp must be on device");
    TT_FATAL(tensor_args.d_h.device() != nullptr, "d_h must be on device");

    auto* device = tensor_args.y_diag_bcthp.device();
    TT_FATAL(tensor_args.states_out_bchpn.device() == device, "states_out_bchpn must share device");
    TT_FATAL(tensor_args.c_blk_bctn.device() == device, "c_blk_bctn must share device");
    TT_FATAL(tensor_args.a_cumsum_bhct.device() == device, "a_cumsum_bhct must share device");
    TT_FATAL(tensor_args.x_orig_blk_bcthp.device() == device, "x_orig_blk_bcthp must share device");
    TT_FATAL(tensor_args.d_h.device() == device, "d_h must share device");
    TT_FATAL(is_supported_dtype(tensor_args.y_diag_bcthp), "ssd_output kernel expects BF16/FLOAT32 y_diag_bcthp");
    TT_FATAL(
        is_supported_dtype(tensor_args.states_out_bchpn), "ssd_output kernel expects BF16/FLOAT32 states_out_bchpn");
    TT_FATAL(is_supported_dtype(tensor_args.c_blk_bctn), "ssd_output kernel expects BF16/FLOAT32 c_blk_bctn");
    TT_FATAL(is_supported_dtype(tensor_args.a_cumsum_bhct), "ssd_output kernel expects BF16/FLOAT32 a_cumsum_bhct");
    TT_FATAL(
        is_supported_dtype(tensor_args.x_orig_blk_bcthp), "ssd_output kernel expects BF16/FLOAT32 x_orig_blk_bcthp");
    TT_FATAL(is_supported_dtype(tensor_args.d_h), "ssd_output kernel expects BF16/FLOAT32 d_h");
    TT_FATAL(tensor_args.y_diag_bcthp.layout() == ttnn::TILE_LAYOUT, "y_diag_bcthp must be TILE layout");
    TT_FATAL(tensor_args.states_out_bchpn.layout() == ttnn::TILE_LAYOUT, "states_out_bchpn must be TILE layout");
    TT_FATAL(tensor_args.c_blk_bctn.layout() == ttnn::TILE_LAYOUT, "c_blk_bctn must be TILE layout");
    TT_FATAL(tensor_args.a_cumsum_bhct.layout() == ttnn::TILE_LAYOUT, "a_cumsum_bhct must be TILE layout");
    TT_FATAL(tensor_args.x_orig_blk_bcthp.layout() == ttnn::TILE_LAYOUT, "x_orig_blk_bcthp must be TILE layout");
    TT_FATAL(tensor_args.d_h.layout() == ttnn::TILE_LAYOUT, "d_h must be TILE layout");

    TT_FATAL(y.rank() == 5, "y_diag_bcthp must be rank 5");
    TT_FATAL(s.rank() == 5, "states_out_bchpn must be rank 5");
    TT_FATAL(c.rank() == 4, "c_blk_bctn must be rank 4");
    TT_FATAL(a.rank() == 4, "a_cumsum_bhct must be rank 4");
    TT_FATAL(x.rank() == 5, "x_orig_blk_bcthp must be rank 5");

    const uint32_t B = y[0];
    const uint32_t C = y[1];
    const uint32_t T = y[2];
    const uint32_t H = y[3];
    const uint32_t P = y[4];
    const uint32_t N = s[4];

    TT_FATAL(s[0] == B && s[1] == C && s[2] == H && s[3] == P, "states_out_bchpn shape mismatch");
    TT_FATAL(c[0] == B && c[1] == C && c[2] == T && c[3] == N, "c_blk_bctn shape mismatch");
    TT_FATAL(a[0] == B && a[1] == H && a[2] == C && a[3] == T, "a_cumsum_bhct shape mismatch");
    TT_FATAL(x[0] == B && x[1] == C && x[2] == T && x[3] == H && x[4] == P, "x_orig_blk_bcthp shape mismatch");
    TT_FATAL(tensor_args.d_h.logical_shape().volume() == H, "d_h volume must match num_heads");
    TT_FATAL(args.seq_len > 0, "seq_len must be positive");
    TT_FATAL(args.seq_len + args.pad_size == C * T, "seq_len + pad_size must equal num_chunks * chunk_size");
    TT_FATAL(!args.memory_config.is_sharded(), "ssd_output output memory_config must be interleaved");
    TT_FATAL(!tensor_args.y_diag_bcthp.memory_config().is_sharded(), "y_diag_bcthp must be interleaved");
    TT_FATAL(!tensor_args.states_out_bchpn.memory_config().is_sharded(), "states_out_bchpn must be interleaved");
    TT_FATAL(!tensor_args.c_blk_bctn.memory_config().is_sharded(), "c_blk_bctn must be interleaved");
    TT_FATAL(!tensor_args.a_cumsum_bhct.memory_config().is_sharded(), "a_cumsum_bhct must be interleaved");
    TT_FATAL(!tensor_args.x_orig_blk_bcthp.memory_config().is_sharded(), "x_orig_blk_bcthp must be interleaved");
    TT_FATAL(!tensor_args.d_h.memory_config().is_sharded(), "d_h must be interleaved");
}

void MambaSSDOutputDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

MambaSSDOutputDeviceOperation::spec_return_value_t MambaSSDOutputDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& y = tensor_args.y_diag_bcthp.logical_shape();
    const uint32_t B = y[0];
    const uint32_t H = y[3];
    const uint32_t P = y[4];
    const auto logical_shape = ttnn::Shape({B, args.seq_len, H * P});
    if (can_use_chunk_output(args, tensor_args)) {
        const uint32_t padded_seq_len = ((args.seq_len + 31) / 32) * 32;
        const uint32_t padded_hidden = (((H * P) + 31) / 32) * 32;
        const auto padded_shape = ttnn::Shape({B, padded_seq_len, padded_hidden});
        return TensorSpec(
            logical_shape,
            tt::tt_metal::TensorLayout::fromPaddedShape(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                args.memory_config,
                logical_shape,
                padded_shape));
    }
    return TensorSpec(
        logical_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            args.memory_config,
            logical_shape,
            logical_shape));
}

MambaSSDOutputDeviceOperation::tensor_return_value_t MambaSSDOutputDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.y_diag_bcthp.device());
}

ttsl::hash::hash_t MambaSSDOutputDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MambaSSDOutputDeviceOperation>(
        args.memory_config,
        args.has_core_grid,
        args.core_grid_x,
        args.core_grid_y,
        args.seq_len,
        args.pad_size,
        can_use_chunk_output(args, tensor_args),
        tensor_args.y_diag_bcthp.dtype(),
        tensor_args.y_diag_bcthp.memory_config(),
        tensor_args.states_out_bchpn.dtype(),
        tensor_args.states_out_bchpn.memory_config(),
        tensor_args.c_blk_bctn.dtype(),
        tensor_args.c_blk_bctn.memory_config(),
        tensor_args.a_cumsum_bhct.dtype(),
        tensor_args.a_cumsum_bhct.memory_config(),
        tensor_args.x_orig_blk_bcthp.dtype(),
        tensor_args.x_orig_blk_bcthp.memory_config(),
        tensor_args.d_h.dtype(),
        tensor_args.d_h.memory_config(),
        tensor_args.y_diag_bcthp.padded_shape(),
        tensor_args.states_out_bchpn.padded_shape(),
        tensor_args.c_blk_bctn.padded_shape(),
        tensor_args.a_cumsum_bhct.padded_shape(),
        tensor_args.x_orig_blk_bcthp.padded_shape(),
        tensor_args.d_h.padded_shape());
}

bool MambaSSDOutputDeviceOperation::skip_launch(
    const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&) {
    return false;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mamba_ssd_output(
    const Tensor& y_diag_bcthp,
    const Tensor& states_out_bchpn,
    const Tensor& c_blk_bctn,
    const Tensor& a_cumsum_bhct,
    const Tensor& x_orig_blk_bcthp,
    const Tensor& d_h,
    uint32_t seq_len,
    uint32_t pad_size,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::MambaSSDOutputDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .has_core_grid = core_grid.has_value(),
        .core_grid_x = core_grid.has_value() ? core_grid->x : 0,
        .core_grid_y = core_grid.has_value() ? core_grid->y : 0,
        .seq_len = seq_len,
        .pad_size = pad_size};
    auto tensor_args = OperationType::tensor_args_t{
        .y_diag_bcthp = y_diag_bcthp,
        .states_out_bchpn = states_out_bchpn,
        .c_blk_bctn = c_blk_bctn,
        .a_cumsum_bhct = a_cumsum_bhct,
        .x_orig_blk_bcthp = x_orig_blk_bcthp,
        .d_h = d_h};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
