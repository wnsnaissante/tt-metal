// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_chunk_scan_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

bool is_supported_dtype(const Tensor& tensor) {
    return tensor.dtype() == ttnn::DataType::BFLOAT16 || tensor.dtype() == ttnn::DataType::FLOAT32;
}

}  // namespace

MambaSSDChunkScanKernelConfig build_mamba_ssd_chunk_scan_kernel_config(
    const MambaSSDChunkScanParams&, const MambaSSDChunkScanInputs& tensor_args) {
    const auto& x = tensor_args.x_blk_bcthp.logical_shape();
    return MambaSSDChunkScanKernelConfig{
        .batch_size = static_cast<uint32_t>(x[0]),
        .num_chunks = static_cast<uint32_t>(x[1]),
        .chunk_size = static_cast<uint32_t>(x[2]),
        .num_heads = static_cast<uint32_t>(x[3]),
        .head_dim = static_cast<uint32_t>(x[4]),
        .state_size = static_cast<uint32_t>(tensor_args.b_blk_bctn.logical_shape()[3]),
        .p_tiles = tt::div_up(static_cast<uint32_t>(x[4]), tt::constants::TILE_WIDTH),
        .t_tiles = tt::div_up(static_cast<uint32_t>(x[2]), tt::constants::TILE_WIDTH)};
}

void MambaSSDChunkScanDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x_blk_bcthp.logical_shape();
    const auto& a = tensor_args.a_blk_bhct.logical_shape();
    const auto& b = tensor_args.b_blk_bctn.logical_shape();
    const auto& c = tensor_args.c_blk_bctn.logical_shape();

    TT_FATAL(tensor_args.x_blk_bcthp.device() != nullptr, "x_blk_bcthp must be on device");
    TT_FATAL(tensor_args.a_blk_bhct.device() != nullptr, "a_blk_bhct must be on device");
    TT_FATAL(tensor_args.b_blk_bctn.device() != nullptr, "b_blk_bctn must be on device");
    TT_FATAL(tensor_args.c_blk_bctn.device() != nullptr, "c_blk_bctn must be on device");

    auto* device = tensor_args.x_blk_bcthp.device();
    TT_FATAL(tensor_args.a_blk_bhct.device() == device, "a_blk_bhct must share device");
    TT_FATAL(tensor_args.b_blk_bctn.device() == device, "b_blk_bctn must share device");
    TT_FATAL(tensor_args.c_blk_bctn.device() == device, "c_blk_bctn must share device");

    TT_FATAL(is_supported_dtype(tensor_args.x_blk_bcthp), "x_blk_bcthp must be BF16/FLOAT32");
    TT_FATAL(is_supported_dtype(tensor_args.a_blk_bhct), "a_blk_bhct must be BF16/FLOAT32");
    TT_FATAL(is_supported_dtype(tensor_args.b_blk_bctn), "b_blk_bctn must be BF16/FLOAT32");
    TT_FATAL(is_supported_dtype(tensor_args.c_blk_bctn), "c_blk_bctn must be BF16/FLOAT32");

    TT_FATAL(tensor_args.x_blk_bcthp.layout() == ttnn::TILE_LAYOUT, "x_blk_bcthp must be TILE layout");
    TT_FATAL(tensor_args.a_blk_bhct.layout() == ttnn::TILE_LAYOUT, "a_blk_bhct must be TILE layout");
    TT_FATAL(tensor_args.b_blk_bctn.layout() == ttnn::TILE_LAYOUT, "b_blk_bctn must be TILE layout");
    TT_FATAL(tensor_args.c_blk_bctn.layout() == ttnn::TILE_LAYOUT, "c_blk_bctn must be TILE layout");

    TT_FATAL(x.rank() == 5, "x_blk_bcthp must be rank 5");
    TT_FATAL(a.rank() == 4, "a_blk_bhct must be rank 4");
    TT_FATAL(b.rank() == 4, "b_blk_bctn must be rank 4");
    TT_FATAL(c.rank() == 4, "c_blk_bctn must be rank 4");

    const uint32_t B = x[0];
    const uint32_t C = x[1];
    const uint32_t T = x[2];
    const uint32_t H = x[3];
    const uint32_t P = x[4];
    const uint32_t N = b[3];

    TT_FATAL(T == 134, "mamba_ssd_chunk_scan v1 supports chunk_size=134 only");
    TT_FATAL(H == 8, "mamba_ssd_chunk_scan v1 supports num_heads=8 only");
    TT_FATAL(P == 64, "mamba_ssd_chunk_scan v1 supports head_dim=64 only");
    TT_FATAL(N == 16, "mamba_ssd_chunk_scan v1 supports state_size=16 only");
    TT_FATAL(a[0] == B && a[1] == H && a[2] == C && a[3] == T, "a_blk_bhct shape mismatch");
    TT_FATAL(b[0] == B && b[1] == C && b[2] == T && b[3] == N, "b_blk_bctn shape mismatch");
    TT_FATAL(c[0] == B && c[1] == C && c[2] == T && c[3] == N, "c_blk_bctn shape mismatch");

    TT_FATAL(!args.memory_config.is_sharded(), "mamba_ssd_chunk_scan outputs must be interleaved");
    TT_FATAL(!tensor_args.x_blk_bcthp.memory_config().is_sharded(), "x_blk_bcthp must be interleaved");
    TT_FATAL(!tensor_args.a_blk_bhct.memory_config().is_sharded(), "a_blk_bhct must be interleaved");
    TT_FATAL(!tensor_args.b_blk_bctn.memory_config().is_sharded(), "b_blk_bctn must be interleaved");
    TT_FATAL(!tensor_args.c_blk_bctn.memory_config().is_sharded(), "c_blk_bctn must be interleaved");
}

void MambaSSDChunkScanDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

MambaSSDChunkScanDeviceOperation::spec_return_value_t MambaSSDChunkScanDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x_blk_bcthp.logical_shape();
    const uint32_t B = x[0];
    const uint32_t C = x[1];
    const uint32_t T = x[2];
    const uint32_t H = x[3];
    const uint32_t P = x[4];
    const uint32_t N = tensor_args.b_blk_bctn.logical_shape()[3];

    auto tile_layout = tt::tt_metal::TensorLayout(
        tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), args.memory_config);
    return {
        TensorSpec(ttnn::Shape({B, C, T, H, P}), tile_layout),
        TensorSpec(ttnn::Shape({B, H, C, P, N}), tile_layout),
        TensorSpec(ttnn::Shape({B, H, C, T}), tile_layout)};
}

MambaSSDChunkScanDeviceOperation::tensor_return_value_t MambaSSDChunkScanDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(specs[0], tensor_args.x_blk_bcthp.device()),
        create_device_tensor(specs[1], tensor_args.x_blk_bcthp.device()),
        create_device_tensor(specs[2], tensor_args.x_blk_bcthp.device())};
}

ttsl::hash::hash_t MambaSSDChunkScanDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<MambaSSDChunkScanDeviceOperation>(
        args.memory_config,
        args.has_core_grid,
        args.core_grid_x,
        args.core_grid_y,
        tensor_args.x_blk_bcthp.dtype(),
        tensor_args.x_blk_bcthp.memory_config(),
        tensor_args.a_blk_bhct.dtype(),
        tensor_args.a_blk_bhct.memory_config(),
        tensor_args.b_blk_bctn.dtype(),
        tensor_args.b_blk_bctn.memory_config(),
        tensor_args.c_blk_bctn.dtype(),
        tensor_args.c_blk_bctn.memory_config(),
        tensor_args.x_blk_bcthp.padded_shape(),
        tensor_args.a_blk_bhct.padded_shape(),
        tensor_args.b_blk_bctn.padded_shape(),
        tensor_args.c_blk_bctn.padded_shape());
}

bool MambaSSDChunkScanDeviceOperation::skip_launch(
    const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&) {
    return false;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::experimental::prim::MambaSSDChunkScanDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        .has_core_grid = core_grid.has_value(),
        .core_grid_x = core_grid.has_value() ? core_grid->x : 0,
        .core_grid_y = core_grid.has_value() ? core_grid->y : 0};
    auto tensor_args = OperationType::tensor_args_t{
        .x_blk_bcthp = x_blk_bcthp, .a_blk_bhct = a_blk_bhct, .b_blk_bctn = b_blk_bctn, .c_blk_bctn = c_blk_bctn};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
