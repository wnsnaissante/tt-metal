// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_device_operation.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/types.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::depthwise_conv1d {

namespace {

constexpr uint32_t kBlockHeight = TILE_HEIGHT;
constexpr uint32_t kKernelDebugVersion = 3;

void set_runtime_arguments(
    Program& program,
    KernelHandle kernel_id,
    const std::vector<CoreCoord>& cores,
    const std::vector<std::vector<uint32_t>>& runtime_args) {
    TT_FATAL(cores.size() == runtime_args.size(), "Core/runtime-arg count mismatch");
    for (size_t i = 0; i < cores.size(); ++i) {
        SetRuntimeArgs(program, kernel_id, cores[i], runtime_args[i]);
    }
}

}  // namespace

DepthwiseConv1dDeviceOperation::ProgramFactory::cached_program_t DepthwiseConv1dDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;

    const uint32_t batch = input_tensor.logical_shape()[0];
    const uint32_t padded_sequence_length = input_tensor.logical_shape()[2];
    const uint32_t channels = attrs.channels;
    const uint32_t kernel_size = attrs.kernel_size;
    const uint32_t sequence_length = attrs.sequence_length;
    const uint32_t width_tiles = channels / TILE_WIDTH;
    const uint32_t blocks_per_batch = (sequence_length + kBlockHeight - 1) / kBlockHeight;
    const uint32_t total_blocks = batch * blocks_per_batch;

    const tt::DataFormat input_df = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat weight_df = datatype_to_dataformat_converter(weight_tensor.dtype());
    const uint32_t stick_nbytes = channels * datum_size(input_df);
    const uint32_t tile_nbytes = tt::tile_size(input_df);

    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    constexpr bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        split_work_to_cores(device_grid, total_blocks, row_major);
    std::vector<CoreCoord> cores = grid_to_cores(num_cores, device_grid.x, device_grid.y, row_major);

    Program program = CreateProgram();

    Buffer* src_buffer = input_tensor.buffer();
    Buffer* weight_buffer = weight_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();

    constexpr uint32_t cb_act_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_act_tiled = tt::CBIndex::c_1;
    constexpr uint32_t cb_weight_rm = tt::CBIndex::c_2;
    constexpr uint32_t cb_weight_tiled = tt::CBIndex::c_3;
    constexpr uint32_t cb_bias_rm = tt::CBIndex::c_4;
    constexpr uint32_t cb_bias_tiled = tt::CBIndex::c_5;
    constexpr uint32_t cb_partial = tt::CBIndex::c_6;
    constexpr uint32_t cb_out_tiled = tt::CBIndex::c_7;
    constexpr uint32_t cb_temp_sum = tt::CBIndex::c_8;
    constexpr uint32_t cb_out_rm0 = tt::CBIndex::c_9;
    constexpr uint32_t cb_out_rm1 = tt::CBIndex::c_10;

    create_cb(cb_act_rm, program, all_cores, stick_nbytes, kBlockHeight * kernel_size * 2, input_df);
    create_cb(cb_act_tiled, program, all_cores, tile_nbytes, width_tiles * 2, input_df);
    create_cb(cb_weight_rm, program, all_cores, stick_nbytes, kBlockHeight * kernel_size, weight_df);
    create_cb(cb_weight_tiled, program, all_cores, tile_nbytes, width_tiles * kernel_size, weight_df);
    create_cb(cb_bias_rm, program, all_cores, stick_nbytes, kBlockHeight * 2, output_df);
    create_cb(cb_bias_tiled, program, all_cores, tile_nbytes, width_tiles, output_df);
    create_cb(cb_partial, program, all_cores, tile_nbytes, width_tiles, output_df);
    create_cb(cb_out_tiled, program, all_cores, tile_nbytes, width_tiles, output_df);
    create_cb(cb_temp_sum, program, all_cores, tile_nbytes, width_tiles, output_df);
    create_cb(cb_out_rm0, program, all_cores, tile_nbytes, width_tiles, output_df);
    create_cb(cb_out_rm1, program, all_cores, tile_nbytes, width_tiles, output_df);

    const bool has_bias = attrs.has_bias;

    std::vector<uint32_t> reader_ct_args = {
        kBlockHeight,
        kernel_size,
        channels,
        padded_sequence_length,
        sequence_length,
        blocks_per_batch,
        stick_nbytes,
        cb_act_rm,
        cb_weight_rm,
        cb_bias_rm,
        static_cast<uint32_t>(has_bias),
        kKernelDebugVersion,
    };
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(weight_buffer).append_to(reader_ct_args);
    TensorAccessorArgs(has_bias ? tensor_args.bias_tensor->buffer() : nullptr).append_to(reader_ct_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/reader_causal_depthwise_conv1d.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> compute_ct_args = {
        width_tiles,
        kBlockHeight,
        kernel_size,
        cb_act_rm,
        cb_act_tiled,
        cb_weight_rm,
        cb_weight_tiled,
        cb_bias_rm,
        cb_bias_tiled,
        cb_partial,
        cb_out_tiled,
        cb_temp_sum,
        cb_out_rm0,
        cb_out_rm1,
        static_cast<uint32_t>(has_bias),
        static_cast<uint32_t>(attrs.silu_activation),
        kKernelDebugVersion,
    };

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/compute_causal_depthwise_conv1d.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_ct_args});

    std::vector<uint32_t> writer_ct_args = {
        kBlockHeight,
        sequence_length,
        blocks_per_batch,
        stick_nbytes,
        width_tiles,
        cb_out_rm0,
        cb_out_rm1,
        kKernelDebugVersion,
    };
    TensorAccessorArgs(dst_buffer).append_to(writer_ct_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/writer_causal_depthwise_conv1d.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    shared_variables_t shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .compute_kernel_id = compute_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .all_cores = all_cores,
        .cores = cores,
        .num_cores = num_cores,
        .g1_numcores = core_group_1.num_cores(),
        .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
        .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
    };

    cached_program_t cached_program{std::move(program), std::move(shared_variables)};
    override_runtime_arguments(cached_program, attrs, tensor_args, output_tensor);
    return cached_program;
}

void DepthwiseConv1dDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    const auto& cores = shared.cores;

    Buffer* src_buffer = tensor_args.input_tensor.buffer();
    Buffer* weight_buffer = tensor_args.weight_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    const uint32_t bias_addr = tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor->buffer()->address() : 0;

    std::vector<std::vector<uint32_t>> reader_runtime_args(cores.size(), std::vector<uint32_t>(5, 0));
    std::vector<std::vector<uint32_t>> compute_runtime_args(cores.size(), std::vector<uint32_t>(1, 0));
    std::vector<std::vector<uint32_t>> writer_runtime_args(cores.size(), std::vector<uint32_t>(3, 0));

    uint32_t next_block = 0;
    for (uint32_t core_idx = 0; core_idx < shared.num_cores; ++core_idx) {
        const uint32_t num_blocks =
            core_idx < shared.g1_numcores ? shared.num_blocks_per_core_group_1 : shared.num_blocks_per_core_group_2;

        reader_runtime_args[core_idx][0] = src_buffer->address();
        reader_runtime_args[core_idx][1] = weight_buffer->address();
        reader_runtime_args[core_idx][2] = bias_addr;
        reader_runtime_args[core_idx][3] = next_block;
        reader_runtime_args[core_idx][4] = num_blocks;

        compute_runtime_args[core_idx][0] = num_blocks;

        writer_runtime_args[core_idx][0] = dst_buffer->address();
        writer_runtime_args[core_idx][1] = next_block;
        writer_runtime_args[core_idx][2] = num_blocks;

        next_block += num_blocks;
    }

    set_runtime_arguments(program, shared.reader_kernel_id, cores, reader_runtime_args);
    set_runtime_arguments(program, shared.compute_kernel_id, cores, compute_runtime_args);
    set_runtime_arguments(program, shared.writer_kernel_id, cores, writer_runtime_args);
}

}  // namespace ttnn::operations::experimental::depthwise_conv1d
