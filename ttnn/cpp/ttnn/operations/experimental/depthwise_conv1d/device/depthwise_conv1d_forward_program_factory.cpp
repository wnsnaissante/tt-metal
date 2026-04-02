// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_forward_program_factory.hpp"

#include <array>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "depthwise_conv1d_forward_device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/reader_forward.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/compute_forward.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/depthwise_conv1d/device/kernels/writer_forward.cpp";

constexpr uint32_t kInputTapCb0 = tt::CBIndex::c_0;
constexpr uint32_t kBiasCbIndex = tt::CBIndex::c_8;
constexpr uint32_t kAccumCb0Index = tt::CBIndex::c_9;
constexpr uint32_t kAccumCb1Index = tt::CBIndex::c_10;
constexpr uint32_t kRawCurrentCbIndex = tt::CBIndex::c_11;
constexpr uint32_t kRawPrefixCbIndex = tt::CBIndex::c_12;
constexpr uint32_t kPartialCbIndex = tt::CBIndex::c_16;
constexpr uint32_t kOutputCbIndex = tt::CBIndex::c_17;
constexpr uint32_t kPreparedCbDepth = 2;
constexpr uint32_t kRawScratchCbDepth = 1;

}  // namespace

DepthwiseConv1dForwardProgramFactory::cached_program_t DepthwiseConv1dForwardProgramFactory::create(
    const DepthwiseConv1dForwardParams& operation_attributes,
    const DepthwiseConv1dForwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& x = tensor_args.x_blf;
    const auto& conv_state = tensor_args.conv_state_bfk;
    const auto& weight = tensor_args.weight_1fk;
    const auto& bias = tensor_args.bias_11f;
    auto* x_buffer = x.buffer();
    auto* conv_state_buffer = conv_state.buffer();
    auto* weight_buffer = weight.buffer();
    auto* bias_buffer = bias.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(x.dtype());
    const auto tile_size = tt::tile_size(data_format);
    const auto compute_grid = x.device()->compute_with_storage_grid_size();

    const auto& x_padded = x.padded_shape();
    const auto& state_padded = conv_state.padded_shape();
    const auto& output_padded = tensor_return_value.padded_shape();
    const uint32_t input_sequence_tiles = x_padded[1] / tt::constants::TILE_HEIGHT;
    const uint32_t output_sequence_tiles = output_padded[2] / tt::constants::TILE_HEIGHT;
    const uint32_t feature_tiles = output_padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t total_tiles = output_padded[0] * output_sequence_tiles * feature_tiles;
    const uint32_t x_batch_stride = input_sequence_tiles * feature_tiles;
    const uint32_t x_sequence_stride = feature_tiles;
    const uint32_t state_kernel_tiles = state_padded[2] / tt::constants::TILE_WIDTH;
    const uint32_t state_batch_stride = (state_padded[1] / tt::constants::TILE_HEIGHT) * state_kernel_tiles;
    const uint32_t state_feature_stride = state_kernel_tiles;

    const bool row_major = false;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_group_1, tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_tiles, row_major);
    auto cores = grid_to_cores(num_cores, compute_grid.x, compute_grid.y, row_major);

    for (uint32_t cb_index = kInputTapCb0; cb_index <= kBiasCbIndex; ++cb_index) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(kPreparedCbDepth * tile_size, {{cb_index, data_format}})
                .set_page_size(cb_index, tile_size));
    }
    for (const auto cb_index :
         std::array<uint32_t, 4>{kAccumCb0Index, kAccumCb1Index, kPartialCbIndex, kOutputCbIndex}) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(tile_size, {{cb_index, data_format}})
                .set_page_size(cb_index, tile_size));
    }
    for (const auto cb_index : std::array<uint32_t, 2>{kRawCurrentCbIndex, kRawPrefixCbIndex}) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(kRawScratchCbDepth * tile_size, {{cb_index, data_format}})
                .set_page_size(cb_index, tile_size));
    }

    std::vector<uint32_t> reader_compile_time_args = {operation_attributes.kernel_size};
    tt::tt_metal::TensorAccessorArgs(*x_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*conv_state_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weight_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*bias_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kOutputCbIndex};
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program, kReaderKernelPath, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program, kWriterKernelPath, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto compute_kernel_id_group_1 = tt::tt_metal::CreateKernel(
        program,
        kComputeKernelPath,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = {operation_attributes.kernel_size}});

    auto compute_kernel_id_group_2 = compute_kernel_id_group_1;
    if (!core_group_2.ranges().empty()) {
        compute_kernel_id_group_2 = tt::tt_metal::CreateKernel(
            program,
            kComputeKernelPath,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
                .compile_args = {operation_attributes.kernel_size}});
    }

    uint32_t next_start_tile = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = tiles_per_core_group_2;
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                x_buffer->address(),
                conv_state_buffer->address(),
                weight_buffer->address(),
                bias_buffer->address(),
                num_tiles_per_core,
                next_start_tile,
                tensor_return_value.logical_shape()[2],
                output_sequence_tiles,
                feature_tiles,
                x_batch_stride,
                x_sequence_stride,
                state_batch_stride,
                state_feature_stride,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output_buffer->address(), num_tiles_per_core, next_start_tile, output_sequence_tiles, feature_tiles});

        const auto compute_kernel_id =
            core_group_1.contains(core) ? compute_kernel_id_group_1 : compute_kernel_id_group_2;
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_per_core});
        next_start_tile += num_tiles_per_core;
    }

    return cached_program_t{
        std::move(program),
        {.cores = std::move(cores),
         .group_1_core_count = core_group_1.num_cores(),
         .tiles_per_core_group_1 = tiles_per_core_group_1,
         .tiles_per_core_group_2 = tiles_per_core_group_2,
         .reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id_group_1 = compute_kernel_id_group_1,
         .compute_kernel_id_group_2 = compute_kernel_id_group_2}};
}

void DepthwiseConv1dForwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DepthwiseConv1dForwardParams&,
    const DepthwiseConv1dForwardInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* x_buffer = tensor_args.x_blf.buffer();
    auto* conv_state_buffer = tensor_args.conv_state_bfk.buffer();
    auto* weight_buffer = tensor_args.weight_1fk.buffer();
    auto* bias_buffer = tensor_args.bias_11f.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    const auto& x_padded = tensor_args.x_blf.padded_shape();
    const auto& state_padded = tensor_args.conv_state_bfk.padded_shape();
    const auto& output_padded = tensor_return_value.padded_shape();
    const uint32_t input_sequence_tiles = x_padded[1] / tt::constants::TILE_HEIGHT;
    const uint32_t output_sequence_tiles = output_padded[2] / tt::constants::TILE_HEIGHT;
    const uint32_t feature_tiles = output_padded[3] / tt::constants::TILE_WIDTH;
    const uint32_t x_batch_stride = input_sequence_tiles * feature_tiles;
    const uint32_t x_sequence_stride = feature_tiles;
    const uint32_t state_kernel_tiles = state_padded[2] / tt::constants::TILE_WIDTH;
    const uint32_t state_batch_stride = (state_padded[1] / tt::constants::TILE_HEIGHT) * state_kernel_tiles;
    const uint32_t state_feature_stride = state_kernel_tiles;

    uint32_t next_start_tile = 0;
    auto& program = cached_program.program;
    for (uint32_t core_index = 0; core_index < cached_program.shared_variables.cores.size(); ++core_index) {
        const auto& core = cached_program.shared_variables.cores.at(core_index);
        const uint32_t num_tiles_per_core = core_index < cached_program.shared_variables.group_1_core_count
                                                ? cached_program.shared_variables.tiles_per_core_group_1
                                                : cached_program.shared_variables.tiles_per_core_group_2;

        auto& reader_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_runtime_args[0] = x_buffer->address();
        reader_runtime_args[1] = conv_state_buffer->address();
        reader_runtime_args[2] = weight_buffer->address();
        reader_runtime_args[3] = bias_buffer->address();
        reader_runtime_args[4] = num_tiles_per_core;
        reader_runtime_args[5] = next_start_tile;
        reader_runtime_args[6] = tensor_return_value.logical_shape()[2];
        reader_runtime_args[7] = output_sequence_tiles;
        reader_runtime_args[8] = feature_tiles;
        reader_runtime_args[9] = x_batch_stride;
        reader_runtime_args[10] = x_sequence_stride;
        reader_runtime_args[11] = state_batch_stride;
        reader_runtime_args[12] = state_feature_stride;

        auto& writer_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_runtime_args[0] = output_buffer->address();
        writer_runtime_args[1] = num_tiles_per_core;
        writer_runtime_args[2] = next_start_tile;
        writer_runtime_args[3] = output_sequence_tiles;
        writer_runtime_args[4] = feature_tiles;

        const auto compute_kernel_id = core_index < cached_program.shared_variables.group_1_core_count
                                           ? cached_program.shared_variables.compute_kernel_id_group_1
                                           : cached_program.shared_variables.compute_kernel_id_group_2;
        auto& compute_runtime_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_id, core);
        compute_runtime_args[0] = num_tiles_per_core;

        next_start_tile += num_tiles_per_core;
    }
}

}  // namespace ttnn::experimental::prim
