// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_output_program_factory.hpp"

#include "mamba_ssd_output_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace {

struct OutputRowWorkSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    CoreRangeSet all_cores = CoreRangeSet{};
    CoreRangeSet core_group_1 = CoreRangeSet{};
    CoreRangeSet core_group_2 = CoreRangeSet{};
    uint32_t rows_per_core_group_1 = 0;
    uint32_t rows_per_core_group_2 = 0;
};

OutputRowWorkSplit split_output_row_work(
    const MambaSSDOutputParams& operation_attributes,
    const MambaSSDOutputKernelConfig& kernel_config,
    tt::tt_metal::IDevice* device) {
    const uint32_t total_rows = kernel_config.batch_size * kernel_config.seq_len;
    (void)operation_attributes;
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_group_1, rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_rows);
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    return OutputRowWorkSplit{
        .num_cores = num_cores,
        .num_cores_y = num_cores_y,
        .all_cores = all_cores,
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .rows_per_core_group_1 = rows_per_core_group_1,
        .rows_per_core_group_2 = rows_per_core_group_2,
    };
}

constexpr uint32_t kDTypeBFloat16 = 0;
constexpr uint32_t kDTypeFloat32 = 1;

uint32_t encode_supported_dtype(const Tensor& tensor) {
    switch (tensor.dtype()) {
        case ttnn::DataType::BFLOAT16: return kDTypeBFloat16;
        case ttnn::DataType::FLOAT32: return kDTypeFloat32;
        default: TT_THROW("mamba_ssd_output kernel supports BF16/FLOAT32 inputs only");
    }
}

struct WorkRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

WorkRange split_linear_work(uint32_t total, uint32_t num_cores, uint32_t core_index) {
    const uint32_t base = total / num_cores;
    const uint32_t rem = total % num_cores;
    const uint32_t count = base + (core_index < rem ? 1 : 0);
    const uint32_t start = core_index * base + std::min(core_index, rem);
    return {.start = start, .count = count};
}

bool can_use_chunk_output(const MambaSSDOutputKernelConfig& kernel_config) {
    const uint32_t p_tiles = tt::div_up(kernel_config.head_dim, tt::constants::TILE_WIDTH);
    const uint32_t n_tiles = tt::div_up(kernel_config.state_size, tt::constants::TILE_WIDTH);
    return kernel_config.batch_size == 32 && kernel_config.num_chunks == 1 && kernel_config.chunk_size == 134 &&
           (kernel_config.seq_len == 134 || kernel_config.seq_len == 133) && kernel_config.num_heads == 8 &&
           kernel_config.head_dim == 64 && kernel_config.hidden_dim == 512 && p_tiles == 2 &&
           kernel_config.state_size <= 32 && n_tiles == 1;
}

}  // namespace

MambaSSDOutputProgramFactory::cached_program_t MambaSSDOutputProgramFactory::create(
    const MambaSSDOutputParams& operation_attributes,
    const MambaSSDOutputInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};
    auto* device = tensor_args.y_diag_bcthp.device();
    const auto kernel_config = build_mamba_ssd_output_kernel_config(operation_attributes, tensor_args);
    const auto work_split = split_output_row_work(operation_attributes, kernel_config, device);

    const auto y_diag_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.y_diag_bcthp.dtype());
    const auto states_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tensor_args.states_out_bchpn.dtype());
    const auto c_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.c_blk_bctn.dtype());
    const auto a_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.a_cumsum_bhct.dtype());
    const auto x_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.x_orig_blk_bcthp.dtype());
    const auto d_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.d_h.dtype());
    const auto out_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto fp32_tile_data_format = tt::DataFormat::Float32;

    const uint32_t y_diag_page_size = tensor_args.y_diag_bcthp.buffer()->aligned_page_size();
    const uint32_t states_page_size = tensor_args.states_out_bchpn.buffer()->aligned_page_size();
    const uint32_t c_page_size = tensor_args.c_blk_bctn.buffer()->aligned_page_size();
    const uint32_t a_page_size = tensor_args.a_cumsum_bhct.buffer()->aligned_page_size();
    const uint32_t x_page_size = tensor_args.x_orig_blk_bcthp.buffer()->aligned_page_size();
    const uint32_t d_page_size = tensor_args.d_h.buffer()->aligned_page_size();
    const uint32_t out_page_size = tensor_return_value.buffer()->aligned_page_size();
    const uint32_t fp32_tile_page_size = tt::tile_size(fp32_tile_data_format);
    const uint32_t n_tiles = tt::div_up(kernel_config.state_size, tt::constants::TILE_WIDTH);
    const uint32_t p_tiles = tt::div_up(kernel_config.head_dim, tt::constants::TILE_WIDTH);
    const uint32_t h_tiles = tt::div_up(kernel_config.num_heads, tt::constants::TILE_HEIGHT);
    const auto& d_shape = tensor_args.d_h.logical_shape();
    const bool d_values_in_rows = d_shape.rank() > 1 && d_shape[d_shape.rank() - 1] == 1;
    const bool use_chunk_output = can_use_chunk_output(kernel_config);

    if (use_chunk_output) {
        const uint32_t total_units = kernel_config.batch_size;
        const auto compute_grid = device->compute_with_storage_grid_size();
        const auto [num_cores, all_cores, _cg1, _cg2, _r1, _r2] =
            tt::tt_metal::split_work_to_cores(compute_grid, total_units);
        const uint32_t num_cores_y = compute_grid.y;

        constexpr uint32_t states_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t c_raw_cb_index = tt::CBIndex::c_1;
        constexpr uint32_t c_t_cb_index = tt::CBIndex::c_2;
        constexpr uint32_t off_tile_cb_index = tt::CBIndex::c_3;
        constexpr uint32_t scratch_y_cb_index = tt::CBIndex::c_4;
        constexpr uint32_t scratch_x_cb_index = tt::CBIndex::c_5;
        constexpr uint32_t scratch_a_cb_index = tt::CBIndex::c_6;
        constexpr uint32_t scratch_d_cb_index = tt::CBIndex::c_7;
        constexpr uint32_t scratch_c_cb_index = tt::CBIndex::c_8;
        constexpr uint32_t decay_cb_index = tt::CBIndex::c_9;
        constexpr uint32_t c_scaled_cb_index = tt::CBIndex::c_10;
        constexpr uint32_t out_row_cb_index = tt::CBIndex::c_16;

        const uint32_t state_tiles = kernel_config.num_heads * p_tiles;

        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(states_page_size * state_tiles, {{states_cb_index, states_data_format}})
                .set_page_size(states_cb_index, states_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(states_page_size, {{c_raw_cb_index, states_data_format}})
                .set_page_size(c_raw_cb_index, states_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(states_page_size, {{c_t_cb_index, states_data_format}})
                .set_page_size(c_t_cb_index, states_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(
                fp32_tile_page_size * state_tiles, {{off_tile_cb_index, fp32_tile_data_format}})
                .set_page_size(off_tile_cb_index, fp32_tile_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(
                fp32_tile_page_size * state_tiles, {{decay_cb_index, fp32_tile_data_format}})
                .set_page_size(decay_cb_index, fp32_tile_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{c_scaled_cb_index, fp32_tile_data_format}})
                .set_page_size(c_scaled_cb_index, fp32_tile_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(y_diag_page_size, {{scratch_y_cb_index, y_diag_data_format}})
                .set_page_size(scratch_y_cb_index, y_diag_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(x_page_size, {{scratch_x_cb_index, x_data_format}})
                .set_page_size(scratch_x_cb_index, x_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(
                a_page_size * kernel_config.num_heads, {{scratch_a_cb_index, a_data_format}})
                .set_page_size(scratch_a_cb_index, a_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(d_page_size, {{scratch_d_cb_index, d_data_format}})
                .set_page_size(scratch_d_cb_index, d_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(c_page_size, {{scratch_c_cb_index, c_data_format}})
                .set_page_size(scratch_c_cb_index, c_page_size));
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(
                out_page_size * kernel_config.num_heads, {{out_row_cb_index, out_data_format}})
                .set_page_size(out_row_cb_index, out_page_size));

        std::vector<uint32_t> reader_compile_args = {
            states_cb_index,
            c_raw_cb_index,
            scratch_c_cb_index,
            decay_cb_index,
            scratch_a_cb_index,
            states_page_size,
            c_page_size,
            a_page_size,
            encode_supported_dtype(tensor_args.c_blk_bctn),
            encode_supported_dtype(tensor_args.a_cumsum_bhct)};
        tt::tt_metal::TensorAccessorArgs(tensor_args.states_out_bchpn.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.c_blk_bctn.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.a_cumsum_bhct.buffer()).append_to(reader_compile_args);

        std::vector<uint32_t> compute_compile_args = {
            states_cb_index, c_raw_cb_index, c_t_cb_index, off_tile_cb_index, decay_cb_index, c_scaled_cb_index};

        std::vector<uint32_t> writer_compile_args = {
            off_tile_cb_index,
            out_row_cb_index,
            scratch_y_cb_index,
            scratch_x_cb_index,
            scratch_a_cb_index,
            scratch_d_cb_index,
            out_page_size,
            y_diag_page_size,
            x_page_size,
            a_page_size,
            d_page_size,
            fp32_tile_page_size,
            encode_supported_dtype(tensor_args.y_diag_bcthp),
            encode_supported_dtype(tensor_args.x_orig_blk_bcthp),
            encode_supported_dtype(tensor_args.a_cumsum_bhct),
            encode_supported_dtype(tensor_args.d_h),
            d_values_in_rows ? 1u : 0u,
        };
        tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.y_diag_bcthp.buffer()).append_to(writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.x_orig_blk_bcthp.buffer()).append_to(writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.a_cumsum_bhct.buffer()).append_to(writer_compile_args);
        tt::tt_metal::TensorAccessorArgs(tensor_args.d_h.buffer()).append_to(writer_compile_args);

        const auto reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            std::string{kReaderChunkKernelPath},
            all_cores,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
        const auto compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            std::string{kComputeChunkKernelPath},
            all_cores,
            tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = true, .compile_args = compute_compile_args});
        const auto writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            std::string{kWriterChunkKernelPath},
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto split_range = split_linear_work(total_units, num_cores, i);
            const WorkRange unit_range{
                .start = split_range.start * kernel_config.num_chunks,
                .count = split_range.count * kernel_config.num_chunks};
            tt::tt_metal::SetRuntimeArgs(
                program,
                reader_kernel_id,
                core,
                {
                    tensor_args.states_out_bchpn.buffer()->address(),
                    tensor_args.c_blk_bctn.buffer()->address(),
                    tensor_args.a_cumsum_bhct.buffer()->address(),
                    unit_range.start,
                    unit_range.count,
                    kernel_config.batch_size,
                    kernel_config.num_chunks,
                    kernel_config.chunk_size,
                    kernel_config.seq_len,
                    kernel_config.num_heads,
                    kernel_config.head_dim,
                    kernel_config.state_size,
                });
            tt::tt_metal::SetRuntimeArgs(
                program,
                compute_kernel_id,
                core,
                {
                    unit_range.count,
                    kernel_config.chunk_size,
                    kernel_config.num_heads,
                    kernel_config.head_dim,
                });
            tt::tt_metal::SetRuntimeArgs(
                program,
                writer_kernel_id,
                core,
                {
                    tensor_return_value.buffer()->address(),
                    tensor_args.y_diag_bcthp.buffer()->address(),
                    tensor_args.x_orig_blk_bcthp.buffer()->address(),
                    tensor_args.a_cumsum_bhct.buffer()->address(),
                    tensor_args.d_h.buffer()->address(),
                    unit_range.start,
                    unit_range.count,
                    kernel_config.batch_size,
                    kernel_config.num_chunks,
                    kernel_config.chunk_size,
                    kernel_config.seq_len,
                    kernel_config.num_heads,
                    kernel_config.head_dim,
                });
        }

        return {
            std::move(program),
            {.reader_kernel_id = reader_kernel_id,
             .compute_kernel_id = compute_kernel_id,
             .writer_kernel_id = writer_kernel_id,
             .use_chunk_output = true}};
    }

    constexpr uint32_t y_diag_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t states_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t c_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t a_cb_index = tt::CBIndex::c_3;
    constexpr uint32_t states_t_cb_index = tt::CBIndex::c_4;
    constexpr uint32_t decay_cb_index = tt::CBIndex::c_5;
    constexpr uint32_t scaled_cb_index = tt::CBIndex::c_6;
    constexpr uint32_t dot_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t scratch_a_cb_index = tt::CBIndex::c_8;
    constexpr uint32_t scratch_c_cb_index = tt::CBIndex::c_9;
    constexpr uint32_t scratch_y_cb_index = tt::CBIndex::c_10;
    constexpr uint32_t out_tile_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t out_row_cb_index = tt::CBIndex::c_17;
    constexpr uint32_t scratch_x_cb_index = tt::CBIndex::c_18;
    constexpr uint32_t scratch_d_cb_index = tt::CBIndex::c_19;

    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{y_diag_cb_index, fp32_tile_data_format}})
            .set_page_size(y_diag_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(states_page_size, {{states_cb_index, states_data_format}})
            .set_page_size(states_cb_index, states_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(states_page_size * n_tiles, {{c_cb_index, states_data_format}})
            .set_page_size(c_cb_index, states_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{a_cb_index, fp32_tile_data_format}})
            .set_page_size(a_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(states_page_size * n_tiles, {{states_t_cb_index, states_data_format}})
            .set_page_size(states_t_cb_index, states_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{decay_cb_index, fp32_tile_data_format}})
            .set_page_size(decay_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{scaled_cb_index, fp32_tile_data_format}})
            .set_page_size(scaled_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{dot_cb_index, fp32_tile_data_format}})
            .set_page_size(dot_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{scratch_a_cb_index, fp32_tile_data_format}})
            .set_page_size(scratch_a_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(c_page_size, {{scratch_c_cb_index, c_data_format}})
            .set_page_size(scratch_c_cb_index, c_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(y_diag_page_size, {{scratch_y_cb_index, y_diag_data_format}})
            .set_page_size(scratch_y_cb_index, y_diag_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(x_page_size * h_tiles * p_tiles, {{scratch_x_cb_index, x_data_format}})
            .set_page_size(scratch_x_cb_index, x_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(d_page_size * h_tiles, {{scratch_d_cb_index, d_data_format}})
            .set_page_size(scratch_d_cb_index, d_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_tile_page_size, {{out_tile_cb_index, fp32_tile_data_format}})
            .set_page_size(out_tile_cb_index, fp32_tile_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        work_split.all_cores,
        tt::tt_metal::CircularBufferConfig(out_page_size, {{out_row_cb_index, out_data_format}})
            .set_page_size(out_row_cb_index, out_page_size));

    std::vector<uint32_t> reader_compile_args = {
        y_diag_cb_index,
        states_cb_index,
        c_cb_index,
        a_cb_index,
        y_diag_page_size,
        states_page_size,
        c_page_size,
        a_page_size,
        encode_supported_dtype(tensor_args.y_diag_bcthp),
        encode_supported_dtype(tensor_args.c_blk_bctn),
        encode_supported_dtype(tensor_args.a_cumsum_bhct),
        encode_supported_dtype(tensor_args.states_out_bchpn),
    };
    std::vector<uint32_t> compute_compile_args = {
        y_diag_cb_index,
        states_cb_index,
        c_cb_index,
        a_cb_index,
        states_t_cb_index,
        decay_cb_index,
        scaled_cb_index,
        dot_cb_index,
        out_tile_cb_index,
    };
    std::vector<uint32_t> writer_compile_args = {
        out_tile_cb_index,
        out_row_cb_index,
        scratch_x_cb_index,
        scratch_d_cb_index,
        out_page_size,
        x_page_size,
        d_page_size,
        encode_supported_dtype(tensor_args.x_orig_blk_bcthp),
        encode_supported_dtype(tensor_args.d_h),
        d_values_in_rows ? 1u : 0u,
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.y_diag_bcthp.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.states_out_bchpn.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.c_blk_bctn.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.a_cumsum_bhct.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.x_orig_blk_bcthp.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.d_h.buffer()).append_to(writer_compile_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kReaderKernelPath},
        work_split.all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kComputeKernelPath},
        work_split.all_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_args});

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kWriterKernelPath},
        work_split.all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    for (uint32_t i = 0, row_start = 0; i < work_split.num_cores; ++i) {
        CoreCoord core = {i / work_split.num_cores_y, i % work_split.num_cores_y};
        uint32_t row_count = 0;
        if (work_split.core_group_1.contains(core)) {
            row_count = work_split.rows_per_core_group_1;
        } else if (work_split.core_group_2.contains(core)) {
            row_count = work_split.rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                tensor_args.y_diag_bcthp.buffer()->address(),
                tensor_args.states_out_bchpn.buffer()->address(),
                tensor_args.c_blk_bctn.buffer()->address(),
                tensor_args.a_cumsum_bhct.buffer()->address(),
                row_count,
                row_start,
                kernel_config.seq_len,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.head_dim,
                kernel_config.state_size,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_id,
            core,
            {
                row_count,
                kernel_config.num_heads,
                kernel_config.head_dim,
                kernel_config.state_size,
                kernel_config.seq_len,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                tensor_return_value.buffer()->address(),
                tensor_args.x_orig_blk_bcthp.buffer()->address(),
                tensor_args.d_h.buffer()->address(),
                row_count,
                row_start,
                kernel_config.seq_len,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.head_dim,
            });
        row_start += row_count;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id},
    };
}

void MambaSSDOutputProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MambaSSDOutputParams& operation_attributes,
    const MambaSSDOutputInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto* device = tensor_args.y_diag_bcthp.device();
    const auto kernel_config = build_mamba_ssd_output_kernel_config(operation_attributes, tensor_args);
    if (cached_program.shared_variables.use_chunk_output) {
        const uint32_t total_units = kernel_config.batch_size;
        const auto compute_grid = device->compute_with_storage_grid_size();
        const auto [num_cores, _all_cores, _cg1, _cg2, _r1, _r2] =
            tt::tt_metal::split_work_to_cores(compute_grid, total_units);
        const uint32_t num_cores_y = compute_grid.y;
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            const auto split_range = split_linear_work(total_units, num_cores, i);
            const WorkRange unit_range{
                .start = split_range.start * kernel_config.num_chunks,
                .count = split_range.count * kernel_config.num_chunks};

            auto& reader_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
            reader_runtime_args[0] = tensor_args.states_out_bchpn.buffer()->address();
            reader_runtime_args[1] = tensor_args.c_blk_bctn.buffer()->address();
            reader_runtime_args[2] = tensor_args.a_cumsum_bhct.buffer()->address();
            reader_runtime_args[3] = unit_range.start;
            reader_runtime_args[4] = unit_range.count;
            reader_runtime_args[5] = kernel_config.batch_size;
            reader_runtime_args[6] = kernel_config.num_chunks;
            reader_runtime_args[7] = kernel_config.chunk_size;
            reader_runtime_args[8] = kernel_config.seq_len;
            reader_runtime_args[9] = kernel_config.num_heads;
            reader_runtime_args[10] = kernel_config.head_dim;
            reader_runtime_args[11] = kernel_config.state_size;

            auto& compute_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.compute_kernel_id, core);
            compute_runtime_args[0] = unit_range.count;
            compute_runtime_args[1] = kernel_config.chunk_size;
            compute_runtime_args[2] = kernel_config.num_heads;
            compute_runtime_args[3] = kernel_config.head_dim;

            auto& writer_runtime_args =
                tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
            writer_runtime_args[0] = tensor_return_value.buffer()->address();
            writer_runtime_args[1] = tensor_args.y_diag_bcthp.buffer()->address();
            writer_runtime_args[2] = tensor_args.x_orig_blk_bcthp.buffer()->address();
            writer_runtime_args[3] = tensor_args.a_cumsum_bhct.buffer()->address();
            writer_runtime_args[4] = tensor_args.d_h.buffer()->address();
            writer_runtime_args[5] = unit_range.start;
            writer_runtime_args[6] = unit_range.count;
            writer_runtime_args[7] = kernel_config.batch_size;
            writer_runtime_args[8] = kernel_config.num_chunks;
            writer_runtime_args[9] = kernel_config.chunk_size;
            writer_runtime_args[10] = kernel_config.seq_len;
            writer_runtime_args[11] = kernel_config.num_heads;
            writer_runtime_args[12] = kernel_config.head_dim;
        }
        return;
    }

    const auto work_split = split_output_row_work(operation_attributes, kernel_config, device);

    for (uint32_t i = 0, row_start = 0; i < work_split.num_cores; ++i) {
        CoreCoord core = {i / work_split.num_cores_y, i % work_split.num_cores_y};
        uint32_t row_count = 0;
        if (work_split.core_group_1.contains(core)) {
            row_count = work_split.rows_per_core_group_1;
        } else if (work_split.core_group_2.contains(core)) {
            row_count = work_split.rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        auto& reader_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_runtime_args[0] = tensor_args.y_diag_bcthp.buffer()->address();
        reader_runtime_args[1] = tensor_args.states_out_bchpn.buffer()->address();
        reader_runtime_args[2] = tensor_args.c_blk_bctn.buffer()->address();
        reader_runtime_args[3] = tensor_args.a_cumsum_bhct.buffer()->address();
        reader_runtime_args[4] = row_count;
        reader_runtime_args[5] = row_start;
        reader_runtime_args[6] = kernel_config.seq_len;
        reader_runtime_args[7] = kernel_config.num_chunks;
        reader_runtime_args[8] = kernel_config.chunk_size;
        reader_runtime_args[9] = kernel_config.num_heads;
        reader_runtime_args[10] = kernel_config.head_dim;
        reader_runtime_args[11] = kernel_config.state_size;

        auto& compute_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.compute_kernel_id, core);
        compute_runtime_args[0] = row_count;
        compute_runtime_args[1] = kernel_config.num_heads;
        compute_runtime_args[2] = kernel_config.head_dim;
        compute_runtime_args[3] = kernel_config.state_size;
        compute_runtime_args[4] = kernel_config.seq_len;
        compute_runtime_args[5] = kernel_config.num_chunks;
        compute_runtime_args[6] = kernel_config.chunk_size;

        auto& writer_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value.buffer()->address();
        writer_runtime_args[1] = tensor_args.x_orig_blk_bcthp.buffer()->address();
        writer_runtime_args[2] = tensor_args.d_h.buffer()->address();
        writer_runtime_args[3] = row_count;
        writer_runtime_args[4] = row_start;
        writer_runtime_args[5] = kernel_config.seq_len;
        writer_runtime_args[6] = kernel_config.num_chunks;
        writer_runtime_args[7] = kernel_config.chunk_size;
        writer_runtime_args[8] = kernel_config.num_heads;
        writer_runtime_args[9] = kernel_config.head_dim;
        row_start += row_count;
    }
}

}  // namespace ttnn::experimental::prim
