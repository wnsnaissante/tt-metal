// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if 0
#include "mamba_ssd_chunk_scan_program_factory.hpp"

#include "mamba_ssd_chunk_scan_device_operation.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kDTypeBFloat16 = 0;
constexpr uint32_t kDTypeFloat32 = 1;
constexpr const char* kUseSfpuYReduceEnv = "MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE";
constexpr const char* kMergePTilesEnv = "MAMBA_CHUNK_SCAN_MERGE_P_TILES";
constexpr const char* kCumsumV2Env = "MAMBA_CHUNK_SCAN_CUMSUM_V2";
constexpr const char* kCumsumV2Env = "MAMBA_CHUNK_SCAN_CUMSUM_V2";

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}
constexpr const char* kMergePTilesEnv = "MAMBA_CHUNK_SCAN_MERGE_P_TILES";

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

uint32_t encode_supported_dtype(const Tensor& tensor) {
    switch (tensor.dtype()) {
        case ttnn::DataType::BFLOAT16: return kDTypeBFloat16;
        case ttnn::DataType::FLOAT32: return kDTypeFloat32;
        default: TT_THROW("mamba_ssd_chunk_scan kernel supports BF16/FLOAT32 inputs only");
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

}  // namespace

MambaSSDChunkScanProgramFactory::cached_program_t MambaSSDChunkScanProgramFactory::create(
    const MambaSSDChunkScanParams& operation_attributes,
    const MambaSSDChunkScanInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};
    auto* device = tensor_args.x_blk_bcthp.device();
    const auto kernel_config = build_mamba_ssd_chunk_scan_kernel_config(operation_attributes, tensor_args);
    const bool use_merge_p_tiles = env_enabled(kMergePTilesEnv) && kernel_config.p_tiles == 2 &&
                                   kernel_config.head_dim == 64 && kernel_config.num_heads == 8;

    const uint32_t scan_units =
        kernel_config.batch_size * kernel_config.num_chunks * (use_merge_p_tiles ? 1 : kernel_config.p_tiles);
    const uint32_t cumsum_units = kernel_config.batch_size * kernel_config.num_heads;
    const uint32_t total_units = std::max(scan_units, cumsum_units);
    const auto compute_grid = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, _cg1, _cg2, _r1, _r2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_units);
    const uint32_t num_cores_y = compute_grid.y;

    const auto x_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.x_blk_bcthp.dtype());
    const auto a_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.a_blk_bhct.dtype());
    const auto b_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.b_blk_bctn.dtype());
    const auto c_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.c_blk_bctn.dtype());
    const auto fp32_format = tt::DataFormat::Float32;

    const uint32_t x_page_size = tensor_args.x_blk_bcthp.buffer()->aligned_page_size();
    const uint32_t a_page_size = tensor_args.a_blk_bhct.buffer()->aligned_page_size();
    const uint32_t b_page_size = tensor_args.b_blk_bctn.buffer()->aligned_page_size();
    const uint32_t c_page_size = tensor_args.c_blk_bctn.buffer()->aligned_page_size();
    const uint32_t fp32_page_size = tt::tile_size(fp32_format);
    const uint32_t x_src_tiles = use_merge_p_tiles ? kernel_config.p_tiles : 1;
    const uint32_t a_scan_src_tiles = kernel_config.num_heads;
    const uint32_t state_acc_tiles = kernel_config.num_heads * (use_merge_p_tiles ? kernel_config.p_tiles : 1);
    const uint32_t y_tile_tiles = use_merge_p_tiles ? kernel_config.p_tiles : 1;

    constexpr uint32_t x_cb = tt::CBIndex::c_0;
    constexpr uint32_t a_cb = tt::CBIndex::c_1;
    constexpr uint32_t b_cb = tt::CBIndex::c_2;
    constexpr uint32_t c_cb = tt::CBIndex::c_3;
    constexpr uint32_t y_cb = tt::CBIndex::c_4;
    constexpr uint32_t state_cb = tt::CBIndex::c_5;
    constexpr uint32_t a_out_cb = tt::CBIndex::c_6;

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(x_page_size, {{x_cb, x_format}}).set_page_size(x_cb, x_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(a_page_size, {{a_cb, a_format}}).set_page_size(a_cb, a_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(b_page_size, {{b_cb, b_format}}).set_page_size(b_cb, b_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(c_page_size, {{c_cb, c_format}}).set_page_size(c_cb, c_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size, {{y_cb, fp32_format}}).set_page_size(y_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * kernel_config.num_heads, {{state_cb, fp32_format}})
            .set_page_size(state_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size, {{a_out_cb, fp32_format}})
            .set_page_size(a_out_cb, fp32_page_size));

    std::vector<uint32_t> writer_compile_args = {
        x_cb,
        a_cb,
        b_cb,
        c_cb,
        y_cb,
        state_cb,
        a_out_cb,
        x_page_size,
        a_page_size,
        b_page_size,
        c_page_size,
        fp32_page_size,
        encode_supported_dtype(tensor_args.x_blk_bcthp),
        encode_supported_dtype(tensor_args.a_blk_bhct),
        encode_supported_dtype(tensor_args.b_blk_bctn),
        encode_supported_dtype(tensor_args.c_blk_bctn),
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.x_blk_bcthp.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.a_blk_bhct.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.b_blk_bctn.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.c_blk_bctn.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[0].buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[1].buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[2].buffer()).append_to(writer_compile_args);

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kWriterKernelPath},
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto cumsum_range = split_linear_work(cumsum_units, num_cores, i);
        const auto scan_range = split_linear_work(scan_units, num_cores, i);
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                tensor_args.x_blk_bcthp.buffer()->address(),
                tensor_args.a_blk_bhct.buffer()->address(),
                tensor_args.b_blk_bctn.buffer()->address(),
                tensor_args.c_blk_bctn.buffer()->address(),
                tensor_return_value[0].buffer()->address(),
                tensor_return_value[1].buffer()->address(),
                tensor_return_value[2].buffer()->address(),
                kernel_config.batch_size,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.head_dim,
                kernel_config.state_size,
                kernel_config.p_tiles,
                kernel_config.t_tiles,
                cumsum_range.start,
                cumsum_range.count,
                scan_range.start,
                scan_range.count,
            });
    }

    return {std::move(program), {.writer_kernel_id = writer_kernel_id, .num_cores = num_cores, .num_cores_y = num_cores_y}};
}

void MambaSSDChunkScanProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MambaSSDChunkScanParams& operation_attributes,
    const MambaSSDChunkScanInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    const auto kernel_config = build_mamba_ssd_chunk_scan_kernel_config(operation_attributes, tensor_args);
    const uint32_t scan_units = kernel_config.batch_size * kernel_config.num_chunks * kernel_config.p_tiles;
    const uint32_t cumsum_units = kernel_config.batch_size * kernel_config.num_heads;

    for (uint32_t i = 0; i < cached_program.shared_variables.num_cores; ++i) {
        CoreCoord core = {i / cached_program.shared_variables.num_cores_y, i % cached_program.shared_variables.num_cores_y};
        const auto cumsum_range = split_linear_work(cumsum_units, cached_program.shared_variables.num_cores, i);
        const auto scan_range = split_linear_work(scan_units, cached_program.shared_variables.num_cores, i);
        auto& args = tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        args[0] = tensor_args.x_blk_bcthp.buffer()->address();
        args[1] = tensor_args.a_blk_bhct.buffer()->address();
        args[2] = tensor_args.b_blk_bctn.buffer()->address();
        args[3] = tensor_args.c_blk_bctn.buffer()->address();
        args[4] = tensor_return_value[0].buffer()->address();
        args[5] = tensor_return_value[1].buffer()->address();
        args[6] = tensor_return_value[2].buffer()->address();
        args[7] = kernel_config.batch_size;
        args[8] = kernel_config.num_chunks;
        args[9] = kernel_config.chunk_size;
        args[10] = kernel_config.num_heads;
        args[11] = kernel_config.head_dim;
        args[12] = kernel_config.state_size;
        args[13] = kernel_config.p_tiles;
        args[14] = kernel_config.t_tiles;
        args[15] = cumsum_range.start;
        args[16] = cumsum_range.count;
        args[17] = scan_range.start;
        args[18] = scan_range.count;
    }
}

}  // namespace ttnn::experimental::prim
#endif

#include "mamba_ssd_chunk_scan_program_factory.hpp"

#include "mamba_ssd_chunk_scan_device_operation.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t kDTypeBFloat16 = 0;
constexpr uint32_t kDTypeFloat32 = 1;
constexpr const char* kUseSfpuYReduceEnv = "MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE";
constexpr const char* kMergePTilesEnv = "MAMBA_CHUNK_SCAN_MERGE_P_TILES";

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

uint32_t encode_supported_dtype(const Tensor& tensor) {
    switch (tensor.dtype()) {
        case ttnn::DataType::BFLOAT16: return kDTypeBFloat16;
        case ttnn::DataType::FLOAT32: return kDTypeFloat32;
        default: TT_THROW("mamba_ssd_chunk_scan kernel supports BF16/FLOAT32 inputs only");
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

}  // namespace

MambaSSDChunkScanProgramFactory::cached_program_t MambaSSDChunkScanProgramFactory::create(
    const MambaSSDChunkScanParams& operation_attributes,
    const MambaSSDChunkScanInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};
    auto* device = tensor_args.x_blk_bcthp.device();
    const auto kernel_config = build_mamba_ssd_chunk_scan_kernel_config(operation_attributes, tensor_args);
    const bool use_merge_p_tiles = env_enabled(kMergePTilesEnv) && kernel_config.p_tiles == 2 &&
                                   kernel_config.head_dim == 64 && kernel_config.num_heads == 8;
    const bool use_cumsum_v2 = env_enabled("MAMBA_CHUNK_SCAN_CUMSUM_V2");

    const uint32_t scan_units =
        kernel_config.batch_size * kernel_config.num_chunks * (use_merge_p_tiles ? 1 : kernel_config.p_tiles);
    const uint32_t cumsum_units =
        kernel_config.batch_size * kernel_config.num_heads * (use_cumsum_v2 ? 1 : kernel_config.t_tiles);
    const uint32_t total_units = std::max(scan_units, cumsum_units);
    const auto compute_grid = device->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, _cg1, _cg2, _r1, _r2] =
        tt::tt_metal::split_work_to_cores(compute_grid, total_units);
    const uint32_t num_cores_y = compute_grid.y;

    const auto x_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.x_blk_bcthp.dtype());
    const auto a_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.a_blk_bhct.dtype());
    const auto b_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.b_blk_bctn.dtype());
    const auto c_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.c_blk_bctn.dtype());
    const auto fp32_format = tt::DataFormat::Float32;

    const uint32_t x_page_size = tensor_args.x_blk_bcthp.buffer()->aligned_page_size();
    const uint32_t a_page_size = tensor_args.a_blk_bhct.buffer()->aligned_page_size();
    const uint32_t b_page_size = tensor_args.b_blk_bctn.buffer()->aligned_page_size();
    const uint32_t c_page_size = tensor_args.c_blk_bctn.buffer()->aligned_page_size();
    const uint32_t fp32_page_size = tt::tile_size(fp32_format);
    const uint32_t x_src_tiles = use_merge_p_tiles ? kernel_config.p_tiles : 1;
    const uint32_t a_scan_src_tiles = kernel_config.num_heads;
    const uint32_t state_acc_tiles = kernel_config.num_heads * (use_merge_p_tiles ? kernel_config.p_tiles : 1);
    const uint32_t y_tile_tiles = use_merge_p_tiles ? kernel_config.p_tiles : 1;

    constexpr uint32_t x_src_cb = tt::CBIndex::c_0;
    constexpr uint32_t a_scan_src_cb = tt::CBIndex::c_1;
    constexpr uint32_t b_src_cb = tt::CBIndex::c_2;
    constexpr uint32_t c_src_cb = tt::CBIndex::c_3;
    constexpr uint32_t decay_cb = tt::CBIndex::c_4;
    constexpr uint32_t x_col_cb = tt::CBIndex::c_5;
    constexpr uint32_t b_row_cb = tt::CBIndex::c_6;
    constexpr uint32_t c_row_cb = tt::CBIndex::c_7;
    constexpr uint32_t zero_state_cb = tt::CBIndex::c_8;
    constexpr uint32_t state_acc_cb = tt::CBIndex::c_9;
    constexpr uint32_t state_scaled_cb = tt::CBIndex::c_10;
    constexpr uint32_t outer_cb = tt::CBIndex::c_11;
    constexpr uint32_t state_next_cb = tt::CBIndex::c_12;
    constexpr uint32_t y_vec_cb = tt::CBIndex::c_13;
    constexpr uint32_t final_state_cb = tt::CBIndex::c_14;
    constexpr uint32_t a_cumsum_src_cb = tt::CBIndex::c_15;
    constexpr uint32_t a_out_cb = tt::CBIndex::c_16;
    constexpr uint32_t y_tile_cb = tt::CBIndex::c_17;
    constexpr uint32_t x_full_cb = tt::CBIndex::c_18;
    constexpr uint32_t reduce_scaler_cb = tt::CBIndex::c_20;

    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(x_page_size * x_src_tiles, {{x_src_cb, x_format}})
            .set_page_size(x_src_cb, x_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(a_page_size * a_scan_src_tiles, {{a_scan_src_cb, a_format}})
            .set_page_size(a_scan_src_cb, a_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(b_page_size, {{b_src_cb, b_format}}).set_page_size(b_src_cb, b_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(c_page_size, {{c_src_cb, c_format}}).set_page_size(c_src_cb, c_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * 2, {{decay_cb, fp32_format}})
            .set_page_size(decay_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * 2, {{x_col_cb, fp32_format}})
            .set_page_size(x_col_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * 2, {{b_row_cb, fp32_format}})
            .set_page_size(b_row_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * 2, {{c_row_cb, fp32_format}})
            .set_page_size(c_row_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size, {{zero_state_cb, fp32_format}})
            .set_page_size(zero_state_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * state_acc_tiles, {{state_acc_cb, fp32_format}})
            .set_page_size(state_acc_cb, fp32_page_size));
    for (uint32_t cb :
         {state_scaled_cb, outer_cb, state_next_cb, final_state_cb, a_out_cb, x_full_cb, reduce_scaler_cb}) {
        tt::tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt::tt_metal::CircularBufferConfig(fp32_page_size, {{cb, fp32_format}}).set_page_size(cb, fp32_page_size));
    }
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * y_tile_tiles, {{y_tile_cb, fp32_format}})
            .set_page_size(y_tile_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(fp32_page_size * 2, {{y_vec_cb, fp32_format}})
            .set_page_size(y_vec_cb, fp32_page_size));
    tt::tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt::tt_metal::CircularBufferConfig(a_page_size, {{a_cumsum_src_cb, a_format}})
            .set_page_size(a_cumsum_src_cb, a_page_size));

    std::vector<uint32_t> reader_compile_args = {
        x_src_cb,
        a_scan_src_cb,
        b_src_cb,
        c_src_cb,
        decay_cb,
        x_col_cb,
        b_row_cb,
        c_row_cb,
        zero_state_cb,
        reduce_scaler_cb,
        x_page_size,
        a_page_size,
        b_page_size,
        c_page_size,
        fp32_page_size,
        encode_supported_dtype(tensor_args.x_blk_bcthp),
        encode_supported_dtype(tensor_args.a_blk_bhct),
        encode_supported_dtype(tensor_args.b_blk_bctn),
        encode_supported_dtype(tensor_args.c_blk_bctn),
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.x_blk_bcthp.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.a_blk_bhct.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.b_blk_bctn.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.c_blk_bctn.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> compute_compile_args = {
        decay_cb,
        x_col_cb,
        b_row_cb,
        c_row_cb,
        zero_state_cb,
        state_acc_cb,
        state_scaled_cb,
        outer_cb,
        state_next_cb,
        y_vec_cb,
        final_state_cb,
        x_full_cb,
        reduce_scaler_cb,
    };
    const bool use_sfpu_y_reduce = env_enabled(kUseSfpuYReduceEnv);
    std::map<std::string, std::string> dataflow_defines;
    if (use_merge_p_tiles) {
        dataflow_defines["MAMBA_CHUNK_SCAN_MERGE_P_TILES"] = "1";
    }
    if (use_cumsum_v2) {
        dataflow_defines["MAMBA_CHUNK_SCAN_CUMSUM_V2"] = "1";
    }
    std::map<std::string, std::string> compute_defines;
    if (use_sfpu_y_reduce) {
        compute_defines["MAMBA_CHUNK_SCAN_USE_SFPU_Y_REDUCE"] = "1";
    }
    if (use_merge_p_tiles) {
        compute_defines["MAMBA_CHUNK_SCAN_MERGE_P_TILES"] = "1";
    }

    std::vector<uint32_t> writer_compile_args = {
        a_cumsum_src_cb,
        y_vec_cb,
        final_state_cb,
        y_tile_cb,
        a_out_cb,
        a_page_size,
        fp32_page_size,
        encode_supported_dtype(tensor_args.a_blk_bhct),
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.a_blk_bhct.buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[0].buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[1].buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_return_value[2].buffer()).append_to(writer_compile_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kReaderKernelPath},
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, dataflow_defines));
    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kComputeKernelPath},
        all_cores,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = use_sfpu_y_reduce, .compile_args = compute_compile_args, .defines = compute_defines});
    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string{kWriterKernelPath},
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, dataflow_defines));

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const auto cumsum_range = split_linear_work(cumsum_units, num_cores, i);
        const auto scan_range = split_linear_work(scan_units, num_cores, i);
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                tensor_args.x_blk_bcthp.buffer()->address(),
                tensor_args.a_blk_bhct.buffer()->address(),
                tensor_args.b_blk_bctn.buffer()->address(),
                tensor_args.c_blk_bctn.buffer()->address(),
                kernel_config.batch_size,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.head_dim,
                kernel_config.state_size,
                kernel_config.p_tiles,
                kernel_config.t_tiles,
                scan_range.start,
                scan_range.count,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_id,
            core,
            {
                scan_range.count,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.p_tiles,
            });
        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                tensor_args.a_blk_bhct.buffer()->address(),
                tensor_return_value[0].buffer()->address(),
                tensor_return_value[1].buffer()->address(),
                tensor_return_value[2].buffer()->address(),
                kernel_config.batch_size,
                kernel_config.num_chunks,
                kernel_config.chunk_size,
                kernel_config.num_heads,
                kernel_config.head_dim,
                kernel_config.state_size,
                kernel_config.p_tiles,
                kernel_config.t_tiles,
                cumsum_range.start,
                cumsum_range.count,
                scan_range.start,
                scan_range.count,
            });
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y,
         .merge_p_tiles = use_merge_p_tiles,
         .cumsum_v2 = use_cumsum_v2}};
}

void MambaSSDChunkScanProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MambaSSDChunkScanParams& operation_attributes,
    const MambaSSDChunkScanInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    const auto kernel_config = build_mamba_ssd_chunk_scan_kernel_config(operation_attributes, tensor_args);
    const bool use_merge_p_tiles = cached_program.shared_variables.merge_p_tiles;
    const bool use_cumsum_v2 = cached_program.shared_variables.cumsum_v2;
    const uint32_t scan_units =
        kernel_config.batch_size * kernel_config.num_chunks * (use_merge_p_tiles ? 1 : kernel_config.p_tiles);
    const uint32_t cumsum_units =
        kernel_config.batch_size * kernel_config.num_heads * (use_cumsum_v2 ? 1 : kernel_config.t_tiles);

    for (uint32_t i = 0; i < cached_program.shared_variables.num_cores; ++i) {
        CoreCoord core = {
            i / cached_program.shared_variables.num_cores_y, i % cached_program.shared_variables.num_cores_y};
        const auto cumsum_range = split_linear_work(cumsum_units, cached_program.shared_variables.num_cores, i);
        const auto scan_range = split_linear_work(scan_units, cached_program.shared_variables.num_cores, i);

        auto& reader_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_args[0] = tensor_args.x_blk_bcthp.buffer()->address();
        reader_args[1] = tensor_args.a_blk_bhct.buffer()->address();
        reader_args[2] = tensor_args.b_blk_bctn.buffer()->address();
        reader_args[3] = tensor_args.c_blk_bctn.buffer()->address();
        reader_args[4] = kernel_config.batch_size;
        reader_args[5] = kernel_config.num_chunks;
        reader_args[6] = kernel_config.chunk_size;
        reader_args[7] = kernel_config.num_heads;
        reader_args[8] = kernel_config.head_dim;
        reader_args[9] = kernel_config.state_size;
        reader_args[10] = kernel_config.p_tiles;
        reader_args[11] = kernel_config.t_tiles;
        reader_args[12] = scan_range.start;
        reader_args[13] = scan_range.count;

        auto& compute_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.compute_kernel_id, core);
        compute_args[0] = scan_range.count;
        compute_args[1] = kernel_config.chunk_size;
        compute_args[2] = kernel_config.num_heads;
        compute_args[3] = kernel_config.p_tiles;

        auto& writer_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_args[0] = tensor_args.a_blk_bhct.buffer()->address();
        writer_args[1] = tensor_return_value[0].buffer()->address();
        writer_args[2] = tensor_return_value[1].buffer()->address();
        writer_args[3] = tensor_return_value[2].buffer()->address();
        writer_args[4] = kernel_config.batch_size;
        writer_args[5] = kernel_config.num_chunks;
        writer_args[6] = kernel_config.chunk_size;
        writer_args[7] = kernel_config.num_heads;
        writer_args[8] = kernel_config.head_dim;
        writer_args[9] = kernel_config.state_size;
        writer_args[10] = kernel_config.p_tiles;
        writer_args[11] = kernel_config.t_tiles;
        writer_args[12] = cumsum_range.start;
        writer_args[13] = cumsum_range.count;
        writer_args[14] = scan_range.start;
        writer_args[15] = scan_range.count;
    }
}

}  // namespace ttnn::experimental::prim
