// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mamba_ssd_output_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDOutputSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle compute_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    bool use_chunk_output = false;
};

struct MambaSSDOutputProgramFactory {
    using shared_variables_t = MambaSSDOutputSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static constexpr std::string_view kReaderKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/reader.cpp";
    static constexpr std::string_view kComputeKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/compute.cpp";
    static constexpr std::string_view kWriterKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/writer.cpp";
    static constexpr std::string_view kReaderChunkKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/reader_chunk.cpp";
    static constexpr std::string_view kComputeChunkKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/compute_chunk.cpp";
    static constexpr std::string_view kWriterChunkKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_output/device/kernels/writer_chunk.cpp";

    static cached_program_t create(
        const MambaSSDOutputParams& operation_attributes,
        const MambaSSDOutputInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MambaSSDOutputParams& operation_attributes,
        const MambaSSDOutputInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
