// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "mamba_ssd_chunk_scan_device_operation_types.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDChunkScanSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    bool merge_p_tiles = false;
    bool cumsum_v2 = false;
};

struct MambaSSDChunkScanProgramFactory {
    using shared_variables_t = MambaSSDChunkScanSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static constexpr std::string_view kWriterKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_chunk_scan/device/kernels/writer.cpp";
    static constexpr std::string_view kReaderKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_chunk_scan/device/kernels/reader.cpp";
    static constexpr std::string_view kComputeKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/ssd_chunk_scan/device/kernels/compute.cpp";

    static cached_program_t create(
        const MambaSSDChunkScanParams& operation_attributes,
        const MambaSSDChunkScanInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MambaSSDChunkScanParams& operation_attributes,
        const MambaSSDChunkScanInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
