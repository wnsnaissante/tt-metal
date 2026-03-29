// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "mamba_ssd_recurrence_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDRecurrenceSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    CoreRangeSet cores = CoreRangeSet{};
};

struct MambaSSDRecurrenceProgramFactory {
    using shared_variables_t = MambaSSDRecurrenceSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static constexpr std::string_view kReaderKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/device/kernels/reader.cpp";
    static constexpr std::string_view kComputeKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/device/kernels/compute.cpp";
    static constexpr std::string_view kWriterKernelPath =
        "ttnn/cpp/ttnn/operations/experimental/mamba/device/kernels/writer.cpp";

    static cached_program_t create(
        const MambaSSDRecurrenceParams& operation_attributes,
        const MambaSSDRecurrenceInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MambaSSDRecurrenceParams& operation_attributes,
        const MambaSSDRecurrenceInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
