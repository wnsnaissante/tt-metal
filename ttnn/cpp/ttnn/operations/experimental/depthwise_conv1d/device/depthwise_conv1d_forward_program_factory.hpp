// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct DepthwiseConv1dForwardParams;
struct DepthwiseConv1dForwardInputs;

struct DepthwiseConv1dForwardSharedVariables {
    std::vector<CoreCoord> cores;
    uint32_t group_1_core_count = 0;
    uint32_t tiles_per_core_group_1 = 0;
    uint32_t tiles_per_core_group_2 = 0;
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id_group_1 = 0;
    tt::tt_metal::KernelHandle compute_kernel_id_group_2 = 0;
};

struct DepthwiseConv1dForwardProgramFactory {
    using shared_variables_t = DepthwiseConv1dForwardSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DepthwiseConv1dForwardParams& operation_attributes,
        const DepthwiseConv1dForwardInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DepthwiseConv1dForwardParams& operation_attributes,
        const DepthwiseConv1dForwardInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
