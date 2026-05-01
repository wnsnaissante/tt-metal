// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "depthwise_conv1d_device_operation_types.hpp"

namespace ttnn::operations::experimental::depthwise_conv1d {

struct DepthwiseConv1dDeviceOperation {
    using operation_attributes_t = DepthwiseConv1dParams;

    struct tensor_args_t {
        const Tensor& input_tensor;                // [B, 1, L_padded, C] — causal-padded, ROW_MAJOR
        const Tensor& weight_tensor;               // [1, 1, K, C] — ROW_MAJOR
        const std::optional<Tensor>& bias_tensor;  // [1, 1, 1, C] — optional
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            tt::tt_metal::CoreRangeSet all_cores;
            std::vector<tt::tt_metal::CoreCoord> cores;
            uint32_t num_cores;
            uint32_t g1_numcores;
            uint32_t num_blocks_per_core_group_1;
            uint32_t num_blocks_per_core_group_2;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::depthwise_conv1d

namespace ttnn::prim {

Tensor depthwise_conv1d_forward(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    uint32_t kernel_size,
    uint32_t channels,
    uint32_t sequence_length,
    const std::optional<Tensor>& bias = std::nullopt,
    bool silu_activation = false);

}  // namespace ttnn::prim
