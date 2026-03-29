// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include "mamba_ssd_recurrence_device_operation_types.hpp"
#include "mamba_ssd_recurrence_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDRecurrenceDeviceOperation {
    using operation_attributes_t = MambaSSDRecurrenceParams;
    using tensor_args_t = MambaSSDRecurrenceInputs;
    using spec_return_value_t = std::array<TensorSpec, 2>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MambaSSDRecurrenceProgramFactory>;
    using shared_variables_t = MambaSSDRecurrenceProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

std::vector<Tensor> mamba_ssd_recurrence_fallback(
    const MambaSSDRecurrenceParams& operation_attributes, const MambaSSDRecurrenceInputs& tensor_args);

MambaSSDRecurrenceKernelConfig build_mamba_ssd_recurrence_kernel_config(
    const MambaSSDRecurrenceParams& operation_attributes, const MambaSSDRecurrenceInputs& tensor_args);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> mamba_ssd_recurrence(
    const Tensor& states_bhcpn,
    const Tensor& initial_states,
    const Tensor& a_end_bhc,
    std::optional<CoreGrid> core_grid = std::nullopt);

}  // namespace ttnn::prim
