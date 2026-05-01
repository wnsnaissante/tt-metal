// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "mamba_ssd_output_device_operation_types.hpp"
#include "mamba_ssd_output_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDOutputDeviceOperation {
    using operation_attributes_t = MambaSSDOutputParams;
    using tensor_args_t = MambaSSDOutputInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MambaSSDOutputProgramFactory>;
    using shared_variables_t = MambaSSDOutputProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

Tensor mamba_ssd_output_fallback(
    const MambaSSDOutputParams& operation_attributes, const MambaSSDOutputInputs& tensor_args);

MambaSSDOutputKernelConfig build_mamba_ssd_output_kernel_config(
    const MambaSSDOutputParams& operation_attributes, const MambaSSDOutputInputs& tensor_args);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mamba_ssd_output(
    const Tensor& y_diag_bcthp,
    const Tensor& states_out_bchpn,
    const Tensor& c_blk_bctn,
    const Tensor& a_cumsum_bhct,
    const Tensor& x_orig_blk_bcthp,
    const Tensor& d_h,
    uint32_t seq_len,
    uint32_t pad_size,
    std::optional<CoreGrid> core_grid = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
