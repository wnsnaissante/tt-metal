// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include "mamba_ssd_chunk_scan_device_operation_types.hpp"
#include "mamba_ssd_chunk_scan_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MambaSSDChunkScanDeviceOperation {
    using operation_attributes_t = MambaSSDChunkScanParams;
    using tensor_args_t = MambaSSDChunkScanInputs;
    using spec_return_value_t = std::array<TensorSpec, 3>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MambaSSDChunkScanProgramFactory>;
    using shared_variables_t = MambaSSDChunkScanProgramFactory::shared_variables_t;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const tensor_return_value_t&);
};

MambaSSDChunkScanKernelConfig build_mamba_ssd_chunk_scan_kernel_config(
    const MambaSSDChunkScanParams& operation_attributes, const MambaSSDChunkScanInputs& tensor_args);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> mamba_ssd_chunk_scan(
    const Tensor& x_blk_bcthp,
    const Tensor& a_blk_bhct,
    const Tensor& b_blk_bctn,
    const Tensor& c_blk_bctn,
    std::optional<CoreGrid> core_grid = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
