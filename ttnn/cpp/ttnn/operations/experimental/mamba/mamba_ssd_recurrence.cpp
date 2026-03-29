// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_recurrence.hpp"

#include "device/mamba_ssd_recurrence_device_operation.hpp"

namespace ttnn::experimental {

std::vector<Tensor> mamba_ssd_recurrence(
    const Tensor& states_bhcpn,
    const Tensor& initial_states,
    const Tensor& a_end_bhc,
    std::optional<CoreGrid> core_grid) {
    return ttnn::prim::mamba_ssd_recurrence(states_bhcpn, initial_states, a_end_bhc, std::move(core_grid));
}

}  // namespace ttnn::experimental
