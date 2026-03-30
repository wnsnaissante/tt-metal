// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mamba_ssd_recurrence.hpp"

#include "device/mamba_ssd_recurrence_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::experimental {

std::vector<Tensor> mamba_ssd_recurrence(
    const Tensor& states_bhcpn,
    const Tensor& initial_states,
    const Tensor& a_end_bhc,
    std::optional<CoreGrid> core_grid) {
    auto outputs = ttnn::prim::mamba_ssd_recurrence(states_bhcpn, initial_states, a_end_bhc, std::move(core_grid));
    auto states_out = outputs.at(0);
    const auto& s = states_bhcpn.logical_shape();
    const uint32_t B = s[0];
    const uint32_t H = s[1];
    const uint32_t C = s[2];
    const uint32_t P = s[3];
    const uint32_t N = s[4];
    const auto mem = states_out.memory_config();

    auto h_prev_last = ttnn::slice(
        states_out,
        ttnn::SmallVector<uint32_t>{0, C - 1, 0, 0, 0},
        ttnn::SmallVector<uint32_t>{B, C, H, P, N},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1, 1},
        mem);
    h_prev_last = ttnn::reshape(h_prev_last, ttnn::Shape({B, H, P, N}), mem);

    auto x_last = ttnn::slice(
        states_bhcpn,
        ttnn::SmallVector<uint32_t>{0, 0, C - 1, 0, 0},
        ttnn::SmallVector<uint32_t>{B, H, C, P, N},
        ttnn::SmallVector<uint32_t>{1, 1, 1, 1, 1},
        mem);
    x_last = ttnn::reshape(x_last, ttnn::Shape({B, H, P, N}), mem);
    if (x_last.dtype() != ttnn::DataType::FLOAT32) {
        x_last = ttnn::typecast(x_last, ttnn::DataType::FLOAT32, mem);
    }

    auto a_last = ttnn::slice(
        a_end_bhc,
        ttnn::SmallVector<uint32_t>{0, 0, C - 1},
        ttnn::SmallVector<uint32_t>{B, H, C},
        ttnn::SmallVector<uint32_t>{1, 1, 1},
        mem);
    a_last = ttnn::reshape(a_last, ttnn::Shape({B, H, 1, 1}), mem);
    if (a_last.dtype() != ttnn::DataType::FLOAT32) {
        a_last = ttnn::typecast(a_last, ttnn::DataType::FLOAT32, mem);
    }

    auto final_state = ttnn::multiply(ttnn::exp(a_last, false, mem), h_prev_last, std::nullopt, mem);
    final_state = ttnn::add(final_state, x_last, std::nullopt, mem);

    return {states_out, final_state};
}

}  // namespace ttnn::experimental
