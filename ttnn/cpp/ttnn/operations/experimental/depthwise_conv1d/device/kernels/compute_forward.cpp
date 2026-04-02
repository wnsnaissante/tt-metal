// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

namespace {

constexpr uint32_t kBiasCbIndex = tt::CBIndex::c_8;
constexpr uint32_t kAccumCb0Index = tt::CBIndex::c_9;
constexpr uint32_t kAccumCb1Index = tt::CBIndex::c_10;
constexpr uint32_t kPartialCbIndex = tt::CBIndex::c_16;
constexpr uint32_t kOutputCbIndex = tt::CBIndex::c_17;
constexpr uint32_t kOneTile = 1;

}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t kernel_size = get_compile_time_arg_val(0);
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_4, kOutputCbIndex);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_wait_front(kBiasCbIndex, kOneTile);
        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            cb_wait_front(tt::CBIndex::c_4 + tap, kOneTile);
        }

        uint32_t accum_in_cb = kAccumCb0Index;
        uint32_t accum_out_cb = kAccumCb1Index;

        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            const uint32_t input_cb = tt::CBIndex::c_0 + tap;
            const uint32_t weight_cb = tt::CBIndex::c_4 + tap;

            cb_wait_front(input_cb, kOneTile);
            const bool first_tap = tap == 0;

            if (first_tap) {
                cb_reserve_back(accum_in_cb, kOneTile);
                pack_reconfig_data_format(accum_in_cb);

                tile_regs_acquire();
                mul_bcast_rows_init_short(input_cb, weight_cb);
                mul_tiles_bcast_rows(input_cb, weight_cb, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, accum_in_cb);
                tile_regs_release();
                cb_push_back(accum_in_cb, kOneTile);

                cb_pop_front(input_cb, kOneTile);
                continue;
            }

            cb_reserve_back(kPartialCbIndex, kOneTile);
            pack_reconfig_data_format(kPartialCbIndex);

            tile_regs_acquire();
            mul_bcast_rows_init_short(input_cb, weight_cb);
            mul_tiles_bcast_rows(input_cb, weight_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, kPartialCbIndex);
            tile_regs_release();
            cb_push_back(kPartialCbIndex, kOneTile);

            cb_pop_front(input_cb, kOneTile);

            cb_wait_front(kPartialCbIndex, kOneTile);
            cb_wait_front(accum_in_cb, kOneTile);
            cb_reserve_back(accum_out_cb, kOneTile);
            pack_reconfig_data_format(accum_out_cb);

            tile_regs_acquire();
            add_tiles_init(kPartialCbIndex, accum_in_cb);
            add_tiles(kPartialCbIndex, accum_in_cb, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, accum_out_cb);
            tile_regs_release();
            cb_push_back(accum_out_cb, kOneTile);

            cb_pop_front(kPartialCbIndex, kOneTile);
            cb_pop_front(accum_in_cb, kOneTile);

            const uint32_t previous_accum_in_cb = accum_in_cb;
            accum_in_cb = accum_out_cb;
            accum_out_cb = previous_accum_in_cb;
        }

        cb_wait_front(accum_in_cb, kOneTile);
        cb_reserve_back(kOutputCbIndex, kOneTile);
        pack_reconfig_data_format(kOutputCbIndex);
        tile_regs_acquire();
        add_bcast_rows_init_short(accum_in_cb, kBiasCbIndex);
        add_tiles_bcast_rows(accum_in_cb, kBiasCbIndex, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, kOutputCbIndex);
        tile_regs_release();

        cb_push_back(kOutputCbIndex, kOneTile);
        cb_pop_front(accum_in_cb, kOneTile);

        for (uint32_t tap = 0; tap < kernel_size; ++tap) {
            cb_pop_front(tt::CBIndex::c_4 + tap, kOneTile);
        }
        cb_pop_front(kBiasCbIndex, kOneTile);
    }
}
