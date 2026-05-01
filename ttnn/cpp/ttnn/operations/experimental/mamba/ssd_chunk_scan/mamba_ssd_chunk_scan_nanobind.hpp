// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::mamba::detail {

void bind_mamba_ssd_chunk_scan(nb::module_& mod);

}  // namespace ttnn::operations::experimental::mamba::detail
