# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_depthwise_conv1d_stateful_signature_exposed():
    doc = ttnn._ttnn.operations.experimental.depthwise_conv1d.__call__.__doc__
    assert "conv_state" in doc
    assert "features: int" in doc
    assert "-> list[ttnn._ttnn.tensor.Tensor]" in doc
