# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def reference_causal_depthwise_conv1d(x_b1lc, weight_c1k, bias=None, silu_activation=False):
    x_bcl = x_b1lc.squeeze(1).permute(0, 2, 1)
    padded = F.pad(x_bcl, (weight_c1k.shape[-1] - 1, 0))
    out = F.conv1d(padded, weight_c1k, bias=bias, groups=x_bcl.shape[1])
    if silu_activation:
        out = F.silu(out)
    return out.permute(0, 2, 1).unsqueeze(1)


def test_depthwise_conv1d_stateful_signature_exposed():
    doc = ttnn._ttnn.operations.experimental.depthwise_conv1d.__call__.__doc__
    assert "conv_state" in doc
    assert "features: int" in doc
    assert "-> list[ttnn._ttnn.tensor.Tensor]" in doc


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("silu_activation", [False, True])
def test_depthwise_conv1d_interleaved(device, with_bias, silu_activation):
    torch.manual_seed(0)

    batch, seqlen, channels, kernel_size = 1, 64, 64, 4
    x = torch.randn((batch, 1, seqlen, channels), dtype=torch.bfloat16)
    w = torch.randn((channels, 1, kernel_size), dtype=torch.bfloat16)
    b = torch.randn((channels,), dtype=torch.bfloat16) if with_bias else None

    tt_x = ttnn.from_torch(x, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tt_w = ttnn.from_torch(w, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(b, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16) if with_bias else None

    tt_out = ttnn.experimental.depthwise_conv1d(tt_x, tt_w, kernel_size, bias=tt_b, silu_activation=silu_activation)

    ref = reference_causal_depthwise_conv1d(x.float(), w.float(), b.float() if b is not None else None, silu_activation)
    assert_with_pcc(ref, ttnn.to_torch(tt_out).float(), 0.999)


def test_depthwise_conv1d_sharded_input(device):
    torch.manual_seed(0)

    batch, seqlen, channels, kernel_size = 1, 96, 64, 4
    required_cores = 2
    available_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    if available_cores < required_cores:
        pytest.skip(f"Not enough cores to run sharded test case (need {required_cores} but have {available_cores})")

    x = torch.randn((batch, 1, seqlen, channels), dtype=torch.bfloat16)
    w = torch.randn((1, 1, channels, kernel_size), dtype=torch.bfloat16)

    shard_mem_config = ttnn.create_sharded_memory_config(
        [batch, 1, seqlen, channels],
        ttnn.CoreGrid(x=2, y=1),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.to_device(tt_x, device, shard_mem_config)
    tt_w = ttnn.from_torch(w, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_out = ttnn.experimental.depthwise_conv1d(tt_x, tt_w, kernel_size)

    ref = reference_causal_depthwise_conv1d(x.float(), w.reshape(channels, 1, kernel_size).float())
    assert tt_out.is_sharded()
    assert tt_out.memory_config() == shard_mem_config
    assert_with_pcc(ref, ttnn.to_torch(tt_out).float(), 0.999)


def test_depthwise_conv1d_accepts_host_weight_and_bias(device):
    torch.manual_seed(0)

    batch, seqlen, channels, kernel_size = 1, 32, 64, 4
    x = torch.randn((batch, 1, seqlen, channels), dtype=torch.bfloat16)
    w = torch.randn((channels, 1, kernel_size), dtype=torch.bfloat16)
    b = torch.randn((channels,), dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(x, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tt_w = ttnn.from_torch(w, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(b, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    tt_out = ttnn.experimental.depthwise_conv1d(tt_x, tt_w, kernel_size, bias=tt_b)

    ref = reference_causal_depthwise_conv1d(x.float(), w.float(), b.float())
    assert_with_pcc(ref, ttnn.to_torch(tt_out).float(), 0.999)
