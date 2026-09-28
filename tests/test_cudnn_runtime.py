# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU image regression for cuDNN engines removed by the vLLM base image."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cudnn_patch_embedding_forward_backward(dtype):
    """Exercise forward and both gradients, not just a successful CUDA import."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("Requires a GPU with BF16 support")
    assert torch.backends.cudnn.is_available()
    with torch.backends.cudnn.flags(enabled=True):
        layer = torch.nn.Conv2d(3, 384, kernel_size=16, stride=16).cuda().to(dtype)
        inputs = torch.randn(2, 3, 256, 256, device="cuda", dtype=dtype,
                             requires_grad=True)
        output = layer(inputs)
        assert output.shape == (2, 384, 16, 16)
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        for gradient in (inputs.grad, layer.weight.grad, layer.bias.grad):
            assert gradient is not None
            assert torch.isfinite(gradient).all()
