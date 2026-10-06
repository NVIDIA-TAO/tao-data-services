# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit runtime requirements for optional container integration tests."""

import importlib
import inspect

import pytest


def pytest_addoption(parser):
    """Allow container validation to require complete integration runtimes."""
    parser.addoption(
        "--require-deft-runtime", action="store_true",
        help="Fail rather than skip when DEFT training dependencies are unavailable.",
    )

    parser.addoption(
        "--require-sparse4d-runtime", action="store_true",
        help="Fail rather than skip when the Sparse4D co-training runtime is unavailable.",
    )


@pytest.fixture
def deft_runtime_unavailable(pytestconfig):
    """Skip unsupported source-only environments, but fail required-runtime runs."""
    def unavailable(reason):
        if pytestconfig.getoption("--require-deft-runtime"):
            pytest.fail(reason)
        pytest.skip(reason)
    return unavailable


@pytest.fixture
def sparse4d_runtime(pytestconfig):
    """Require co-training capabilities, not just an installed Sparse4D module."""
    def unavailable(reason):
        if pytestconfig.getoption("--require-sparse4d-runtime"):
            pytest.fail(reason)
        pytest.skip(reason)

    try:
        dataset_module = importlib.import_module("nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset")
        transforms = importlib.import_module("nvidia_tao_pytorch.cv.sparse4d.dataloader.transforms")
    except ImportError as exc:
        unavailable(f"Sparse4D integration runtime unavailable: {exc}")

    dataset_class = getattr(dataset_module, "Omniverse3DDetTrackDataset", None)
    if dataset_class is None or "lazy_load" not in inspect.signature(dataset_class).parameters:
        unavailable("Sparse4D integration requires Omniverse3DDetTrackDataset with lazy_load support")
    if not hasattr(transforms, "LoadRTDETR2D"):
        unavailable("Sparse4D integration requires the co-training LoadRTDETR2D transform")
    return dataset_module, transforms
