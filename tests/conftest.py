# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit runtime requirements for DEFT container validation."""

import pytest


def pytest_addoption(parser):
    """Allow container validation to require the complete DEFT runtime."""
    parser.addoption(
        "--require-deft-runtime", action="store_true",
        help="Fail rather than skip when DEFT training dependencies are unavailable.",
    )


@pytest.fixture
def deft_runtime_unavailable(pytestconfig):
    """Skip unsupported source-only environments, but fail required-runtime runs."""
    def unavailable(reason):
        if pytestconfig.getoption("--require-deft-runtime"):
            pytest.fail(reason)
        pytest.skip(reason)
    return unavailable
