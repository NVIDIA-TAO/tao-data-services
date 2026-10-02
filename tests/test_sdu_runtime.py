# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require CI's installed SDU to match the declared base-image dependency."""

from importlib import import_module, metadata
from pathlib import Path
import re

import pytest


def test_sdu_version_matches_base_image_requirement():
    """Fail on stale images even when their legacy SDU API still works."""
    requirements = Path(__file__).resolve().parents[1] / "docker" / "requirements-pip.txt"
    pins = re.findall(r"^spatialai-data-utils==([^\s#]+)$", requirements.read_text(), re.MULTILINE)
    assert len(pins) == 1, "Declare exactly one SDU pin in docker/requirements-pip.txt"
    installed = metadata.version("spatialai-data-utils")
    assert installed == pins[0], (
        f"Installed SDU {installed} != required {pins[0]}; rebuild the base image "
        "and update docker/manifest.json and release/docker/Dockerfile.release."
    )


@pytest.mark.parametrize("module,symbol", [
    ("spatialai_data_utils.constants", "FPS"),
    ("spatialai_data_utils.loaders.calibration", "load_calib"),
    ("spatialai_data_utils.datasets.scenes", "get_cam_names_in_scene"),
])
def test_sdu_real_runtime_imports(module, symbol):
    """Exercise published SDU imports, without replacing them with mocks."""
    assert getattr(import_module(module), symbol) is not None
