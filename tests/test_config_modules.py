# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests covering the nvidia_tao_ds.config dataclass move and default_specs aliasing."""

import importlib

import pytest


LOCAL_CONFIG_MODULES = [
    "analytics",
    "annotations",
    "augmentation",
    "image",
]


@pytest.mark.parametrize("module", LOCAL_CONFIG_MODULES)
def test_local_experiment_config_imports_and_constructs(module):
    """Each moved task config exposes a default-constructible ExperimentConfig.

    Mirrors the dynamic import path used at runtime by default_specs.py and
    api/app.py: f"nvidia_tao_ds.config.{module}.default_config".
    """
    mod = importlib.import_module(f"nvidia_tao_ds.config.{module}.default_config")
    assert hasattr(mod, "ExperimentConfig")
    assert mod.ExperimentConfig() is not None


def test_annotations_slice_and_merge_configs():
    """SliceConfig and MergeConfig live alongside the annotations default config."""
    from nvidia_tao_ds.config.annotations.slice_config import SliceConfig
    from nvidia_tao_ds.config.annotations.merge_config import MergeConfig
    assert SliceConfig() is not None
    assert MergeConfig() is not None


def test_auto_label_config_imports_when_tao_core_available():
    """auto_label.default_config still pulls grounding_dino / mal configs from tao-core.

    Skipped when nvidia_tao_core is not installed — that chain moves with the
    separate tao-pytorch MR.
    """
    pytest.importorskip("nvidia_tao_core")
    mod = importlib.import_module("nvidia_tao_ds.config.auto_label.default_config")
    assert hasattr(mod, "ExperimentConfig")
    assert mod.ExperimentConfig() is not None


def test_default_specs_config_root_is_local():
    """CONFIG_ROOT points at nvidia_tao_ds/config/, not tao-core's config directory."""
    from nvidia_tao_ds.core.utils import default_specs
    assert default_specs.CONFIG_ROOT.endswith("nvidia_tao_ds/config")
    assert "nvidia_tao_core" not in default_specs.CONFIG_ROOT


def test_get_supported_modules_includes_analytics_alias():
    """get_supported_modules surfaces 'analytics' via the data_analytics alias.

    The config dir is 'analytics' but the DS module dir is 'data_analytics'; without
    CONFIG_TO_DS_MODULE_ALIASES the strict-equality intersection silently dropped it.
    """
    from nvidia_tao_ds.core.utils import default_specs
    supported = default_specs.get_supported_modules()
    assert "analytics" in supported
    # Sanity: a non-aliased module also surfaces correctly.
    assert "annotations" in supported
