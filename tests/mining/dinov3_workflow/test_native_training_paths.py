# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify DEFT output paths against TAO's real training-directory handling."""

import builtins
from pathlib import Path
import re

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nvidia_tao_ds.mining.dinov3.workflow.native_actions import (
    build_training_spec,
    finalize_training,
)


@pytest.mark.parametrize("train_config", [
    {},
    {"results_dir": None},
    {"results_dir": ""},
    {"results_dir": "${results_dir}/train"},
    {"results_dir": "/unrelated/old-training-run"},
    {"results_dir": "relative-training-run"},
], ids=["omitted", "null", "empty", "shipped-interpolation", "absolute", "relative"])
def test_native_training_uses_controller_stage_directory(tmp_path, train_config, deft_runtime_unavailable):
    """Controller-owned paths must survive native TAO's results-dir rewrite."""
    # The DEFT publish helpers ship with the stacked TAO PyTorch DINOv3 runtime,
    # which is not present in every CI image. Import them lazily -- a module-level
    # import would abort collection for the WHOLE suite instead of skipping this
    # one test. Mirrors the guard in test_workflow.py.
    try:
        import torch
        from nvidia_tao_pytorch.core.utilities import update_results_dir
        from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
        from nvidia_tao_pytorch.ssl.dinov3.utils.runtime_spec import (
            publish_runtime_spec,
        )
        from nvidia_tao_pytorch.ssl.dinov3.utils.refinement_attestation import (
            publish_native_attestation,
        )
    except ModuleNotFoundError as error:
        if error.name not in (
            "torch", "nvidia_tao_pytorch", "nvidia_tao_pytorch.core",
            "nvidia_tao_pytorch.core.utilities", "nvidia_tao_pytorch.core.decorators",
            "nvidia_tao_pytorch.core.decorators.workflow",
            "nvidia_tao_pytorch.ssl", "nvidia_tao_pytorch.ssl.dinov3",
            "nvidia_tao_pytorch.ssl.dinov3.utils",
            "nvidia_tao_pytorch.ssl.dinov3.utils.runtime_spec",
            "nvidia_tao_pytorch.ssl.dinov3.utils.refinement_attestation",
        ):
            raise
        deft_runtime_unavailable("requires the stacked TAO PyTorch DINOv3 DEFT runtime")

    base_spec = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create({
        "results_dir": "/unrelated/base-run",
        "dataset": {"batch_size": 1},
        "train": train_config,
    }), base_spec)
    original_spec = base_spec.read_bytes()
    manifest = tmp_path / "training.parquet"
    pq.write_table(pa.table({
        "sample_id": ["sample"], "storage_type": ["file"], "path": ["/data/sample.jpg"],
    }), manifest)
    parent = tmp_path / "original.pth"
    torch.save({"weight": torch.ones(1)}, parent)
    output = tmp_path / "rounds" / "round_001" / "train"

    spec_path, _, contract = build_training_spec(
        base_spec=base_spec, manifest=manifest, parent_checkpoint=parent,
        passes=1, output_dir=output, num_nodes=1, gpus_per_node=1,
        checkpoint_policy="base_checkpoint_each_round",
    )
    spec = OmegaConf.load(spec_path)
    assert spec.results_dir == spec.train.results_dir == str(output)
    # This is the actual helper called by DINOv3's monitor_status decorator.
    update_results_dir(spec, "train")
    update_results_dir(spec, "train")
    assert spec.results_dir == spec.train.results_dir == str(output)
    assert Path(contract["runtime_spec"]) == Path(spec.results_dir) / "experiment.yaml"
    assert spec.train.pretrained_model_path == str(parent)
    assert spec.train.auto_resume is False
    assert base_spec.read_bytes() == original_spec
    with pytest.raises(RuntimeError, match=re.escape(str(output / "train_implementation_audit.json"))):
        finalize_training(output)

    @monitor_status(name="DINOv3", mode="train", write_experiment_spec=False)
    def publish_outputs(config):
        publish_runtime_spec(config, config.results_dir)
        publish_native_attestation("train", config.results_dir)

    publish_outputs(spec)
    for name in ("experiment.yaml", "train_implementation_audit.json", "status.json"):
        assert (output / name).is_file()
    assert not (output / "train").exists()


@pytest.mark.parametrize("missing,optional", [
    ("torch", True), ("nvidia_tao_pytorch", True),
    ("nvidia_tao_pytorch.core.utilities", True),
    ("google.protobuf", False),
])
def test_training_import_gate_is_narrow(monkeypatch, tmp_path, missing, optional):
    """Only absent optional runtime packages may use the skip/fail policy."""
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("missing test dependency", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    expected = pytest.fail.Exception if optional else ModuleNotFoundError
    with pytest.raises(expected):
        test_native_training_uses_controller_stage_directory(tmp_path, {}, pytest.fail)
