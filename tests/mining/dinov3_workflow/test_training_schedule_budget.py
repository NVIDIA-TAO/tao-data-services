# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check per-candidate schedule warnings without changing training policy."""

import warnings

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nvidia_tao_ds.mining.dinov3.workflow.native_actions import build_training_spec


@pytest.fixture
def prepare_round(tmp_path, deft_runtime_unavailable):
    """Compose real native specs from synthetic manifest rows and a parent file."""
    try:
        from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
    except ModuleNotFoundError as error:
        if not error.name.startswith("nvidia_tao_pytorch"):
            raise
        deft_runtime_unavailable("requires the TAO DINOv3 runtime")
    schema = OmegaConf.structured(ExperimentConfig())
    if "train_manifest" not in schema.dataset:
        deft_runtime_unavailable("requires the DINOv3 manifest training schema")

    def prepare(rows=768, passes=2, nodes=1, gpus=1, schedules=None, output="round"):
        base = tmp_path / "base.yaml"
        OmegaConf.save(OmegaConf.create({
            "dataset": {"batch_size": 16},
            "train": {"schedulers": schedules or {}},
        }), base)
        original = base.read_bytes()
        manifest = tmp_path / "training.parquet"
        pq.write_table(pa.table({
            "sample_id": [str(i) for i in range(rows)],
            "storage_type": ["file"] * rows,
            "path": [f"/data/{i}.jpg" for i in range(rows)],
        }), manifest)
        parent = tmp_path / "original.pth"
        parent.write_bytes(b"spec-composition-only")
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            path, _, contract, diagnostic = build_training_spec(
                base_spec=base, manifest=manifest, parent_checkpoint=parent,
                passes=passes, output_dir=tmp_path / output,
                num_nodes=nodes, gpus_per_node=gpus,
                checkpoint_policy="base_checkpoint_each_round",
            )
        assert base.read_bytes() == original
        prepared = OmegaConf.load(path)
        expected = OmegaConf.merge(schema, OmegaConf.load(base))
        expected.train.num_nodes = nodes
        expected.train.num_gpus = gpus
        assert OmegaConf.to_container(prepared.train.schedulers) == OmegaConf.to_container(
            expected.train.schedulers, resolve=True,
        )
        assert prepared.train.pretrained_model_path == str(parent)
        assert prepared.train.resume_training_checkpoint_path is None
        assert prepared.train.auto_resume is False
        assert contract["scheduler_policy"] == "preserve_resolved_base_spec"
        messages = [
            str(item.message) for item in emitted
            if item.category is UserWarning and
            str(item.message).startswith("DINOv3 DEFT training round")
        ]
        assert messages == ([diagnostic["message"]] if diagnostic else [])
        return contract, messages

    return prepare


@pytest.mark.parametrize("rows,steps", [(768, 96), (1536, 192)])
def test_short_round_reports_both_warmups_and_last_layer_freeze(prepare_round, rows, steps):
    """The reported 96/192-step budgets must not silently accept long schedules."""
    contract, messages = prepare_round(rows=rows, schedules={
        "learning_rate": {"warm_up_steps": 10000},
        "last_layer_learning_rate": {"warm_up_steps": 10000, "freeze_steps": 1250},
    })
    assert contract["total_optimizer_steps"] == steps
    assert len(messages) == 1
    for text in (
        f"total_optimizer_steps={steps}",
        "train.schedulers.learning_rate.warm_up_steps=10000",
        "train.schedulers.last_layer_learning_rate.warm_up_steps=10000",
        "train.schedulers.last_layer_learning_rate.freeze_steps=1250",
        "Scheduler settings are preserved", "training.base_spec",
    ):
        assert text in messages[0]


@pytest.mark.parametrize("name,field", [
    ("learning_rate", "warm_up_steps"),
    ("last_layer_learning_rate", "warm_up_steps"),
    ("last_layer_learning_rate", "freeze_steps"),
])
@pytest.mark.parametrize("threshold,warn", [(0, False), (20, False), (95, False), (96, True), (97, True)])
def test_schedule_boundaries_are_independent(prepare_round, name, field, threshold, warn):
    """Equality still covers all steps, but disabled and completed phases do not."""
    schedules = {
        key: {"warm_up_steps": 0}
        for key in ("learning_rate", "last_layer_learning_rate")
    }
    schedules["last_layer_learning_rate"]["freeze_steps"] = 0
    schedules[name][field] = threshold
    _, messages = prepare_round(schedules=schedules)
    assert len(messages) == int(warn)
    if warn:
        assert f"train.schedulers.{name}.{field}={threshold}" in messages[0]


def test_resolved_schedule_uses_padded_distributed_budget(prepare_round):
    """Warnings use resolved topology and the same padded budget as the contract."""
    contract, messages = prepare_round(rows=65, nodes=2, gpus=2, schedules={
        "learning_rate": {"warm_up_steps": "${eval:'${train.num_nodes} * ${train.num_gpus}'}"},
        "last_layer_learning_rate": {"warm_up_steps": 0, "freeze_steps": 0},
    })
    assert contract["total_optimizer_steps"] == 4
    assert len(messages) == 1
    assert "learning_rate.warm_up_steps=4" in messages[0]


def test_warning_identifies_each_candidate(prepare_round):
    """Equal-budget candidates have distinct warnings and the same initializer."""
    schedules = {"learning_rate": {"warm_up_steps": 10000}}
    _, first = prepare_round(schedules=schedules, output="round_001")
    _, second = prepare_round(schedules=schedules, output="round_002")
    assert "round_001" in first[0]
    assert "round_002" in second[0]
    assert first != second


def test_warning_includes_partially_covered_phases(prepare_round):
    """An emitted warning quantifies phases that do not cover the entire round."""
    contract, messages = prepare_round(rows=10400, schedules={
        "learning_rate": {"warm_up_steps": 10000},
        "last_layer_learning_rate": {"warm_up_steps": 20, "freeze_steps": 1250},
    })
    assert contract["total_optimizer_steps"] == 1300
    assert len(messages) == 1
    assert "last_layer_learning_rate.freeze_steps=1250 covers 96.2%" in messages[0]


def test_unrelated_dependency_warnings_do_not_change_schedule_assertions(prepare_round, monkeypatch):
    """Unrelated UserWarnings and deprecations must not inflate warning counts."""
    from nvidia_tao_ds.mining.dinov3.workflow import native_actions

    original = native_actions._experiment_spec

    def noisy_spec(*args, **kwargs):
        warnings.warn("dependency deprecation", DeprecationWarning)
        warnings.warn("unrelated dependency warning", UserWarning)
        return original(*args, **kwargs)

    monkeypatch.setattr(native_actions, "_experiment_spec", noisy_spec)
    _, messages = prepare_round(schedules={
        "learning_rate": {"warm_up_steps": 20},
        "last_layer_learning_rate": {"warm_up_steps": 20, "freeze_steps": 0},
    })
    assert messages == []
