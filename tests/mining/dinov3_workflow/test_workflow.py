# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and local end-to-end tests for DINOv3 SSL DEFT."""

from __future__ import annotations

from contextlib import contextmanager
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf
import pandas as pd
import pytest
import torch
import yaml

from nvidia_tao_ds.mining.dinov3.contracts import artifact_content_id


DATA_SERVICES = Path(__file__).resolve().parents[3]
SCRIPTS = DATA_SERVICES
WORKFLOW_ROOT = DATA_SERVICES / "nvidia_tao_ds/mining/dinov3/workflow"

from nvidia_tao_ds.mining.dinov3.workflow import RefinementWorkflow, WorkflowConfig  # noqa: E402
from nvidia_tao_ds.mining.dinov3.workflow import cli as workflow_cli  # noqa: E402
from nvidia_tao_ds.mining.dinov3.workflow import controller as workflow_controller  # noqa: E402
from nvidia_tao_ds.mining.dinov3.workflow import native_actions as workflow_native_actions  # noqa: E402
from nvidia_tao_ds.mining.dinov3.workflow.controller import (  # noqa: E402
    StageFailure,
    _empty_search_stop_reason,
    _file_identity,
    _is_supported_search_proof,
    _module_source,
    _source_payload_contract,
    _validate_continuation_source_lineage,
    _verified_embedding_store,
)
from nvidia_tao_ds.mining.dinov3.workflow.config import (  # noqa: E402
    canonical_digest,
    training_allocation,
)
from nvidia_tao_ds.mining.dinov3.workflow.execution import (  # noqa: E402
    StageRequest,
    StageResult,
    build_runner,
    client_job_id,
)
from nvidia_tao_ds.mining.dinov3.workflow.native_actions import (  # noqa: E402
    resolved_training_batch_size,
)


@pytest.fixture(autouse=True)
def _allocated_test_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


@pytest.fixture(autouse=True)
def _compatible_native_config(monkeypatch: pytest.MonkeyPatch):
    """Keep DS unit tests independent of an unpublished TAO image rebuild."""
    from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig

    installed = OmegaConf.structured(ExperimentConfig())
    if "grit_score" in installed and "train_manifest" in installed.dataset:
        return

    defaults = OmegaConf.create(
        {
            "results_dir": None,
            "model": {"distill": {"enable": False}},
            "dataset": {
                "batch_size": 1,
                "train_manifest": None,
                "train_dataset": {"images_dir": None},
            },
            "train": {
                "num_epochs": 1,
                "num_nodes": 1,
                "num_gpus": 1,
                "gpu_ids": [0],
                "pretrained_model_path": None,
                "resume_training_checkpoint_path": None,
                "auto_resume": False,
                "checkpoint_interval": 1,
                "checkpoint_interval_unit": "epoch",
            },
            "grit_score": {
                "results_dir": None,
                "input_parquet": None,
                "checkpoint": None,
                "base_spec": None,
                "batch_size": 1,
                "workers": 0,
                "device": "cpu",
                "amp": False,
                "neighbor_backend": "torch_exact",
                "neighbor_device": "cpu",
                "work_dir": None,
            },
        }
    )

    def compose(base_spec: str | Path):
        base_path = Path(base_spec).expanduser().resolve()
        return base_path, OmegaConf.merge(defaults, OmegaConf.load(base_path))

    monkeypatch.setattr(workflow_native_actions, "_experiment_spec", compose)


FAKE = Path(__file__).parent / "fixtures" / "fake_actions.py"
FAKE_RUNNER = Path(__file__).parent / "fixtures" / "fake_runner.py"


def _fake_training_metadata(checkpoint: str | Path) -> dict:
    return json.loads(
        (Path(checkpoint).parent / "fake_training.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.parametrize("path", [WORKFLOW_ROOT / "controller.py", FAKE])
def test_parquet_reads_disable_background_prefetch(path: Path) -> None:
    """Arrow 23 background read-ahead can abort short-lived leaves on exit."""
    calls = [
        node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd" and node.func.attr == "read_parquet"
    ]
    assert calls
    for call in calls:
        assert any(
            keyword.arg == "pre_buffer" and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False for keyword in call.keywords
        ), f"{path}:{call.lineno} must disable asynchronous read-ahead"


def _committed_payload(
    root: Path,
    name: str,
    artifact_type: str,
    payload: dict,
    inputs: list[dict] | None = None,
) -> tuple[Path, dict]:
    root.mkdir(exist_ok=True)
    manifest = root / name
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    artifact = {
        "artifact_type": artifact_type,
        "schema_version": "1.0",
        "producer": {"action": "test", "version": "1.0"},
        "inputs": inputs or [],
        "payload": payload,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    artifact["artifact_id"] = artifact_content_id(artifact)
    (root / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    (root / "_SUCCESS").write_text(
        artifact["artifact_id"] + "\n", encoding="utf-8"
    )
    return manifest, artifact


def _payload_contract_lineage(
    path: Path, *, row_count: int
) -> tuple[dict, dict]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    identity = {
        "uri": path.resolve().as_uri(),
        "bytes": len(raw),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "role": "source_payload_contract",
    }
    binding = {
        "sha256": identity["sha256"],
        "datasets_digest": canonical_digest(contract["datasets"]),
        "locator_audit": {
            "algorithm": "sha256_canonical_jsonl_v1",
            "digest": "sha256:" + "c" * 64,
            "row_count": row_count,
        },
    }
    return binding, identity


def _registered_source_payload(
    tmp_path: Path, shard: Path, contract_path: Path
) -> tuple[dict, dict]:
    binding, contract_input = _payload_contract_lineage(
        contract_path, row_count=int(pd.read_parquet(shard).shape[0])
    )
    shards = [
        {
            "relative_path": shard.name,
            "bytes": shard.stat().st_size,
            "sha256": "sha256:"
            + hashlib.sha256(shard.read_bytes()).hexdigest(),
        }
    ]
    payload = {
        "root_uri": tmp_path.resolve().as_uri(),
        "shards": shards,
        "row_count": int(pd.read_parquet(shard).shape[0]),
        "shard_count": 1,
        "embedding_dim": 2,
        "encoder": {"name": "test"},
        "fingerprint_method": "sha256",
        "inventory_digest": canonical_digest(shards),
        "source_payload_contract": binding,
    }
    return payload, contract_input


def _config(tmp_path: Path, strategy: str) -> WorkflowConfig:
    base = tmp_path / "base.pth"
    base.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "dinov3.yaml"
    base_spec.write_text(
        "model: {}\ndataset:\n  batch_size: 2\n", encoding="utf-8"
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text("[]", encoding="utf-8")
    benchmark_units = tmp_path / "benchmark_units.parquet"
    pd.DataFrame(
        {"sample_id": ["sealed-1"], "acquisition_unit_id": ["sealed-1"]}
    ).to_parquet(
        benchmark_units, index=False
    )
    source = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "embedding": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            "storage_type": ["file"] * 3,
            "path": [
                str(tmp_path / "data/s1.jpg"),
                str(tmp_path / "data/s2.jpg"),
                str(tmp_path / "data/s3.jpg"),
            ],
            "member": [None] * 3,
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    source_payload_contract = tmp_path / "source_payload_contract.json"
    source_payload_contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-source",
                        "version": "v1",
                        "root_uri": tmp_path.resolve().as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if strategy == "grit_score":
        targets = pd.DataFrame(
            {
                "sample_id": ["q1", "q2"],
                "task": ["aoi", "aoi"],
                "role": ["query", "query"],
                "grit_score": [0.9, 0.1],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
                "acquisition_unit_id": ["target-q1", "target-q2"],
            }
        )
    else:
        targets = pd.DataFrame(
            {
                "sample_id": ["q1", "q2", "q3", "q4"],
                "task": ["classification", "classification", "segmentation", "segmentation"],
                "weakness_score": [0.9, 0.1, 0.8, 0.2],
                "embedding": [[1.0, 0.0], [0.0, 1.0], [0.95, 0.05], [0.0, 1.0]],
                "acquisition_unit_id": [
                    "target-q1", "target-q2", "target-q3", "target-q4"
                ],
            }
        )
    target_path = tmp_path / "targets.parquet"
    targets.to_parquet(target_path, index=False)

    pythonpath = str(DATA_SERVICES)
    value = {
        "schema_version": "1.0",
        "workflow": {"strategy": strategy, "max_rounds": 1},
        "model": {"base_checkpoint": str(base)},
        "data": {
            "target_manifest": str(target_path),
            "source_store_validation": "full_sha256",
            "benchmark_manifest": str(benchmark),
            "benchmark_acquisition_units": str(benchmark_units),
            "source_payload_contract": str(source_payload_contract),
            "source_parts": [str(source_path)],
        },
        "mining": {
            "top_k_per_target": 1,
            "min_similarity": 0.8,
            "parent_history_policy": "exclude",
        },
        "training": {
            "base_spec": str(base_spec),
            "passes_per_round": 2,
            "checkpoint_policy": "base_checkpoint_each_round",
        },
        "actions": {
            "score": {
                "implementation_files": [str(FAKE)],
                "command": [
                    sys.executable,
                    str(FAKE),
                    "score",
                    "--input",
                    "{target_manifest}",
                    "--checkpoint",
                    "{checkpoint}",
                    "--output-dir",
                    "{output_dir}",
                    "--strategy",
                    strategy,
                ]
            },
            "data": {
                "command": [
                    sys.executable,
                    "-m",
                    "nvidia_tao_ds.mining.dinov3.internal.refinement",
                ]
            },
            "train": {
                "implementation_files": [str(FAKE)],
                "command": [
                    sys.executable,
                    str(FAKE),
                    "train",
                    "--manifest",
                    "{training_manifest}",
                    "--checkpoint",
                    "{checkpoint}",
                    "--passes",
                    "{passes}",
                    "--output-dir",
                    "{output_dir}",
                ],
            },
            "evaluate": {
                "implementation_files": [str(FAKE)],
                "command": [
                    sys.executable,
                    str(FAKE),
                    "evaluate",
                    "--benchmark",
                    "{benchmark_manifest}",
                    "--checkpoint",
                    "{checkpoint}",
                    "--output-dir",
                    "{output_dir}",
                    "--scope",
                    "{evaluation_scope}",
                ],
                "metrics": "{output_dir}/metrics.json",
            },
        },
        "execution": {
            "backend": "local",
            "environment": {
                "PYTHONPATH": pythonpath,
                "CUDA_VISIBLE_DEVICES": "0",
            },
        },
        "output": {"run_dir": str(tmp_path / "run")},
    }
    if strategy == "grit_score":
        value["grit"] = {"target_fraction": 0.5}
    else:
        value["multi_task"] = {
            "tasks": ["classification", "segmentation"],
            "targets_per_task": 1,
        }
    return WorkflowConfig.from_dict(value)


def test_source_payload_contract_requires_immutable_versions(tmp_path: Path) -> None:
    path = tmp_path / "source_payload_contract.json"
    contract = {
        "schema_version": "1.0",
        "immutability": "immutable",
        "datasets": [
            {
                "dataset_id": "wfm-aoi",
                "version": "snapshot-1",
                "root_uri": "file:///datasets/wfm-aoi",
            }
        ],
    }
    path.write_text(json.dumps(contract), encoding="utf-8")
    assert _source_payload_contract(path) == contract

    contract["immutability"] = "mutable"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable payloads"):
        _source_payload_contract(path)


@pytest.mark.parametrize(
    ("embedding_dim", "error"),
    [
        (3, "embedding dimension differs"),
        (True, "requires integer embedding_dim"),
        ("2", "requires integer embedding_dim"),
        (2.9, "requires integer embedding_dim"),
    ],
)
def test_target_embedding_dimension_mismatch_fails_before_launch(
    tmp_path: Path, embedding_dim, error: str,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    shard = tmp_path / "source.parquet"
    payload, contract_input = _registered_source_payload(
        tmp_path, shard, Path(value["data"]["source_payload_contract"])
    )
    manifest, _ = _committed_payload(
        tmp_path / "store",
        "embedding_store.json",
        "embedding_store",
        payload,
        inputs=[contract_input],
    )
    target_contract = tmp_path / "target_embedding_contract.json"
    target_contract.write_text(
        json.dumps({
            "encoder": {"name": "test"}, "embedding_dim": embedding_dim,
        }),
        encoding="utf-8",
    )
    value["data"].pop("source_parts")
    value["data"].update({
        "source_store_manifest": str(manifest),
        "target_embedding_contract": str(target_contract),
    })
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))

    class MustNotRun:
        @staticmethod
        def run(_request):
            raise AssertionError("target contract mismatch launched a stage")

    workflow.runner = MustNotRun()
    with pytest.raises(ValueError, match=error):
        workflow.execute()


def test_embedding_store_verifies_every_shard_identity(tmp_path: Path) -> None:
    shard = tmp_path / "source.parquet"
    pd.DataFrame({"sample_id": ["source-1"]}).to_parquet(shard, index=False)
    digest = "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest()
    payload = {
        "root_uri": tmp_path.resolve().as_uri(),
        "shards": [
            {
                "relative_path": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": digest,
            }
        ],
        "row_count": 1,
        "shard_count": 1,
        "embedding_dim": 2,
        "encoder": {"name": "test"},
    }
    manifest, _ = _committed_payload(
        tmp_path / "store", "embedding_store.json", "embedding_store", payload
    )
    _, _, paths, verified = _verified_embedding_store(manifest)
    assert paths == [shard.resolve()]
    assert verified["shards"][0]["sha256"] == digest

    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        _verified_embedding_store(manifest)


def test_sealed_inventory_requires_and_rechecks_content_seal(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "embedding": [[1.0, 0.0]],
            "path": ["/data/a.jpg"],
            "storage_type": ["file"],
        }
    ).to_parquet(shard, index=False)
    stat = shard.stat()
    shard_identity = {
        "relative_path": shard.name,
        "bytes": stat.st_size,
        "rows": 1,
        "sha256": "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest(),
    }
    seal = {
        "relative_path": shard.name,
        "bytes": stat.st_size,
        "sha256": shard_identity["sha256"],
        "stat": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
    }
    payload = {
        "root_uri": tmp_path.resolve().as_uri(),
        "shards": [shard_identity],
        "row_count": 1,
        "shard_count": 1,
        "embedding_dim": 2,
        "encoder": {"name": "test"},
        "inventory_digest": canonical_digest([shard_identity]),
        "content_verification": {
            "algorithm": "sha256_each_shard_with_posix_stat_v1",
            "inventory_digest": canonical_digest([shard_identity]),
            "shard_seal_digest": canonical_digest([seal]),
            "shards": [seal],
        },
    }
    manifest, _ = _committed_payload(
        tmp_path / "store", "embedding_store.json", "embedding_store", payload
    )

    _verified_embedding_store(manifest, content_validation="sealed_inventory")
    # Lustre may round timestamps; exercise a distinct metadata identity rather
    # than assuming a one-millisecond change is observable on every filesystem.
    changed_mtime = shard.stat().st_mtime_ns + 2_000_000_000
    os.utime(shard, ns=(shard.stat().st_atime_ns, changed_mtime))

    with pytest.raises(ValueError, match="differs from its content seal"):
        _verified_embedding_store(
            manifest, content_validation="sealed_inventory"
        )


def test_registered_store_requires_shard_digest(tmp_path: Path) -> None:
    shard = tmp_path / "source.parquet"
    pd.DataFrame({"sample_id": ["source-1"]}).to_parquet(shard, index=False)
    payload = {
        "root_uri": tmp_path.resolve().as_uri(),
        "shards": [{"relative_path": shard.name, "bytes": shard.stat().st_size}],
        "row_count": 1,
        "shard_count": 1,
        "embedding_dim": 2,
        "encoder": {"name": "test"},
    }
    manifest, _ = _committed_payload(
        tmp_path / "store", "embedding_store.json", "embedding_store", payload
    )
    with pytest.raises(ValueError, match="bytes and SHA-256"):
        _verified_embedding_store(manifest)


def test_registered_store_requires_payload_contract_lineage(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    shard = tmp_path / "source.parquet"
    payload, _ = _registered_source_payload(
        tmp_path, shard, Path(value["data"]["source_payload_contract"])
    )
    payload.pop("source_payload_contract")
    manifest, _ = _committed_payload(
        tmp_path / "store", "embedding_store.json", "embedding_store", payload
    )
    target_contract = tmp_path / "target_embedding_contract.json"
    target_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )
    value["data"].pop("source_parts")
    value["data"].update(
        {
            "source_store_manifest": str(manifest),
            "target_embedding_contract": str(target_contract),
        }
    )

    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    with pytest.raises(ValueError, match="not bound"):
        workflow.validate()


def test_store_is_reverified_after_controller_lock(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    original = tmp_path / "source.parquet"
    replacement = tmp_path / "replacement.parquet"
    replacement_frame = pd.read_parquet(original).copy()
    replacement_frame["sample_id"] = ["r1", "r2", "r3"]
    replacement_frame.to_parquet(replacement, index=False)
    contract_path = Path(value["data"]["source_payload_contract"])
    original_payload, contract_input = _registered_source_payload(
        tmp_path, original, contract_path
    )
    store_dir = tmp_path / "store"
    manifest, _ = _committed_payload(
        store_dir,
        "embedding_store.json",
        "embedding_store",
        original_payload,
        inputs=[contract_input],
    )
    target_contract = tmp_path / "target_embedding_contract.json"
    target_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )
    value["data"].pop("source_parts")
    value["data"].update(
        {
            "source_store_manifest": str(manifest),
            "target_embedding_contract": str(target_contract),
        }
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    replacement_payload, replacement_input = _registered_source_payload(
        tmp_path, replacement, contract_path
    )

    @contextmanager
    def mutate_while_waiting():
        _committed_payload(
            store_dir,
            "embedding_store.json",
            "embedding_store",
            replacement_payload,
            inputs=[replacement_input],
        )
        yield

    workflow.store.controller_lock = mutate_while_waiting
    state = workflow.execute()
    assert state["status"] == "complete"
    lock = json.loads((tmp_path / "run/data.lock.json").read_text())
    assert lock["source_store"]["shards"][0]["relative_path"] == replacement.name


def test_search_rejects_store_replaced_after_initialization(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    source = tmp_path / "source.parquet"
    contract_path = Path(value["data"]["source_payload_contract"])
    payload, contract_input = _registered_source_payload(
        tmp_path, source, contract_path
    )
    store_dir = tmp_path / "store"
    manifest, _ = _committed_payload(
        store_dir,
        "embedding_store.json",
        "embedding_store",
        payload,
        inputs=[contract_input],
    )
    target_contract = tmp_path / "target_embedding_contract.json"
    target_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )
    value["data"].pop("source_parts")
    value["data"].update(
        {
            "source_store_manifest": str(manifest),
            "target_embedding_contract": str(target_contract),
        }
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    original_score = workflow._score  # pylint: disable=protected-access
    mutated = False

    def score_then_replace(*args, **kwargs):
        nonlocal mutated
        result = original_score(*args, **kwargs)
        if not mutated:
            replacement = pd.read_parquet(source)
            replacement["sample_id"] = ["new-1", "new-2", "new-3"]
            replacement.to_parquet(source, index=False)
            replacement_payload, replacement_input = _registered_source_payload(
                tmp_path, source, contract_path
            )
            _committed_payload(
                store_dir,
                "embedding_store.json",
                "embedding_store",
                replacement_payload,
                inputs=[replacement_input],
            )
            mutated = True
        return result

    workflow._score = score_then_replace  # pylint: disable=protected-access
    with pytest.raises(StageFailure, match="does not bind locked source store"):
        workflow.execute()


@pytest.mark.parametrize("strategy", ["grit_score", "multi_task_round_robin"])
def test_local_workflow_smoke_and_resume(tmp_path: Path, strategy: str) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, strategy))
    state = workflow.execute()
    assert state["status"] == "complete"
    assert state["stop_reason"] == "max_rounds"
    assert (tmp_path / "run" / "final_model.json").is_file()
    assert (tmp_path / "run" / "report.html").is_file()
    report = (tmp_path / "run" / "report.html").read_text(encoding="utf-8")
    assert "Round Summary" in report
    assert "Task Distribution" in report
    assert "Evaluation Metrics" in report
    assert "Failures And Retries" in report
    events_before = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")

    resumed = workflow.execute()
    assert resumed["status"] == "complete"
    assert (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8") == events_before


def test_metric_patience_stops_and_delivers_best_checkpoint(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["workflow"].update(
        {
            "max_rounds": 4,
            "persistent_target_rounds": 4,
            "early_stopping": {
                "patience": 2,
                "min_delta": 0.01,
                "metrics": [{"task": "smoke", "name": "score"}],
            },
        }
    )
    value["mining"].update(
        {"min_similarity": -1.0, "hard_min_similarity": -1.0}
    )
    source_path = Path(value["data"]["source_parts"][0])
    source = pd.read_parquet(source_path)
    source = pd.concat(
        [
            source,
            pd.DataFrame(
                {
                    "sample_id": ["s4", "s5", "s6"],
                    "embedding": [[0.8, 0.2], [0.7, 0.3], [0.3, 0.7]],
                    "storage_type": ["file"] * 3,
                    "path": [str(tmp_path / f"data/s{i}.jpg") for i in range(4, 7)],
                    "member": [None] * 3,
                }
            ),
        ],
        ignore_index=True,
    )
    source.to_parquet(source_path, index=False)
    value["execution"]["environment"]["FAKE_METRIC_VALUES"] = (
        "1.0,1.2,1.1,1.15,1.3"
    )

    state = RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()

    assert state["status"] == "complete"
    assert state["stop_reason"] == "metric_patience"
    assert state["selected_round"] == 1
    assert len(state["completed_rounds"]) == 3
    assert state["early_stopping"]["rounds_without_improvement"] == 2
    assert state["early_stopping"]["best_score"] == pytest.approx(1.2)
    final_model = json.loads(
        (tmp_path / "run" / "final_model.json").read_text(encoding="utf-8")
    )
    assert final_model["selected_round"] == 1
    assert final_model["checkpoint"].endswith(
        "rounds/round_001/train/checkpoint.pth"
    )
    assert final_model["latest_checkpoint"].endswith(
        "rounds/round_003/train/checkpoint.pth"
    )
    assert "Early Stopping" in (
        tmp_path / "run" / "report.html"
    ).read_text(encoding="utf-8")


def test_early_stopping_requires_evaluation(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["workflow"]["early_stopping"] = {
        "metrics": [{"task": "smoke", "name": "score"}]
    }
    value["actions"]["evaluate"]["command"] = []
    with pytest.raises(ValueError, match="requires actions.evaluate.command"):
        WorkflowConfig.from_dict(value)


def test_balanced_multitask_workflow_publishes_unique_and_training_views(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["multi_task"]["policy"] = "balanced"
    state = RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()

    materialize = tmp_path / "run/rounds/round_001/materialize"
    cumulative = pd.read_parquet(materialize / "training_manifest.parquet")
    training = pd.read_parquet(
        materialize / "balanced_training_manifest.parquet"
    )
    assert state["current_training_manifest"] == str(
        materialize / "training_manifest.parquet"
    )
    assert cumulative["sample_id"].is_unique
    assert "query_task" in cumulative
    assert set(cumulative["sample_id"]).issubset(set(training["sample_id"]))
    assert training.groupby("query_task").size().nunique() == 1
    checkpoint = _fake_training_metadata(state["current_checkpoint"])
    assert checkpoint["training_rows"] == len(training)


def test_resume_rejects_changed_source_shard(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    workflow.execute()
    source = tmp_path / "source.parquet"
    frame = pd.read_parquet(source)
    frame.loc[len(frame)] = {
        "sample_id": "changed",
        "embedding": [1.0, 0.0],
        "storage_type": "file",
        "path": str(tmp_path / "data/changed.jpg"),
        "member": None,
    }
    frame.to_parquet(source, index=False)

    with pytest.raises(RuntimeError, match="Input data or checkpoint changed"):
        workflow.execute()


@pytest.mark.parametrize("strategy", ["grit_score", "multi_task_round_robin"])
def test_every_round_trains_from_immutable_base_checkpoint(
    tmp_path: Path, strategy: str
) -> None:
    value = _config(tmp_path, strategy).to_dict()
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    training_manifest = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "sample_id": ["source-1"],
            "storage_type": ["file"],
            "path": ["/data/source-1.jpg"],
            "member": [None],
        }
    ).to_parquet(training_manifest, index=False)
    state = workflow._initialize()  # pylint: disable=protected-access

    workflow._train(  # pylint: disable=protected-access
        state, 1, tmp_path / "run/rounds/round_001", training_manifest
    )
    workflow._train(  # pylint: disable=protected-access
        state, 2, tmp_path / "run/rounds/round_002", training_manifest
    )

    expected_base = str(Path(value["model"]["base_checkpoint"]).absolute())
    round_one = _fake_training_metadata(
        tmp_path / "run/rounds/round_001/train/checkpoint.pth"
    )
    round_two = _fake_training_metadata(
        tmp_path / "run/rounds/round_002/train/checkpoint.pth"
    )
    assert round_one["parent"] == expected_base
    assert round_two["parent"] == expected_base
    assert round_two["training_rows"] == round_one["training_rows"]
    assert state["current_checkpoint"].endswith(
        "rounds/round_002/train/checkpoint.pth"
    )


def test_checkpoint_policy_defaults_to_base_checkpoint_each_round(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["training"].pop("checkpoint_policy")

    config = WorkflowConfig.from_dict(value)

    assert config.value["training"]["checkpoint_policy"] == (
        "base_checkpoint_each_round"
    )
    assert "warm_start" not in config.value["training"]


def test_resolved_training_batch_size_composes_interpolation(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "interpolated.yaml"
    spec.write_text(
        "train:\n  num_gpus: 3\n"
        "dataset:\n  batch_size: ${train.num_gpus}\n",
        encoding="utf-8",
    )
    assert resolved_training_batch_size(spec) == 3


@pytest.mark.parametrize("field", ["checkpoint", "contract"])
def test_native_train_output_paths_cannot_be_overridden(
    tmp_path: Path, field: str,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["train"][field] = "/tmp/not-native"
    with pytest.raises(ValueError, match="owns fixed checkpoint/contract paths"):
        WorkflowConfig.from_dict(value)



def test_training_allocation_graduates_without_dropping_update_floor() -> None:
    training = {
        "passes_per_round": 12,
        "node_scaling": {
            "allowed_nodes": [1, 2, 4, 8],
            "gpus_per_node": 8,
            "target_optimizer_updates": 2304,
        },
    }

    expected = {
        24_576: 1,
        49_152: 2,
        98_304: 4,
        196_608: 8,
    }
    for rows, nodes in expected.items():
        allocation = training_allocation(
            training,
            manifest_rows=rows,
            batch_size_per_gpu=16,
        )
        assert allocation["nodes"] == nodes
        assert allocation["total_optimizer_steps"] == 2304


def test_dynamic_training_allocation_is_recorded_and_passed_to_leaf(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["training"]["node_scaling"] = {
        "allowed_nodes": [1, 2, 4],
        "gpus_per_node": 2,
        "target_optimizer_updates": 16,
    }
    execution_environment = value["execution"]["environment"]
    value["execution"] = {
        "backend": "external",
        "runner_command": ["unused"],
        "environment": execution_environment,
        "capabilities": {
            "gang_scheduling": True,
            "gang_retry": True,
            "attempt_scoped_launch_id": True,
            "shared_filesystem": True,
        },
    }
    value["actions"]["train"]["command"].extend(
        [
            "--num-nodes",
            "{training_nodes}",
            "--gpus-per-node",
            "{training_gpus_per_node}",
        ]
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))

    class SimulatedGangRunner:
        @staticmethod
        def run(request: StageRequest) -> StageResult:
            environment = os.environ.copy()
            environment.update(request.environment)
            completed = subprocess.run(
                request.command,
                cwd=request.workdir,
                env=environment,
                check=False,
            )
            return StageResult(
                state="COMPLETE" if completed.returncode == 0 else "ERROR",
                client_job_id=request.client_job_id,
                backend_ref="simulated:1",
                return_code=completed.returncode,
                log_path=None,
                native_state="SUCCEEDED",
                attempt_id="simulated-attempt-1",
            )

    workflow.runner = SimulatedGangRunner()
    manifest = tmp_path / "dynamic-training.parquet"
    pd.DataFrame(
        {
            "sample_id": [f"source-{index}" for index in range(128)],
            "storage_type": ["file"] * 128,
            "path": [f"/data/source-{index}.jpg" for index in range(128)],
            "member": [None] * 128,
        }
    ).to_parquet(manifest, index=False)
    state = workflow._initialize()  # pylint: disable=protected-access

    workflow._train(  # pylint: disable=protected-access
        state, 1, tmp_path / "run/rounds/round_001", manifest
    )

    allocation = json.loads(
        (
            tmp_path
            / "run/rounds/round_001/train/training_allocation.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint = _fake_training_metadata(
        tmp_path / "run/rounds/round_001/train/checkpoint.pth"
    )
    assert allocation["nodes"] == 4
    assert allocation["gpus_per_node"] == 2
    assert allocation["total_optimizer_steps"] == 16
    assert checkpoint["nodes"] == 4
    assert checkpoint["gpus_per_node"] == 2


@pytest.mark.parametrize("warm_start", [False, True])
def test_legacy_warm_start_is_rejected(
    tmp_path: Path, warm_start: bool,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["training"]["warm_start"] = warm_start

    with pytest.raises(ValueError, match="every candidate"):
        WorkflowConfig.from_dict(value)


def test_previous_round_checkpoint_policy_is_rejected(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["training"]["checkpoint_policy"] = "previous_round_checkpoint"

    with pytest.raises(ValueError, match="every candidate"):
        WorkflowConfig.from_dict(value)


def test_audited_ann_search_is_staged_and_resumable(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    source = tmp_path / "source.parquet"
    contract_binding, contract_input = _payload_contract_lineage(
        Path(value["data"]["source_payload_contract"]), row_count=3
    )
    source_payload = {
        "root_uri": tmp_path.resolve().as_uri(),
        "shards": [
            {
                "relative_path": source.name,
                "bytes": source.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        "row_count": 3,
        "shard_count": 1,
        "embedding_dim": 2,
        "encoder": {"name": "test"},
        "inventory_digest": "sha256:source-inventory",
        "source_payload_contract": contract_binding,
    }
    source_store, source_artifact = _committed_payload(
        tmp_path / "source_store",
        "embedding_store.json",
        "embedding_store",
        source_payload,
        inputs=[contract_input],
    )
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps(
            {
                "encoder": {"name": "test"},
                "embedding_dim": 2,
                "source_store_manifest": {
                    "inventory_digest": source_payload["inventory_digest"]
                },
            }
        ),
        encoding="utf-8",
    )
    dense_payload = {
        "source_store_artifact_id": source_artifact["artifact_id"],
        "source_inventory_digest": source_payload["inventory_digest"],
        "vector_inventory_digest": "sha256:vector-inventory",
    }
    dense_manifest, dense_artifact = _committed_payload(
        tmp_path / "dense",
        "dense_store.json",
        "dense_vector_store",
        dense_payload,
    )
    ann_payload = {
        "dense_store_artifact_id": dense_artifact["artifact_id"],
        "source_inventory_digest": source_payload["inventory_digest"],
    }
    ann_manifest, ann_artifact = _committed_payload(
        tmp_path / "ann", "ann_index.json", "ann_index", ann_payload
    )
    audit_payload = {
        "passed": True,
        "index_artifact_id": ann_artifact["artifact_id"],
        "dense_store_artifact_id": dense_artifact["artifact_id"],
        "source_inventory_digest": source_payload["inventory_digest"],
        "vector_inventory_digest": dense_payload["vector_inventory_digest"],
        "n_probes": 4,
        "ann_candidate_count": 10,
    }
    audit_manifest, _ = _committed_payload(
        tmp_path / "audit",
        "ann_audit.json",
        "ann_recall_audit",
        audit_payload,
    )
    identity_manifest, _ = _committed_payload(
        tmp_path / "identity",
        "identity_audit.json",
        "source_identity_audit",
        {
            "source_store_artifact_id": source_artifact["artifact_id"],
            "source_inventory_digest": source_payload["inventory_digest"],
            "row_count": source_payload["row_count"],
            "hash_algorithm": "blake2b-128",
            "sorted_identity_sha256": "sha256:" + "b" * 64,
            "duplicate_hash_count": 0,
            "unique": True,
            "collision_policy": "fail_closed_on_blake2b128_collision",
            "implementation_contract_digest": "sha256:"
            + "a" * 64,
            "proof_digest": canonical_digest(
                {
                    "source_store_artifact_id": source_artifact["artifact_id"],
                    "source_inventory_digest": source_payload["inventory_digest"],
                    "row_count": source_payload["row_count"],
                    "hash_algorithm": "blake2b-128",
                    "sorted_identity_sha256": "sha256:" + "b" * 64,
                    "duplicate_hash_count": 0,
                }
            ),
        },
    )
    value["data"].pop("source_parts")
    value["data"].update(
        {
            "source_store_manifest": str(source_store),
            "target_embedding_contract": str(query_contract),
            "source_identity_audit": str(identity_manifest),
        }
    )
    search_artifacts = {
        "dense_store_manifest": str(dense_manifest),
        "ann_index_manifest": str(ann_manifest),
        "ann_audit_manifest": str(audit_manifest),
    }
    value["actions"]["search"] = {
        "backend": "audited_ann",
        **search_artifacts,
        "n_probes": 4,
        "ann_candidates": 10,
        "candidate": {
            "command": ["python", str(FAKE)],
            "resources": {"gpus": 1},
        },
        "rerank": {
            "command": ["python3", str(FAKE)],
            "resources": {"gpus": 1},
            "device": "cpu",
        },
    }
    fail_once = tmp_path / "rerank_failed_once"
    value["execution"]["environment"]["FAKE_RERANK_FAIL_ONCE"] = str(
        fail_once
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    plan = workflow.plan()
    assert "search_candidates" in plan["round_stages"]
    for stage in ("candidate", "rerank"):
        assert workflow.config.value["actions"]["search"][stage]["command"][0] == sys.executable
    assert plan["approval_contract"]["mining"]["ann"] == {
        "index_manifest": search_artifacts["ann_index_manifest"],
        "audit_manifest": search_artifacts["ann_audit_manifest"],
        "n_probes": 4,
        "candidate_count": 10,
        "decision_rule": "exact_float32_cosine_rerank",
        "underfill_stop": "search_budget_exhausted",
    }
    with pytest.raises(RuntimeError, match="search job"):
        workflow.execute()
    failed = workflow.status()
    assert failed["status"] == "failed"
    assert "round_001/search_candidates" in failed["completed_stages"]
    assert fail_once.is_file()
    candidate_path = (
        tmp_path
        / "run/rounds/round_001/search_candidates/ann_candidates.npz"
    )
    candidate_bytes = candidate_path.read_bytes()
    candidate_path.write_bytes(candidate_bytes[:-1] + bytes([candidate_bytes[-1] ^ 1]))
    with pytest.raises(ValueError, match="payload changed"):
        workflow.execute()
    candidate_path.write_bytes(candidate_bytes)
    state = workflow.execute()
    assert state["status"] == "complete"
    assert "round_001/search_candidates" in state["completed_stages"]
    assert "round_001/search" in state["completed_stages"]
    data_lock = json.loads(
        (tmp_path / "run" / "data.lock.json").read_text(encoding="utf-8")
    )
    assert data_lock["indexed_search"]["n_probes"] == 4
    exclusion = workflow._exclusion_manifest(state, 2)
    assert exclusion is not None
    exclusion_frame = pd.read_parquet(exclusion)
    assert exclusion_frame[
        [
            "sample_id",
            "source_row_id",
            "dense_store_artifact_id",
            "source_inventory_digest",
        ]
    ].to_dict(
        orient="records"
    ) == [
        {
            "sample_id": "ann-source",
            "source_row_id": 0,
            "dense_store_artifact_id": dense_artifact["artifact_id"],
            "source_inventory_digest": source_payload["inventory_digest"],
        }
    ]
    events_before = (tmp_path / "run" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    workflow.execute()
    assert (tmp_path / "run" / "events.jsonl").read_text(
        encoding="utf-8"
    ) == events_before


def test_completed_checkpoint_tamper_is_rejected(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    state = workflow.execute()
    Path(state["current_checkpoint"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint (size|digest)"):
        workflow.execute()


def test_custom_dense_search_repairs_stale_exclusion_lineage(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    parent = tmp_path / "parent.parquet"
    pd.DataFrame({"sample_id": pd.Series(dtype="str")}).to_parquet(
        parent, index=False
    )
    value["data"]["previous_training_manifest"] = str(parent)
    value["mining"]["parent_history_policy"] = "exclude"
    value["actions"]["search"] = {
        "backend": "custom",
        "command": [sys.executable, str(FAKE), "search"],
        "parameters": {"dense_store_manifest": str(tmp_path / "dense.json")},
    }
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    current = tmp_path / "current.parquet"
    pd.DataFrame(
        {
            "sample_id": ["source-1"],
            "source_row_id": [17],
            "dense_store_artifact_id": ["sha256:dense"],
            "source_inventory_digest": ["sha256:source"],
        }
    ).to_parquet(current, index=False)
    stale = tmp_path / "run/lineage/excluded_round_002.parquet"
    stale.parent.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["source-1"]}).to_parquet(stale, index=False)

    exclusion = workflow._exclusion_manifest(  # pylint: disable=protected-access
        {"current_training_manifest": str(current)}, 2
    )

    assert exclusion == stale
    assert pd.read_parquet(exclusion).to_dict(orient="records") == [
        {
            "sample_id": "source-1",
            "source_row_id": 17,
            "dense_store_artifact_id": "sha256:dense",
            "source_inventory_digest": "sha256:source",
        }
    ]


def test_parent_history_requires_parent_training_manifest(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["workflow"]["initialization"] = "parent_history"

    with pytest.raises(
        ValueError, match="requires data.previous_training_manifest"
    ):
        WorkflowConfig.from_dict(value)


def test_parent_history_is_explicit_in_plan(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    parent = tmp_path / "parent.parquet"
    pd.DataFrame({"sample_id": pd.Series(dtype="str")}).to_parquet(
        parent, index=False
    )
    value["workflow"]["initialization"] = "parent_history"
    value["data"]["previous_training_manifest"] = str(parent)

    config = WorkflowConfig.from_dict(value)

    assert config.plan()["initialization"] == "base_checkpoint_with_parent_data_history"


def test_parent_history_materializes_cumulative_manifest(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    parent = tmp_path / "parent.parquet"
    pd.DataFrame(
        {
            "sample_id": ["parent-source"],
            "storage_type": ["file"],
            "path": [str(tmp_path / "data/parent.jpg")],
            "member": [None],
        }
    ).to_parquet(parent, index=False)
    value["workflow"]["initialization"] = "parent_history"
    value["data"]["previous_training_manifest"] = str(parent)
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))

    state = workflow.execute()

    manifest = pd.read_parquet(state["current_training_manifest"])
    assert len(manifest) == 2
    assert "parent-source" in set(manifest["sample_id"].astype(str))
    artifact = json.loads(
        (
            tmp_path
            / "run/rounds/round_001/materialize/artifact.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["payload"]["previous_rows"] == 1
    assert artifact["payload"]["row_count"] == 2


def test_checkpoint_symlink_name_is_preserved_for_training(tmp_path: Path) -> None:
    config = _config(tmp_path, "grit_score").to_dict()
    blob = tmp_path / "checkpoint-blob"
    blob.write_text("base", encoding="utf-8")
    alias = tmp_path / "model.safetensors"
    alias.symlink_to(blob)
    config["model"]["base_checkpoint"] = str(alias)
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(config))
    state = workflow.execute()
    assert state["status"] == "complete"
    lock = json.loads(
        (tmp_path / "run/data.lock.json").read_text(encoding="utf-8")
    )
    assert lock["base_checkpoint"]["path"] == str(alias.absolute())
    assert lock["base_checkpoint"]["resolved_path"] == str(blob.resolve())
    checkpoint = _fake_training_metadata(state["current_checkpoint"])
    assert checkpoint["parent"] == str(alias.absolute())


def test_completed_native_training_is_finalized_without_relaunch_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    manifest = tmp_path / "training.parquet"
    pd.DataFrame({
        "sample_id": ["source-1"],
        "storage_type": ["file"],
        "path": ["/data/source-1.jpg"],
        "member": [None],
    }).to_parquet(manifest, index=False)
    state = workflow._initialize()  # pylint: disable=protected-access
    actual_finalize = workflow_controller.finalize_training
    attempts = 0

    def crash_once(output_dir):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated controller crash")
        return actual_finalize(output_dir)

    monkeypatch.setattr(workflow_controller, "finalize_training", crash_once)
    round_dir = tmp_path / "run/rounds/round_001"
    with pytest.raises(StageFailure, match="simulated controller crash"):
        workflow._train(state, 1, round_dir, manifest)  # pylint: disable=protected-access

    class MustNotRun:
        @staticmethod
        def run(_request):
            raise AssertionError("completed native training was relaunched")

    workflow.runner = MustNotRun()
    checkpoint = workflow._train(  # pylint: disable=protected-access
        state, 1, round_dir, manifest
    )
    assert checkpoint.is_file()
    completed = state["completed_stages"]["round_001/train"]
    assert completed["job"]["native_state"] == "FINALIZED_WITHOUT_RELAUNCH"
    terminal_marker = checkpoint.parent / "terminal_teacher.json"
    terminal_marker.write_text(terminal_marker.read_text() + "\n")
    with pytest.raises(RuntimeError, match="training commit is invalid"):
        actual_finalize(checkpoint.parent)


@pytest.mark.parametrize("balanced_training", [False, True])
@pytest.mark.parametrize("tamper_attestation", [False, True])
def test_sealed_training_continuation_is_adopted_then_evaluated(
    tmp_path: Path, balanced_training: bool, tamper_attestation: bool,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    previous_run = tmp_path / "previous-run"
    train_dir = previous_run / "rounds/round_002/train"
    materialize_dir = previous_run / "rounds/round_002/materialize"
    train_dir.mkdir(parents=True)
    materialize_dir.mkdir(parents=True)
    manifest = materialize_dir / "training_manifest.parquet"
    source = pd.read_parquet(tmp_path / "source.parquet")
    source.to_parquet(manifest, index=False)
    training_input_manifest = manifest
    if balanced_training:
        training_input_manifest = (
            materialize_dir / "balanced_training_manifest.parquet"
        )
        pd.concat([source, source], ignore_index=True).to_parquet(
            training_input_manifest, index=False
        )
    checkpoint = train_dir / "checkpoint.pth"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    runtime_spec = train_dir / "experiment.yaml"
    runtime_spec.write_text("runtime: accepted\n", encoding="utf-8")
    prepared_spec = train_dir / "refinement_input.yaml"
    prepared_spec.write_text("prepared: accepted\n", encoding="utf-8")
    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    previous_lock = RefinementWorkflow(
        WorkflowConfig.from_dict(value)
    )._data_lock()  # pylint: disable=protected-access
    adapter_files = previous_lock["adapter_files"]
    entrypoint_sha256 = "sha256:" + "1" * 64
    adapter_files["train.module.native"] = {
        "path": "/installed/native/train.py",
        "sha256": entrypoint_sha256,
    }
    adapter_files["train.implementation.native.test_leaf.py"] = {
        "path": "/installed/native/test_leaf.py",
        "sha256": "sha256:" + "2" * 64,
    }
    implementation_prefix = "train.implementation."
    implementation_values = {
        key.removeprefix(implementation_prefix): record["sha256"]
        for key, record in adapter_files.items()
        if key.startswith(implementation_prefix)
    }
    implementation_sha256 = canonical_digest(implementation_values)
    implementation_audit = train_dir / "train_implementation_audit.json"
    implementation_audit.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "subtask": "train",
                "entrypoint_sha256": entrypoint_sha256,
                "implementation_sha256": implementation_sha256,
            }
        ),
        encoding="utf-8",
    )

    allocation = training_allocation(
        value["training"],
        manifest_rows=len(pd.read_parquet(training_input_manifest)),
        batch_size_per_gpu=2,
        fixed_nodes=1,
        fixed_gpus_per_node=1,
    )
    contract_path = train_dir / "training_contract.json"
    contract = {
        "schema_version": "1.0",
        "base_spec": str(Path(value["training"]["base_spec"]).resolve()),
        "base_spec_sha256": digest(Path(value["training"]["base_spec"])),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest(checkpoint),
        "runtime_spec": str(runtime_spec.resolve()),
        "runtime_spec_bytes": runtime_spec.stat().st_size,
        "runtime_spec_sha256": digest(runtime_spec),
        "implementation_audit": str(implementation_audit.resolve()),
        "implementation_audit_bytes": implementation_audit.stat().st_size,
        "implementation_audit_sha256": digest(implementation_audit),
        "prepared_spec": str(prepared_spec.resolve()),
        "prepared_spec_sha256": digest(prepared_spec),
        "manifest": str(training_input_manifest.resolve()),
        "manifest_sha256": digest(training_input_manifest),
        "manifest_rows": len(pd.read_parquet(training_input_manifest)),
        "parent_checkpoint": str(Path(value["model"]["base_checkpoint"]).resolve()),
        "parent_checkpoint_sha256": digest(Path(value["model"]["base_checkpoint"])),
        "requested_data_passes": 2,
        "num_nodes": allocation["nodes"],
        "gpus_per_node": allocation["gpus_per_node"],
        "world_size": allocation["world_size"],
        "total_optimizer_steps": allocation["total_optimizer_steps"],
        "round_checkpoint_policy": "final_ema_teacher",
        "source_checkpoint_name": (
            "teacher_epoch_001_step_"
            f"{allocation['total_optimizer_steps']:05d}.pth"
        ),
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    commit_path = train_dir / "training_commit.json"
    commit_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "training_contract_sha256": digest(contract_path),
                "checkpoint_sha256": digest(checkpoint),
                "runtime_spec_sha256": digest(runtime_spec),
                "implementation_audit_sha256": digest(implementation_audit),
            }
        ),
        encoding="utf-8",
    )
    success = train_dir / "_SUCCESS"
    success.write_text(digest(commit_path) + "\n", encoding="utf-8")
    previous_data_lock = previous_run / "data.lock.json"
    previous_release_lock = previous_run / "release.lock.json"
    previous_data_lock.write_text(json.dumps(previous_lock), encoding="utf-8")
    previous_release_lock.write_text('{"old": "release"}', encoding="utf-8")
    persistent_targets = {
        "weak-sample": {
            "consecutive_rounds": 2,
            "last_round": 2,
            "tasks": ["classification"],
        }
    }
    (previous_run / "state.json").write_text(
        json.dumps(
            {
                "completed_stages": {"round_002/train": {}},
                "persistent_targets": persistent_targets,
            }
        ),
        encoding="utf-8",
    )

    value["workflow"].update({"start_round": 2, "max_rounds": 2})
    value["model"]["initial_scoring_checkpoint"] = str(checkpoint)
    value["data"]["previous_training_manifest"] = str(manifest)
    value["continuation"] = {
        "previous_run_dir": str(previous_run),
        "adopted_round": 2,
        "training_contract": str(contract_path),
        "training_commit": str(commit_path),
        "success_marker": str(success),
        "previous_data_lock": str(previous_data_lock),
        "previous_release_lock": str(previous_release_lock),
    }
    value["output"]["run_dir"] = str(tmp_path / "continuation-run")
    value["execution"]["workdir"] = str(tmp_path / "continuation-run")
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))

    if tamper_attestation:
        audit = json.loads(implementation_audit.read_text(encoding="utf-8"))
        audit["implementation_sha256"] = "sha256:" + "0" * 64
        implementation_audit.write_text(json.dumps(audit), encoding="utf-8")
        contract["implementation_audit_bytes"] = implementation_audit.stat().st_size
        contract["implementation_audit_sha256"] = digest(implementation_audit)
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["training_contract_sha256"] = digest(contract_path)
        commit["implementation_audit_sha256"] = digest(implementation_audit)
        commit_path.write_text(json.dumps(commit), encoding="utf-8")
        success.write_text(digest(commit_path) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="implementation differs"):
            workflow.adopt_training()
        return

    adopted = workflow.adopt_training()
    assert adopted["current_checkpoint"] == str(checkpoint.resolve())
    assert adopted["active_jobs"] == {}
    assert adopted["persistent_targets"] == persistent_targets
    assert "round_002/train" in adopted["completed_stages"]
    assert "round_002/evaluate" not in adopted["completed_stages"]
    assert "round_002" not in adopted["completed_rounds"]

    completed = workflow.execute()
    assert "round_002/evaluate" in completed["completed_stages"]
    assert "round_002" in completed["completed_rounds"]
    assert not any(
        key.startswith("round_002/")
        and key not in {"round_002/train", "round_002/evaluate"}
        for key in completed["completed_stages"]
    )


def test_round_one_continuation_contract_is_allowed(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["model"]["initial_scoring_checkpoint"] = str(tmp_path / "checkpoint")
    value["data"]["previous_training_manifest"] = str(tmp_path / "manifest")
    value["continuation"] = {
        "previous_run_dir": str(tmp_path / "previous-run"),
        "adopted_round": 1,
        "training_contract": str(tmp_path / "training-contract.json"),
        "training_commit": str(tmp_path / "training-commit.json"),
        "success_marker": str(tmp_path / "_SUCCESS"),
        "previous_data_lock": str(tmp_path / "data.lock.json"),
        "previous_release_lock": str(tmp_path / "release.lock.json"),
    }

    config = WorkflowConfig.from_dict(value)

    assert config.value["continuation"]["adopted_round"] == 1


def test_continuation_accepts_only_payload_bound_parent_store(
    tmp_path: Path,
) -> None:
    shard = {
        "relative_path": "part.parquet",
        "bytes": 10,
        "rows": 2,
        "sha256": "sha256:" + "a" * 64,
    }
    parent_payload = {
        "inventory_digest": canonical_digest([shard]),
        "row_count": 2,
        "shard_count": 1,
        "encoder": {"name": "test-encoder"},
        "shards": [shard],
    }
    parent_manifest, parent_artifact = _committed_payload(
        tmp_path / "parent-store",
        "embedding_store.json",
        "embedding_store",
        parent_payload,
    )
    parent_bytes = parent_manifest.read_bytes()
    parent_identity = {
        "uri": parent_manifest.resolve().as_uri(),
        "bytes": len(parent_bytes),
        "sha256": "sha256:" + hashlib.sha256(parent_bytes).hexdigest(),
    }
    bound_payload = {
        **parent_payload,
        "parent_store_artifact_id": parent_artifact["artifact_id"],
    }
    bound_manifest, bound_artifact = _committed_payload(
        tmp_path / "bound-store",
        "embedding_store.json",
        "embedding_store",
        bound_payload,
        inputs=[
            {
                **parent_identity,
                "role": "source_embedding_store",
                "artifact_id": parent_artifact["artifact_id"],
            }
        ],
    )
    bound_bytes = bound_manifest.read_bytes()
    bound_identity = {
        "uri": bound_manifest.resolve().as_uri(),
        "bytes": len(bound_bytes),
        "sha256": "sha256:" + hashlib.sha256(bound_bytes).hexdigest(),
    }
    previous_lock = {
        "source_store": {
            "manifest": parent_identity,
            "artifact_id": parent_artifact["artifact_id"],
            "declared_inventory_digest": parent_payload["inventory_digest"],
            "row_count": 2,
            "shard_count": 1,
            "encoder": parent_payload["encoder"],
        }
    }
    current_lock = {
        "source_store": {
            "manifest": bound_identity,
            "artifact_id": bound_artifact["artifact_id"],
            "declared_inventory_digest": bound_payload["inventory_digest"],
            "row_count": 2,
            "shard_count": 1,
            "encoder": bound_payload["encoder"],
        }
    }

    transition = _validate_continuation_source_lineage(
        previous_lock=previous_lock,
        current_lock=current_lock,
        source_store_manifest=str(bound_manifest),
    )

    assert transition == {
        "mode": "bound_store",
        "parent_artifact_id": parent_artifact["artifact_id"],
        "bound_artifact_id": bound_artifact["artifact_id"],
    }
    legacy_previous_lock = {
        "source_store": {
            "manifest": str(parent_manifest.resolve()),
            "manifest_sha256": parent_identity["sha256"],
            "artifact_id": parent_artifact["artifact_id"],
            "inventory_digest": parent_payload["inventory_digest"],
            "row_count": 2,
            "shard_count": 1,
            "encoder": parent_payload["encoder"],
        }
    }
    assert _validate_continuation_source_lineage(
        previous_lock=legacy_previous_lock,
        current_lock=current_lock,
        source_store_manifest=str(bound_manifest),
    ) == transition
    unrelated_payload = {
        **bound_payload,
        "encoder": {"name": "unrelated-encoder"},
    }
    unrelated_manifest, unrelated_artifact = _committed_payload(
        tmp_path / "unrelated-store",
        "embedding_store.json",
        "embedding_store",
        unrelated_payload,
        inputs=bound_artifact["inputs"],
    )
    unrelated_bytes = unrelated_manifest.read_bytes()
    unrelated_lock = {
        "source_store": {
            "manifest": {
                "uri": unrelated_manifest.resolve().as_uri(),
                "bytes": len(unrelated_bytes),
                "sha256": "sha256:"
                + hashlib.sha256(unrelated_bytes).hexdigest(),
            },
            "artifact_id": unrelated_artifact["artifact_id"],
            "declared_inventory_digest": unrelated_payload["inventory_digest"],
            "row_count": 2,
            "shard_count": 1,
            "encoder": unrelated_payload["encoder"],
        }
    }
    with pytest.raises(ValueError, match="bound_inventory"):
        _validate_continuation_source_lineage(
            previous_lock=previous_lock,
            current_lock=unrelated_lock,
            source_store_manifest=str(unrelated_manifest),
        )


def test_continuation_accepts_exact_unchanged_registered_store(
    tmp_path: Path,
) -> None:
    payload = {
        "inventory_digest": "sha256:" + "a" * 64,
        "row_count": 2,
        "shard_count": 1,
        "encoder": {"name": "test-encoder"},
        "shards": [],
    }
    manifest, artifact = _committed_payload(
        tmp_path / "store",
        "embedding_store.json",
        "embedding_store",
        payload,
    )
    manifest_bytes = manifest.read_bytes()
    source_store = {
        "manifest": {
            "uri": manifest.resolve().as_uri(),
            "bytes": len(manifest_bytes),
            "sha256": "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "artifact_id": artifact["artifact_id"],
        "declared_inventory_digest": payload["inventory_digest"],
        "row_count": 2,
        "shard_count": 1,
        "encoder": payload["encoder"],
    }

    transition = _validate_continuation_source_lineage(
        previous_lock={"source_store": source_store},
        current_lock={"source_store": source_store},
        source_store_manifest=str(manifest),
    )

    assert transition == {
        "mode": "unchanged_store",
        "artifact_id": artifact["artifact_id"],
    }


def test_persistent_target_suppression_is_terminal(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "multi_task_round_robin"))
    state = {"persistent_targets": {}}

    def update(round_index: int, sample_ids: list[str]) -> None:
        selection = tmp_path / f"selection-{round_index}.parquet"
        pd.DataFrame({"sample_id": sample_ids}).to_parquet(selection, index=False)
        workflow._update_persistence(state, selection, round_index)

    update(1, ["persistent", "transient"])
    update(2, ["persistent"])
    update(3, ["persistent"])
    update(4, ["continuing"])

    record = state["persistent_targets"]["persistent"]
    assert record["suppressed"] is True
    assert record["suppressed_round"] == 3
    assert record["streak"] == 3
    payload = json.loads(workflow._suppressed_path(state).read_text(encoding="utf-8"))
    assert payload["sample_ids"] == ["persistent"]


def test_dense_exact_search_proof_controls_validation_and_stop_reason() -> None:
    proof = "exact_all_dense_rows_float32"
    assert _is_supported_search_proof(proof)
    assert not _is_supported_search_proof("exact_some_dense_rows_float32")
    assert _empty_search_stop_reason(
        {
            "eligible_source_rows": 73_428_403,
            "search_proof": proof,
            "underfill_exhaustion_proven": True,
        }
    ) == "radius_exhausted"
    assert _empty_search_stop_reason(
        {
            "eligible_source_rows": 73_428_403,
            "search_proof": proof,
            "underfill_exhaustion_proven": False,
        }
    ) == "search_budget_exhausted"
    assert _empty_search_stop_reason(
        {
            "eligible_source_rows": 0,
            "search_proof": proof,
            "underfill_exhaustion_proven": True,
        }
    ) == "pool_exhausted"


def test_plan_has_no_output_side_effects(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    assert not (tmp_path / "run").exists()


def test_weighted_multitask_plan_preserves_total_and_explains_data(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["multi_task"] = {
        "tasks": ["classification", "segmentation"],
        "targets_per_round": 12,
        "task_weights": {"segmentation": 2.0},
    }
    plan = WorkflowConfig.from_dict(value).plan()
    approval = plan["approval_contract"]
    assert approval["selection"] == {
        "method": "within_task_normalized_weakness_round_robin",
        "targets_per_round": 12,
        "task_weights": {"segmentation": 2.0},
        "task_quotas": {"classification": 4, "segmentation": 8},
        "unfilled_task_budget": "redistributed",
        "training_replay": "uniform over the cumulative unique manifest",
    }
    assert "cumulative Parquet" in approval["data"]["training_manifest"]
    assert approval["cache_and_artifacts"]["image_payload_copies"] == (
        "none by the controller"
    )
    assert approval["loop"]["max_rounds"] == 1
    assert approval["monitoring"].startswith("Remain attached")


def test_balanced_multitask_plan_preserves_quotas_and_balances_replay(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["multi_task"] = {
        "policy": "balanced",
        "tasks": ["classification", "segmentation"],
        "targets_per_round": 12,
    }
    selection = WorkflowConfig.from_dict(value).plan()["approval_contract"][
        "selection"
    ]
    assert selection["method"] == (
        "balanced_within_task_normalized_weakness_round_robin"
    )
    assert selection["task_quotas"] == {"classification": 6, "segmentation": 6}
    assert selection["unfilled_task_budget"] == "preserved"
    assert selection["training_replay"].startswith("oversample each task")


def test_multitask_weight_validation_is_strict(tmp_path: Path) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["multi_task"]["task_weights"] = {"unknown": 2.0}
    with pytest.raises(ValueError, match="unknown tasks"):
        WorkflowConfig.from_dict(value)

    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["multi_task"]["task_weights"] = {"segmentation": 0.0}
    with pytest.raises(ValueError, match="finite and positive"):
        WorkflowConfig.from_dict(value)


def test_data_lock_accepts_actions_without_parameter_blocks(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    lock = workflow._data_lock()
    assert any(name.startswith("score.command") for name in lock["adapter_files"])
    assert any(name.startswith("data.module") for name in lock["adapter_files"])
    assert not any(name.startswith("data.parameters") for name in lock["adapter_files"])
    assert workflow.plan()["strategy"] == "grit_score"
    assert not (tmp_path / "run").exists()


def test_release_lock_covers_builtin_ds_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def record(path: Path) -> str:
        observed.append(path.resolve())
        return "sha256:" + "0" * 64

    monkeypatch.setattr(workflow_controller, "_sha256", record)
    assert workflow_controller._workflow_source_digest().startswith("sha256:")
    assert (DATA_SERVICES / "nvidia_tao_ds/mining/dinov3/materialize.py").resolve() in observed
    assert (WORKFLOW_ROOT / "controller.py").resolve() in observed



def test_native_lock_covers_dinov3_and_inherited_nvdinov2_runtime(
    tmp_path: Path,
) -> None:
    try:
        from nvidia_tao_pytorch.ssl.dinov3.utils.refinement_attestation import (
            native_attestation,
        )
    except ModuleNotFoundError as error:
        expected = (
            "nvidia_tao_pytorch.ssl.dinov3.utils.refinement_attestation"
        )
        if error.name != expected:
            raise
        pytest.skip("requires the stacked TAO PyTorch DINOv3 DEFT runtime")

    import nvidia_tao_pytorch

    pytorch = Path(nvidia_tao_pytorch.__file__).resolve().parents[1]
    assert pytorch.is_dir()
    value = _config(tmp_path, "grit_score").to_dict()
    value["execution"]["environment"]["PYTHONPATH"] = os.pathsep.join(
        [str(DATA_SERVICES), str(pytorch)]
    )
    value["actions"]["score"].update(
        command=["dinov3", "grit_score", "-e", "{score_config}"],
        implementation_files=[],
        settings={
            "device": "cpu",
            "neighbor_backend": "torch_exact",
            "neighbor_device": "cpu",
        },
    )
    value["actions"]["train"].update(
        command=["dinov3", "train", "-e", "{training_spec}"],
        implementation_files=[],
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    lock = workflow._data_lock()
    keys = set(lock["adapter_files"])
    assert any(key.endswith("ssl/dinov3/model/pl_model.py") for key in keys)
    assert any(key.endswith("ssl/nvdinov2/model/pl_model.py") for key in keys)
    assert any(key.endswith("ssl/nvdinov2/dataloader/transform.py") for key in keys)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "data.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    for action in ("score", "train"):
        subtask = "grit_score" if action == "score" else action
        expected = native_attestation(subtask)
        assert workflow._locked_action_entrypoint(action) == expected[
            "entrypoint_sha256"
        ]
        assert workflow._locked_action_implementation(action) == expected[
            "implementation_sha256"
        ]


def test_config_change_requires_fork(tmp_path: Path) -> None:
    config = _config(tmp_path, "grit_score")
    workflow = RefinementWorkflow(config)
    workflow.execute()
    changed = config.to_dict()
    changed["mining"]["top_k_per_target"] = 2
    with pytest.raises(RuntimeError, match="changed|match"):
        RefinementWorkflow(WorkflowConfig.from_dict(changed)).execute()


def test_score_requires_complete_target_identity_coverage(tmp_path: Path) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["execution"]["environment"]["FAKE_DROP_SCORE_ROW"] = "1"
    with pytest.raises(StageFailure, match="exactly cover target"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()


def test_score_rejects_changed_target_embedding(tmp_path: Path) -> None:
    value = _config(tmp_path, "multi_task_round_robin").to_dict()
    value["execution"]["environment"]["FAKE_CHANGE_SCORE_EMBEDDING"] = "1"
    with pytest.raises(StageFailure, match="immutable target embeddings"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()


def test_score_rejects_unapproved_implementation(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["execution"]["environment"]["FAKE_IMPLEMENTATION_SHA256"] = (
        "sha256:" + "a" * 64
    )
    with pytest.raises(StageFailure, match="does not bind the current request"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()


def test_custom_actions_require_implementation_closures(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["score"]["implementation_files"] = []
    with pytest.raises(ValueError, match="Custom actions.score"):
        WorkflowConfig.from_dict(value)

    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["train"]["implementation_files"] = []
    with pytest.raises(ValueError, match="Custom actions.train"):
        WorkflowConfig.from_dict(value)

    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["evaluate"]["implementation_files"] = []
    with pytest.raises(ValueError, match="Custom actions.evaluate"):
        WorkflowConfig.from_dict(value)


def test_cuda_grit_requires_declared_gpu_faiss_capability(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["score"].update(
        {
            "command": ["dinov3", "grit_score", "-e", "{score_config}"],
            "implementation_files": [],
            "settings": {
                "device": "cuda",
                "neighbor_backend": "faiss_exact",
                "neighbor_device": "cuda",
            },
        }
    )
    with pytest.raises(ValueError, match="capabilities.gpu_faiss"):
        WorkflowConfig.from_dict(value)

    value["execution"]["capabilities"] = {"gpu_faiss": True}
    assert WorkflowConfig.from_dict(value).value["execution"]["capabilities"] == {
        "gpu_faiss": True
    }


@pytest.mark.parametrize("profile", ["local", "external", "containerized"])
def test_generated_stage_request_matches_published_contract_schema(
    tmp_path: Path, profile: str,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["evaluate"]["command"] = []
    if profile != "local":
        value["execution"].update(
            backend="external", runner_command=["unused-platform-runner"],
        )
        value["execution"].setdefault("capabilities", {}).update(
            shared_filesystem=True
        )
    if profile == "containerized":
        value["execution"].update(
            container_images={
                "pytorch": "registry/pyt:reviewed",
                "data_services": "registry/ds:reviewed",
            },
        )
        value["execution"]["capabilities"]["containers"] = True
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    assert workflow.config.value["training"]["checkpoint_policy"] == "base_checkpoint_each_round"

    class CapturingRunner:
        request: StageRequest | None = None

        def run(self, request: StageRequest) -> StageResult:
            self.request = request
            return StageResult(
                state="FAILED",
                client_job_id=request.client_job_id,
                backend_ref="test",
                return_code=9,
                log_path=None,
                native_state="FAILED",
            )

        @staticmethod
        def cancel(client_job_id: str) -> dict:
            return {"client_job_id": client_job_id}

        @staticmethod
        def logs(client_job_id: str, cursor: str | None = None) -> dict:
            return {"client_job_id": client_job_id, "cursor": cursor}

    runner = CapturingRunner()
    workflow.runner = runner
    with pytest.raises(StageFailure, match="score job .* ended in FAILED"):
        workflow.execute()
    assert runner.request is not None
    contract = runner.request.execution_contract
    if profile == "containerized":
        assert contract["container_image"] == "registry/pyt:reviewed"
        assert "containers" in contract["required_capabilities"]
    schema = json.loads(
        (WORKFLOW_ROOT / "schemas/stage-request.schema.yaml").read_text(
            encoding="utf-8"
        )
    )["properties"]["execution_contract"]
    assert set(schema["required"]).issubset(contract)
    assert set(contract).issubset(schema["properties"])
    expected_capabilities = {
        "local": [],
        "external": ["shared_filesystem"],
        "containerized": ["containers", "shared_filesystem"],
    }
    assert contract["required_capabilities"] == expected_capabilities[profile]


def test_local_runner_probes_required_gpu_faiss_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CpuOnlyFaiss:
        @staticmethod
        def IndexFlatIP(dimension: int) -> object:
            return {"dimension": dimension}

    monkeypatch.setitem(sys.modules, "faiss", CpuOnlyFaiss())
    request = StageRequest(
        client_job_id="gpu-faiss-preflight",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "raise SystemExit(99)"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={"required_capabilities": ["gpu_faiss"]},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    with pytest.raises(RuntimeError, match="cannot satisfy required gpu_faiss"):
        runner.run(request)
    assert not (tmp_path / "jobs/gpu-faiss-preflight.json").exists()

def test_score_rejects_target_changed_after_request_snapshot(
    tmp_path: Path,
) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    delegate = workflow.runner
    target_path = Path(workflow.config.value["data"]["target_manifest"])

    class MutatingRunner:
        mutated = False

        def run(self, request: StageRequest) -> StageResult:
            if request.stage == "score" and not self.mutated:
                frame = pd.read_parquet(target_path)
                embeddings = frame["embedding"].tolist()
                embeddings[0] = [0.75, 0.25]
                frame["embedding"] = embeddings
                frame.to_parquet(target_path, index=False)
                self.mutated = True
            return delegate.run(request)

        def cancel(self, client_job_id: str) -> dict:
            return delegate.cancel(client_job_id)

        def logs(self, client_job_id: str, cursor: str | None = None) -> dict:
            return delegate.logs(client_job_id, cursor)

    workflow.runner = MutatingRunner()
    with pytest.raises(StageFailure, match="does not bind the current request"):
        workflow.execute()


def test_failed_resume_revalidates_completed_training_before_scoring(
    tmp_path: Path,
) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    delegate = workflow.runner

    class FailingEvaluationRunner:
        def run(self, request: StageRequest) -> StageResult:
            if request.stage == "evaluate" and request.round_index == 1:
                return StageResult(
                    state="FAILED",
                    client_job_id=request.client_job_id,
                    backend_ref="test",
                    return_code=9,
                    log_path=None,
                    native_state="FAILED",
                )
            return delegate.run(request)

        def cancel(self, client_job_id: str) -> dict:
            return delegate.cancel(client_job_id)

        def logs(self, client_job_id: str, cursor: str | None = None) -> dict:
            return delegate.logs(client_job_id, cursor)

    workflow.runner = FailingEvaluationRunner()
    with pytest.raises(StageFailure, match="evaluate job .* ended in FAILED"):
        workflow.execute()

    state = workflow.status()
    checkpoint = Path(
        state["completed_stages"]["round_001/train"]["outputs"]["checkpoint"]
    )
    checkpoint.write_text("tampered", encoding="utf-8")
    workflow.runner = delegate
    with pytest.raises(ValueError, match="checkpoint (size|digest)"):
        workflow.execute()


@pytest.mark.parametrize("failed_stage", ["train", "evaluate"])
def test_retry_preserves_completed_round_inputs(
    tmp_path: Path, failed_stage: str,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    delegate = workflow.runner
    calls = []

    class RetryRunner:
        fail = True

        def run(self, request: StageRequest) -> StageResult:
            calls.append((request.round_index, request.stage))
            if self.fail and request.stage == failed_stage and request.round_index == 1:
                return StageResult(
                    state="ERROR", client_job_id=request.client_job_id,
                    backend_ref="test", return_code=9, log_path=None,
                    native_state="ERROR",
                )
            return delegate.run(request)

    runner = RetryRunner()
    workflow.runner = runner
    with pytest.raises(StageFailure, match=f"{failed_stage} job .* ended in ERROR"):
        workflow.execute()
    state = workflow.status()
    original_inputs = state["round_inputs"]["1"]
    assert original_inputs["current_checkpoint"] == str(Path(value["model"]["base_checkpoint"]).absolute())
    assert original_inputs["current_training_manifest"] is None
    request_path = tmp_path / "run/rounds/round_001/score/score_request.json"
    original_request = request_path.read_bytes()
    original_mtime = request_path.stat().st_mtime_ns
    runner.fail = False
    calls.clear()
    assert workflow.execute()["status"] == "complete"
    expected = [(1, "train"), (1, "evaluate")] if failed_stage == "train" else [(1, "evaluate")]
    assert calls == expected
    assert request_path.read_bytes() == original_request
    assert request_path.stat().st_mtime_ns == original_mtime
    assert workflow.status()["round_inputs"]["1"] == original_inputs


def test_cancel_does_not_write_state_owned_by_running_controller(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    workflow._initialize()
    before = workflow.store.state_path.read_bytes()
    with workflow.store.controller_lock():
        assert workflow.cancel()["status"] == "canceling"
        assert workflow.store.state_path.read_bytes() == before
    assert workflow.execute()["status"] == "canceled"


def test_target_embeddings_are_required_before_approval(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    target = Path(value["data"]["target_manifest"])
    frame = pd.read_parquet(target).drop(columns=["embedding"])
    frame.to_parquet(target, index=False)
    with pytest.raises(ValueError, match="missing columns.*embedding"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).validate()


def test_target_identities_must_be_unique_before_approval(tmp_path: Path) -> None:
    config = _config(tmp_path, "grit_score")
    value = config.to_dict()
    target = Path(value["data"]["target_manifest"])
    frame = pd.read_parquet(target)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(
        target, index=False
    )
    with pytest.raises(ValueError, match="globally unique"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).validate()


def test_module_resolution_checks_every_python_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    module = second / "example" / "adapter.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")

    assert _module_source("example.adapter", [first, second]) == module.resolve()


def test_file_identity_rejects_same_descriptor_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"content")
    real_fstat = os.fstat
    calls = 0

    def changing_fstat(file_descriptor: int):
        nonlocal calls
        calls += 1
        result = real_fstat(file_descriptor)
        if calls == 2:
            values = {
                name: getattr(result, name)
                for name in (
                    "st_dev",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            }
            values["st_size"] += 1
            return type("ChangedStat", (), values)()
        return result

    monkeypatch.setattr(os, "fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed while it was read"):
        _file_identity(path)


def test_evaluation_rejects_nonfinite_base_metric(tmp_path: Path) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["execution"]["environment"]["FAKE_NONFINITE_METRIC"] = "1"
    with pytest.raises(StageFailure, match="value is not numeric"):
        RefinementWorkflow(WorkflowConfig.from_dict(value)).execute()


def test_trace_contains_stage_jobs(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "multi_task_round_robin"))
    workflow.execute()
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed = [event for event in events if event["status"] == "complete"]
    assert {event["stage"] for event in completed}.issuperset(
        {
            "score", "select_targets", "search", "materialize", "train",
            "evaluate", "round_complete", "loop_stop"
        }
    )


def test_published_json_schemas_parse() -> None:
    for name in (
        "workflow.schema.json",
        "stage-request.schema.json",
        "metrics.schema.json",
    ):
        value = json.loads((WORKFLOW_ROOT / "schemas" / name.replace(".json", ".yaml")).read_text(encoding="utf-8"))
        assert value["$schema"].endswith("2020-12/schema")


def test_local_runner_rejects_multinode_training_configuration(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["training"]["node_scaling"] = {
        "allowed_nodes": [1, 2],
        "gpus_per_node": 8,
        "target_optimizer_updates": 100,
    }

    with pytest.raises(ValueError, match="backend=external"):
        WorkflowConfig.from_dict(value)


def test_local_runner_rejects_multinode_nontraining_action(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["score"]["resources"] = {"nodes": 2, "gpus_per_node": 8}

    with pytest.raises(ValueError, match="score"):
        WorkflowConfig.from_dict(value)

    value["actions"]["score"].update(
        {
            "execution_mode": "adapter_managed",
            "wrapper_resources": {"nodes": 1, "cpus": 1},
        }
    )
    resolved = WorkflowConfig.from_dict(value)
    assert resolved.to_dict()["actions"]["score"]["execution_mode"] == (
        "adapter_managed"
    )


def test_local_runner_checks_visible_gpu_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    request = StageRequest(
        client_job_id="local-gpu-check",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "pass"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1, "gpus_per_node": 2},
        execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    with pytest.raises(RuntimeError, match="requested 2, visible 1"):
        runner.run(request)


def test_local_runner_uses_pinned_gpu_and_populates_lightning_rendezvous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("must not initialize CUDA in the controller"))
    observed = tmp_path / "node-environment.json"
    names = [
        "WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK",
        "MASTER_ADDR", "MASTER_PORT",
    ]
    probe = (
        "import json,os,sys; "
        "json.dump({name: os.environ.get(name) for name in "
        f"{names!r}}}, open(sys.argv[1], \"w\", encoding=\"utf-8\"))"
    )
    request = StageRequest(
        client_job_id="local-gpu-runtime",
        run_id="run",
        round_index=1,
        stage="train",
        command=[sys.executable, "-c", probe, str(observed)],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={
            "WORLD_SIZE": "2",
            "NUM_GPU_PER_NODE": "8",
            "NODE_RANK": "1",
            "MASTER_ADDR": "unreachable.invalid",
            "MASTER_PORT": "1",
        },
        resources={"nodes": 1, "gpus": 1},
        execution_contract={
            "node_environment": {
                "contract": "lightning_node_entrypoint_v1"
            }
        },
    )
    result = build_runner({"backend": "local"}, tmp_path).run(request)
    assert result.state == "COMPLETE"
    environment = json.loads(observed.read_text(encoding="utf-8"))
    assert environment == {
        "WORLD_SIZE": "1",
        "NUM_GPU_PER_NODE": "1",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": environment["MASTER_PORT"],
    }
    assert 1024 <= int(environment["MASTER_PORT"]) <= 65535


def test_local_runner_requires_declared_scratch_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TAO_TEST_SCRATCH", raising=False)
    request = StageRequest(
        client_job_id="local-scratch-unmapped",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "pass"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={
            "nodes": 1,
            "local_scratch": {"path_environment": "TAO_TEST_SCRATCH"},
        },
        execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    with pytest.raises(RuntimeError, match="no mapped local scratch"):
        runner.run(request)


def test_local_runner_honors_cancel_before_process_start(tmp_path: Path) -> None:
    request = StageRequest(
        client_job_id="local-canceled",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "raise RuntimeError('must not run')"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={},
    )
    (tmp_path / "cancel.requested").write_text("{}", encoding="utf-8")
    runner = build_runner({"backend": "local"}, tmp_path)
    assert runner.run(request).native_state == "CANCELED_BEFORE_START"


def test_local_job_cancel_does_not_create_global_run_intent(
    tmp_path: Path,
) -> None:
    request = StageRequest(
        client_job_id="local-complete",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "pass"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    assert runner.run(request).state == "COMPLETE"

    assert runner.cancel(request.client_job_id)["state"] == "COMPLETE"
    assert not (tmp_path / "cancel.requested").exists()


def test_local_runner_process_identity_rejects_pid_reuse(tmp_path: Path) -> None:
    runner = build_runner({"backend": "local"}, tmp_path)
    start_time = runner._process_start_time(os.getpid())
    assert start_time is not None
    assert runner._alive(os.getpid(), start_time)
    assert not runner._alive(os.getpid(), start_time + 1)


def test_local_runner_observes_cancel_intent_while_process_runs(
    tmp_path: Path,
) -> None:
    request = StageRequest(
        client_job_id="local-intent-cancel",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    results = []
    worker = threading.Thread(target=lambda: results.append(runner.run(request)))
    worker.start()
    record = tmp_path / "jobs/local-intent-cancel.json"
    for _ in range(100):
        if record.is_file():
            break
        time.sleep(0.01)
    assert record.is_file()
    (tmp_path / "cancel.requested").touch()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert results[0].state == "CANCELED"


def test_local_runner_cancels_owned_process_group(tmp_path: Path) -> None:
    request = StageRequest(
        client_job_id="local-running-cancel",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    results = []
    worker = threading.Thread(target=lambda: results.append(runner.run(request)))
    worker.start()
    record = tmp_path / "jobs/local-running-cancel.json"
    for _ in range(100):
        if record.is_file():
            break
        time.sleep(0.01)
    assert record.is_file()
    assert runner.cancel(request.client_job_id)["state"] == "CANCELED"
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert results[0].state == "CANCELED"


@pytest.mark.parametrize("grandchild", [False, True])
def test_local_cancel_waits_for_sigterm_resistant_descendant(
    tmp_path: Path, grandchild: bool,
) -> None:
    pid_path = tmp_path / "resistant.pid"
    child_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    command = [sys.executable, "-c", child_code]
    if grandchild:
        command = [
            sys.executable, "-c",
            f"import subprocess; subprocess.Popen({command!r})",
        ]
    request = StageRequest(
        client_job_id="resistant-cancel", run_id="run", round_index=1,
        stage="train", command=command, workdir=str(tmp_path),
        results_dir=str(tmp_path), environment={}, resources={}, execution_contract={},
    )
    runner = build_runner({"backend": "local"}, tmp_path)
    results = []
    worker = threading.Thread(target=lambda: results.append(runner.run(request)))
    worker.start()
    child_pid = None
    child_start = None
    try:
        deadline = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_path.exists()
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        child_start = runner._process_start_time(child_pid)
        assert child_start is not None
        assert runner.cancel(request.client_job_id)["state"] == "CANCELED"
        assert not runner._alive(child_pid, child_start)
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert results[0].state == "CANCELED"
    finally:
        if child_pid is not None and runner._alive(child_pid, child_start):
            os.kill(child_pid, signal.SIGKILL)
        runner.cancel(request.client_job_id)
        worker.join(timeout=10)


def test_local_runner_adopts_terminal_record_after_controller_crash(
    tmp_path: Path,
) -> None:
    request = StageRequest(
        client_job_id="local-crash-adoption",
        run_id="run",
        round_index=1,
        stage="score",
        command=[sys.executable, "-c", "import time; time.sleep(1)"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={"nodes": 1},
        execution_contract={},
    )
    code = (
        "import json,sys; "
        "from pathlib import Path; "
        "from nvidia_tao_ds.mining.dinov3.workflow.execution import LocalRunner,StageRequest; "
        "LocalRunner(Path(sys.argv[1])).run(StageRequest(**json.loads(sys.argv[2])))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPTS), environment.get("PYTHONPATH", "")]
    )
    controller = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "jobs"),
            json.dumps(request.__dict__),
        ],
        env=environment,
        stderr=subprocess.PIPE,
        text=True,
    )
    record = tmp_path / "jobs/local-crash-adoption.json"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if record.is_file() and json.loads(
            record.read_text(encoding="utf-8")
        ).get("state") == "RUNNING":
            break
        time.sleep(0.05)
    if not record.is_file():
        controller.kill()
        _, stderr = controller.communicate(timeout=5)
        pytest.fail(f"Controller did not publish its RUNNING record: {stderr}")
    controller.kill()
    controller.communicate(timeout=5)
    for _ in range(300):
        if json.loads(record.read_text(encoding="utf-8")).get("state") == "COMPLETE":
            break
        time.sleep(0.01)
    runner = build_runner({"backend": "local"}, tmp_path)
    assert runner.run(request).state == "COMPLETE"


def test_adapter_managed_uses_one_wrapper_and_records_delegated_allocation(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    Path(value["training"]["base_spec"]).write_text(
        "model: {}\n"
        "dataset:\n  batch_size: 2\n"
        "train:\n  num_nodes: 4\n  num_gpus: 8\n"
        "  gpu_ids: [7, 6, 5, 4, 3, 2, 1, 0]\n",
        encoding="utf-8",
    )
    value["training"]["node_scaling"] = {
        "allowed_nodes": [1, 2],
        "gpus_per_node": 1,
        "target_optimizer_updates": 4,
    }
    value["actions"]["train"].update(
        {
            "execution_mode": "adapter_managed",
            "wrapper_resources": {"nodes": 1, "cpus": 1},
        }
    )
    value["actions"]["train"]["command"].extend(
        [
            "--num-nodes",
            "{training_nodes}",
            "--gpus-per-node",
            "{training_gpus_per_node}",
        ]
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    state = workflow._initialize()  # pylint: disable=protected-access
    manifest = tmp_path / "native-leaf-training.parquet"
    pd.DataFrame({
        "sample_id": [f"s{index}" for index in range(16)],
        "storage_type": ["file"] * 16,
        "path": [f"/data/s{index}.png" for index in range(16)],
    }).to_parquet(manifest, index=False)
    observed = {}

    class AdapterManagedRunner:
        @staticmethod
        def run(request: StageRequest) -> StageResult:
            observed["request"] = request
            environment = os.environ.copy()
            environment.update(request.environment)
            if request.environment.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
                environment["PYTHONPATH"] = os.pathsep.join(
                    [request.environment["PYTHONPATH"], os.environ["PYTHONPATH"]]
                )
            completed = subprocess.run(
                request.command,
                cwd=request.workdir,
                env=environment,
                check=False,
            )
            return StageResult(
                state="COMPLETE" if completed.returncode == 0 else "ERROR",
                client_job_id=request.client_job_id,
                backend_ref="native-test",
                return_code=completed.returncode,
                log_path=None,
                native_state="COMPLETE",
            )

    workflow.runner = AdapterManagedRunner()
    workflow._train(  # pylint: disable=protected-access
        state, 1, tmp_path / "run/rounds/round_001", manifest
    )

    request = observed["request"]
    assert request.resources["nodes"] == 1
    assert request.execution_contract["attempt_scope"] == "process"
    assert request.execution_contract["adapter_managed_resources"] == {
        "nodes": 2,
        "gpus_per_node": 1,
        "world_size": 2,
    }
    assert request.execution_contract["node_environment"] == {
        "contract": "lightning_node_entrypoint_v1",
        "world_size_semantics": "nodes",
        "required_for_multinode": [
            "WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK",
            "MASTER_ADDR", "MASTER_PORT",
        ],
        "forbidden_at_entrypoint": [
            "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_*",
        ],
    }
    allocation = json.loads(
        (tmp_path / "run/rounds/round_001/train/training_allocation.json").read_text()
    )
    assert allocation["nodes"] == 2

    prepared = yaml.safe_load(
        (tmp_path / "run/rounds/round_001/train/refinement_input.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert prepared["train"]["num_nodes"] == 2
    assert prepared["train"]["num_gpus"] == 1
    assert prepared["train"]["gpu_ids"] == [0]


@pytest.mark.parametrize("execution_mode", ["runner", "adapter_managed"])
def test_training_preserves_non_gpu_resource_contract(
    tmp_path: Path, execution_mode: str,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    configured = {
        "nodes": 1, "gpus_per_node": 1, "world_size": 999,
        "cpus": 12, "memory": "32Gi", "time_limit": "01:00:00",
        "local_scratch": {"path_environment": "TAO_LOCAL_SCRATCH"},
    }
    value["actions"]["train"].update(
        resources=configured, execution_mode=execution_mode,
        wrapper_resources={"nodes": 1, "cpus": 1},
    )
    value["training"]["node_scaling"] = {
        "allowed_nodes": [1, 2], "gpus_per_node": 1,
        "target_optimizer_updates": 1,
    }
    value["execution"].update(
        backend="external", runner_command=[sys.executable, str(FAKE_RUNNER)],
        capabilities={name: True for name in (
            "gang_scheduling", "gang_retry", "attempt_scoped_launch_id", "shared_filesystem",
        )},
    )
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    state = workflow._initialize()
    manifest = tmp_path / "resource-training.parquet"
    pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(8)],
        "storage_type": ["file"] * 8,
        "path": [f"/data/s{i}.png" for i in range(8)],
    }).to_parquet(manifest, index=False)
    observed = []

    class CaptureRunner:
        def run(self, request: StageRequest) -> StageResult:
            observed.append(request)
            return StageResult(
                state="ERROR", client_job_id=request.client_job_id,
                backend_ref="captured-only", return_code=1,
                log_path=None, native_state="ERROR",
            )

    workflow.runner = CaptureRunner()
    with pytest.raises(StageFailure, match="train job .* ended in ERROR"):
        workflow._train(state, 1, tmp_path / "run/rounds/round_001", manifest)
    request = observed[0]
    expected = {**configured, "nodes": 2, "gpus_per_node": 1, "world_size": 2}
    if execution_mode == "runner":
        assert request.resources == expected
        assert request.execution_contract["adapter_managed_resources"] is None
    else:
        assert request.resources == {"nodes": 1, "cpus": 1}
        assert request.execution_contract["adapter_managed_resources"] == expected


def test_unsupported_train_backend_is_rejected(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["train"].pop("execution_mode")
    value["actions"]["train"]["backend"] = "native_leaf"

    with pytest.raises(ValueError, match="execution_mode"):
        WorkflowConfig.from_dict(value)


def test_multinode_external_runner_requires_gang_capabilities(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"]["train"]["resources"] = {
        "nodes": 2,
        "gpus_per_node": 8,
    }
    value["execution"] = {
        "backend": "external",
        "runner_command": ["runner"],
        "capabilities": {"shared_filesystem": True},
    }

    with pytest.raises(ValueError, match="missing required capabilities"):
        WorkflowConfig.from_dict(value)


def test_external_four_verb_runner_and_unknown_guard(tmp_path: Path) -> None:
    request = StageRequest(
        client_job_id="stable-id",
        run_id="run",
        round_index=1,
        stage="train",
        command=["ignored"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        environment={},
        resources={"gpus": 8},
        execution_contract={
            "membership": "static",
            "attempt_scope": "process",
            "retry_scope": "process",
            "attempt_id_scope": "backend_attempt",
            "launch_id_environment": None,
        },
    )
    runner = build_runner(
        {
            "backend": "external",
            "runner_command": [sys.executable, str(FAKE_RUNNER), "complete"],
            "poll_seconds": 0,
        },
        tmp_path,
    )
    assert runner.run(request).state == "COMPLETE"
    assert runner.logs("stable-id")["cursor"] == "12"
    assert runner.cancel("stable-id")["state"] == "CANCELED"

    unknown = build_runner(
        {
            "backend": "external",
            "runner_command": [sys.executable, str(FAKE_RUNNER), "unknown"],
            "poll_seconds": 0,
        },
        tmp_path / "unknown",
    )
    with pytest.raises(RuntimeError, match="not resubmitting"):
        unknown.run(request)


@pytest.mark.parametrize("before_submit", [True, False])
def test_external_runner_observes_cancel_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, before_submit: bool,
) -> None:
    runner = build_runner(
        {"backend": "external", "runner_command": ["runner"], "poll_seconds": 0},
        tmp_path,
    )
    intent = tmp_path / "cancel.requested"
    verbs = []

    def call(verb: str, *arguments: str) -> dict:
        verbs.append(verb)
        client_id = arguments[-1] if "--client-job-id" in arguments else None
        if verb == "status" and "cancel" not in verbs:
            intent.touch()
            return {"state": "RUNNING", "client_job_id": client_id}
        return {"state": "CANCELED", "client_job_id": client_id}

    monkeypatch.setattr(runner, "_call", call)
    if before_submit:
        intent.touch()
    request = StageRequest(
        client_job_id="cancel-id", run_id="run", round_index=1,
        stage="train", command=["ignored"], workdir=str(tmp_path),
        results_dir=str(tmp_path / "results"), environment={}, resources={},
        execution_contract={"attempt_scope": "process"},
    )
    assert runner.run(request).state == "CANCELED"
    assert verbs == ([] if before_submit else ["submit", "status", "cancel", "status"])


def test_sigterm_handoff_creates_workflow_cancellation_intent(
    tmp_path: Path,
) -> None:
    previous = signal.getsignal(signal.SIGTERM)

    class Workflow:
        run_dir = tmp_path

        @staticmethod
        def execute() -> dict:
            os.kill(os.getpid(), signal.SIGTERM)
            assert (tmp_path / "cancel.requested").is_file()
            return {"status": "canceled"}

    assert workflow_cli._execute_with_termination_handoff(Workflow()) == {
        "status": "canceled"
    }
    assert signal.getsignal(signal.SIGTERM) is previous



def test_external_runner_verbs_have_bounded_timeout(tmp_path: Path) -> None:
    request = StageRequest(
        client_job_id="timeout",
        run_id="run",
        round_index=1,
        stage="score",
        command=["ignored"],
        workdir=str(tmp_path),
        results_dir=str(tmp_path),
        environment={},
        resources={},
        execution_contract={"attempt_scope": "process"},
    )
    runner = build_runner(
        {
            "backend": "external",
            "runner_command": [
                sys.executable,
                "-c",
                "import time; time.sleep(10)",
            ],
            "poll_seconds": 0.01,
            "call_timeout_seconds": 0.05,
        },
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="submit timed out"):
        runner.run(request)


def test_cancellation_waits_for_terminal_backend_acknowledgement(
    tmp_path: Path,
) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    state = workflow._initialize()  # pylint: disable=protected-access
    state["active_jobs"] = {"round_001/train": {"client_job_id": "job-1"}}
    workflow.store.save(state)

    class CancelRunner:
        terminal = False

        def cancel(self, client_job_id: str) -> dict:
            return {
                "state": "CANCELED" if self.terminal else "PENDING",
                "client_job_id": client_job_id,
            }

    runner = CancelRunner()
    workflow.runner = runner
    first = workflow.cancel()
    assert first["status"] == "canceling"
    assert workflow.status()["active_jobs"] == {
        "round_001/train": {"client_job_id": "job-1"}
    }

    runner.terminal = True
    second = workflow.cancel()
    assert second["status"] == "canceled"
    assert workflow.status()["active_jobs"] == {}
    assert workflow.execute()["status"] == "canceled"


def test_local_cancel_terminalizes_missing_prestart_record(
    tmp_path: Path,
) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    state = workflow._initialize()  # pylint: disable=protected-access
    state["status"] = "running"
    state["active_jobs"] = {
        "round_001/train": {"client_job_id": "missing-prestart"}
    }
    workflow.store.save(state)

    result = workflow.cancel()

    assert result["status"] == "canceled"
    assert result["jobs"]["round_001/train"]["native_state"] == (
        "CANCELED_BEFORE_START"
    )
    assert workflow.status()["active_jobs"] == {}
    record = json.loads(
        (tmp_path / "run/jobs/missing-prestart.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["state"] == "CANCELED"
    assert record["native_state"] == "CANCELED_BEFORE_START"
    assert workflow.execute()["status"] == "canceled"


def test_local_cancel_reconciles_canceling_record_after_process_exit(
    tmp_path: Path,
) -> None:
    runner = build_runner({"backend": "local"}, tmp_path)
    record = tmp_path / "jobs/dead-canceling.json"
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "state": "CANCELING",
                "client_job_id": "dead-canceling",
                "backend_ref": "pid:999999999:start:1",
                "return_code": None,
                "log_path": None,
                "native_state": "CANCELING",
                "attempt": 1,
                "attempt_id": "local-dead",
            }
        ),
        encoding="utf-8",
    )

    assert runner.cancel("dead-canceling")["state"] == "CANCELED"
    assert json.loads(record.read_text(encoding="utf-8"))["state"] == "CANCELED"


def test_cancel_before_state_initialization_persists_intent(tmp_path: Path) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    assert workflow.cancel() == {"status": "canceling", "jobs": {}}
    assert (workflow.run_dir / "cancel.requested").is_file()
    assert workflow.execute()["status"] == "canceled"


def test_client_job_identity_is_scoped_to_durable_run_directory() -> None:
    common = {
        "run_id": "user-supplied",
        "round_index": 1,
        "stage": "train",
        "command": ["constant-command"],
    }
    first = client_job_id(run_scope="/runs/first", **common)
    second = client_job_id(run_scope="/runs/second", **common)
    assert first != second
    assert first == client_job_id(run_scope="/runs/first", **common)


@pytest.mark.parametrize(
    ("action", "command"),
    [("score", ["dinov3", "grit_score"]), ("train", ["dinov3", "train"])],
)
def test_builtin_native_actions_reject_extra_implementation_files(
    tmp_path: Path, action: str, command: list[str],
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["actions"][action]["command"] = command
    value["actions"][action]["implementation_files"] = [str(FAKE)]
    value["actions"]["score"]["settings"]["neighbor_backend"] = "torch_exact"
    with pytest.raises(ValueError, match="owns its implementation closure"):
        WorkflowConfig.from_dict(value)


def test_external_backend_requires_shared_filesystem_without_images(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path, "grit_score").to_dict()
    value["execution"].update(
        backend="external", runner_command=["runner"], capabilities={}
    )
    with pytest.raises(ValueError, match="shared_filesystem=true"):
        WorkflowConfig.from_dict(value)


def test_external_runner_rejects_invalid_state_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = StageRequest(
        client_job_id="identity-test", run_id="run", round_index=1,
        stage="score", command=["ignored"], workdir=str(tmp_path),
        results_dir=str(tmp_path), environment={}, resources={},
        execution_contract={"attempt_scope": "process"},
    )
    runner = build_runner(
        {"backend": "external", "runner_command": ["runner"], "poll_seconds": 0},
        tmp_path,
    )
    monkeypatch.setattr(
        runner, "_call",
        lambda verb, *_args: ({} if verb == "submit" else {
            "client_job_id": request.client_job_id, "state": "FAILED",
        }),
    )
    with pytest.raises(RuntimeError, match="invalid state"):
        runner.run(request)
    monkeypatch.setattr(
        runner, "_call",
        lambda *_args: {"client_job_id": "other", "state": "CANCELED"},
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.cancel(request.client_job_id)

    monkeypatch.setattr(
        runner, "_call",
        lambda *_args: {
            "client_job_id": request.client_job_id,
            "state": "CANCEL_REQUEST_ACCEPTED",
        },
    )
    with pytest.raises(RuntimeError, match="invalid cancel state"):
        runner.cancel(request.client_job_id)


def test_controller_retains_job_after_mismatched_cancel_acknowledgement(
    tmp_path: Path,
) -> None:
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    state = workflow._initialize()  # pylint: disable=protected-access
    state["status"] = "running"
    state["active_jobs"] = {
        "round_001/train": {"client_job_id": "expected-job"}
    }
    workflow.store.save(state)

    class WrongRunner:
        @staticmethod
        def cancel(_client_job_id: str) -> dict:
            return {"state": "CANCELED", "client_job_id": "other-job"}

    workflow.runner = WrongRunner()
    result = workflow.cancel()
    assert result["status"] == "canceling"
    assert "round_001/train" in workflow.status()["active_jobs"]


@pytest.mark.parametrize("recipe,strategy", [("grit_score", "grit_score"),
                                            ("multi_task_round_robin", "multi_task_round_robin")])
def test_shipped_recipe_plan_and_all_command_placeholders(tmp_path, recipe, strategy):
    """Exercise both distributed recipes, not just their YAML syntax."""
    value = yaml.safe_load((WORKFLOW_ROOT / "recipes" / f"{recipe}.yaml").read_text())
    fixtures = _config(tmp_path, strategy).to_dict()
    for section in ("model", "data", "output"):
        value[section] = fixtures[section]
    value["training"]["base_spec"] = fixtures["training"]["base_spec"]
    if strategy == "multi_task_round_robin":
        value["multi_task"] = fixtures["multi_task"]
        value["actions"]["score"]["command"][0] = str(FAKE)
        value["actions"]["score"]["implementation_files"] = [str(FAKE)]
    workflow = RefinementWorkflow(WorkflowConfig.from_dict(value))
    assert workflow.plan()["strategy"] == strategy
    variables = {name: str(tmp_path / name) for name in (
        "checkpoint", "target_manifest", "output_dir", "score_config", "training_spec",
        "head_config", "training_manifest", "benchmark_manifest", "round",
    )}
    for action in workflow.config.value["actions"].values():
        command = workflow._command(action["command"], variables)
        assert not any("{" in token for token in command)
    assert workflow.config.value["actions"]["data"]["command"][0] == sys.executable


def test_controller_patch_drift_is_recorded_but_compatibility_drift_fails(tmp_path, monkeypatch):
    """A harmless controller patch must not strand a durable run."""
    workflow = RefinementWorkflow(_config(tmp_path, "grit_score"))
    workflow._initialize()
    monkeypatch.setattr(workflow_controller, "_workflow_source_digest", lambda: "patched-controller")
    with pytest.warns(RuntimeWarning, match="provenance changed"):
        workflow._initialize()
    events = [json.loads(line) for line in workflow.store.events_path.read_text().splitlines()]
    assert events[-1]["status"] == "provenance_drift"
    release = workflow.run_dir / "release.lock.json"
    changed = json.loads(release.read_text())
    changed["workflow_version"] = "incompatible"
    release.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="compatibility version"):
        workflow._initialize()
