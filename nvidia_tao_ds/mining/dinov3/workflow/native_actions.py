# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and seal native TAO DINOv3 actions for DEFT rounds."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import warnings
from typing import Any

from omegaconf import OmegaConf
import pyarrow.parquet as pq

from ..contracts import canonical_digest, write_json_atomic
from .snapshots import file_sha256


TRAINING_ALGORITHM_VERSION = "dinov3_deft_native_train_v1"
GRIT_ALGORITHM_VERSION = "dinov3_deft_native_grit_v1"


def _configured_absolute(path: str | Path) -> Path:
    """Make a configured path absolute without dereferencing its filename alias."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else (Path.cwd() / value).absolute()


def _experiment_spec(base_spec: str | Path):
    """Compose a user spec over the installed TAO DINOv3 structured defaults."""
    try:
        # Keep non-training DS actions usable without importing the TAO runtime.
        from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - production image preflight
        raise RuntimeError(
            "The Data Services container must include the TAO DINOv3 runtime"
        ) from error
    base_path = Path(base_spec).expanduser().resolve()
    return base_path, OmegaConf.merge(
        OmegaConf.structured(ExperimentConfig()), OmegaConf.load(base_path)
    )


def resolved_training_batch_size(base_spec: str | Path) -> int:
    """Read the effective per-GPU batch size from the composed TAO spec."""
    _, spec = _experiment_spec(base_spec)
    try:
        value = int(spec.dataset.batch_size)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "The resolved DINOv3 dataset.batch_size must be a positive integer"
        ) from error
    if value <= 0:
        raise ValueError(
            "The resolved DINOv3 dataset.batch_size must be a positive integer"
        )
    return value


def _write_yaml_atomic(path: Path, config) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = OmegaConf.to_yaml(config, resolve=True)
    if path.is_file() and path.read_text(encoding="utf-8") == serialized:
        return path
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_text_atomic(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def build_grit_spec(
    *,
    base_spec: str | Path,
    input_parquet: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    settings: dict[str, Any],
) -> Path:
    """Write a complete native ``dinov3 grit_score`` experiment spec."""
    base_path, spec = _experiment_spec(base_spec)
    output = Path(output_dir).resolve()
    spec.results_dir = str(output)
    spec.grit_score.results_dir = str(output)
    spec.grit_score.input_parquet = str(Path(input_parquet).resolve())
    checkpoint_path = _configured_absolute(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint does not exist: {checkpoint_path}")
    spec.grit_score.checkpoint = str(checkpoint_path)
    spec.grit_score.base_spec = str(base_path)
    valid = set(spec.grit_score.keys())
    unknown = set(settings).difference(valid)
    if unknown:
        raise ValueError(f"Unknown native DINOv3 GRIT settings: {sorted(unknown)}")
    for name, value in settings.items():
        spec.grit_score[name] = value
    destination = output / "grit_score.yaml"
    return _write_yaml_atomic(destination, spec)


def build_training_spec(
    *,
    base_spec: str | Path,
    manifest: str | Path,
    parent_checkpoint: str | Path,
    passes: int,
    output_dir: str | Path,
    num_nodes: int,
    gpus_per_node: int,
    checkpoint_policy: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write one native DINOv3 train spec while preserving scheduler policy."""
    if passes <= 0 or num_nodes <= 0 or gpus_per_node <= 0:
        raise ValueError("passes, num_nodes, and gpus_per_node must be positive")
    base_path, spec = _experiment_spec(base_spec)
    manifest_path = Path(manifest).resolve()
    parent_path = _configured_absolute(parent_checkpoint)
    output = Path(output_dir).resolve()
    if not manifest_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("DINOv3 training inputs must be regular files")
    parquet = pq.ParquetFile(manifest_path)
    required = {"sample_id", "storage_type", "path"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Training manifest is missing columns: {sorted(missing)}")
    rows = int(parquet.metadata.num_rows)
    if rows <= 0:
        raise ValueError("Training manifest must contain at least one row")
    if bool(spec.model.distill.enable):
        raise ValueError("DEFT training requires model.distill.enable=false")

    spec.results_dir = str(output)
    spec.dataset.train_manifest = str(manifest_path)
    spec.train.num_epochs = int(passes)
    spec.train.num_nodes = int(num_nodes)
    spec.train.num_gpus = int(gpus_per_node)
    spec.train.gpu_ids = list(range(gpus_per_node))
    spec.train.pretrained_model_path = str(parent_path)
    spec.train.resume_training_checkpoint_path = None
    spec.train.auto_resume = False

    batch_size = resolved_training_batch_size(base_path)
    world_size = num_nodes * gpus_per_node
    steps_per_pass = math.ceil(math.ceil(rows / world_size) / batch_size)
    total_steps = steps_per_pass * passes
    spec_path = _write_yaml_atomic(output / "refinement_input.yaml", spec)
    contract = {
        "schema_version": "1.0",
        "algorithm_version": TRAINING_ALGORITHM_VERSION,
        "checkpoint_policy": checkpoint_policy,
        "base_spec": str(base_path),
        "base_spec_sha256": file_sha256(base_path),
        "prepared_spec": str(spec_path),
        "prepared_spec_sha256": file_sha256(spec_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_rows": rows,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": file_sha256(parent_path),
        "requested_data_passes": passes,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "batch_size_per_gpu": batch_size,
        "steps_per_pass": steps_per_pass,
        "total_optimizer_steps": total_steps,
        "scheduler_policy": "preserve_resolved_base_spec",
        "checkpoint_interval": int(spec.train.checkpoint_interval),
        "checkpoint_interval_unit": str(spec.train.checkpoint_interval_unit),
        "round_checkpoint_policy": "final_ema_teacher",
        "runtime_spec": str(output / "experiment.yaml"),
    }
    contract["request_digest"] = canonical_digest(contract)
    contract_path = output / "training_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        changed = {
            name: {"expected": value, "actual": existing.get(name)}
            for name, value in contract.items()
            if existing.get(name) != value
        }
        if changed:
            raise RuntimeError(
                "Prepared DINOv3 training request conflicts with existing contract: "
                f"{json.dumps(changed, sort_keys=True)}"
            )
        return spec_path, contract_path, existing
    write_json_atomic(contract_path, contract)
    return spec_path, contract_path, contract


def _atomic_publish_checkpoint(source: Path, destination: Path) -> str:
    """Publish by hard link when possible, otherwise by an atomic metadata copy."""
    if destination.exists():
        if destination.is_file() and file_sha256(destination) == file_sha256(source):
            return "existing_identical"
        raise RuntimeError(f"Refusing to replace conflicting checkpoint: {destination}")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    temporary.unlink()
    method = "hardlink"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            method = "copy2"
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return method


def _validate_teacher_checkpoint(path: Path) -> None:
    """Safely verify that a terminal teacher export is a nonempty tensor state."""
    try:
        import torch  # pylint: disable=import-outside-toplevel

        value = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as error:
        raise RuntimeError(
            f"Terminal DINOv3 teacher checkpoint is not safely loadable: {path}"
        ) from error
    if not isinstance(value, dict) or not value or any(
        not isinstance(name, str) or not torch.is_tensor(tensor)
        for name, tensor in value.items()
    ):
        raise RuntimeError(
            "Terminal DINOv3 teacher checkpoint must be a nonempty tensor state dict"
        )


def finalize_training(output_dir: str | Path) -> Path:
    """Safely validate and seal the exact terminal native teacher checkpoint."""
    output = Path(output_dir).resolve()
    contract_path = output / "training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint = output / "checkpoint.pth"
    audit_path = output / "train_implementation_audit.json"
    commit_path = output / "training_commit.json"
    marker = output / "_SUCCESS"
    if not audit_path.is_file():
        raise RuntimeError("Native DINOv3 train produced no implementation audit")
    if all(path.is_file() for path in (checkpoint, commit_path, marker)):
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        expected = {
            "terminal_teacher_manifest_sha256": file_sha256(output / "terminal_teacher.json"),
            "training_contract_sha256": file_sha256(contract_path),
            "checkpoint_sha256": file_sha256(checkpoint),
            "runtime_spec_sha256": file_sha256(contract["runtime_spec"]),
            "implementation_audit_sha256": file_sha256(audit_path),
        }
        if any(commit.get(name) != value for name, value in expected.items()):
            raise RuntimeError("Existing native DINOv3 training commit is invalid")
        if marker.read_text(encoding="utf-8").strip() != file_sha256(commit_path):
            raise RuntimeError("Existing native DINOv3 success marker is invalid")
        _validate_teacher_checkpoint(checkpoint)
        return checkpoint
    if file_sha256(contract["prepared_spec"]) != contract["prepared_spec_sha256"]:
        raise RuntimeError("Prepared DINOv3 training spec changed during execution")
    expected_step = int(contract["total_optimizer_steps"])
    terminal = json.loads((output / "terminal_teacher.json").read_text(encoding="utf-8"))
    if terminal.get("schema_version") != "1.0" or terminal.get("model") != "teacher":
        raise ValueError("Unsupported terminal teacher manifest")
    filename = terminal.get("filename", "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Terminal teacher filename must be a direct child of the training output")
    source = output / filename
    if (not source.is_file() or source.is_symlink() or
            source.stat().st_size != terminal.get("bytes") or
            file_sha256(source) != terminal.get("sha256")):
        raise ValueError("Terminal teacher checkpoint does not match its manifest")
    actual_step = terminal.get("global_step")
    if isinstance(actual_step, bool) or not isinstance(actual_step, int) or actual_step <= 0:
        raise ValueError("Terminal teacher global_step must be a positive integer")
    if actual_step != expected_step:
        warnings.warn(f"Terminal teacher completed {actual_step} steps; planning estimate was {expected_step}",
                      RuntimeWarning, stacklevel=2)
    _validate_teacher_checkpoint(source)
    runtime_spec = Path(contract["runtime_spec"])
    if not runtime_spec.is_file():
        raise RuntimeError("Native DINOv3 train produced no runtime experiment spec")
    publication_method = _atomic_publish_checkpoint(source, checkpoint)
    contract.update({
        "terminal_teacher_manifest_sha256": file_sha256(output / "terminal_teacher.json"),
        "actual_optimizer_steps": actual_step,
        "actual_terminal_epoch": terminal.get("epoch"),
        "source_checkpoint_name": source.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_publication_method": publication_method,
        "runtime_spec_sha256": file_sha256(runtime_spec),
        "runtime_spec_bytes": runtime_spec.stat().st_size,
        "implementation_audit": str(audit_path),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_audit_bytes": audit_path.stat().st_size,
    })
    contract["commit_record"] = str(commit_path)
    write_json_atomic(contract_path, contract)
    commit = {
        "terminal_teacher_manifest_sha256": contract["terminal_teacher_manifest_sha256"],
        "schema_version": "1.0",
        "algorithm_version": TRAINING_ALGORITHM_VERSION,
        "training_contract_sha256": file_sha256(contract_path),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "runtime_spec_sha256": contract["runtime_spec_sha256"],
        "implementation_audit_sha256": contract[
            "implementation_audit_sha256"
        ],
    }
    write_json_atomic(commit_path, commit)
    _write_text_atomic(marker, file_sha256(commit_path) + "\n")
    return checkpoint
