# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic DINOv3 SSL DEFT state machine."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import html
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .config import (
    WorkflowConfig,
    canonical_digest,
    multitask_budgets,
    training_allocation,
)
from .execution import StageRequest, build_runner, client_job_id
from .containers import stage_image
from .state import ControllerBusy, StateStore


class _FormatValues(dict):
    def __missing__(self, key: str) -> str:
        raise ValueError(f"Action command references unknown placeholder {{{key}}}")


class StageFailure(RuntimeError):
    """A leaf action failed or violated its declared output contract."""

    def __init__(self, stage: str, message: str):
        """Bind the workflow configuration or durable runtime paths."""
        super().__init__(message)
        self.stage = stage


class RunCanceled(RuntimeError):
    """Cancellation intent was observed after a platform action returned."""


EXACT_SEARCH_PROOFS = frozenset(
    {
        "exact_all_declared_shards",
        "exact_all_dense_rows_float32",
    }
)


def _is_supported_search_proof(proof: str) -> bool:
    return proof in EXACT_SEARCH_PROOFS or proof.startswith("ann_audited_")


def _empty_search_stop_reason(search: dict[str, Any]) -> str:
    if int(search.get("eligible_source_rows", -1)) == 0:
        return "pool_exhausted"
    if (
        search.get("search_proof") in EXACT_SEARCH_PROOFS and
        search.get("underfill_exhaustion_proven") is True
    ):
        return "radius_exhausted"
    return "search_budget_exhausted"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, name)) for name in _STAT_FIELDS)


def _stable_file_snapshot(
    path: Path, *, capture_bytes: bool = False
) -> tuple[Path, os.stat_result, bytes | None, str]:
    """Read and hash one immutable file-descriptor snapshot."""
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    chunks = [] if capture_bytes else None
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"Input changed while it was read: {resolved}")
    if _stat_identity(resolved.stat()) != _stat_identity(after):
        raise ValueError(f"Input path changed while it was read: {resolved}")
    raw = b"".join(chunks) if chunks is not None else None
    if raw is not None and len(raw) != after.st_size:
        raise ValueError(f"Input size changed while it was read: {resolved}")
    return resolved, after, raw, "sha256:" + digest.hexdigest()


def _sha256(path: Path) -> str:
    return _stable_file_snapshot(path)[3]


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    hexadecimal = value.removeprefix("sha256:")
    return len(hexadecimal) == 64 and all(
        character in "0123456789abcdef" for character in hexadecimal
    )


def _file_identity(path: Path) -> dict[str, Any]:
    resolved, stat, _, digest = _stable_file_snapshot(path)
    return {
        "uri": resolved.as_uri(),
        "bytes": stat.st_size,
        "sha256": digest,
    }


def _json_snapshot(path: Path) -> tuple[Any, dict[str, Any]]:
    """Read and identify one stable JSON file descriptor snapshot."""
    resolved, stat, raw, digest = _stable_file_snapshot(
        path, capture_bytes=True
    )
    assert raw is not None
    return json.loads(raw), {
        "uri": resolved.as_uri(),
        "bytes": stat.st_size,
        "sha256": digest,
    }


def _source_payload_contract_snapshot(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable dataset versions behind source locators."""
    contract, identity = _json_snapshot(path)
    if not isinstance(contract, dict):
        raise ValueError("Source payload contract must be a JSON object")
    if contract.get("schema_version") != "1.0":
        raise ValueError("Source payload contract schema_version must be 1.0")
    if contract.get("immutability") != "immutable":
        raise ValueError("Source payload contract must declare immutable payloads")
    datasets = contract.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Source payload contract requires at least one dataset")
    identities: set[tuple[str, str, str]] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"Source payload dataset {index} must be an object")
        fields = {
            name: dataset.get(name)
            for name in ("dataset_id", "version", "root_uri")
        }
        if not all(isinstance(item, str) and item.strip() for item in fields.values()):
            raise ValueError(
                "Each source payload dataset requires nonempty dataset_id, "
                "version, and root_uri strings"
            )
        parsed = urlparse(fields["root_uri"])
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("Source payload roots must be local file:// URIs")
        dataset_identity = (
            fields["dataset_id"],
            fields["version"],
            fields["root_uri"],
        )
        if dataset_identity in identities:
            raise ValueError("Source payload contract contains a duplicate dataset")
        identities.add(dataset_identity)
    return contract, identity


def _source_payload_contract(path: Path) -> dict[str, Any]:
    contract, _ = _source_payload_contract_snapshot(path)
    return contract


def _audit_source_locators(
    parts: list[Path], payload_contract: dict[str, Any]
) -> dict[str, Any]:
    """Bind legacy embedding Parquets to contracted payload roots."""
    roots = [
        Path(unquote(urlparse(item["root_uri"]).path)).resolve()
        for item in payload_contract["datasets"]
    ]
    root_cache: dict[str, bool] = {}
    digest = hashlib.sha256()
    row_count = 0
    for part in parts:
        parquet = pq.ParquetFile(part)
        columns = set(parquet.schema_arrow.names)
        if not {"storage_type", "path"}.issubset(columns):
            raise ValueError(f"Source shard lacks canonical locators: {part}")
        selected = ["storage_type", "path"]
        if "member" in columns:
            selected.append("member")
        for batch in parquet.iter_batches(batch_size=65536, columns=selected):
            frame = batch.to_pandas()
            members = frame["member"] if "member" in frame else [None] * len(frame)
            for storage_type, locator, member in zip(
                frame["storage_type"], frame["path"], members
            ):
                locator_text = str(locator)
                belongs = root_cache.get(locator_text)
                if belongs is None:
                    parsed = urlparse(locator_text)
                    if parsed.scheme:
                        if parsed.scheme != "file" or parsed.netloc not in {
                            "",
                            "localhost",
                        }:
                            raise ValueError(
                                f"Source locator is not local: {locator_text}"
                            )
                        locator_path = Path(unquote(parsed.path))
                    else:
                        locator_path = Path(locator_text)
                    if not locator_path.is_absolute():
                        raise ValueError(
                            f"Source locator is not absolute: {locator_text}"
                        )
                    resolved = locator_path.resolve()
                    belongs = any(resolved.is_relative_to(root) for root in roots)
                    root_cache[locator_text] = belongs
                if not belongs:
                    raise ValueError(
                        "Source locator is outside every contracted root: "
                        f"{locator_text}"
                    )
                record = {
                    "storage_type": str(storage_type),
                    "path": locator_text,
                    "member": None if pd.isna(member) else str(member),
                }
                digest.update(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8") +
                    b"\n"
                )
                row_count += 1
    return {
        "algorithm": "sha256_canonical_jsonl_v1",
        "digest": "sha256:" + digest.hexdigest(),
        "row_count": row_count,
    }


def _verified_embedding_store(
    manifest_path: Path,
    *,
    content_validation: str = "full_sha256",
) -> tuple[dict[str, Any], dict[str, Any], list[Path], dict[str, Any]]:
    """Verify every local embedding shard named by a committed store."""
    payload, artifact, manifest_identity = _committed_artifact_snapshot(
        manifest_path, artifact_type="embedding_store"
    )
    parsed = urlparse(payload.get("root_uri", ""))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Exact local search requires a local file:// store root")
    root = Path(unquote(parsed.path)).resolve()
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Embedding store requires at least one shard")
    if int(payload.get("shard_count", -1)) != len(shards):
        raise ValueError("Embedding store shard_count differs from its inventory")

    paths: list[Path] = []
    inventory: list[dict[str, Any]] = []
    relative_paths: set[str] = set()
    content_verification = payload.get("content_verification", {})
    content_seals = content_verification.get("shards", [])
    sealed_by_path: dict[str, dict[str, Any]] = {}
    if content_validation == "sealed_inventory":
        if (
            content_verification.get("algorithm") !=
            "sha256_each_shard_with_posix_stat_v1" or
            content_verification.get("inventory_digest") !=
            payload.get("inventory_digest") or
            not isinstance(content_seals, list) or
            content_verification.get("shard_seal_digest") !=
            canonical_digest(content_seals)
        ):
            raise ValueError(
                "sealed_inventory requires a valid content-verification seal"
            )
        for seal in content_seals:
            if not isinstance(seal, dict):
                raise ValueError("Content-verification shard seal must be an object")
            name = seal.get("relative_path")
            if not isinstance(name, str) or name in sealed_by_path:
                raise ValueError("Content-verification shard paths must be unique")
            sealed_by_path[name] = seal
        if len(sealed_by_path) != len(shards):
            raise ValueError("Content-verification seal does not cover every shard")
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"Embedding shard {index} must be an object")
        relative_value = shard.get("relative_path")
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError(f"Embedding shard {index} requires relative_path")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Embedding shard escapes the store root: {relative}")
        normalized = relative.as_posix()
        if normalized in relative_paths:
            raise ValueError(f"Duplicate embedding shard: {normalized}")
        relative_paths.add(normalized)
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Embedding shard escapes the store root: {relative}")
        declared_bytes = shard.get("bytes")
        declared_sha256 = shard.get("sha256")
        if (
            isinstance(declared_sha256, str) and
            len(declared_sha256) == 64 and
            all(character in "0123456789abcdef" for character in declared_sha256)
        ):
            declared_sha256 = "sha256:" + declared_sha256
        if (
            isinstance(declared_bytes, bool) or
            not isinstance(declared_bytes, int) or
            declared_bytes < 0 or
            not _is_sha256(declared_sha256)
        ):
            raise ValueError(
                f"Embedding shard {normalized} requires bytes and SHA-256 identity"
            )
        if not resolved.is_file():
            raise ValueError(f"Missing embedding shard: {resolved}")
        observed_stat = resolved.stat()
        actual_bytes = observed_stat.st_size
        if actual_bytes != declared_bytes:
            raise ValueError(
                f"Embedding shard identity changed (size mismatch): {resolved}"
            )
        if content_validation == "full_sha256":
            actual_sha256 = _sha256(resolved)
            if actual_sha256 != declared_sha256:
                raise ValueError(f"Embedding shard digest changed: {resolved}")
        elif content_validation == "sealed_inventory":
            seal = sealed_by_path.get(normalized)
            observed_seal = {
                "relative_path": normalized,
                "bytes": actual_bytes,
                "sha256": declared_sha256,
                "stat": {
                    "device": observed_stat.st_dev,
                    "inode": observed_stat.st_ino,
                    "mtime_ns": observed_stat.st_mtime_ns,
                    "ctime_ns": observed_stat.st_ctime_ns,
                },
            }
            if seal != observed_seal:
                raise ValueError(
                    f"Embedding shard differs from its content seal: {resolved}"
                )
        else:
            raise ValueError(
                f"Unsupported embedding-store validation: {content_validation}"
            )
        paths.append(resolved)
        inventory.append(
            {
                "relative_path": normalized,
                "bytes": declared_bytes,
                "sha256": declared_sha256,
            }
        )
    return payload, artifact, paths, {
        "manifest_identity": manifest_identity,
        "fingerprint_method": (
            "sha256_bytes_v1"
            if content_validation == "full_sha256"
            else "sealed_manifest_inventory_v1"
        ),
        "content_validation": content_validation,
        "inventory_digest": canonical_digest(inventory),
        "content_seal_digest": (
            content_verification.get("shard_seal_digest")
            if content_validation == "sealed_inventory"
            else None
        ),
        "payload_bytes": sum(item["bytes"] for item in inventory),
        "shards": inventory,
    }


def _validate_store_payload_binding(
    source_payload: dict[str, Any],
    source_artifact: dict[str, Any],
    payload_contract: dict[str, Any],
    payload_contract_identity: dict[str, Any],
) -> None:
    binding = source_payload.get("source_payload_contract", {})
    expected_input = {
        **payload_contract_identity,
        "role": "source_payload_contract",
    }
    locator_audit = binding.get("locator_audit", {})
    audit_algorithm = locator_audit.get("algorithm")
    valid_locator_audit = (
        audit_algorithm == "sha256_canonical_jsonl_v1" and
        _is_sha256(locator_audit.get("digest"))
    ) or (
        audit_algorithm ==
        "sealed_shard_inventory_and_canonical_root_prefix_v1" and
        locator_audit.get("inventory_digest") ==
        source_payload.get("inventory_digest") and
        _is_sha256(locator_audit.get("inventory_digest"))
    )
    if (
        binding.get("sha256") != payload_contract_identity["sha256"] or
        binding.get("datasets_digest") !=
        canonical_digest(payload_contract["datasets"]) or
        expected_input not in source_artifact.get("inputs", []) or
        not valid_locator_audit or
        int(locator_audit.get("row_count", -1)) !=
        int(source_payload.get("row_count", -2))
    ):
        raise ValueError(
            "Embedding store is not bound to the source payload contract"
        )


def _validate_continuation_source_lineage(
    *,
    previous_lock: dict[str, Any],
    current_lock: dict[str, Any],
    source_store_manifest: str | None,
) -> dict[str, Any]:
    """Prove that a continuation uses the prior run's exact source lineage."""
    mismatches = {}
    if source_store_manifest is None:
        previous_source = previous_lock.get("source_store")
        current_source = current_lock.get("source_store")
        if (
            not isinstance(previous_source, dict) or
            previous_source.get("fingerprint_method") != "sha256_bytes_v1" or
            previous_source != current_source
        ):
            mismatches["source_parts"] = {
                "expected": previous_source,
                "actual": current_source,
            }
        transition = {
            "mode": "source_parts",
            "identities_digest": canonical_digest(current_source),
        }
    else:
        previous_source = previous_lock.get("source_store")
        current_source = current_lock.get("source_store")
        if not isinstance(previous_source, dict):
            raise ValueError(
                "Continuation from a registered store requires the prior "
                "data lock's source_store"
            )
        if not isinstance(current_source, dict):
            raise ValueError(
                "Continuation from a registered store requires the current "
                "data lock's source_store"
            )
        previous_manifest_value = previous_source.get("manifest")
        if isinstance(previous_manifest_value, dict):
            parsed = urlparse(str(previous_manifest_value.get("uri", "")))
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise ValueError(
                    "Continuation parent store manifest must be a local file URI"
                )
            previous_manifest = Path(unquote(parsed.path)).resolve()
            previous_manifest_sha = previous_manifest_value.get("sha256")
        elif isinstance(previous_manifest_value, str):
            previous_manifest = Path(previous_manifest_value).resolve()
            previous_manifest_sha = previous_source.get("manifest_sha256")
        else:
            raise ValueError(
                "Continuation prior data lock has no source-store manifest"
            )
        source_manifest = Path(source_store_manifest).expanduser().resolve()
        if (
            previous_source == current_source and
            source_manifest == previous_manifest
        ):
            source_payload, source_artifact, source_identity = (
                _committed_artifact_snapshot(
                    source_manifest, artifact_type="embedding_store"
                )
            )
            expected_current = {
                "manifest_sha256": source_identity["sha256"],
                "artifact_id": source_artifact["artifact_id"],
                "declared_inventory_digest": source_payload.get(
                    "inventory_digest"
                ),
                "row_count": source_payload.get("row_count"),
                "shard_count": source_payload.get("shard_count"),
                "encoder": source_payload.get("encoder"),
            }
            observed_current = {
                "manifest_sha256": current_source.get("manifest", {}).get(
                    "sha256"
                ),
                **{
                    name: current_source.get(name)
                    for name in expected_current
                    if name != "manifest_sha256"
                },
            }
            if observed_current != expected_current:
                raise ValueError(
                    "Continuation source transition is not linked to the prior "
                    "data lock: " +
                    json.dumps(
                        {
                            "current_store": {
                                "expected": expected_current,
                                "actual": observed_current,
                            }
                        },
                        sort_keys=True,
                    )
                )
            return {
                "mode": "unchanged_store",
                "artifact_id": source_artifact["artifact_id"],
            }
        parent_payload, parent_artifact, parent_identity = (
            _committed_artifact_snapshot(
                previous_manifest, artifact_type="embedding_store"
            )
        )
        source_payload, source_artifact, source_identity = (
            _committed_artifact_snapshot(
                source_manifest, artifact_type="embedding_store"
            )
        )
        parent_input = next(
            (
                item
                for item in source_artifact.get("inputs", [])
                if item.get("role") == "source_embedding_store"
            ),
            None,
        )
        if previous_manifest_sha != parent_identity["sha256"]:
            mismatches["previous_manifest_sha256"] = {
                "expected": previous_manifest_sha,
                "actual": parent_identity["sha256"],
            }
        for name, previous_name in (
            ("inventory_digest", "declared_inventory_digest"),
            ("row_count", "row_count"),
            ("shard_count", "shard_count"),
            ("encoder", "encoder"),
        ):
            previous_value = previous_source.get(previous_name)
            if name == "inventory_digest" and previous_value is None:
                previous_value = previous_source.get("inventory_digest")
            if previous_value != parent_payload.get(name):
                mismatches[name] = {
                    "expected": previous_value,
                    "actual": parent_payload.get(name),
                }
        expected_parent_input = {
            **parent_identity,
            "role": "source_embedding_store",
            "artifact_id": parent_artifact["artifact_id"],
        }
        if (
            source_payload.get("parent_store_artifact_id") !=
            parent_artifact["artifact_id"] or
            parent_input != expected_parent_input
        ):
            mismatches["parent_store"] = {
                "expected": parent_artifact["artifact_id"],
                "actual": source_payload.get("parent_store_artifact_id"),
            }
        additive_binding_fields = {
            "content_verification",
            "parent_store_artifact_id",
            "source_payload_contract",
        }
        parent_inventory = {
            name: value
            for name, value in parent_payload.items()
            if name not in additive_binding_fields
        }
        bound_inventory = {
            name: value
            for name, value in source_payload.items()
            if name not in additive_binding_fields
        }
        if bound_inventory != parent_inventory:
            mismatches["bound_inventory"] = {
                "expected_digest": canonical_digest(parent_inventory),
                "actual_digest": canonical_digest(bound_inventory),
            }
        expected_current = {
            "manifest_sha256": source_identity["sha256"],
            "artifact_id": source_artifact["artifact_id"],
            "declared_inventory_digest": source_payload.get("inventory_digest"),
            "row_count": source_payload.get("row_count"),
            "shard_count": source_payload.get("shard_count"),
            "encoder": source_payload.get("encoder"),
        }
        observed_current = {
            "manifest_sha256": current_source.get("manifest", {}).get("sha256"),
            **{
                name: current_source.get(name)
                for name in expected_current
                if name != "manifest_sha256"
            },
        }
        if observed_current != expected_current:
            mismatches["current_store"] = {
                "expected": expected_current,
                "actual": observed_current,
            }
        transition = {
            "mode": "bound_store",
            "parent_artifact_id": parent_artifact["artifact_id"],
            "bound_artifact_id": source_artifact["artifact_id"],
        }
    if mismatches:
        raise ValueError(
            "Continuation source transition is not linked to the prior data lock: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return transition


def _committed_artifact(
    manifest_path: Path, *, artifact_type: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, artifact, _ = _committed_artifact_snapshot(
        manifest_path, artifact_type=artifact_type
    )
    return payload, artifact


def _committed_artifact_snapshot(
    manifest_path: Path, *, artifact_type: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = manifest_path.resolve()
    payload, payload_identity = _json_snapshot(manifest)
    artifact_path = manifest.with_name("artifact.json")
    success_path = manifest.with_name("_SUCCESS")
    if not artifact_path.is_file() or not success_path.is_file():
        raise ValueError(f"Artifact is not committed: {manifest}")
    artifact, _ = _json_snapshot(artifact_path)
    expected_id = canonical_digest(
        {
            name: artifact[name]
            for name in (
                "artifact_type",
                "schema_version",
                "producer",
                "inputs",
                "payload",
            )
        }
    )
    if artifact.get("artifact_id") != expected_id:
        raise ValueError(f"Artifact identity is invalid: {artifact_path}")
    if artifact.get("artifact_type") != artifact_type:
        raise ValueError(
            f"Expected {artifact_type}, found {artifact.get('artifact_type')}"
        )
    if artifact.get("payload") != payload:
        raise ValueError(f"Artifact payload differs from {manifest}")
    if success_path.read_text(encoding="utf-8").strip() != expected_id:
        raise ValueError(f"Artifact marker differs from {artifact_path}")
    return payload, artifact, payload_identity


def _validate_output_artifact(
    output_dir: Path,
    *,
    artifact_type: str,
    payload_files: dict[str, Path],
) -> dict[str, Any]:
    artifact_path = output_dir / "artifact.json"
    marker_path = output_dir / "_SUCCESS"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_id = canonical_digest(
        {
            name: artifact[name]
            for name in (
                "artifact_type",
                "schema_version",
                "producer",
                "inputs",
                "payload",
            )
        }
    )
    if artifact.get("artifact_id") != expected_id:
        raise ValueError(f"Output artifact identity is invalid: {artifact_path}")
    if artifact.get("artifact_type") != artifact_type:
        raise ValueError(f"Output artifact type changed: {artifact_path}")
    if marker_path.read_text(encoding="utf-8").strip() != expected_id:
        raise ValueError(f"Output success marker changed: {marker_path}")
    for payload_name, path in payload_files.items():
        if artifact["payload"].get(payload_name) != _file_identity(path):
            raise ValueError(f"Output payload changed after commit: {path}")
    return artifact


def _checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    requested = path.expanduser().absolute()
    resolved = requested.resolve()
    if resolved.is_file():
        _, _, _, digest = _stable_file_snapshot(resolved)
        return {
            "path": str(requested),
            "resolved_path": str(resolved),
            "method": "sha256",
            "sha256": digest,
        }
    for name in ("model.safetensors", "pytorch_model.bin", "model.pth"):
        candidate = resolved / name
        if candidate.is_file():
            _, _, _, digest = _stable_file_snapshot(candidate)
            return {
                "path": str(requested),
                "resolved_path": str(resolved),
                "checkpoint_file": name,
                "method": "sha256",
                "sha256": digest,
            }
    raise ValueError(f"Cannot fingerprint DINOv3 checkpoint: {resolved}")


def _workflow_source_digest() -> str:
    root = Path(__file__).resolve().parent
    included_suffixes = {".json", ".md", ".py", ".yaml", ".yml"}
    ignored_parts = {"__pycache__", ".pytest_cache", "tests"}
    values = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and
        path.suffix in included_suffixes and
        ignored_parts.isdisjoint(path.relative_to(root).parts)
    }
    return canonical_digest(values)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_source(
    module_name: str, search_paths: list[str | Path] | None = None
) -> Path | None:
    """Resolve a Python module without importing customer code."""
    relative = Path(*module_name.split("."))
    entries = [*(search_paths or []), *sys.path]
    for entry in entries:
        root = Path(entry or os.curdir)
        for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _execution_python_paths(value: dict[str, Any]) -> list[Path]:
    """Mirror the worker's configured Python search path for provenance."""
    execution = value["execution"]
    workdir = Path(execution["workdir"]).expanduser().resolve()
    configured = execution.get("environment", {}).get("PYTHONPATH", "")
    paths = []
    for entry in str(configured).split(os.pathsep):
        if not entry:
            continue
        path = Path(entry).expanduser()
        paths.append((path if path.is_absolute() else workdir / path).resolve())
    return paths


def _target_identity_set(path: Path, strategy: str) -> set[tuple[str, str]]:
    """Stream and validate the exact target identity population."""
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = {"sample_id", "task"}
    if missing := required.difference(names):
        raise ValueError(f"Target manifest is missing columns: {sorted(missing)}")
    columns = ["sample_id", "task"]
    if "role" in names:
        columns.append("role")
    identities: set[tuple[str, str]] = set()
    for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
        frame = batch.to_pandas()
        if strategy == "grit_score" and "role" in frame:
            frame = frame.loc[frame["role"].astype(str) == "query"]
        if frame[["sample_id", "task"]].isnull().any().any():
            raise ValueError("Target sample/task identities contain null values")
        batch_identities = list(
            zip(frame["sample_id"].astype(str), frame["task"].astype(str))
        )
        if len(batch_identities) != len(set(batch_identities)):
            raise ValueError("Target sample/task identities are not unique")
        overlap = identities.intersection(batch_identities)
        if overlap:
            raise ValueError("Target sample/task identities are not unique")
        identities.update(batch_identities)
    if not identities:
        raise ValueError("Target manifest has no scoring identities")
    return identities


def _validate_score_embedding_binding(
    path: Path,
    strategy: str,
    score_frame: pd.DataFrame,
    score_embeddings: np.ndarray,
) -> None:
    """Require score rows to carry the target manifest's exact embeddings."""
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if "embedding" not in names:
        raise StageFailure("score", "Target manifest has no immutable embedding column")
    columns = ["sample_id", "task", "embedding"]
    if "role" in names:
        columns.append("role")
    output_positions = {
        identity: index
        for index, identity in enumerate(
            zip(
                score_frame["sample_id"].astype(str),
                score_frame["task"].astype(str),
            )
        )
    }
    seen: set[tuple[str, str]] = set()
    for batch in parquet.iter_batches(columns=columns, batch_size=4_096):
        frame = batch.to_pandas()
        if strategy == "grit_score" and "role" in frame:
            frame = frame.loc[frame["role"].astype(str) == "query"]
        if frame.empty:
            continue
        identities = list(
            zip(frame["sample_id"].astype(str), frame["task"].astype(str))
        )
        if len(identities) != len(set(identities)) or seen.intersection(identities):
            raise StageFailure("score", "Target sample/task identities are not unique")
        if unknown := set(identities).difference(output_positions):
            raise StageFailure(
                "score",
                f"Target identities are absent from score output: {len(unknown)}",
            )
        try:
            expected = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise StageFailure(
                "score", "Target embeddings are not a numeric matrix"
            ) from exc
        positions = np.asarray(
            [output_positions[identity] for identity in identities], dtype=np.int64
        )
        observed = score_embeddings[positions]
        if (
            expected.ndim != 2 or
            expected.shape != observed.shape or
            not np.isfinite(expected).all() or
            not np.array_equal(expected, observed)
        ):
            raise StageFailure(
                "score",
                "Score embeddings differ from the immutable target embeddings",
            )
        seen.update(identities)
    if seen != set(output_positions):
        raise StageFailure(
            "score", "Score output does not exactly cover target embeddings"
        )


class RefinementWorkflow:
    """Bare Python API used by both the CLI and conversational skill."""

    def __init__(self, config: WorkflowConfig):
        """Bind the workflow configuration or durable runtime paths."""
        self.config = config
        self.run_dir = config.run_dir
        self.store = StateStore(self.run_dir)
        self.runner = build_runner(config.value["execution"], self.run_dir)
        self._source_store_verification: tuple[
            dict[str, Any], dict[str, Any], list[Path], dict[str, Any]
        ] | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "RefinementWorkflow":
        """Load and validate a workflow configuration from YAML."""
        return cls(WorkflowConfig.from_file(path))

    def validate(self, *, require_paths: bool = True) -> dict[str, Any]:
        """Validate configuration, source provenance and required input paths."""
        if require_paths:
            # Revalidate once per approval/resume, then reuse for data.lock.
            self._source_store_verification = None
        value = self.config.value
        paths = {
            "base_checkpoint": Path(value["model"]["base_checkpoint"]).expanduser(),
            "base_spec": Path(value["training"]["base_spec"]).expanduser(),
            "target_manifest": Path(value["data"]["target_manifest"]).expanduser(),
            "source_payload_contract": Path(
                value["data"]["source_payload_contract"]
            ).expanduser(),
        }
        initial_scoring = value["model"].get("initial_scoring_checkpoint")
        if initial_scoring:
            paths["initial_scoring_checkpoint"] = Path(initial_scoring).expanduser()
        if value["data"].get("source_store_manifest"):
            paths["source_store_manifest"] = Path(
                value["data"]["source_store_manifest"]
            ).expanduser()
            paths["target_embedding_contract"] = Path(
                value["data"]["target_embedding_contract"]
            ).expanduser()
        else:
            paths.update(
                {
                    f"source_part_{index}": Path(path).expanduser()
                    for index, path in enumerate(value["data"]["source_parts"])
                }
            )
        search = value["actions"]["search"]
        if search["backend"] == "audited_ann":
            paths["source_identity_audit"] = Path(
                value["data"]["source_identity_audit"]
            ).expanduser()
            for name in (
                "dense_store_manifest",
                "ann_index_manifest",
                "ann_audit_manifest",
            ):
                paths[f"search_{name}"] = Path(search[name]).expanduser()
        if value["data"].get("benchmark_manifest"):
            paths["benchmark_manifest"] = Path(
                value["data"]["benchmark_manifest"]
            ).expanduser()
        if value["data"].get("previous_training_manifest"):
            paths["previous_training_manifest"] = Path(
                value["data"]["previous_training_manifest"]
            ).expanduser()
        if value.get("continuation"):
            for name in (
                "training_contract",
                "training_commit",
                "success_marker",
                "previous_data_lock",
                "previous_release_lock",
            ):
                paths[f"continuation_{name}"] = Path(
                    value["continuation"][name]
                ).expanduser()
        if value["data"].get("benchmark_acquisition_units"):
            paths["benchmark_acquisition_units"] = Path(
                value["data"]["benchmark_acquisition_units"]
            ).expanduser()
        for action_name, action in value["actions"].items():
            for index, configured in enumerate(
                action.get("implementation_files", [])
            ):
                paths[f"{action_name}_implementation_{index}"] = Path(
                    configured
                ).expanduser()
        missing = [name for name, path in paths.items() if not path.exists()]
        if (
            require_paths and
            not missing and
            not value["data"].get("source_store_manifest")
        ):
            for index, part in enumerate(self._source_parts()):
                if not part.is_file():
                    missing.append(f"registered_source_part_{index}")
        if require_paths and missing:
            raise ValueError(f"Missing configured inputs: {missing}")
        if require_paths:
            _source_payload_contract(paths["source_payload_contract"])
            _target_identity_set(paths["target_manifest"], self.config.strategy)
        parent = value["data"].get("previous_training_manifest")
        if (
            require_paths and
            search["backend"] == "audited_ann" and
            parent
        ):
            parquet = pq.ParquetFile(Path(parent).expanduser())
            lineage = {
                "source_row_id",
                "dense_store_artifact_id",
                "source_inventory_digest",
            }
            if parquet.metadata.num_rows and not lineage.issubset(
                parquet.schema_arrow.names
            ):
                raise ValueError(
                    "Nonempty indexed parent history requires corpus lineage"
                )
        if (
            require_paths and
            value["actions"]["evaluate"]["command"] and
            value["actions"]["evaluate"]["scope"] == "sealed_benchmark"
        ):
            self._validate_benchmark_disjointness()
        return {
            "valid": not missing,
            "missing": missing,
            "plan": (
                self.plan(_cache_source_verification=True)
                if not missing
                else self.config.plan()
            ),
        }

    def plan(
        self, *, _cache_source_verification: bool = False
    ) -> dict[str, Any]:
        """Describe the resolved stages, resources and stopping policy."""
        if not _cache_source_verification:
            self._source_store_verification = None
        plan = self.config.plan()
        data = self.config.value["data"]

        def parquet_inventory(configured: str) -> dict[str, Any]:
            path = Path(configured).expanduser().resolve()
            return {
                "path": str(path),
                "rows": int(pq.ParquetFile(path).metadata.num_rows),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }

        inventory: dict[str, Any] = {
            "targets": parquet_inventory(data["target_manifest"]),
        }
        parent = data.get("previous_training_manifest")
        if parent:
            inventory["parent_training"] = parquet_inventory(parent)
        store: dict[str, Any] | None = None
        store_artifact: dict[str, Any] | None = None
        legacy_source_paths: list[Path] = []
        if data.get("source_store_manifest"):
            store_path = Path(data["source_store_manifest"]).expanduser().resolve()
            store, store_artifact, _, verified = self._verified_source_store(
                cache=_cache_source_verification
            )
            inventory["source"] = {
                "path": str(store_path),
                "manifest_sha256": verified["manifest_identity"]["sha256"],
                "rows": int(store["row_count"]),
                "shards": int(store["shard_count"]),
                "embedding_dim": int(store["embedding_dim"]),
                "payload_bytes": verified["payload_bytes"],
                "verified_inventory_digest": verified["inventory_digest"],
            }
        else:
            legacy_source_paths = [
                Path(path).expanduser().resolve() for path in data["source_parts"]
            ]
            parts = [parquet_inventory(path) for path in data["source_parts"]]
            inventory["source"] = {
                "parts": parts,
                "rows": sum(part["rows"] for part in parts),
                "payload_bytes": sum(part["bytes"] for part in parts),
            }
        payload_contract_path = Path(
            data["source_payload_contract"]
        ).expanduser().resolve()
        payload_contract, payload_identity = _source_payload_contract_snapshot(
            payload_contract_path
        )
        inventory["source_payloads"] = {
            "path": str(payload_contract_path),
            "sha256": payload_identity["sha256"],
            "datasets": payload_contract["datasets"],
        }
        if parent:
            inventory["parent_training"]["locator_audit"] = (
                _audit_source_locators(
                    [Path(parent).expanduser().resolve()], payload_contract
                )
            )
        if data.get("source_store_manifest"):
            assert store is not None and store_artifact is not None
            _validate_store_payload_binding(
                store,
                store_artifact,
                payload_contract,
                payload_identity,
            )
        else:
            inventory["source"]["locator_audit"] = _audit_source_locators(
                legacy_source_paths, payload_contract
            )
        plan["approval_contract"]["data"]["inventory"] = inventory
        return plan

    def _verified_source_store(
        self,
        *,
        cache: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], list[Path], dict[str, Any]]:
        if cache and self._source_store_verification is not None:
            return self._source_store_verification
        manifest = Path(
            self.config.value["data"]["source_store_manifest"]
        ).expanduser()
        verification = _verified_embedding_store(
            manifest,
            content_validation=self.config.value["data"][
                "source_store_validation"
            ],
        )
        if cache:
            self._source_store_verification = verification
        return verification

    def _source_parts(self) -> list[Path]:
        data = self.config.value["data"]
        if data.get("source_parts"):
            return [Path(path).expanduser().resolve() for path in data["source_parts"]]
        if self._source_store_verification is None:
            self._verified_source_store(cache=True)
        assert self._source_store_verification is not None
        return list(self._source_store_verification[2])

    def _exclusion_manifest(
        self, state: dict[str, Any], round_index: int
    ) -> Path | None:
        """Build a lineage-preserving union of parent and prior-round samples."""
        configured = self.config.value["data"].get("previous_training_manifest")
        inputs = (
            [Path(configured).expanduser().resolve()]
            if configured
            else []
        )
        round_inputs = state.get("round_inputs", {}).get(str(round_index), state)
        if round_inputs["current_training_manifest"]:
            current = Path(round_inputs["current_training_manifest"]).resolve()
            if current not in inputs:
                inputs.append(current)
        if not inputs:
            return None
        if len(inputs) == 1:
            return inputs[0]
        destination = (
            self.run_dir /
            "lineage" /
            f"excluded_round_{round_index:03d}.parquet"
        )
        search = self.config.value["actions"]["search"]
        requires_corpus_lineage = (
            search["backend"] == "audited_ann" or
            bool(search.get("dense_store_manifest")) or
            bool(search.get("parameters", {}).get("dense_store_manifest"))
        )
        lineage_columns = [
            "source_row_id",
            "dense_store_artifact_id",
            "source_inventory_digest",
        ]
        if destination.is_file():
            parquet = pq.ParquetFile(destination)
            present = set(parquet.schema_arrow.names)
            if (
                not requires_corpus_lineage or
                parquet.metadata.num_rows == 0 or
                set(lineage_columns).issubset(present)
            ):
                return destination
        frames = []
        for path in inputs:
            parquet = pq.ParquetFile(path)
            columns = ["sample_id"]
            present = set(parquet.schema_arrow.names)
            if requires_corpus_lineage and parquet.metadata.num_rows:
                if missing := set(lineage_columns).difference(present):
                    raise ValueError(
                        "Nonempty row-ID search exclusions require corpus lineage: "
                        f"{path}: {sorted(missing)}"
                    )
            columns.extend(name for name in lineage_columns if name in present)
            frame = pd.read_parquet(path, columns=columns, pre_buffer=False)
            if requires_corpus_lineage:
                defaults = {
                    "source_row_id": pd.Series(dtype="int64"),
                    "dense_store_artifact_id": pd.Series(dtype="str"),
                    "source_inventory_digest": pd.Series(dtype="str"),
                }
                for name, empty in defaults.items():
                    if name not in frame:
                        frame[name] = empty
            frames.append(frame)
        identities = pd.concat(frames, ignore_index=True)
        identities["sample_id"] = identities["sample_id"].astype(str)
        identities = identities.drop_duplicates("sample_id", keep="first")
        if requires_corpus_lineage and len(identities):
            missing_lineage = identities[lineage_columns].isna().any(axis=1)
            if missing_lineage.any():
                raise ValueError(
                    "Nonempty row-ID search exclusions contain incomplete corpus "
                    f"lineage: rows={int(missing_lineage.sum())}"
                )
            identities["source_row_id"] = identities["source_row_id"].astype(
                "int64"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        identities.to_parquet(temporary, index=False)
        temporary.replace(destination)
        return destination

    def _data_lock(self) -> dict[str, Any]:
        value = self.config.value
        data = value["data"]
        lock: dict[str, Any] = {
            "schema_version": "1.0",
            "base_checkpoint": _checkpoint_fingerprint(
                Path(value["model"]["base_checkpoint"])
            ),
            "target_manifest": {
                "path": str(Path(data["target_manifest"]).expanduser().resolve()),
                "sha256": _sha256(Path(data["target_manifest"]).expanduser()),
            },
            "base_spec": {
                "path": str(
                    Path(value["training"]["base_spec"]).expanduser().resolve()
                ),
                "sha256": _sha256(
                    Path(value["training"]["base_spec"]).expanduser()
                ),
            },
        }
        if value["model"].get("initial_scoring_checkpoint"):
            lock["initial_scoring_checkpoint"] = _checkpoint_fingerprint(
                Path(value["model"]["initial_scoring_checkpoint"])
            )
        payload_contract_path = Path(
            data["source_payload_contract"]
        ).expanduser().resolve()
        payload_contract, payload_contract_identity = (
            _source_payload_contract_snapshot(payload_contract_path)
        )
        lock["source_payload_contract"] = {
            "identity": payload_contract_identity,
            "contract": payload_contract,
        }
        source_payload: dict[str, Any] | None = None
        source_artifact: dict[str, Any] | None = None
        if data.get("source_store_manifest"):
            source_payload, source_artifact, _, verified = (
                self._verified_source_store(cache=True)
            )
            _validate_store_payload_binding(
                source_payload,
                source_artifact,
                payload_contract,
                payload_contract_identity,
            )
            lock["source_store"] = {
                "manifest": verified["manifest_identity"],
                "artifact_id": source_artifact["artifact_id"],
                "declared_inventory_digest": source_payload.get(
                    "inventory_digest"
                ),
                "verified_inventory_digest": verified["inventory_digest"],
                "fingerprint_method": verified["fingerprint_method"],
                "encoder": source_payload.get("encoder"),
                "shard_count": source_payload.get("shard_count"),
                "row_count": source_payload.get("row_count"),
                "payload_bytes": verified["payload_bytes"],
                "shards": verified["shards"],
            }
            target_contract = Path(
                data["target_embedding_contract"]
            ).expanduser().resolve()
            lock["target_embedding_contract"] = _file_identity(target_contract)
        else:
            legacy_parts = self._source_parts()
            inventory = [_file_identity(path) for path in legacy_parts]
            lock["source_store"] = {
                "fingerprint_method": "sha256_bytes_v1",
                "inventory_digest": canonical_digest(inventory),
                "shards": inventory,
                "locator_audit": _audit_source_locators(
                    legacy_parts, payload_contract
                ),
            }
        if data.get("previous_training_manifest"):
            previous = Path(data["previous_training_manifest"]).expanduser().resolve()
            lock["previous_training_manifest"] = {
                "path": str(previous),
                "sha256": _sha256(previous),
                "locator_audit": _audit_source_locators(
                    [previous], payload_contract
                ),
            }
        if value.get("continuation"):
            continuation = value["continuation"]
            lock["continuation"] = {
                "previous_run_dir": str(
                    Path(continuation["previous_run_dir"]).expanduser().resolve()
                ),
                "adopted_round": int(continuation["adopted_round"]),
                "artifacts": {
                    name: _file_identity(Path(continuation[name]).expanduser())
                    for name in (
                        "training_contract",
                        "training_commit",
                        "success_marker",
                        "previous_data_lock",
                        "previous_release_lock",
                    )
                },
            }
        if data.get("benchmark_manifest"):
            benchmark = Path(data["benchmark_manifest"]).expanduser().resolve()
            lock["benchmark_manifest"] = {
                "path": str(benchmark),
                "sha256": _sha256(benchmark),
            }
        if data.get("benchmark_acquisition_units"):
            units = Path(data["benchmark_acquisition_units"]).expanduser().resolve()
            lock["benchmark_acquisition_units"] = {
                "path": str(units),
                "sha256": _sha256(units),
            }
        search = value["actions"]["search"]
        if search["backend"] == "audited_ann":
            if source_payload is None or source_artifact is None:
                raise ValueError("Audited ANN search requires a verified source store")
            dense_payload, dense_artifact, dense_identity = (
                _committed_artifact_snapshot(
                    Path(search["dense_store_manifest"]),
                    artifact_type="dense_vector_store",
                )
            )
            ann_payload, ann_artifact, ann_identity = (
                _committed_artifact_snapshot(
                    Path(search["ann_index_manifest"]),
                    artifact_type="ann_index",
                )
            )
            audit_payload, audit_artifact, audit_identity = (
                _committed_artifact_snapshot(
                    Path(search["ann_audit_manifest"]),
                    artifact_type="ann_recall_audit",
                )
            )
            if dense_payload.get("source_store_artifact_id") != source_artifact[
                "artifact_id"
            ]:
                raise ValueError("Dense store belongs to another source store")
            if dense_payload.get("source_inventory_digest") != source_payload.get(
                "inventory_digest"
            ):
                raise ValueError("Dense store source inventory differs")
            if ann_payload.get("dense_store_artifact_id") != dense_artifact[
                "artifact_id"
            ]:
                raise ValueError("ANN index belongs to another dense store")
            if audit_payload.get("index_artifact_id") != ann_artifact["artifact_id"]:
                raise ValueError("ANN audit belongs to another index")
            if audit_payload.get("dense_store_artifact_id") != dense_artifact[
                "artifact_id"
            ]:
                raise ValueError("ANN audit belongs to another dense store")
            if audit_payload.get("source_inventory_digest") != source_payload.get(
                "inventory_digest"
            ):
                raise ValueError("ANN audit source inventory differs")
            if audit_payload.get("vector_inventory_digest") != dense_payload.get(
                "vector_inventory_digest"
            ):
                raise ValueError("ANN audit vector inventory differs")
            if not audit_payload.get("passed"):
                raise ValueError("ANN audit did not pass")
            if int(audit_payload.get("n_probes", -1)) != int(search["n_probes"]):
                raise ValueError("ANN audit n_probes differs from the run")
            if int(audit_payload.get("ann_candidate_count", -1)) != int(
                search["ann_candidates"]
            ):
                raise ValueError("ANN audit candidate depth differs from the run")
            target_contract = json.loads(
                Path(data["target_embedding_contract"]).read_text(encoding="utf-8")
            )
            if target_contract.get("encoder") != source_payload.get("encoder"):
                raise ValueError("Target embedding encoder differs from the source")
            contract_source = target_contract.get("source_store_manifest", {})
            if contract_source.get("inventory_digest") != source_payload.get(
                "inventory_digest"
            ):
                raise ValueError("Target embedding contract names another source")
            identity_audit_path = data.get("source_identity_audit")
            if not identity_audit_path:
                raise ValueError("Audited ANN search requires data.source_identity_audit")
            identity_payload, identity_artifact = _committed_artifact(
                Path(identity_audit_path), artifact_type="source_identity_audit"
            )
            if identity_payload.get("source_store_artifact_id") != source_artifact[
                "artifact_id"
            ]:
                raise ValueError("Source identity audit belongs to another store")
            if identity_payload.get("source_inventory_digest") != source_payload.get(
                "inventory_digest"
            ):
                raise ValueError("Source identity audit inventory differs")
            if int(identity_payload.get("row_count", -1)) != int(
                source_payload.get("row_count", -2)
            ):
                raise ValueError("Source identity audit row count differs")
            if identity_payload.get("hash_algorithm") != "blake2b-128":
                raise ValueError("Source identity audit hash algorithm is unsupported")
            if identity_payload.get("collision_policy") != (
                "fail_closed_on_blake2b128_collision"
            ):
                raise ValueError("Source identity audit collision policy is unsafe")
            if int(identity_payload.get("duplicate_hash_count", -1)) != 0:
                raise ValueError("Source identity audit found duplicate sample IDs")
            if not identity_payload.get("unique"):
                raise ValueError("Source identity audit found duplicate sample IDs")
            if not _is_sha256(identity_payload.get("sorted_identity_sha256")):
                raise ValueError("Source identity audit sorted digest is invalid")
            if not _is_sha256(
                identity_payload.get("implementation_contract_digest")
            ):
                raise ValueError(
                    "Source identity audit implementation digest is invalid"
                )
            identity_proof = {
                "source_store_artifact_id": source_artifact["artifact_id"],
                "source_inventory_digest": source_payload["inventory_digest"],
                "row_count": int(source_payload["row_count"]),
                "hash_algorithm": "blake2b-128",
                "sorted_identity_sha256": identity_payload.get(
                    "sorted_identity_sha256"
                ),
                "duplicate_hash_count": 0,
            }
            if identity_payload.get("proof_digest") != canonical_digest(
                identity_proof
            ):
                raise ValueError("Source identity audit proof digest is invalid")
            lock["indexed_search"] = {
                "dense_store_manifest": {
                    **dense_identity,
                    "artifact_id": dense_artifact["artifact_id"],
                },
                "ann_index_manifest": {
                    **ann_identity,
                    "artifact_id": ann_artifact["artifact_id"],
                },
                "ann_audit_manifest": {
                    **audit_identity,
                    "artifact_id": audit_artifact["artifact_id"],
                },
            }
            lock["indexed_search"].update(
                {
                    "source_store_artifact_id": source_artifact["artifact_id"],
                    "dense_store_artifact_id": dense_artifact["artifact_id"],
                    "ann_index_artifact_id": ann_artifact["artifact_id"],
                    "ann_audit_artifact_id": audit_artifact["artifact_id"],
                    "source_identity_audit_artifact_id": identity_artifact[
                        "artifact_id"
                    ],
                    "source_identity_proof_digest": identity_payload[
                        "proof_digest"
                    ],
                    "source_identity_audit": {
                        "path": str(Path(identity_audit_path).resolve()),
                        "sha256": _sha256(Path(identity_audit_path)),
                    },
                    "n_probes": int(search["n_probes"]),
                    "ann_candidates": int(search["ann_candidates"]),
                }
            )
        adapter_files = {}
        module_paths = _execution_python_paths(value)
        for action_name in ("score", "data", "search", "train", "evaluate"):
            action_command = value["actions"][action_name].get("command", [])
            for index, token in enumerate(action_command):
                candidate = Path(str(token)).expanduser()
                if candidate.is_file():
                    adapter_files[f"{action_name}.command.{index}"] = {
                        "path": str(candidate.resolve()),
                        "sha256": _sha256(candidate),
                    }
                if token == "-m" and index + 1 < len(action_command):
                    module_name = str(action_command[index + 1])
                    module = _module_source(module_name, module_paths)
                    if module is None:
                        raise ValueError(
                            f"Cannot resolve {action_name} module {module_name!r} "
                            "from execution.environment.PYTHONPATH"
                        )
                    adapter_files[f"{action_name}.module.{index + 1}"] = {
                        "module": module_name,
                        "path": str(module),
                        "sha256": _sha256(module),
                    }
                    if (
                        action_name == "score" and
                        module_name ==
                        "nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli"
                    ):
                        for implementation in (
                            module,
                            module.with_name("grit.py"),
                            module.with_name("grit_pipeline.py"),
                        ):
                            if not implementation.is_file():
                                raise ValueError(
                                    "Built-in GRIT implementation closure is incomplete: "
                                    f"{implementation}"
                                )
                            key = (
                                f"{action_name}.implementation."
                                f"{implementation.name}"
                            )
                            adapter_files[key] = {
                                "path": str(implementation),
                                "sha256": _sha256(implementation),
                            }
            if action_name == "search":
                for search_stage in ("candidate", "rerank"):
                    stage_command = value["actions"]["search"].get(
                        search_stage, {}
                    ).get("command", [])
                    for index, token in enumerate(stage_command):
                        candidate = Path(str(token)).expanduser()
                        if candidate.is_file():
                            key = f"search.{search_stage}.command.{index}"
                            adapter_files[key] = {
                                "path": str(candidate.resolve()),
                                "sha256": _sha256(candidate),
                            }
                        if token == "-m" and index + 1 < len(stage_command):
                            module_name = str(stage_command[index + 1])
                            module = _module_source(module_name, module_paths)
                            if module is None:
                                raise ValueError(
                                    "Cannot resolve search "
                                    f"{search_stage} module {module_name!r} from "
                                    "execution.environment.PYTHONPATH"
                                )
                            key = f"search.{search_stage}.module.{index + 1}"
                            adapter_files[key] = {
                                "module": module_name,
                                "path": str(module),
                                "sha256": _sha256(module),
                            }
            for name, configured in value["actions"][action_name].get(
                "parameters", {}
            ).items():
                candidate = Path(str(configured)).expanduser()
                if candidate.is_file():
                    adapter_files[f"{action_name}.{name}"] = {
                        "path": str(candidate.resolve()),
                        "sha256": _sha256(candidate),
                    }
            for index, configured in enumerate(
                value["actions"][action_name].get("implementation_files", [])
            ):
                candidate = Path(configured).expanduser().resolve()
                adapter_files[
                    f"{action_name}.implementation.explicit.{index}"
                ] = {
                    "path": str(candidate),
                    "sha256": _sha256(candidate),
                }
        lock["adapter_files"] = adapter_files
        return lock

    def _validate_benchmark_disjointness(self) -> None:
        data = self.config.value["data"]
        column = data["acquisition_unit_column"]
        targets = pd.read_parquet(data["target_manifest"], pre_buffer=False)
        benchmark = pd.read_parquet(data["benchmark_acquisition_units"], pre_buffer=False)
        if column not in targets or column not in benchmark:
            raise ValueError(
                "Target and benchmark-unit manifests must contain the declared "
                f"acquisition unit column {column!r}"
            )
        overlap = set(targets[column].astype(str)).intersection(
            benchmark[column].astype(str)
        )
        if overlap:
            examples = sorted(overlap)[:10]
            raise ValueError(
                "Adaptive targets overlap the sealed benchmark by acquisition unit: "
                f"count={len(overlap)}, examples={examples}"
            )

    def _initialize(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.run_dir / "input.yaml"
        resolved_path = self.run_dir / "resolved_run.json"
        if not input_path.exists():
            input_path.write_text(
                yaml.safe_dump(self.config.to_dict(), sort_keys=False), encoding="utf-8"
            )
        if resolved_path.exists():
            existing = json.loads(resolved_path.read_text(encoding="utf-8"))
            if canonical_digest(existing) != self.config.digest:
                raise RuntimeError("resolved_run.json does not match requested config")
        else:
            _atomic_json(resolved_path, self.config.to_dict())
        data_lock_path = self.run_dir / "data.lock.json"
        requested_data_lock = self._data_lock()
        if data_lock_path.exists():
            existing_data_lock = json.loads(data_lock_path.read_text(encoding="utf-8"))
            if canonical_digest(existing_data_lock) != canonical_digest(requested_data_lock):
                raise RuntimeError("Input data or checkpoint changed; fork the run")
        else:
            _atomic_json(data_lock_path, requested_data_lock)
        release = {
            "workflow": "tao-run-dinov3-ssl-deft",
            "workflow_version": "0.1.0",
            "schema_version": self.config.value["schema_version"],
            "python": sys.version.split()[0],
            "workflow_source_digest": _workflow_source_digest(),
            "installed_packages": {
                name: _package_version(name)
                for name in ("nvidia-tao-pytorch", "nvidia-tao-core", "nvidia-tao-ds")
            },
            "components": self.config.value.get("release", {}),
        }
        release_path = self.run_dir / "release.lock.json"
        if release_path.exists():
            existing_release = json.loads(release_path.read_text(encoding="utf-8"))
            if canonical_digest(existing_release) != canonical_digest(release):
                raise RuntimeError("Workflow release changed; fork the run")
        else:
            _atomic_json(release_path, release)
        run_id = self.config.value["workflow"].get("run_id") or self.config.digest[7:23]
        state = self.store.initialize(run_id=run_id, config_digest=self.config.digest)
        if state["current_checkpoint"] is None:
            initial_checkpoint = self.config.value["model"].get(
                "initial_scoring_checkpoint"
            ) or self.config.value["model"]["base_checkpoint"]
            state["current_checkpoint"] = str(
                Path(initial_checkpoint)
                .expanduser()
                .absolute()
            )
            self.store.save(state)
        return state

    def _command(
        self,
        template: list[str],
        values: dict[str, Any],
    ) -> list[str]:
        formatted = _FormatValues({key: str(value) for key, value in values.items()})
        return [str(token).format_map(formatted) for token in template]

    def _run_stage(
        self,
        state: dict[str, Any],
        *,
        round_index: int,
        stage: str,
        command: list[str],
        output_paths: dict[str, Path],
        resources: dict[str, Any] | None = None,
        stage_environment: dict[str, str] | None = None,
        validate_outputs: Callable[[], None] | None = None,
        commit_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if (self.run_dir / "cancel.requested").exists():
            self._reconcile_cancellation(state)
            raise RunCanceled(f"Cancellation requested for {state['run_id']}")
        key = f"round_{round_index:03d}/{stage}"
        if key in state["completed_stages"]:
            for name, path in output_paths.items():
                if not path.exists():
                    raise StageFailure(
                        stage, f"Cached {key} output {name} is missing: {path}"
                    )
            if validate_outputs:
                validate_outputs()
            return
        output_root = self.run_dir / "rounds" / f"round_{round_index:03d}" / stage
        output_root.mkdir(parents=True, exist_ok=True)
        job_id = client_job_id(
            run_id=state["run_id"],
            round_index=round_index,
            stage=stage,
            command=command,
        )
        action_name = {
            "select_targets": "data",
            "search_candidates": "search",
            "materialize": "data",
        }.get(stage, stage)
        action = self.config.value["actions"].get(action_name, {})
        desired_resources = (
            resources
            if resources is not None
            else action.get("resources", {})
        )
        adapter_managed = action.get("execution_mode") == "adapter_managed"
        resolved_resources = (
            action.get("wrapper_resources", {"nodes": 1})
            if adapter_managed
            else desired_resources
        )
        managed_resources = desired_resources if adapter_managed else None
        nodes = int(resolved_resources.get("nodes", 1))
        environment = {
            str(key): str(value)
            for key, value in self.config.value["execution"].get(
                "environment", {}
            ).items()
        }
        environment.update(stage_environment or {})
        request = StageRequest(
            client_job_id=job_id,
            run_id=state["run_id"],
            round_index=round_index,
            stage=stage,
            command=command,
            workdir=str(self.config.value["execution"]["workdir"]),
            results_dir=str(output_root),
            environment=environment,
            resources=resolved_resources,
            execution_contract={
                **({"container_image": stage_image(self.config.value["actions"], stage)}
                   if stage_image(self.config.value["actions"], stage) else {}),
                "membership": "static",
                "attempt_scope": "gang" if nodes > 1 else "process",
                "retry_scope": "cancel_all_then_restart" if nodes > 1 else "process",
                "attempt_id_scope": "backend_attempt",
                "launch_id_environment": (
                    "TAO_REFINEMENT_LAUNCH_ID" if nodes > 1 else None
                ),
                "adapter_managed_resources": managed_resources,
                "required_capabilities": (
                    (["containers"] if stage_image(self.config.value["actions"], stage) else [])
                ) + (
                    ["gpu_faiss"]
                    if action_name == "score" and
                    self.config.value["execution"].get(
                        "capabilities", {}
                    ).get("gpu_faiss") is True
                    else []
                ),
            },
        )
        state["active_jobs"][key] = {"client_job_id": job_id}
        self.store.save(state)
        self.store.append_event(
            state,
            round_index=round_index,
            stage=stage,
            status="submitted",
            extra={"job": {"client_job_id": job_id}, "command_digest": canonical_digest(command)},
        )
        result = self.runner.run(request)
        if (self.run_dir / "cancel.requested").exists():
            latest = self.store.load()
            latest.get("active_jobs", {}).pop(key, None)
            latest["status"] = (
                "canceling" if latest.get("active_jobs") else "canceled"
            )
            self.store.save(latest)
            raise RunCanceled(f"Cancellation requested for {state['run_id']}")
        if result.state != "COMPLETE":
            raise StageFailure(
                stage,
                f"{stage} job {job_id} ended in {result.state}; log={result.log_path}"
            )
        missing = {name: str(path) for name, path in output_paths.items() if not path.exists()}
        if missing:
            raise StageFailure(
                stage, f"{stage} completed without required outputs: {missing}"
            )
        if validate_outputs:
            try:
                validate_outputs()
            except StageFailure:
                raise
            except Exception as exc:
                raise StageFailure(stage, str(exc)) from exc
        if commit_state:
            commit_state(state)
        self.store.complete_stage(
            state,
            round_index=round_index,
            stage=stage,
            outputs={name: str(path) for name, path in output_paths.items()},
            job=asdict(result),
        )

    def _suppressed_path(self, state: dict[str, Any]) -> Path:
        limit = int(self.config.value["workflow"]["persistent_target_rounds"])
        sample_ids = sorted(
            sample_id
            for sample_id, record in state["persistent_targets"].items()
            if record.get("suppressed") or int(record.get("streak", 0)) >= limit
        )
        path = self.run_dir / "artifacts" / "suppressed_targets.json"
        _atomic_json(
            path,
            {
                "schema_version": "1.0",
                "reason": "persistent_weak_target",
                "threshold_rounds": limit,
                "sample_ids": sample_ids,
            },
        )
        return path

    def _update_persistence(
        self, state: dict[str, Any], selection_path: Path, round_index: int
    ) -> None:
        current = set(pd.read_parquet(selection_path, pre_buffer=False)["sample_id"].astype(str))
        for sample_id, record in state["persistent_targets"].items():
            if sample_id not in current and not record.get("suppressed"):
                record["streak"] = 0
        for sample_id in current:
            record = state["persistent_targets"].setdefault(
                sample_id, {"streak": 0, "first_round": round_index}
            )
            if record.get("suppressed"):
                continue
            previous_round = int(record.get("last_round", round_index - 1))
            record["streak"] = int(record["streak"]) + 1 if previous_round == round_index - 1 else 1
            record["last_round"] = round_index
            if int(record["streak"]) >= int(
                self.config.value["workflow"]["persistent_target_rounds"]
            ):
                record["suppressed"] = True
                record["suppressed_round"] = round_index

    def _locked_action_entrypoint(self, action_name: str) -> str:
        """Return the approved entrypoint digest for one configured action."""
        lock = json.loads(
            (self.run_dir / "data.lock.json").read_text(encoding="utf-8")
        )
        entries = lock.get("adapter_files", {})
        modules = [
            value["sha256"]
            for key, value in entries.items()
            if key.startswith(f"{action_name}.") and
            ".module." in key
        ]
        commands = [
            value["sha256"]
            for key, value in entries.items()
            if key.startswith(f"{action_name}.") and ".command." in key
        ]
        candidates = modules or commands[-1:]
        if not candidates:
            raise ValueError(
                f"No locked entrypoint exists for action {action_name!r}"
            )
        return candidates[0]

    def _locked_action_implementation(self, action_name: str) -> str:
        """Return the approved dependency-closure digest for one action."""
        lock = json.loads(
            (self.run_dir / "data.lock.json").read_text(encoding="utf-8")
        )
        entries = lock.get("adapter_files", {})
        implementations = [
            value
            for key, value in entries.items()
            if key.startswith(f"{action_name}.implementation.")
        ]
        if not implementations:
            raise ValueError(
                f"No locked implementation closure exists for action {action_name!r}"
            )
        by_name = {
            Path(value["path"]).name: value["sha256"] for value in implementations
        }
        if len(by_name) != len(implementations):
            raise ValueError(
                f"Implementation files for {action_name!r} have duplicate basenames"
            )
        if len(by_name) == 1:
            return next(iter(by_name.values()))
        return canonical_digest(by_name)

    def _score(self, state: dict[str, Any], round_index: int, round_dir: Path) -> Path:
        output_dir = round_dir / "score"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = "grit_scores.parquet" if self.config.strategy == "grit_score" else "task_scores.parquet"
        output_path = output_dir / output_name
        commit_path = output_dir / "score_commit.json"
        values = {
            "checkpoint": state.get("round_inputs", {}).get(
                str(round_index), state
            )["current_checkpoint"],
            "target_manifest": Path(self.config.value["data"]["target_manifest"]).resolve(),
            "output_dir": output_dir,
            "round": round_index,
        }
        values.update(self.config.value["actions"]["score"]["parameters"])
        request_path = output_dir / "score_request.json"
        score_request = {
            "schema_version": "1.0",
            "strategy": self.config.strategy,
            "round": round_index,
            "target_manifest": _file_identity(values["target_manifest"]),
            "checkpoint": _checkpoint_fingerprint(Path(values["checkpoint"])),
            "base_spec": _file_identity(
                Path(self.config.value["training"]["base_spec"])
            ),
            "settings": self.config.value["actions"]["score"].get(
                "settings", {}
            ),
            "parameters": self.config.value["actions"]["score"].get(
                "parameters", {}
            ),
            "entrypoint_sha256": self._locked_action_entrypoint("score"),
            "implementation_sha256": self._locked_action_implementation(
                "score"
            ),
        }
        _atomic_json(request_path, score_request)
        request_sha256 = _sha256(request_path)
        if self.config.strategy == "grit_score":
            score_config = output_dir / "grit_score.yaml"
            score_config.write_text(
                yaml.safe_dump(
                    {
                        "input_parquet": str(values["target_manifest"]),
                        "output_dir": str(output_dir),
                        "checkpoint": str(values["checkpoint"]),
                        "base_spec": str(self.config.value["training"]["base_spec"]),
                        "request_sha256": request_sha256,
                        **self.config.value["actions"]["score"].get("settings", {}),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            values["score_config"] = score_config
        command = self._command(self.config.value["actions"]["score"]["command"], values)

        def validate_scores() -> None:
            frame = pd.read_parquet(output_path, pre_buffer=False)
            score_column = (
                "grit_score"
                if self.config.strategy == "grit_score"
                else "weakness_score"
            )
            required = {"sample_id", "task", score_column, "embedding"}
            missing = required.difference(frame.columns)
            if missing:
                raise StageFailure("score", f"Score columns missing: {sorted(missing)}")
            if frame.empty:
                raise StageFailure("score", "Score output is empty")
            if frame[list(required)].isnull().any().any():
                raise StageFailure("score", "Score required columns contain null values")
            if frame.duplicated(["sample_id", "task"]).any():
                raise StageFailure("score", "Score sample/task identities are not unique")
            numeric = pd.to_numeric(frame[score_column], errors="coerce").to_numpy()
            if not np.isfinite(numeric).all():
                raise StageFailure("score", f"{score_column} must contain finite numbers")
            try:
                embeddings = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise StageFailure("score", "Score embeddings are not a numeric matrix") from exc
            if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
                raise StageFailure("score", "Score embeddings must be a finite 2D matrix")
            expected_identities = _target_identity_set(
                Path(self.config.value["data"]["target_manifest"]),
                self.config.strategy,
            )
            actual_identities = set(
                zip(frame["sample_id"].astype(str), frame["task"].astype(str))
            )
            if actual_identities != expected_identities:
                raise StageFailure(
                    "score",
                    "Score output does not exactly cover target sample/task identities: "
                    f"missing={len(expected_identities - actual_identities)}, "
                    f"extra={len(actual_identities - expected_identities)}",
                )
            _validate_score_embedding_binding(
                Path(self.config.value["data"]["target_manifest"]),
                self.config.strategy,
                frame,
                embeddings,
            )
            target_contract_path = self.config.value["data"].get(
                "target_embedding_contract"
            )
            if target_contract_path:
                target_contract, _ = _json_snapshot(Path(target_contract_path))
                if int(target_contract.get("embedding_dim", -1)) != int(
                    embeddings.shape[1]
                ):
                    raise StageFailure(
                        "score",
                        "Score embedding dimension differs from the target contract",
                    )
            if self.config.strategy == "multi_task_round_robin":
                actual = set(frame["task"].astype(str))
                expected = set(map(str, self.config.value["multi_task"]["tasks"]))
                if actual != expected:
                    raise StageFailure(
                        "score",
                        "Task-score output does not match multi_task.tasks: "
                        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
                    )
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            expected_commit = {
                "input_sha256": score_request["target_manifest"]["sha256"],
                "checkpoint_sha256": score_request["checkpoint"]["sha256"],
                "output_sha256": _sha256(output_path),
                "request_sha256": request_sha256,
                "entrypoint_sha256": score_request["entrypoint_sha256"],
                "implementation_sha256": score_request[
                    "implementation_sha256"
                ],
            }
            if any(commit.get(name) != value for name, value in expected_commit.items()):
                raise StageFailure(
                    "score", "Score commit does not bind the current request"
                )
            marker = output_dir / "_SUCCESS"
            if marker.read_text(encoding="utf-8").strip() != _sha256(commit_path):
                raise StageFailure("score", "Score success marker does not seal its commit")

        self._run_stage(
            state,
            round_index=round_index,
            stage="score",
            command=command,
            output_paths={
                "scores": output_path,
                "commit": commit_path,
                "success": output_dir / "_SUCCESS",
                "request": request_path,
            },
            stage_environment={
                "TAO_REFINEMENT_REQUEST_SHA256": request_sha256,
                "TAO_REFINEMENT_ENTRYPOINT_SHA256": score_request[
                    "entrypoint_sha256"
                ],
                "TAO_REFINEMENT_IMPLEMENTATION_SHA256": score_request[
                    "implementation_sha256"
                ],
            },
            validate_outputs=validate_scores,
        )
        return output_path

    def _select(
        self,
        state: dict[str, Any],
        round_index: int,
        round_dir: Path,
        scores: Path,
    ) -> Path:
        output_dir = round_dir / "select_targets"
        selection = output_dir / "selected_targets.parquet"
        suppressed = self._suppressed_path(
            state.get("round_inputs", {}).get(str(round_index), state)
        )
        data_command = list(
            map(str, self.config.value["actions"]["data"]["command"])
        )
        if self.config.strategy == "grit_score":
            command = [
                *data_command,
                "select-grit",
                "--scores",
                str(scores),
                "--fraction",
                str(self.config.value["grit"]["target_fraction"]),
            ]
        else:
            multitask = self.config.value["multi_task"]
            total = sum(multitask_budgets(multitask).values())
            command = [
                *data_command,
                "select-multitask",
                "--scores",
                str(scores),
                "--total",
                str(total),
            ]
            if multitask.get("task_weights"):
                command.extend(
                    [
                        "--task-weights-json",
                        json.dumps(multitask["task_weights"], sort_keys=True),
                    ]
                )
            if multitask["policy"] == "balanced":
                command.extend(
                    [
                        "--configured-tasks-json",
                        json.dumps(multitask["tasks"], sort_keys=True),
                        "--preserve-unfilled-budget",
                    ]
                )
        command.extend(
            ["--exclude-targets", str(suppressed), "--output-dir", str(output_dir)]
        )

        def validate_selection() -> None:
            artifact = _validate_output_artifact(
                output_dir,
                artifact_type="target_selection",
                payload_files={},
            )
            identity = _file_identity(selection)
            if any(
                artifact["payload"].get(name) != identity[name]
                for name in ("uri", "bytes", "sha256")
            ):
                raise ValueError(
                    f"Target selection payload changed after commit: {selection}"
                )

        self._run_stage(
            state,
            round_index=round_index,
            stage="select_targets",
            command=command,
            output_paths={
                "selection": selection,
                "artifact": output_dir / "artifact.json",
                "success": output_dir / "_SUCCESS",
            },
            commit_state=lambda current: self._update_persistence(
                current, selection, round_index
            ),
            validate_outputs=validate_selection,
        )
        return selection

    def _validate_search_input_lineage(self, artifact: dict[str, Any]) -> None:
        """Require the search leaf to prove it used the locked corpus."""
        lock = json.loads(
            (self.run_dir / "data.lock.json").read_text(encoding="utf-8")
        )
        inputs = artifact.get("inputs", [])
        if not isinstance(inputs, list):
            raise ValueError("Search artifact inputs must be a list")

        def require_input(expected: dict[str, Any], label: str) -> None:
            if not any(
                isinstance(item, dict) and
                all(item.get(name) == value for name, value in expected.items())
                for item in inputs
            ):
                raise ValueError(
                    f"Search artifact does not bind locked {label} identity"
                )

        search = self.config.value["actions"]["search"]
        data = self.config.value["data"]
        if search["backend"] == "audited_ann":
            dense = lock["indexed_search"]["dense_store_manifest"]
            require_input(
                {**dense, "role": "dense_vector_store"},
                "dense vector store",
            )
            require_input(
                {
                    **lock["indexed_search"]["ann_index_manifest"],
                    "role": "ann_index",
                },
                "ANN index",
            )
            require_input(
                {
                    **lock["indexed_search"]["ann_audit_manifest"],
                    "role": "ann_recall_audit",
                },
                "ANN recall audit",
            )
        elif data.get("source_store_manifest"):
            source = lock["source_store"]
            require_input(
                {
                    **source["manifest"],
                    "role": "source_store_manifest",
                    "artifact_id": source["artifact_id"],
                    "inventory_digest": source["declared_inventory_digest"],
                },
                "source store",
            )
        else:
            for index, shard in enumerate(lock["source_store"]["shards"]):
                require_input(
                    {**shard, "role": "source_shard"},
                    f"source shard {index}",
                )
        if data.get("source_store_manifest"):
            require_input(
                {
                    **lock["target_embedding_contract"],
                    "role": "query_embedding_contract",
                },
                "query embedding contract",
            )

    def _search(
        self,
        state: dict[str, Any],
        round_index: int,
        round_dir: Path,
        queries: Path,
    ) -> tuple[Path, dict[str, Any]]:
        output_dir = round_dir / "search"
        neighbors = output_dir / "neighbors.parquet"
        summary_path = output_dir / "search_summary.json"
        action = self.config.value["actions"]["search"]
        mining = self.config.value["mining"]
        data = self.config.value["data"]
        exclusion_manifest = self._exclusion_manifest(state, round_index)
        search_resources = None
        if action["backend"] == "audited_ann":
            candidate_dir = round_dir / "search_candidates"
            candidate_path = candidate_dir / "ann_candidates.npz"
            candidate_command = [
                *map(str, action["candidate"]["command"]),
                "ann-candidates",
                "--queries",
                str(queries),
                "--ann-index-manifest",
                str(action["ann_index_manifest"]),
                "--query-embedding-contract",
                str(data["target_embedding_contract"]),
                "--ann-audit-manifest",
                str(action["ann_audit_manifest"]),
                "--n-probes",
                str(action["n_probes"]),
                "--ann-candidates",
                str(action["ann_candidates"]),
                "--output-dir",
                str(candidate_dir),
            ]

            def validate_candidates() -> None:
                artifact = _validate_output_artifact(
                    candidate_dir,
                    artifact_type="ann_candidates",
                    payload_files={"candidates": candidate_path},
                )
                candidate_summary = json.loads(
                    (candidate_dir / "candidate_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                if int(candidate_summary["n_probes"]) != int(action["n_probes"]):
                    raise ValueError("ANN candidate n_probes changed")
                if int(candidate_summary["candidate_count"]) != int(
                    action["ann_candidates"]
                ):
                    raise ValueError("ANN candidate depth changed")
                if artifact["payload"] != candidate_summary:
                    raise ValueError("ANN candidate summary differs from its artifact")
                indexed = self._data_lock()["indexed_search"]
                expected = {
                    "index_artifact_id": indexed["ann_index_artifact_id"],
                    "audit_artifact_id": indexed["ann_audit_artifact_id"],
                    "dense_store_artifact_id": indexed[
                        "dense_store_artifact_id"
                    ],
                }
                for name, value in expected.items():
                    if candidate_summary.get(name) != value:
                        raise ValueError(
                            f"ANN candidate {name} differs from data.lock"
                        )

            self._run_stage(
                state,
                round_index=round_index,
                stage="search_candidates",
                command=candidate_command,
                output_paths={
                    "candidates": candidate_path,
                    "summary": candidate_dir / "candidate_summary.json",
                    "artifact": candidate_dir / "artifact.json",
                    "success": candidate_dir / "_SUCCESS",
                },
                resources=action["candidate"]["resources"],
                validate_outputs=validate_candidates,
            )
            command = [
                *map(str, action["rerank"]["command"]),
                "ann-rerank",
                "--queries",
                str(queries),
                "--candidates",
                str(candidate_path),
                "--dense-store-manifest",
                str(action["dense_store_manifest"]),
                "--ann-index-manifest",
                str(action["ann_index_manifest"]),
                "--ann-audit-manifest",
                str(action["ann_audit_manifest"]),
                "--query-embedding-contract",
                str(data["target_embedding_contract"]),
                "--top-k",
                str(mining["top_k_per_target"]),
                "--min-similarity",
                str(mining["min_similarity"]),
                "--hard-min-similarity",
                str(mining["hard_min_similarity"]),
                "--similarity-step",
                str(mining["similarity_step"]),
                "--duplicate-similarity",
                str(mining["duplicate_similarity"]),
                "--candidate-multiplier",
                str(mining["candidate_multiplier"]),
                "--device",
                str(action["rerank"].get("device", "cuda:0")),
                "--output-dir",
                str(output_dir),
            ]
            if exclusion_manifest:
                command.extend(["--exclude", str(exclusion_manifest)])
            search_resources = action["rerank"]["resources"]
        elif action["backend"] == "custom":
            values = {
                "queries": queries,
                "source_store_manifest": data.get("source_store_manifest", ""),
                "target_embedding_contract": data.get(
                    "target_embedding_contract", ""
                ),
                "exclude_manifest": exclusion_manifest or "",
                "top_k": mining["top_k_per_target"],
                "min_similarity": mining["min_similarity"],
                "hard_min_similarity": mining["hard_min_similarity"],
                "similarity_step": mining["similarity_step"],
                "duplicate_similarity": mining["duplicate_similarity"],
                "candidate_multiplier": mining["candidate_multiplier"],
                "output_dir": output_dir,
                "round": round_index,
                **action["parameters"],
            }
            command = self._command(action["command"], values)
        else:
            data_command = list(
                map(str, self.config.value["actions"]["data"]["command"])
            )
            command = [
                *data_command,
                "exact-search",
                "--queries",
                str(queries),
                "--top-k",
                str(mining["top_k_per_target"]),
                "--min-similarity",
                str(mining["min_similarity"]),
                "--hard-min-similarity",
                str(mining["hard_min_similarity"]),
                "--similarity-step",
                str(mining["similarity_step"]),
                "--duplicate-similarity",
                str(mining["duplicate_similarity"]),
                "--candidate-multiplier",
                str(mining["candidate_multiplier"]),
                "--output-dir",
                str(output_dir),
            ]
            for source_part in self._source_parts():
                command.extend(["--source-part", str(source_part)])
            if data.get("source_store_manifest"):
                command.extend(
                    [
                        "--source-store-manifest",
                        str(data["source_store_manifest"]),
                        "--query-embedding-contract",
                        str(data["target_embedding_contract"]),
                    ]
                )
            if exclusion_manifest:
                command.extend(["--exclude", str(exclusion_manifest)])

        def validate_search() -> None:
            artifact = _validate_output_artifact(
                output_dir,
                artifact_type="neighbor_selection",
                payload_files={"neighbors": neighbors, "summary": summary_path},
            )
            self._validate_search_input_lineage(artifact)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            required = {
                "row_count",
                "eligible_source_rows",
                "search_proof",
                "adaptive_radius",
                "underfill_exhaustion_proven",
            }
            missing = required.difference(summary)
            if missing:
                raise ValueError(f"Search summary is missing fields: {sorted(missing)}")
            proof = str(summary["search_proof"])
            if not _is_supported_search_proof(proof):
                raise ValueError(f"Unsupported search proof: {proof}")
            adaptive = summary["adaptive_radius"]
            if not isinstance(adaptive.get("query_stats"), list):
                raise ValueError("Search summary has no per-query adaptive-radius audit")
            if not isinstance(summary["underfill_exhaustion_proven"], bool):
                raise ValueError("Search summary has no Boolean underfill proof")
            if artifact["payload"].get("search_proof") != proof:
                raise ValueError("Search artifact proof differs from its summary")
        self._run_stage(
            state,
            round_index=round_index,
            stage="search",
            command=command,
            output_paths={
                "neighbors": neighbors,
                "summary": summary_path,
                "artifact": output_dir / "artifact.json",
                "success": output_dir / "_SUCCESS",
            },
            resources=search_resources,
            validate_outputs=validate_search,
        )
        return neighbors, json.loads(summary_path.read_text(encoding="utf-8"))

    def _materialize(
        self,
        state: dict[str, Any],
        round_index: int,
        round_dir: Path,
        delta: Path,
        targets: Path,
    ) -> tuple[Path, Path]:
        output_dir = round_dir / "materialize"
        manifest = output_dir / "training_manifest.parquet"
        training_view = manifest
        data_command = list(
            map(str, self.config.value["actions"]["data"]["command"])
        )
        command = [
            *data_command,
            "materialize",
            "--delta",
            str(delta),
            "--output-dir",
            str(output_dir),
        ]
        previous_manifest = state.get("round_inputs", {}).get(
            str(round_index), state
        )["current_training_manifest"] or self.config.value[
            "data"
        ].get("previous_training_manifest")
        if previous_manifest:
            command.extend(["--previous", previous_manifest])
        balanced = (
            self.config.strategy == "multi_task_round_robin" and
            self.config.value["multi_task"]["policy"] == "balanced"
        )
        if balanced:
            training_view = output_dir / "balanced_training_manifest.parquet"
            command.extend(
                [
                    "--query-manifest",
                    str(targets),
                    "--balance-column",
                    "query_task",
                ]
            )

        def validate_manifest() -> None:
            artifact = _validate_output_artifact(
                output_dir,
                artifact_type="training_manifest",
                payload_files={},
            )
            payload = artifact["payload"]
            identity = _file_identity(manifest)
            if payload.get("manifest_uri") != identity["uri"]:
                raise ValueError("Training manifest URI changed after commit")
            if payload.get("manifest_sha256") != identity["sha256"]:
                raise ValueError("Training manifest payload changed after commit")
            if int(payload.get("row_count", -1)) != len(pd.read_parquet(manifest, pre_buffer=False)):
                raise ValueError("Training manifest row count changed after commit")
            for name in ("row_count", "delta_rows", "replayed_rows", "previous_rows"):
                if not isinstance(payload.get(name), int) or payload[name] < 0:
                    raise ValueError(
                        f"Training manifest requires nonnegative integer {name}"
                    )
            if payload["previous_rows"] + payload["delta_rows"] != payload["row_count"]:
                raise ValueError(
                    "Training manifest transition counts are inconsistent"
                )
            if balanced:
                view_identity = _file_identity(training_view)
                if payload.get("training_view") != view_identity:
                    raise ValueError("Balanced training view changed after commit")
                if int(payload.get("training_view_rows", -1)) != int(
                    pq.ParquetFile(training_view).metadata.num_rows
                ):
                    raise ValueError("Balanced training view row count changed")
                balance = payload.get("balance", {})
                if balance.get("column") != "query_task":
                    raise ValueError("Balanced training view lost task provenance")

        self._run_stage(
            state,
            round_index=round_index,
            stage="materialize",
            command=command,
            output_paths={
                "training_manifest": manifest,
                **({"balanced_training_manifest": training_view} if balanced else {}),
                "artifact": output_dir / "artifact.json",
                "success": output_dir / "_SUCCESS",
            },
            commit_state=lambda current: current.update(
                {"current_training_manifest": str(manifest)}
            ),
            validate_outputs=validate_manifest,
        )
        return manifest, training_view

    def _train(
        self,
        state: dict[str, Any],
        round_index: int,
        round_dir: Path,
        training_manifest: Path,
    ) -> Path:
        output_dir = round_dir / "train"
        checkpoint_template = self.config.value["actions"]["train"].get(
            "checkpoint", "{output_dir}/checkpoint.pth"
        )
        checkpoint_policy = self.config.value["training"]["checkpoint_policy"]
        initialization_checkpoint = (
            state.get("round_inputs", {}).get(str(round_index), state)["current_checkpoint"]
            if checkpoint_policy == "previous_round_checkpoint"
            else str(
                Path(self.config.value["model"]["base_checkpoint"])
                .expanduser()
                .absolute()
            )
        )
        base_spec_path = Path(
            self.config.value["training"]["base_spec"]
        ).expanduser()
        base_spec = yaml.safe_load(
            base_spec_path.read_text(encoding="utf-8")
        ) or {}
        try:
            batch_size_per_gpu = int(base_spec["dataset"]["batch_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "training.base_spec must define a positive dataset.batch_size"
            ) from error
        train_resources = self.config.value["actions"]["train"].get(
            "resources", {}
        )
        allocation = training_allocation(
            self.config.value["training"],
            manifest_rows=int(pq.ParquetFile(training_manifest).metadata.num_rows),
            batch_size_per_gpu=batch_size_per_gpu,
            fixed_nodes=int(train_resources.get("nodes", 1)),
            fixed_gpus_per_node=int(
                train_resources.get("gpus_per_node", 1)
            ),
        )
        reference_world_size = int(train_resources.get("nodes", 1)) * int(
            train_resources.get("gpus_per_node", 1)
        )
        allocation["lr_scaling_rule"] = self.config.value["training"][
            "lr_scaling_rule"
        ]
        allocation["lr_reference_world_size"] = reference_world_size
        allocation_path = output_dir / "training_allocation.json"
        _atomic_json(allocation_path, allocation)
        values = {
            "checkpoint": initialization_checkpoint,
            "training_manifest": training_manifest,
            "base_spec": self.config.value["training"].get("base_spec", ""),
            "output_dir": output_dir,
            "round": round_index,
            "passes": self.config.value["training"]["passes_per_round"],
            "training_nodes": allocation["nodes"],
            "training_gpus_per_node": allocation["gpus_per_node"],
            "training_world_size": allocation["world_size"],
            "training_optimizer_updates": allocation["total_optimizer_steps"],
            "training_lr_scaling_rule": allocation["lr_scaling_rule"],
            "training_lr_reference_world_size": reference_world_size,
            "checkpoint_policy": checkpoint_policy,
        }
        checkpoint = Path(str(checkpoint_template).format_map(_FormatValues(values)))
        contract_template = self.config.value["actions"]["train"]["contract"]
        contract_path = Path(
            str(contract_template).format_map(_FormatValues(values))
        )
        commit_path = contract_path.with_name("training_commit.json")
        command = self._command(self.config.value["actions"]["train"]["command"], values)

        def validate_checkpoint() -> None:
            self._validate_training_outputs(
                checkpoint,
                contract_path,
                commit_path,
                output_dir / "_SUCCESS",
            )

        self._run_stage(
            state,
            round_index=round_index,
            stage="train",
            command=command,
            output_paths={
                "checkpoint": checkpoint,
                "contract": contract_path,
                "commit": commit_path,
                "allocation": allocation_path,
                "success": output_dir / "_SUCCESS",
            },
            resources={
                **train_resources,
                "nodes": allocation["nodes"],
                "gpus_per_node": allocation["gpus_per_node"],
                "world_size": allocation["world_size"],
            },
            validate_outputs=validate_checkpoint,
            commit_state=lambda current: current.update(
                {"current_checkpoint": str(checkpoint)}
            ),
        )
        return checkpoint

    @staticmethod
    def _validate_training_outputs(
        checkpoint: Path,
        contract_path: Path,
        commit_path: Path,
        marker: Path,
    ) -> None:
        """Verify that a published checkpoint still matches its commit record."""
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        recorded = Path(str(contract.get("checkpoint", ""))).resolve()
        if recorded != checkpoint.resolve():
            raise ValueError(
                "Training contract checkpoint does not match the configured output"
            )
        if int(contract.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size:
            raise ValueError("Published checkpoint size does not match its contract")
        expected = str(contract.get("checkpoint_sha256", ""))
        if not expected.startswith("sha256:") or _sha256(checkpoint) != expected:
            raise ValueError("Published checkpoint digest does not match its contract")
        runtime_spec = Path(str(contract.get("runtime_spec", "")))
        runtime_expected = str(contract.get("runtime_spec_sha256", ""))
        if (
            not runtime_spec.is_file() or
            runtime_spec.stat().st_size != int(contract.get("runtime_spec_bytes", -1)) or
            _sha256(runtime_spec) != runtime_expected
        ):
            raise ValueError("TAO runtime spec does not match its contract")
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit_expectations = {
            "training_contract_sha256": _sha256(contract_path),
            "checkpoint_sha256": expected,
            "runtime_spec_sha256": runtime_expected,
        }
        if any(
            commit.get(name) != value
            for name, value in commit_expectations.items()
        ):
            raise ValueError("Training commit does not seal the final contract")
        if marker.read_text(encoding="utf-8").strip() != _sha256(commit_path):
            raise ValueError("Training success marker does not seal the final commit")

    def _validate_completed_training(self, state: dict[str, Any]) -> None:
        for key, record in state.get("completed_stages", {}).items():
            if not key.endswith("/train"):
                continue
            outputs = record.get("outputs", {})
            required = {"checkpoint", "contract", "commit", "success"}
            if missing := required.difference(outputs):
                raise RuntimeError(
                    f"Cached training stage {key} lacks committed outputs: {sorted(missing)}"
                )
            self._validate_training_outputs(
                Path(outputs["checkpoint"]),
                Path(outputs["contract"]),
                Path(outputs["commit"]),
                Path(outputs["success"]),
            )

    def _validate_adopted_training(self) -> dict[str, Any]:
        """Validate an immutable partial-round training result for continuation."""
        continuation = self.config.value.get("continuation")
        if not continuation:
            raise ValueError("This workflow has no continuation contract")
        adopted_round = int(continuation["adopted_round"])
        checkpoint = Path(
            self.config.value["model"]["initial_scoring_checkpoint"]
        ).expanduser().resolve()
        manifest = Path(
            self.config.value["data"]["previous_training_manifest"]
        ).expanduser().resolve()
        contract_path = Path(continuation["training_contract"]).expanduser().resolve()
        commit_path = Path(continuation["training_commit"]).expanduser().resolve()
        marker = Path(continuation["success_marker"]).expanduser().resolve()
        previous_run = Path(continuation["previous_run_dir"]).expanduser().resolve()
        if self.run_dir == previous_run:
            raise ValueError("Continuation requires a new output.run_dir")
        previous_data_lock = Path(
            continuation["previous_data_lock"]
        ).expanduser().resolve()
        previous_release_lock = Path(
            continuation["previous_release_lock"]
        ).expanduser().resolve()
        previous_state_path = previous_run / "state.json"
        round_root = previous_run / "rounds" / f"round_{adopted_round:03d}"
        expected_paths = {
            "checkpoint": round_root / "train/checkpoint.pth",
            "training_manifest": round_root / "materialize/training_manifest.parquet",
            "training_contract": round_root / "train/training_contract.json",
            "training_commit": round_root / "train/training_commit.json",
            "success_marker": round_root / "train/_SUCCESS",
            "previous_data_lock": previous_run / "data.lock.json",
            "previous_release_lock": previous_run / "release.lock.json",
            "previous_state": previous_state_path,
        }
        actual_paths = {
            "checkpoint": checkpoint,
            "training_manifest": manifest,
            "training_contract": contract_path,
            "training_commit": commit_path,
            "success_marker": marker,
            "previous_data_lock": previous_data_lock,
            "previous_release_lock": previous_release_lock,
            "previous_state": previous_state_path,
        }
        for name, expected_path in expected_paths.items():
            expected = expected_path.resolve()
            if actual_paths[name] != expected:
                raise ValueError(
                    f"Continuation {name} must be {expected}; got {actual_paths[name]}"
                )
        self._validate_training_outputs(
            checkpoint, contract_path, commit_path, marker
        )
        previous_state = json.loads(previous_state_path.read_text(encoding="utf-8"))
        completed_training_key = f"round_{adopted_round:03d}/train"
        if completed_training_key not in previous_state.get("completed_stages", {}):
            raise ValueError(
                "Continuation state does not contain the adopted training stage: "
                f"{completed_training_key}"
            )
        persistent_targets = previous_state.get("persistent_targets", {})
        if not isinstance(persistent_targets, dict):
            raise ValueError("Continuation persistent_targets must be a mapping")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        training_input_manifest = Path(
            str(contract.get("manifest", ""))
        ).expanduser().resolve()
        if training_input_manifest != manifest:
            balanced_manifest = (
                round_root / "materialize/balanced_training_manifest.parquet"
            ).resolve()
            if training_input_manifest != balanced_manifest:
                raise ValueError(
                    "Continuation training input must be the unique lineage "
                    "manifest or its balanced training view"
                )
        if not training_input_manifest.is_file():
            raise ValueError(
                "Continuation training input manifest is missing: "
                f"{training_input_manifest}"
            )
        manifest_rows = int(
            pq.ParquetFile(training_input_manifest).metadata.num_rows
        )
        base_spec = Path(self.config.value["training"]["base_spec"]).expanduser()
        base_spec_value = yaml.safe_load(base_spec.read_text(encoding="utf-8")) or {}
        batch_size = int(base_spec_value["dataset"]["batch_size"])
        resources = self.config.value["actions"]["train"].get("resources", {})
        allocation = training_allocation(
            self.config.value["training"],
            manifest_rows=manifest_rows,
            batch_size_per_gpu=batch_size,
            fixed_nodes=int(resources.get("nodes", 1)),
            fixed_gpus_per_node=int(resources.get("gpus_per_node", 1)),
        )
        passes = int(self.config.value["training"]["passes_per_round"])
        parent_checkpoint = Path(self.config.value["model"]["base_checkpoint"])
        if (
            self.config.value["training"]["checkpoint_policy"] ==
            "previous_round_checkpoint" and
            adopted_round > 1
        ):
            previous_key = f"round_{adopted_round - 1:03d}/train"
            previous_outputs = previous_state.get("completed_stages", {}).get(
                previous_key, {}
            ).get("outputs", {})
            configured_parent = previous_outputs.get("checkpoint")
            if not configured_parent:
                raise ValueError(
                    "Warm-start continuation has no sealed preceding checkpoint: "
                    f"{previous_key}"
                )
            parent_checkpoint = Path(configured_parent)
        expected = {
            "manifest": str(training_input_manifest),
            "manifest_sha256": _sha256(training_input_manifest),
            "manifest_rows": manifest_rows,
            "base_spec_sha256": _sha256(base_spec),
            "parent_checkpoint_sha256": _checkpoint_fingerprint(
                parent_checkpoint
            )["sha256"],
            "requested_data_passes": passes,
            "num_nodes": allocation["nodes"],
            "gpus_per_node": allocation["gpus_per_node"],
            "world_size": allocation["world_size"],
            "total_optimizer_steps": allocation["total_optimizer_steps"],
            "round_checkpoint_policy": "final_ema_teacher",
            "source_checkpoint_name": (
                f"teacher_epoch_{passes - 1:03d}_step_"
                f"{allocation['total_optimizer_steps']:05d}.pth"
            ),
        }
        mismatches = {
            name: {"expected": value, "actual": contract.get(name)}
            for name, value in expected.items()
            if contract.get(name) != value
        }
        if Path(str(contract.get("checkpoint", ""))).resolve() != checkpoint:
            mismatches["checkpoint"] = {
                "expected": str(checkpoint),
                "actual": contract.get("checkpoint"),
            }
        if mismatches:
            raise ValueError(
                "Continuation training contract does not match this workflow: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )

        previous_lock = json.loads(previous_data_lock.read_text(encoding="utf-8"))
        current_lock = json.loads(
            (self.run_dir / "data.lock.json").read_text(encoding="utf-8")
        )
        source_lineage_transition = _validate_continuation_source_lineage(
            previous_lock=previous_lock,
            current_lock=current_lock,
            source_store_manifest=self.config.value["data"].get(
                "source_store_manifest"
            ),
        )
        return {
            "schema_version": "1.0",
            "adopted_round": adopted_round,
            "previous_run_dir": str(previous_run),
            "checkpoint": _file_identity(checkpoint),
            "training_manifest": {
                **_file_identity(manifest),
                "rows": int(pq.ParquetFile(manifest).metadata.num_rows),
            },
            "training_input_manifest": {
                **_file_identity(training_input_manifest),
                "rows": manifest_rows,
            },
            "training_contract": _file_identity(contract_path),
            "training_commit": _file_identity(commit_path),
            "success_marker": _file_identity(marker),
            "previous_data_lock": _file_identity(
                previous_data_lock
            ),
            "previous_release_lock": _file_identity(
                previous_release_lock
            ),
            "previous_state": _file_identity(previous_state_path),
            "persistent_targets": persistent_targets,
            "source_lineage_transition": {
                **source_lineage_transition,
            },
            "new_data_lock": _file_identity(self.run_dir / "data.lock.json"),
            "new_release_lock": _file_identity(self.run_dir / "release.lock.json"),
            "allocation": allocation,
        }

    def adopt_training(self) -> dict[str, Any]:
        """Adopt a sealed training result without invoking any training runner."""
        self.validate(require_paths=True)
        with self.store.controller_lock():
            self._source_store_verification = None
            state = self._initialize()
            adoption = self._validate_adopted_training()
            round_index = int(adoption["adopted_round"])
            key = f"round_{round_index:03d}/train"
            adoption_path = self.run_dir / "continuation_adoption.json"
            if key in state["completed_stages"]:
                if not adoption_path.is_file():
                    raise RuntimeError("Adopted training state lacks its provenance record")
                existing = json.loads(adoption_path.read_text(encoding="utf-8"))
                if canonical_digest(existing) != canonical_digest(adoption):
                    raise RuntimeError("Continuation adoption provenance changed")
                return state
            if state["completed_stages"] or state["completed_rounds"]:
                raise RuntimeError("Training adoption requires a fresh continuation run")
            state.update(
                {
                    "current_round": round_index,
                    "current_checkpoint": str(
                        Path(self.config.value["model"]["initial_scoring_checkpoint"])
                        .expanduser()
                        .resolve()
                    ),
                    "current_training_manifest": str(
                        Path(self.config.value["data"]["previous_training_manifest"])
                        .expanduser()
                        .resolve()
                    ),
                    "adopted_partial_round": round_index,
                    "active_jobs": {},
                    "persistent_targets": adoption["persistent_targets"],
                }
            )
            _atomic_json(adoption_path, adoption)
            continuation = self.config.value["continuation"]
            self.store.complete_stage(
                state,
                round_index=round_index,
                stage="train",
                outputs={
                    "checkpoint": state["current_checkpoint"],
                    "contract": str(Path(continuation["training_contract"]).resolve()),
                    "commit": str(Path(continuation["training_commit"]).resolve()),
                    "success": str(Path(continuation["success_marker"]).resolve()),
                    "adoption": str(adoption_path),
                },
                job={"adopted": True, "previous_run": adoption["previous_run_dir"]},
            )
            return state

    def _evaluate(
        self,
        state: dict[str, Any],
        round_index: int,
        round_dir: Path,
        checkpoint: Path,
    ) -> Path | None:
        action = self.config.value["actions"]["evaluate"]
        if not action["command"]:
            return None
        output_dir = round_dir / "evaluate"
        values = {
            "checkpoint": checkpoint,
            "benchmark_manifest": Path(
                self.config.value["data"]["benchmark_manifest"]
            ).expanduser().resolve(),
            "output_dir": output_dir,
            "round": round_index,
            "evaluation_scope": action["scope"],
        }
        values.update(action["parameters"])
        metrics = Path(str(action["metrics"]).format_map(_FormatValues(values)))
        commit = Path(str(action["commit"]).format_map(_FormatValues(values)))
        marker = output_dir / "_SUCCESS"
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / "evaluation_request.json"
        evaluation_request = {
            "schema_version": "1.0",
            "round": round_index,
            "scope": action["scope"],
            "checkpoint": _checkpoint_fingerprint(checkpoint),
            "benchmark": _file_identity(values["benchmark_manifest"]),
            "parameters": action["parameters"],
            "entrypoint_sha256": self._locked_action_entrypoint("evaluate"),
            "implementation_sha256": self._locked_action_implementation(
                "evaluate"
            ),
        }
        _atomic_json(request_path, evaluation_request)
        request_sha256 = _sha256(request_path)
        command = self._command(action["command"], values)

        def validate_metrics() -> None:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "1.0":
                raise ValueError("Evaluation metrics require schema_version=1.0")
            if payload.get("evaluation_scope") != action["scope"]:
                raise ValueError(
                    "Evaluation metrics scope does not match the configured scope"
                )
            records = payload.get("metrics")
            if not isinstance(records, list) or not records:
                raise ValueError("Evaluation metrics must contain a non-empty metrics list")
            required = {"task", "name", "value", "higher_is_better", "sample_count"}
            for index, record in enumerate(records):
                missing = required.difference(record)
                if missing:
                    raise ValueError(
                        f"Evaluation metric {index} is missing fields: {sorted(missing)}"
                    )
                if (
                    isinstance(record["value"], bool) or
                    not isinstance(record["value"], (int, float)) or
                    not np.isfinite(float(record["value"]))
                ):
                    raise ValueError(f"Evaluation metric {index} value is not numeric")
                if not isinstance(record["higher_is_better"], bool):
                    raise ValueError(
                        f"Evaluation metric {index} higher_is_better is not boolean"
                    )
                if int(record["sample_count"]) <= 0:
                    raise ValueError(f"Evaluation metric {index} sample_count is not positive")
            identities = [
                (str(record["task"]), str(record["name"]))
                for record in records
            ]
            if len(identities) != len(set(identities)):
                raise ValueError("Evaluation task/name identities are not unique")
            metric_contract = {
                identity: (
                    bool(record["higher_is_better"]),
                    int(record["sample_count"]),
                )
                for identity, record in zip(identities, records)
            }
            baseline_metrics = (
                self.run_dir / "rounds/round_000/evaluate/metrics.json"
            )
            if round_index > 0:
                baseline = json.loads(
                    baseline_metrics.read_text(encoding="utf-8")
                )["metrics"]
                baseline_contract = {
                    (str(record["task"]), str(record["name"])): (
                        bool(record["higher_is_better"]),
                        int(record["sample_count"]),
                    )
                    for record in baseline
                }
                if metric_contract != baseline_contract:
                    raise ValueError(
                        "Evaluation metric coverage differs from the base checkpoint"
                    )
            seal = json.loads(commit.read_text(encoding="utf-8"))
            expected_seal = {
                "schema_version": "1.0",
                "evaluation_scope": action["scope"],
                "checkpoint": evaluation_request["checkpoint"],
                "metrics": _file_identity(metrics),
                "benchmark": evaluation_request["benchmark"],
                "request_sha256": request_sha256,
                "entrypoint_sha256": evaluation_request[
                    "entrypoint_sha256"
                ],
                "implementation_sha256": evaluation_request[
                    "implementation_sha256"
                ],
            }
            if any(seal.get(name) != value for name, value in expected_seal.items()):
                raise ValueError("Evaluation commit does not seal this request and metrics")
            if marker.read_text(encoding="utf-8").strip() != _sha256(commit):
                raise ValueError("Evaluation success marker does not seal its commit")
        self._run_stage(
            state,
            round_index=round_index,
            stage="evaluate",
            command=command,
            output_paths={
                "metrics": metrics,
                "commit": commit,
                "success": marker,
                "request": request_path,
            },
            stage_environment={
                "TAO_REFINEMENT_REQUEST_SHA256": request_sha256,
                "TAO_REFINEMENT_ENTRYPOINT_SHA256": evaluation_request[
                    "entrypoint_sha256"
                ],
                "TAO_REFINEMENT_IMPLEMENTATION_SHA256": evaluation_request[
                    "implementation_sha256"
                ],
            },
            validate_outputs=validate_metrics,
        )
        return metrics

    def _record_early_stopping(
        self,
        state: dict[str, Any],
        *,
        round_index: int,
        checkpoint: Path,
        training_manifest: str | None,
        metrics_path: Path | None,
    ) -> bool:
        """Record metric patience and the best evaluated checkpoint."""
        config = self.config.value["workflow"].get("early_stopping")
        if config is None:
            return False
        if metrics_path is None:
            raise RuntimeError("Early stopping requires evaluation metrics")

        tracker = state.setdefault(
            "early_stopping",
            {
                "history": {},
                "rounds_without_improvement": 0,
                "should_stop": False,
            },
        )
        round_key = f"round_{round_index:03d}"
        if round_key in tracker["history"]:
            return bool(tracker["history"][round_key]["should_stop"])

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        available = {
            (str(metric["task"]), str(metric["name"])): metric
            for metric in payload["metrics"]
        }
        components = []
        weighted_score = 0.0
        total_weight = 0.0
        for requested in config["metrics"]:
            identity = (requested["task"], requested["name"])
            if identity not in available:
                raise ValueError(
                    "Early-stopping metric is absent from evaluation output: "
                    f"{identity[0]}:{identity[1]}"
                )
            metric = available[identity]
            value = float(metric["value"])
            direction = 1.0 if metric["higher_is_better"] else -1.0
            weight = float(requested["weight"])
            contribution = direction * value
            weighted_score += weight * contribution
            total_weight += weight
            components.append(
                {
                    "task": identity[0],
                    "name": identity[1],
                    "value": value,
                    "higher_is_better": bool(metric["higher_is_better"]),
                    "weight": weight,
                }
            )
        score = weighted_score / total_weight

        best_score = tracker.get("best_score")
        if best_score is None or score > float(best_score):
            tracker.update(
                {
                    "best_score": score,
                    "best_round": round_index,
                    "best_checkpoint": str(checkpoint.expanduser().resolve()),
                    "best_training_manifest": (
                        str(Path(training_manifest).expanduser().resolve())
                        if training_manifest
                        else None
                    ),
                }
            )

        reference_score = tracker.get("reference_score")
        significant = (
            reference_score is None or
            score > float(reference_score) + float(config["min_delta"])
        )
        if significant:
            tracker["reference_score"] = score
            tracker["rounds_without_improvement"] = 0
        elif round_index > 0:
            tracker["rounds_without_improvement"] = int(
                tracker["rounds_without_improvement"]
            ) + 1
        should_stop = (
            round_index > 0 and
            int(tracker["rounds_without_improvement"]) >=
            int(config["patience"])
        )
        tracker["should_stop"] = should_stop
        tracker["history"][round_key] = {
            "score": score,
            "components": components,
            "significant_improvement": significant,
            "rounds_without_improvement": tracker[
                "rounds_without_improvement"
            ],
            "should_stop": should_stop,
        }
        self.store.save(state)
        self.store.append_event(
            state,
            round_index=round_index,
            stage="early_stopping",
            status="complete",
            extra={
                "score": score,
                "best_score": tracker["best_score"],
                "best_round": tracker["best_round"],
                "rounds_without_improvement": tracker[
                    "rounds_without_improvement"
                ],
                "should_stop": should_stop,
            },
        )
        return should_stop

    def _stop(self, state: dict[str, Any], reason: str) -> None:
        tracker = state.get("early_stopping", {})
        selected_checkpoint = tracker.get(
            "best_checkpoint", state["current_checkpoint"]
        )
        selected_manifest = tracker.get(
            "best_training_manifest", state["current_training_manifest"]
        )
        selected_round = tracker.get("best_round", state["current_round"])
        state["status"] = "complete"
        state["stop_reason"] = reason
        state["selected_checkpoint"] = selected_checkpoint
        state["selected_training_manifest"] = selected_manifest
        state["selected_round"] = selected_round
        self.store.save(state)
        self.store.append_event(
            state,
            round_index=int(state["current_round"]),
            stage="loop_stop",
            status="complete",
            extra={"stop_reason": reason},
        )
        _atomic_json(
            self.run_dir / "final_model.json",
            {
                "schema_version": "1.0",
                "run_id": state["run_id"],
                "strategy": self.config.strategy,
                "checkpoint": selected_checkpoint,
                "training_manifest": selected_manifest,
                "selected_round": selected_round,
                "latest_checkpoint": state["current_checkpoint"],
                "stop_reason": reason,
                "config_digest": state["config_digest"],
            },
        )
        self._render_report(state)

    def _render_report(self, state: dict[str, Any]) -> None:
        def stage_outputs(round_index: int, stage: str) -> dict[str, Any]:
            record = state["completed_stages"].get(
                f"round_{round_index:03d}/{stage}", {}
            )
            return record.get("outputs", {})

        def read_json(path_value: Any) -> dict[str, Any]:
            if not path_value:
                return {}
            path = Path(str(path_value))
            if not path.is_file():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}

        def read_parquet(path_value: Any) -> pd.DataFrame:
            if not path_value:
                return pd.DataFrame()
            try:
                return pd.read_parquet(path_value, pre_buffer=False)
            except (OSError, ValueError):
                return pd.DataFrame()

        round_indices = sorted(
            {
                int(key.split("/")[0].removeprefix("round_"))
                for key in state["completed_stages"]
                if key.startswith("round_")
            }
        )
        round_rows = []
        distribution_rows = []
        metric_rows = []
        baseline_payload = read_json(
            stage_outputs(0, "evaluate").get("metrics")
        )
        baseline_values = {
            (str(metric["task"]), str(metric["name"])): float(metric["value"])
            for metric in baseline_payload.get("metrics", [])
        }
        for round_index in round_indices:
            selection_outputs = stage_outputs(round_index, "select_targets")
            search_outputs = stage_outputs(round_index, "search")
            materialize_outputs = stage_outputs(round_index, "materialize")
            train_outputs = stage_outputs(round_index, "train")
            evaluate_outputs = stage_outputs(round_index, "evaluate")

            selection_path = selection_outputs.get("selection")
            selected = read_parquet(selection_path)
            neighbors_path = search_outputs.get("neighbors")
            neighbors = read_parquet(neighbors_path)
            selected_counts = (
                selected.groupby(selected["task"].astype(str)).size().to_dict()
                if not selected.empty and "task" in selected
                else {}
            )
            mined_tasks = pd.Series(dtype=str)
            for task_column in ("task", "query_task"):
                if task_column in neighbors:
                    mined_tasks = neighbors[task_column].astype(str)
                    break
            if (
                mined_tasks.empty and
                "query_id" in neighbors and
                {"sample_id", "task"}.issubset(selected.columns)
            ):
                query_tasks = selected.set_index("sample_id")["task"].astype(str)
                mined_tasks = neighbors["query_id"].astype(str).map(query_tasks)
            mined_counts = mined_tasks.dropna().value_counts().to_dict()
            for task in sorted(set(selected_counts) | set(mined_counts)):
                distribution_rows.append(
                    f'<tr><td>{round_index}</td><td>{html.escape(str(task))}</td><td>{int(selected_counts.get(task, 0))}</td><td>{int(mined_counts.get(task, 0))}</td></tr>'
                )

            similarities = (
                pd.to_numeric(neighbors["cosine_similarity"], errors="coerce")
                .dropna()
                .to_numpy()
                if "cosine_similarity" in neighbors
                else np.asarray([])
            )
            similarity_summary = (
                f'{float(np.min(similarities)):.3f} / {float(np.median(similarities)):.3f} / {float(np.max(similarities)):.3f}'
                if len(similarities)
                else "-"
            )
            materialize = read_json(materialize_outputs.get("artifact")).get(
                "payload", {}
            )
            allocation = read_json(train_outputs.get("allocation"))
            metrics = read_json(evaluate_outputs.get("metrics"))
            metric_summary = []
            for metric in metrics.get("metrics", []):
                name = f"{metric.get('task', 'all')}:{metric.get('name', 'metric')}"
                value = float(metric["value"])
                identity = (
                    str(metric.get("task", "all")),
                    str(metric.get("name", "metric")),
                )
                raw_delta = value - baseline_values.get(identity, value)
                gain = raw_delta if metric.get("higher_is_better", True) else -raw_delta
                metric_summary.append(f"{name}={value:.4g} (gain {gain:+.3g})")
                metric_rows.append(
                    f"<tr><td>{round_index}</td><td>{html.escape(str(metric.get('task', 'all')))}</td><td>{html.escape(str(metric.get('name', 'metric')))}</td><td>{value:.6g}</td><td>{gain:+.6g}</td><td>{('higher' if metric.get('higher_is_better', True) else 'lower')}</td><td>{int(metric.get('sample_count', 0))}</td></tr>"
                )
            allocation_summary = (
                f"{allocation.get('nodes', 0)} x "
                f"{allocation.get('gpus_per_node', 0)} GPU, "
                f"{allocation.get('total_optimizer_steps', 0)} updates"
                if allocation
                else "-"
            )
            round_rows.append(
                f"<tr><td>{round_index}</td><td>{len(selected)}</td><td>{len(neighbors)}</td><td>{similarity_summary}</td><td>{int(materialize.get('delta_rows', 0))}</td><td>{int(materialize.get('replayed_rows', 0))}</td><td>{html.escape(allocation_summary)}</td><td>{html.escape(', '.join(metric_summary) or '-')}</td></tr>"
            )

        stage_rows = []
        for key, record in sorted(state["completed_stages"].items()):
            stage_rows.append(
                f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(record.get('completed_at', 'unknown')))}</td><td><code>{html.escape(json.dumps(record.get('outputs', {}), sort_keys=True))}</code></td></tr>"
            )
        failures = []
        submissions: dict[tuple[int, str], int] = {}
        if self.store.events_path.is_file():
            for line in self.store.events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                key = (int(event.get("round", 0)), str(event.get("stage", "")))
                if event.get("status") == "submitted":
                    submissions[key] = submissions.get(key, 0) + 1
                if event.get("status") == "error":
                    failures.append(event)
        retry_count = sum(max(0, count - 1) for count in submissions.values())
        failure_rows = [
            f"<tr><td>{int(event.get('round', 0))}</td><td>{html.escape(str(event.get('stage', '')))}</td><td>{html.escape(str(event.get('timestamp', '')))}</td><td>{html.escape(str(event.get('error', '')))}</td></tr>"
            for event in failures
        ]
        early_stopping = state.get("early_stopping", {})
        early_stopping_rows = []
        for round_key, record in sorted(
            early_stopping.get("history", {}).items()
        ):
            components = ", ".join(
                f"{component['task']}:{component['name']}={component['value']:.6g}"
                for component in record.get("components", [])
            )
            early_stopping_rows.append(
                f"<tr><td>{int(round_key.removeprefix('round_'))}</td><td>{float(record['score']):.6g}</td><td>{html.escape(components)}</td><td>{int(record['rounds_without_improvement'])}</td><td>{('yes' if record['should_stop'] else 'no')}</td></tr>"
            )
        early_stopping_config = self.config.value["workflow"].get(
            "early_stopping"
        )
        early_stopping_note = (
            f"Weighted, direction-normalized score; patience={int(early_stopping_config['patience'])} round(s), minimum meaningful change={float(early_stopping_config['min_delta']):.6g}. The best evaluated checkpoint is retained for delivery."
            if early_stopping_config
            else "Disabled; terminal delivery uses the latest checkpoint."
        )
        document = f"""<!doctype html><html><head><meta charset="utf-8">\n<title>DINOv3 SSL DEFT Report</title><style>\nbody{{font:14px Arial,sans-serif;max-width:1400px;margin:32px auto;color:#202124;line-height:1.45}}\nh1{{font-size:28px;margin-bottom:4px}}h2{{font-size:19px;margin-top:30px;border-bottom:2px solid #333;padding-bottom:5px}}\n.summary{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));border:1px solid #bbb}}\n.summary div{{padding:12px;border-right:1px solid #bbb}}.summary div:last-child{{border-right:0}}\n.label{{font-size:11px;text-transform:uppercase;color:#666}}.value{{font-size:18px;font-weight:600}}\ntable{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #bbb;padding:7px;text-align:left;vertical-align:top}}\nth{{background:#f1f3f4}}code{{font-size:12px;overflow-wrap:anywhere}}.note{{color:#555}}@media print{{body{{margin:10mm}}}}\n</style></head><body>\n<h1>DINOv3 SSL DEFT</h1><p class="note">Run <code>{html.escape(state['run_id'])}</code> using <code>{html.escape(self.config.strategy)}</code>.</p>\n<div class="summary"><div><span class="label">Status</span><br><span class="value">{html.escape(state['status'])}</span></div>\n<div><span class="label">Stop reason</span><br><span class="value">{html.escape(str(state.get('stop_reason')))}</span></div>\n<div><span class="label">Completed rounds</span><br><span class="value">{len(state.get('completed_rounds', {}))}</span></div>\n<div><span class="label">Retries / failures</span><br><span class="value">{retry_count} / {len(failures)}</span></div>\n<div><span class="label">Checkpoint policy</span><br><span class="value">{html.escape(self.config.value['training']['checkpoint_policy'])}</span></div>\n<div><span class="label">Selected round</span><br><span class="value">{html.escape(str(state.get('selected_round', early_stopping.get('best_round', '-'))))}</span></div></div>\n<h2>Round Summary</h2><table><thead><tr><th>Round</th><th>Weak targets</th><th>Mined rows</th><th>Similarity min / median / max</th><th>Novel rows</th><th>Replay removed</th><th>Training</th><th>Evaluation</th></tr></thead><tbody>{''.join(round_rows)}</tbody></table>\n<h2>Task Distribution</h2><table><thead><tr><th>Round</th><th>Task</th><th>Weak targets</th><th>Mined rows</th></tr></thead><tbody>{''.join(distribution_rows)}</tbody></table>\n<h2>Evaluation Metrics</h2><p class="note">Round 0 is the immutable base checkpoint. Gain is direction-normalized versus that baseline. Scope: <strong>{html.escape(self.config.value['actions']['evaluate']['scope'])}</strong>. Diagnostic replay is not held-out evidence.</p><table><thead><tr><th>Round</th><th>Task</th><th>Metric</th><th>Value</th><th>Gain vs base</th><th>Direction</th><th>Samples</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>\n<h2>Early Stopping</h2><p class="note">{html.escape(early_stopping_note)}</p><table><thead><tr><th>Round</th><th>Score</th><th>Components</th><th>Rounds without improvement</th><th>Stop</th></tr></thead><tbody>{''.join(early_stopping_rows)}</tbody></table>\n<h2>Failures And Retries</h2><table><thead><tr><th>Round</th><th>Stage</th><th>Time</th><th>Error</th></tr></thead><tbody>{''.join(failure_rows)}</tbody></table>\n<h2>Artifact Trace</h2><table><thead><tr><th>Stage</th><th>Completed</th><th>Outputs</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></body></html>"""
        temporary = self.run_dir / "report.html.tmp"
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(self.run_dir / "report.html")

    def execute(self) -> dict[str, Any]:
        # Each terminal state exits explicitly; keep the state-machine boundaries visible.
        # pylint: disable=too-many-return-statements
        """Run or resume until a scientific stop condition is reached."""
        self.validate(require_paths=True)
        with self.store.controller_lock():
            # Validation happens before lock acquisition. Reverify all source
            # inputs here so waiting for another controller cannot create a
            # mixed data.lock, then retain those exact paths for every round.
            self._source_store_verification = None
            state = self._initialize()
            if state["status"] in {"complete", "canceled"}:
                if state["status"] == "complete":
                    self._validate_completed_training(state)
                return state
            if (
                state["status"] == "canceling" or
                (self.run_dir / "cancel.requested").exists()
            ):
                state, _ = self._reconcile_cancellation(state)
                return state
            if state["status"] == "failed":
                state["status"] = "running"
                state.pop("failure", None)
                self.store.save(state)
            # Recheck every committed checkpoint before any resumed downstream
            # stage can consume it.
            self._validate_completed_training(state)
            if self.config.value["actions"]["evaluate"]["command"]:
                baseline_metrics = self._evaluate(
                    state,
                    0,
                    self.run_dir / "rounds/round_000",
                    Path(self.config.value["model"]["base_checkpoint"])
                    .expanduser()
                    .resolve(),
                )
                self._record_early_stopping(
                    state,
                    round_index=0,
                    checkpoint=Path(
                        self.config.value["model"]["base_checkpoint"]
                    ),
                    training_manifest=None,
                    metrics_path=baseline_metrics,
                )
            configured_start = int(self.config.value["workflow"]["start_round"])
            if configured_start > 1 and not state.get("adopted_partial_round"):
                adopted_key = f"round_{configured_start:03d}/train"
                if adopted_key not in state.get("completed_stages", {}):
                    raise RuntimeError(
                        "Continuation training has not been adopted; run "
                        "adopt-training before run or resume"
                    )
            max_rounds = int(self.config.value["workflow"]["max_rounds"])
            current_round = int(state["current_round"])
            current_round_key = f"round_{current_round:03d}"
            start_round = (
                current_round + 1
                if current_round and
                current_round_key in state.get("completed_rounds", {})
                else max(current_round, configured_start)
            )
            if state.get("early_stopping", {}).get("should_stop"):
                self._stop(state, "metric_patience")
                return state
            for round_index in range(start_round, max_rounds + 1):
                if (self.run_dir / "cancel.requested").exists():
                    state["status"] = "canceled"
                    self.store.save(state)
                    return state
                state["current_round"] = round_index
                # Stage commits advance the current model/manifest/persistence.
                # Retrying this round must still reconstruct its original inputs.
                state.setdefault("round_inputs", {}).setdefault(
                    str(round_index),
                    {
                        "current_checkpoint": state["current_checkpoint"],
                        "current_training_manifest": state["current_training_manifest"],
                        "persistent_targets": json.loads(json.dumps(state["persistent_targets"])),
                    },
                )
                self.store.save(state)
                round_dir = self.run_dir / "rounds" / f"round_{round_index:03d}"
                try:
                    if int(state.get("adopted_partial_round", -1)) == round_index:
                        checkpoint = Path(state["current_checkpoint"])
                        metrics = self._evaluate(
                            state, round_index, round_dir, checkpoint
                        )
                        should_stop = self._record_early_stopping(
                            state,
                            round_index=round_index,
                            checkpoint=checkpoint,
                            training_manifest=state.get(
                                "current_training_manifest"
                            ),
                            metrics_path=metrics,
                        )
                        self.store.complete_round(state, round_index=round_index)
                        state.pop("adopted_partial_round", None)
                        self.store.save(state)
                        self._render_report(state)
                        if should_stop:
                            self._stop(state, "metric_patience")
                            return state
                        continue
                    scores = self._score(state, round_index, round_dir)
                    targets = self._select(state, round_index, round_dir, scores)
                    if pd.read_parquet(targets, pre_buffer=False).empty:
                        self._stop(state, "no_actionable_targets")
                        return state
                    neighbors, search = self._search(
                        state, round_index, round_dir, targets
                    )
                    if pd.read_parquet(neighbors, pre_buffer=False).empty:
                        reason = _empty_search_stop_reason(search)
                        self._stop(state, reason)
                        return state
                    _, training_view = self._materialize(
                        state, round_index, round_dir, neighbors, targets
                    )
                    materialize_artifact = json.loads(
                        (round_dir / "materialize" / "artifact.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if int(materialize_artifact["payload"].get("delta_rows", -1)) == 0:
                        self._stop(state, "no_novel_samples")
                        return state
                    checkpoint = self._train(
                        state, round_index, round_dir, training_view
                    )
                    metrics = self._evaluate(
                        state, round_index, round_dir, checkpoint
                    )
                    should_stop = self._record_early_stopping(
                        state,
                        round_index=round_index,
                        checkpoint=checkpoint,
                        training_manifest=state.get("current_training_manifest"),
                        metrics_path=metrics,
                    )
                    self.store.complete_round(state, round_index=round_index)
                    self._render_report(state)
                    if should_stop:
                        self._stop(state, "metric_patience")
                        return state
                except RunCanceled:
                    return self.store.load()
                except Exception as exc:
                    self.store.fail_stage(
                        state,
                        round_index=round_index,
                        stage=exc.stage if isinstance(exc, StageFailure) else "controller",
                        error=str(exc),
                    )
                    try:
                        self._render_report(state)
                    except Exception as report_error:  # recovery must preserve root cause
                        _atomic_json(
                            self.run_dir / "report_error.json",
                            {
                                "schema_version": "1.0",
                                "error": str(report_error),
                                "root_failure": str(exc),
                            },
                        )
                    raise
            self._stop(state, "max_rounds")
            return state

    def status(self) -> dict[str, Any]:
        """Read the durable workflow status without executing stages."""
        return self.store.load()

    def _reconcile_cancellation(
        self, state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        results: dict[str, Any] = {}
        for key, job in list(state.get("active_jobs", {}).items()):
            try:
                result = self.runner.cancel(job["client_job_id"])
            except Exception as exc:  # backend acknowledgement remains pending
                result = {"state": "UNKNOWN", "error": str(exc)}
            results[key] = result
            if str(result.get("state", "UNKNOWN")).upper() in {
                "COMPLETE",
                "ERROR",
                "CANCELED",
            }:
                state["active_jobs"].pop(key, None)
        state["status"] = "canceling" if state.get("active_jobs") else "canceled"
        state["cancellation"] = {"jobs": results}
        self.store.save(state)
        self.store.append_event(
            state,
            round_index=int(state.get("current_round", 0)),
            stage="cancel",
            status=state["status"],
            extra={"jobs": results},
        )
        return state, results

    def cancel(self) -> dict[str, Any]:
        """Reconcile cancellation intent with the run-owned process or backend job."""
        state = self.store.load()
        if not state:
            return {"status": "not_started", "jobs": {}}
        if state.get("status") in {"complete", "canceled"}:
            return {"status": state.get("status"), "jobs": {}}
        intent = self.run_dir / "cancel.requested"
        intent.touch(exist_ok=True)
        try:
            with self.store.controller_lock():
                state, results = self._reconcile_cancellation(self.store.load())
                return {"status": state["status"], "jobs": results}
        except ControllerBusy:
            # The running controller remains the sole state writer. Cancel may
            # signal its backend, but must never publish a stale state snapshot.
            results = {}
            for key, job in self.store.load().get("active_jobs", {}).items():
                try:
                    results[key] = self.runner.cancel(job["client_job_id"])
                except Exception as exc:
                    results[key] = {"state": "UNKNOWN", "error": str(exc)}
            return {"status": "canceling", "jobs": results}

    def logs(self, client_job_id: str, cursor: str | None = None) -> dict[str, Any]:
        """Read one stage log through the configured runner contract."""
        return self.runner.logs(client_job_id, cursor)
