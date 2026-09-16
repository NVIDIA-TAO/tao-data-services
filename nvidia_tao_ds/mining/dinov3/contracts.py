# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned artifact contracts shared by data-refinement actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
IDENTITY_VERSION = "2.0"
COSINE_TOLERANCE = 1e-6


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible value."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def shard_inventory_digest(shards: list[dict[str, Any]]) -> str:
    """Identify shard content and ordering independently of its POSIX binding."""
    portable = [
        {
            name: value
            for name, value in shard.items()
            if name != "posix_identity" and not name.endswith("_posix_identity")
        }
        for shard in shards
    ]
    return canonical_digest(portable)


_LOCAL_BINDING_FIELDS = frozenset({
    "root_uri",
    "source_root_uri",
    "result_uri",
    "manifest_uri",
    "vector_uri",
    "source_store_manifest",
    "dense_store_manifest",
    "ann_index_manifest",
    "ann_audit_manifest",
})


def semantic_content(value: Any, *, _context: str | None = None) -> Any:
    """Remove only recognized local bindings from a v2 content identity.

    A field merely named ``uri`` is not automatically local: model and encoder
    descriptors may use one as semantic configuration. File identities are
    recognizable by their content digest and byte count, while the other
    excluded names are reserved artifact-binding fields in this contract.
    """
    if isinstance(value, dict):
        content_identity_record = "sha256" in value and "bytes" in value
        file_identity_record = "uri" in value and content_identity_record
        return {
            key: semantic_content(item, _context=key)
            for key, item in value.items()
            if (
                key != "created_at" and
                not (
                    key in _LOCAL_BINDING_FIELDS and _context == "payload"
                ) and
                not key.endswith("_posix_identity") and
                not (key == "uri" and file_identity_record) and
                not (
                    key in {"posix_identity", "stat"} and
                    content_identity_record
                ) and
                not (
                    key in {"bytes", "sha256"} and
                    file_identity_record and
                    "artifact_id" in value
                ) and
                not (
                    key == "implementation_sha256" and _context == "producer"
                )
            )
        }
    if isinstance(value, list):
        return [semantic_content(item, _context=_context) for item in value]
    return value


def validate_serialized_artifact(manifest: dict[str, Any]) -> None:
    """Enforce the packaged artifact schema without an optional dependency."""
    required = {
        "artifact_id", "artifact_type", "schema_version", "producer", "inputs",
        "payload", "created_at",
    }
    allowed = required | {"audit_id", "identity_version"}
    if set(manifest).difference(allowed) or required.difference(manifest):
        raise ValueError("Artifact manifest fields do not match schema version 1.0")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Artifact schema_version must be 1.0")
    if not isinstance(manifest["artifact_type"], str) or not manifest["artifact_type"]:
        raise ValueError("Artifact type must be a non-empty string")
    if not isinstance(manifest["producer"], dict) or not isinstance(manifest["inputs"], list):
        raise ValueError("Artifact producer and inputs have invalid types")
    if any(not isinstance(item, dict) for item in manifest["inputs"]):
        raise ValueError("Artifact inputs must contain objects")
    if not isinstance(manifest["payload"], dict):
        raise ValueError("Artifact payload must be a mapping")
    identity_version = manifest.get("identity_version")
    if identity_version not in {None, IDENTITY_VERSION}:
        raise ValueError(f"Unsupported artifact identity_version: {identity_version}")
    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
    for name in ("artifact_id", "audit_id"):
        if name in manifest and not digest_pattern.fullmatch(str(manifest[name])):
            raise ValueError(f"Artifact {name} is not a SHA-256 identity")
    try:
        created_at = str(manifest["created_at"])
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if "T" not in created_at or timestamp.tzinfo is None:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("Artifact created_at is not an ISO-8601 timestamp") from error
    if "audit_id" in manifest:
        expected_audit = canonical_digest({
            name: manifest[name]
            for name in (
                "artifact_type", "schema_version", "identity_version", "producer",
                "inputs", "payload",
            )
            if name in manifest
        })
        if manifest["audit_id"] != expected_audit:
            raise ValueError("Artifact audit identity is invalid")


def artifact_content_id(manifest: dict[str, Any]) -> str:
    """Recompute the portable identity fields of a serialized artifact manifest."""
    if "artifact_id" in manifest:
        validate_serialized_artifact(manifest)
    fields = {
        name: manifest[name]
        for name in ("artifact_type", "schema_version", "producer", "inputs", "payload")
    }
    if manifest.get("identity_version") is None:
        return canonical_digest(fields)
    fields["identity_version"] = manifest["identity_version"]
    return canonical_digest(semantic_content(fields))


def artifact_audit_id(manifest: dict[str, Any]) -> str:
    """Bind the exact implementation audit fields and local publication record."""
    return canonical_digest({
        name: manifest[name]
        for name in (
            "artifact_type", "schema_version", "identity_version", "producer",
            "inputs", "payload",
        )
        if name in manifest
    })


def vector_matrix(values: Any, *, label: str = "Embedding") -> np.ndarray:
    """Validate and normalize one finite, nonzero float32 vector matrix."""
    try:
        matrix64 = np.asarray(
            values.tolist() if hasattr(values, "tolist") else values,
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} column must contain numeric vectors") from error
    if matrix64.ndim != 2 or matrix64.shape[1] == 0:
        raise ValueError(f"{label} column must contain equal-length vectors")
    if not np.isfinite(matrix64).all():
        raise ValueError(f"{label} column contains non-finite values")
    scales = np.max(np.abs(matrix64), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError(f"{label} vectors must have nonzero norms")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        norms64 = scales * np.sqrt(
            np.sum(np.square(matrix64 / scales), axis=1, keepdims=True)
        )
    # The search contract is float32. Values whose norm cannot be represented
    # in that contract are rejected before conversion rather than overflowing
    # in a backend-specific NumPy/Torch operation.
    if (
        not np.isfinite(norms64).all() or
        np.any(norms64 > np.sqrt(np.finfo(np.float32).max))
    ):
        raise ValueError(f"{label} norms contain non-finite values")
    normalized = (matrix64 / norms64).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{label} normalization produced non-finite values")
    return normalized


def cosine_at_or_above(value: Any, threshold: float) -> Any:
    """Apply one numerical boundary policy to scalar or array cosine scores."""
    return value >= float(threshold) - COSINE_TOLERANCE


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    """Normalize the POSIX fields used to detect file replacement or mutation."""
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _stable_file_snapshot(path: str | Path) -> tuple[Path, dict[str, int], str]:
    """Hash a stable open-file snapshot and reject concurrent mutation."""
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = _stat_identity(os.fstat(stream.fileno()))
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
        after = _stat_identity(os.fstat(stream.fileno()))
    if before != after or _stat_identity(resolved.stat()) != after:
        raise RuntimeError(f"File changed while its identity was captured: {resolved}")
    return resolved, after, "sha256:" + digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one stable artifact payload."""
    return _stable_file_snapshot(path)[2]


def file_posix_identity(path: str | Path) -> dict[str, int]:
    """Return the stable POSIX fields used for low-cost mutation detection."""
    return _stat_identity(Path(path).stat())


def file_identity(path: str | Path, *, role: str | None = None) -> dict[str, Any]:
    """Return a content-bound local file identity for artifact lineage."""
    resolved, posix, digest = _stable_file_snapshot(path)
    value = {
        "uri": resolved.as_uri(),
        "bytes": posix["bytes"],
        "sha256": digest,
    }
    if role is not None:
        value["role"] = role
    return value


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Write durable JSON with a unique same-directory rename source."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_text_atomic(path: str | Path, value: str) -> Path:
    """Durably replace text using a unique same-directory temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=f".{destination.name}.", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def require_uncommitted_output(output_dir: str | Path) -> Path:
    """Refuse to mutate a directory that already contains a committed artifact."""
    root = Path(output_dir)
    success = root / "_SUCCESS"
    if success.exists():
        raise RuntimeError(
            f"Refusing to overwrite committed output directory: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


_PUBLICATION_STATE = threading.local()


@contextmanager
def artifact_publication(output_dir: str | Path):
    """Serialize a complete artifact production transaction per thread."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = str(root.resolve())
    held = getattr(_PUBLICATION_STATE, "held", {})
    if key in held:
        held[key][1] += 1
        try:
            yield root
        finally:
            held[key][1] -= 1
        return
    lock = (root / ".artifact.lock").open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if not hasattr(_PUBLICATION_STATE, "held"):
        _PUBLICATION_STATE.held = held
    held[key] = [lock, 1]
    try:
        yield root
    finally:
        held.pop(key)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable identity and provenance for one completed action output."""

    artifact_type: str
    producer: dict[str, Any]
    inputs: list[dict[str, Any]]
    payload: dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    identity_version: str = IDENTITY_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def artifact_id(self) -> str:
        """Identify portable semantic content, excluding local/audit bindings."""
        return artifact_content_id(
            {
                "artifact_type": self.artifact_type,
                "schema_version": self.schema_version,
                "identity_version": self.identity_version,
                "producer": self.producer,
                "inputs": self.inputs,
                "payload": self.payload,
            }
        )

    @property
    def audit_id(self) -> str:
        """Bind the complete producer implementation and local publication record."""
        return artifact_audit_id({
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "identity_version": self.identity_version,
            "producer": self.producer,
            "inputs": self.inputs,
            "payload": self.payload,
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialize with the derived artifact identity."""
        value = asdict(self)
        value["artifact_id"] = self.artifact_id
        value["audit_id"] = self.audit_id
        return value

    def commit(
        self,
        output_dir: str | Path,
        *,
        json_payloads: dict[str, Any] | None = None,
        file_publications: list[tuple[str | Path, str | Path]] | None = None,
    ) -> Path:
        """Publish payloads, the manifest, and success marker under one lock."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        root_resolved = root.resolve()
        serialized = self.to_dict()
        validate_serialized_artifact(serialized)
        payloads = json_payloads or {}
        publications = file_publications or []

        def checked_destination(value: str | Path) -> Path:
            destination = Path(value)
            if not destination.is_absolute():
                destination = root / destination
            destination = destination.resolve()
            if destination.parent != root_resolved:
                raise ValueError(
                    f"Artifact payload must be a direct child of {root}: {destination}"
                )
            return destination

        existing_path = root / "artifact.json"
        with artifact_publication(root):
            if existing_path.exists():
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                if (
                    artifact_content_id(existing) != existing.get("artifact_id") or
                    existing.get("artifact_id") != self.artifact_id
                ):
                    raise RuntimeError(
                        f"Refusing conflicting reuse of committed output directory: {root}"
                    )
                for name, value in payloads.items():
                    payload_path = checked_destination(name)
                    if (
                        not payload_path.is_file() or
                        json.loads(
                            payload_path.read_text(encoding="utf-8")
                        ) != value
                    ):
                        raise RuntimeError(
                            f"Committed artifact payload differs: {payload_path}"
                        )
                for source_value, destination_value in publications:
                    source = Path(source_value).resolve()
                    destination = checked_destination(destination_value)
                    if not destination.is_file():
                        raise RuntimeError(
                            "Committed artifact file payload is missing: "
                            f"{destination_value}"
                        )
                    if (
                        not source.is_file() or
                        destination.stat().st_size != source.stat().st_size or
                        file_sha256(destination) != file_sha256(source)
                    ):
                        raise RuntimeError(
                            f"Committed artifact file payload differs: {destination}"
                        )
                success = root / "_SUCCESS"
                if (
                    success.exists() and
                    success.read_text(encoding="utf-8").strip() !=
                    self.artifact_id
                ):
                    raise RuntimeError(
                        f"Artifact success marker does not match {existing_path}"
                    )
                if not success.exists():
                    write_text_atomic(success, self.artifact_id + "\n")
                return existing_path
            if (root / "_SUCCESS").exists():
                raise RuntimeError(
                    f"Artifact success marker exists without a manifest: {root}"
                )
            resolved_publications = [
                (Path(source_value).resolve(), checked_destination(destination_value))
                for source_value, destination_value in publications
            ]
            for source, destination in resolved_publications:
                if destination.exists():
                    if (
                        not source.is_file() or
                        destination.stat().st_size != source.stat().st_size or
                        file_sha256(destination) != file_sha256(source)
                    ):
                        raise RuntimeError(
                            f"Interrupted artifact payload conflicts: {destination}"
                        )
                elif not source.is_file():
                    raise RuntimeError(f"Artifact payload source is missing: {source}")
            for name, value in payloads.items():
                payload_path = checked_destination(name)
                if (
                    payload_path.exists() and
                    (
                        not payload_path.is_file() or
                        json.loads(
                            payload_path.read_text(encoding="utf-8")
                        ) != value
                    )
                ):
                    raise RuntimeError(
                        f"Interrupted artifact payload conflicts: {payload_path}"
                    )
            for source, destination in resolved_publications:
                if destination.exists():
                    continue
                source.replace(destination)
            for name, value in payloads.items():
                payload_path = checked_destination(name)
                if not payload_path.exists():
                    write_json_atomic(payload_path, value)
            manifest_path = write_json_atomic(existing_path, serialized)
            write_text_atomic(root / "_SUCCESS", self.artifact_id + "\n")
            return manifest_path
