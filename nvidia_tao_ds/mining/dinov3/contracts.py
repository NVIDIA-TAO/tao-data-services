# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned artifact contracts shared by data-refinement actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible value."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    """Write JSON atomically on a POSIX-compatible filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
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


@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable identity and provenance for one completed action output."""

    artifact_type: str
    producer: dict[str, Any]
    inputs: list[dict[str, Any]]
    payload: dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def artifact_id(self) -> str:
        """Identify semantic content while excluding creation time."""
        return canonical_digest(
            {
                "artifact_type": self.artifact_type,
                "schema_version": self.schema_version,
                "producer": self.producer,
                "inputs": self.inputs,
                "payload": self.payload,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize with the derived artifact identity."""
        value = asdict(self)
        value["artifact_id"] = self.artifact_id
        return value

    def commit(self, output_dir: str | Path) -> Path:
        """Publish the manifest before the final success marker."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        existing_path = root / "artifact.json"
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("artifact_id") != self.artifact_id:
                raise RuntimeError(
                    f"Refusing conflicting reuse of committed output directory: {root}"
                )
            success = root / "_SUCCESS"
            if success.exists() and success.read_text(encoding="utf-8").strip() != self.artifact_id:
                raise RuntimeError(f"Artifact success marker does not match {existing_path}")
            if not success.exists():
                temporary = root / "_SUCCESS.tmp"
                temporary.write_text(self.artifact_id + "\n", encoding="utf-8")
                temporary.replace(success)
            return existing_path
        manifest_path = write_json_atomic(root / "artifact.json", self.to_dict())
        success_path = root / "_SUCCESS"
        temporary = root / "_SUCCESS.tmp"
        temporary.write_text(self.artifact_id + "\n", encoding="utf-8")
        temporary.replace(success_path)
        return manifest_path
