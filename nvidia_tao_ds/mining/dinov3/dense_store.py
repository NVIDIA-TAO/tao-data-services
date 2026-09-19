# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable low-inode exact-vector companion for indexed mining."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    ArtifactManifest,
    artifact_content_id,
    canonical_digest,
    file_identity,
    file_posix_identity,
    file_sha256,
    require_uncommitted_output,
    vector_matrix,
    write_json_atomic,
)


_PLAN_NAME = "dense_store_plan.json"
_MANIFEST_NAME = "dense_store.json"
_PARTIAL_VECTOR_NAME = "vectors.f32.partial"
_VECTOR_NAME = "vectors.f32"


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Embedding store must use a local file URI: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _committed_payload(manifest_path: str | Path) -> tuple[dict, dict]:
    """Load a content-bound artifact without rehashing its entire payload."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = path.with_name("artifact.json")
    marker_path = path.with_name("_SUCCESS")
    if not artifact_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Manifest is not a committed artifact: {path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_id = artifact_content_id(artifact)
    if artifact.get("artifact_id") != expected_id:
        raise ValueError(f"Artifact identity is invalid: {artifact_path}")
    if artifact.get("payload") != payload:
        raise ValueError(f"Artifact payload differs from {path}")
    if marker_path.read_text(encoding="utf-8").strip() != expected_id:
        raise ValueError(f"Artifact success marker is inconsistent: {marker_path}")
    return payload, artifact


def initialize_dense_store(
    *, source_store_manifest: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Create a deterministic row layout and one sparse destination file."""
    destination = require_uncommitted_output(output_dir)
    source, source_artifact = _committed_payload(source_store_manifest)
    if source_artifact.get("artifact_type") != "embedding_store":
        raise ValueError("Dense vectors require an embedding-store artifact")
    if source.get("fingerprint_method") != "sha256":
        raise ValueError("Dense vectors require SHA-256 source fingerprints")
    dimension = int(source["embedding_dim"])
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive")

    row_start = 0
    shards = []
    for index, shard in enumerate(source.get("shards", [])):
        rows = int(shard["rows"])
        if rows <= 0 or not shard.get("sha256"):
            raise ValueError(f"Invalid source shard at index {index}")
        shards.append(
            {
                "index": index,
                "relative_path": str(shard["relative_path"]),
                "source_bytes": int(shard["bytes"]),
                "source_rows": rows,
                "source_sha256": str(shard["sha256"]),
                "source_posix_identity": shard.get("posix_identity"),
                "row_start": row_start,
                "row_stop": row_start + rows,
                "byte_start": row_start * dimension * np.dtype("<f4").itemsize,
                "byte_stop": (row_start + rows) *
                dimension *
                np.dtype("<f4").itemsize,
            }
        )
        row_start += rows
    if not shards or row_start != int(source["row_count"]):
        raise ValueError("Embedding-store shard rows do not reconcile")

    plan = {
        "schema_version": "1.0",
        "source_store_manifest": str(Path(source_store_manifest).resolve()),
        "source_store_artifact_id": source_artifact["artifact_id"],
        "source_inventory_digest": source["inventory_digest"],
        "source_root_uri": source["root_uri"],
        "id_column": source["id_column"],
        "embedding_column": source["embedding_column"],
        "locator_schema": source["locator_schema"],
        "locator_defaults": source.get("locator_defaults", {}),
        "encoder": source["encoder"],
        "normalization": "l2_float32",
        "dtype": "<f4",
        "embedding_dim": dimension,
        "row_count": row_start,
        "vector_bytes": row_start * dimension * np.dtype("<f4").itemsize,
        "vector_partial_path": str((destination / _PARTIAL_VECTOR_NAME).resolve()),
        "progress_dir": str((destination / "progress").resolve()),
        "shards": shards,
    }
    plan["plan_digest"] = canonical_digest(plan)
    plan_path = destination / _PLAN_NAME
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError(f"Dense-store plan conflicts with {plan_path}")
    else:
        write_json_atomic(plan_path, plan)

    progress = destination / "progress"
    progress.mkdir(exist_ok=True)
    vector_path = destination / _PARTIAL_VECTOR_NAME
    if vector_path.exists():
        if vector_path.stat().st_size != int(plan["vector_bytes"]):
            raise RuntimeError(f"Dense vector allocation has the wrong size: {vector_path}")
    else:
        with vector_path.open("wb") as stream:
            stream.truncate(int(plan["vector_bytes"]))
    return plan


def _vectors_from_array(array: pa.Array, dimension: int) -> np.ndarray:
    if array.null_count:
        raise ValueError("Embedding column contains null vectors")
    if (
        pa.types.is_fixed_size_list(array.type) and
        int(array.type.list_size) != dimension
    ):
        raise ValueError(
            f"Embedding dimension changed from {dimension} to {array.type.list_size}"
        )
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or
            pa.types.is_fixed_size_list(array.type)):
        raise ValueError("Embedding column must contain lists of numbers")
    if not pa.types.is_fixed_size_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        if not np.all(np.diff(offsets) == dimension):
            raise ValueError("Embedding column must contain equal-length vectors")
    values = array.flatten()
    if values.null_count or not (pa.types.is_floating(values.type) or pa.types.is_integer(values.type)):
        raise ValueError("Embedding values must be non-null numbers")
    matrix = values.to_numpy(zero_copy_only=False).reshape(len(array), dimension)
    result = vector_matrix(matrix, label="Embedding")
    if result.shape != (len(array), dimension):
        raise ValueError("Embedding column must contain equal-length vectors")
    return np.ascontiguousarray(result, dtype="<f4")


def _pwrite_all(file_descriptor: int, payload: memoryview, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(file_descriptor, payload[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite made no progress")
        written += count


def _file_range_sha256(path: Path, start: int, stop: int) -> str:
    """Hash one declared byte range without reading adjacent shard ranges."""
    digest = hashlib.sha256()
    remaining = stop - start
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"Dense vector range is truncated: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_source_shard(source_path: Path, shard: dict[str, Any]) -> None:
    """Validate source bytes, rows, and committed content identity."""
    current_identity = file_posix_identity(source_path)
    # POSIX identity is a fast-path hint, not corpus identity. A copied shard or
    # chmod may change inode/ctime; verify its declared content below instead.
    if current_identity["bytes"] != int(shard["source_bytes"]):
        raise RuntimeError(f"Source shard size changed: {source_path}")
    parquet = pq.ParquetFile(source_path)
    if parquet.metadata.num_rows != int(shard["source_rows"]):
        raise RuntimeError(f"Source shard row count changed: {source_path}")
    observed_sha = file_sha256(source_path).removeprefix("sha256:")
    expected_sha = str(shard["source_sha256"]).removeprefix("sha256:")
    if observed_sha != expected_sha:
        raise RuntimeError(f"Source shard content changed: {source_path}")


def materialize_dense_shards(
    *, plan_path: str | Path, shard_indexes: Iterable[int], batch_rows: int = 8192
) -> dict[str, Any]:
    """Write assigned source shards into their disjoint global vector ranges."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    plan_file = Path(plan_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan.get(
        "plan_digest"
    ):
        raise ValueError("Dense-store plan identity is invalid")
    source_root = _local_path(plan["source_root_uri"])
    vector_path = Path(plan["vector_partial_path"])
    if vector_path.stat().st_size != int(plan["vector_bytes"]):
        raise RuntimeError("Dense vector allocation changed after planning")
    progress_dir = Path(plan["progress_dir"])
    progress_dir.mkdir(exist_ok=True)
    dimension = int(plan["embedding_dim"])
    embedding_column = str(plan["embedding_column"])
    requested = sorted(set(int(value) for value in shard_indexes))
    if not requested:
        raise ValueError("At least one shard index is required")

    completed = []
    skipped = []
    descriptor = os.open(vector_path, os.O_WRONLY)
    try:
        for shard_index in requested:
            if shard_index < 0 or shard_index >= len(plan["shards"]):
                raise IndexError(f"Shard index is out of range: {shard_index}")
            shard = plan["shards"][shard_index]
            marker = progress_dir / f"part-{shard_index:06d}.json"
            source_path = source_root / shard["relative_path"]
            if marker.exists():
                record = json.loads(marker.read_text(encoding="utf-8"))
                if record.get("plan_digest") != plan["plan_digest"]:
                    raise RuntimeError(f"Progress marker conflicts with plan: {marker}")
                _validate_source_shard(source_path, shard)
                observed_vector_sha = _file_range_sha256(
                    vector_path,
                    int(shard["byte_start"]),
                    int(shard["byte_stop"]),
                )
                if observed_vector_sha != record.get("vector_sha256"):
                    raise RuntimeError(f"Dense vector range changed: {marker}")
                skipped.append(shard_index)
                continue
            _validate_source_shard(source_path, shard)
            parquet = pq.ParquetFile(source_path)

            digest = hashlib.sha256()
            rows_written = 0
            maximum_norm_error = 0.0
            for batch in parquet.iter_batches(
                batch_size=batch_rows, columns=[embedding_column]
            ):
                column = batch.column(0)
                vectors = _vectors_from_array(column, dimension)
                norm_error = float(
                    np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0))
                )
                maximum_norm_error = max(maximum_norm_error, norm_error)
                payload = memoryview(vectors).cast("B")
                byte_offset = int(shard["byte_start"]) + (
                    rows_written * dimension * np.dtype("<f4").itemsize
                )
                _pwrite_all(descriptor, payload, byte_offset)
                digest.update(payload)
                rows_written += len(vectors)
            if rows_written != int(shard["source_rows"]):
                raise RuntimeError(f"Dense write did not reconcile rows: {source_path}")
            record = {
                "plan_digest": plan["plan_digest"],
                "shard_index": shard_index,
                "source_sha256": shard["source_sha256"],
                "rows": rows_written,
                "byte_start": shard["byte_start"],
                "byte_stop": shard["byte_stop"],
                "vector_sha256": "sha256:" + digest.hexdigest(),
                "maximum_norm_error": maximum_norm_error,
            }
            write_json_atomic(marker, record)
            completed.append(shard_index)
    finally:
        os.close(descriptor)
    return {
        "plan_digest": plan["plan_digest"],
        "completed": completed,
        "skipped": skipped,
    }


def finalize_dense_store(*, plan_path: str | Path) -> dict[str, Any]:
    """Commit the dense vector file after every declared range is complete."""
    plan_file = Path(plan_path).resolve()
    destination = plan_file.parent
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan.get(
        "plan_digest"
    ):
        raise ValueError("Dense-store plan identity is invalid")
    marker_records = []
    for shard in plan["shards"]:
        marker = Path(plan["progress_dir"]) / f"part-{int(shard['index']):06d}.json"
        if not marker.is_file():
            raise RuntimeError(f"Dense vector shard is incomplete: {shard['index']}")
        record = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "plan_digest": plan["plan_digest"],
            "shard_index": shard["index"],
            "source_sha256": shard["source_sha256"],
            "rows": shard["source_rows"],
            "byte_start": shard["byte_start"],
            "byte_stop": shard["byte_stop"],
        }
        if any(record.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"Dense progress marker is invalid: {marker}")
        source_path = _local_path(plan["source_root_uri"]) / shard["relative_path"]
        _validate_source_shard(source_path, shard)
        marker_records.append(record)

    partial = Path(plan["vector_partial_path"])
    vector_path = destination / _VECTOR_NAME
    if vector_path.exists() and partial.exists():
        raise RuntimeError("Both partial and committed dense vectors exist")
    vector_source = partial if partial.exists() else vector_path
    if partial.exists():
        if partial.stat().st_size != int(plan["vector_bytes"]):
            raise RuntimeError("Dense vector file has the wrong size")
    if vector_source.stat().st_size != int(plan["vector_bytes"]):
        raise RuntimeError("Committed dense vector file has the wrong size")
    for shard, record in zip(plan["shards"], marker_records):
        observed = _file_range_sha256(
            vector_source,
            int(shard["byte_start"]),
            int(shard["byte_stop"]),
        )
        if observed != record["vector_sha256"]:
            raise RuntimeError(
                f"Committed dense vector range changed: {shard['index']}"
            )

    vector_posix_identity = file_posix_identity(vector_source)
    # A same-filesystem rename changes ctime while preserving the inode,
    # byte count, and content mtime. Bind the fields that survive atomic
    # publication so the fast reopen gate remains useful after the move.
    vector_posix_identity.pop("ctime_ns")
    payload = {
        "source_store_manifest": plan["source_store_manifest"],
        "source_store_artifact_id": plan["source_store_artifact_id"],
        "source_inventory_digest": plan["source_inventory_digest"],
        "source_root_uri": plan["source_root_uri"],
        "id_column": plan["id_column"],
        "embedding_column": plan["embedding_column"],
        "locator_schema": plan["locator_schema"],
        "locator_defaults": plan["locator_defaults"],
        "encoder": plan["encoder"],
        "normalization": plan["normalization"],
        "dtype": plan["dtype"],
        "embedding_dim": plan["embedding_dim"],
        "row_count": plan["row_count"],
        "vector_uri": vector_path.resolve().as_uri(),
        "vector_bytes": plan["vector_bytes"],
        "vector_posix_identity": vector_posix_identity,
        "vector_inventory_digest": canonical_digest(
            [record["vector_sha256"] for record in marker_records]
        ),
        "shards": plan["shards"],
        "random_access_contract": "global_row_id_contiguous_float32_v1",
    }
    artifact = ArtifactManifest(
        artifact_type="dense_vector_store",
        producer={"action": "materialize_dense_store", "version": "1.0"},
        inputs=[{
            **file_identity(
                plan["source_store_manifest"], role="embedding_store"
            ),
            "artifact_id": plan["source_store_artifact_id"],
        }],
        payload=payload,
    )
    success = destination / "_SUCCESS"
    manifest_path = destination / _MANIFEST_NAME
    if success.exists():
        existing_payload, existing_artifact = _committed_payload(manifest_path)
        if existing_payload != payload or existing_artifact.get(
            "artifact_id"
        ) != artifact.artifact_id:
            raise RuntimeError(
                f"Dense-store finalization conflicts with committed output: {destination}"
            )
        return existing_artifact
    publications = [(partial, vector_path)] if partial.exists() else []
    artifact.commit(
        destination,
        json_payloads={_MANIFEST_NAME: payload},
        file_publications=publications,
    )
    return artifact.to_dict()


def load_dense_store(manifest_path: str | Path) -> tuple[dict, dict]:
    """Validate manifest seals and cheap payload metadata, not payload bytes.

    Timestamp-coalescing filesystems can hide same-size in-place writes.
    Use verify_dense_store_integrity at release/audit boundaries to validate
    content; this fast path assumes immutable storage after that validation.
    """
    payload, artifact = _committed_payload(manifest_path)
    if artifact.get("artifact_type") != "dense_vector_store":
        raise ValueError("Expected a dense-vector-store artifact")
    vector_path = _local_path(payload["vector_uri"])
    current_identity = file_posix_identity(vector_path)
    if current_identity["bytes"] != int(payload["vector_bytes"]):
        raise RuntimeError(f"Dense vector payload size changed: {vector_path}")
    expected_identity = payload.get("vector_posix_identity")
    if expected_identity is not None and any(
        current_identity.get(name) != value
        for name, value in expected_identity.items()
    ):
        raise RuntimeError(f"Dense vector payload identity changed: {vector_path}")
    if "locator_schema" not in payload or "locator_defaults" not in payload:
        source, source_artifact = _committed_payload(
            payload["source_store_manifest"]
        )
        if source_artifact["artifact_id"] != payload["source_store_artifact_id"]:
            raise ValueError("Dense and source-store artifact identities differ")
        if source["inventory_digest"] != payload["source_inventory_digest"]:
            raise ValueError("Dense and source-store inventories differ")
        payload = dict(payload)
        payload["locator_schema"] = source["locator_schema"]
        payload["locator_defaults"] = source.get("locator_defaults", {})
    return payload, artifact


def verify_dense_store_integrity(
    manifest_path: str | Path, *, workers: int = 1
) -> dict[str, Any]:
    """Rehash every source shard and dense range for a release/audit gate."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    path = Path(manifest_path).resolve()
    payload, artifact = load_dense_store(path)
    source_root = _local_path(payload["source_root_uri"])
    vector_path = _local_path(payload["vector_uri"])
    progress_dir = path.parent / "progress"

    def verify(shard: dict[str, Any]) -> str:
        _validate_source_shard(source_root / shard["relative_path"], shard)
        marker_path = progress_dir / f"part-{int(shard['index']):06d}.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Dense integrity marker is missing: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        observed = _file_range_sha256(
            vector_path,
            int(shard["byte_start"]),
            int(shard["byte_stop"]),
        )
        if observed != marker.get("vector_sha256"):
            raise RuntimeError(f"Dense vector range changed: {marker_path}")
        return observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        vector_hashes = list(executor.map(verify, payload["shards"]))
    observed_inventory = canonical_digest(vector_hashes)
    if observed_inventory != payload["vector_inventory_digest"]:
        raise RuntimeError("Dense vector inventory digest changed")
    return {
        "dense_store_artifact_id": artifact["artifact_id"],
        "source_store_artifact_id": payload["source_store_artifact_id"],
        "source_inventory_digest": payload["source_inventory_digest"],
        "vector_inventory_digest": observed_inventory,
        "row_count": int(payload["row_count"]),
        "embedding_dim": int(payload["embedding_dim"]),
        "verified_source_shards": len(payload["shards"]),
    }
