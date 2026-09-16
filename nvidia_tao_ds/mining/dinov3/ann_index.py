# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent cuVS IVF-PQ index construction for refinement mining."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

from .contracts import (
    ArtifactManifest,
    artifact_content_id,
    file_identity,
    file_sha256,
    require_uncommitted_output,
    write_json_atomic,
)
from .dense_store import load_dense_store


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"ANN artifacts must use local file URIs: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _cuvs_version() -> str:
    for package in ("cuvs-cu13", "cuvs-cu12", "cuvs"):
        try:
            return version(package)
        except PackageNotFoundError:
            continue
    return "unknown"


def build_cuvs_ivf_pq_index(
    *,
    dense_store_manifest: str | Path,
    output_dir: str | Path,
    n_lists: int = 32768,
    pq_dim: int = 160,
    pq_bits: int = 8,
    kmeans_n_iters: int = 20,
    kmeans_trainset_fraction: float = 0.03,
    force_random_rotation: bool = True,
) -> dict[str, Any]:
    """Build and commit a sharded single-node multi-GPU IVF-PQ index."""
    if n_lists <= 0 or pq_dim <= 0 or pq_bits <= 0:
        raise ValueError("n_lists, pq_dim, and pq_bits must be positive")
    if pq_dim * pq_bits % 8:
        raise ValueError("pq_dim * pq_bits must be divisible by 8")
    if kmeans_n_iters <= 0:
        raise ValueError("kmeans_n_iters must be positive")
    if not 0.0 < kmeans_trainset_fraction <= 1.0:
        raise ValueError("kmeans_trainset_fraction must be in (0, 1]")

    dense, dense_artifact = load_dense_store(dense_store_manifest)
    destination = require_uncommitted_output(output_dir)
    vector_path = _local_path(dense["vector_uri"])
    vectors = np.memmap(
        vector_path,
        mode="r",
        dtype=np.dtype(dense["dtype"]),
        shape=(int(dense["row_count"]), int(dense["embedding_dim"])),
    )
    if pq_dim > int(dense["embedding_dim"]):
        raise ValueError("pq_dim cannot exceed the embedding dimension")

    try:
        from cuvs.neighbors.mg import ivf_pq  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError(
            "cuVS is required to build the ANN index; install cuvs-cu12 or cuvs-cu13"
        ) from error

    parameters = {
        "distribution_mode": "sharded",
        "n_lists": n_lists,
        "metric": "sqeuclidean",
        "kmeans_n_iters": kmeans_n_iters,
        "kmeans_trainset_fraction": kmeans_trainset_fraction,
        "pq_bits": pq_bits,
        "pq_dim": pq_dim,
        "codebook_kind": "subspace",
        "force_random_rotation": force_random_rotation,
        "conservative_memory_allocation": True,
    }
    index = ivf_pq.build(ivf_pq.IndexParams(**parameters), vectors)
    temporary = destination / "index.cuvs.partial"
    index_path = destination / "index.cuvs"
    ivf_pq.save(index, str(temporary))
    temporary.replace(index_path)
    del index
    del vectors

    payload = {
        "backend": "cuvs_mg_ivf_pq",
        "backend_version": _cuvs_version(),
        "dense_store_manifest": str(Path(dense_store_manifest).resolve()),
        "dense_store_artifact_id": dense_artifact["artifact_id"],
        "source_store_artifact_id": dense["source_store_artifact_id"],
        "source_inventory_digest": dense["source_inventory_digest"],
        "vector_inventory_digest": dense["vector_inventory_digest"],
        "encoder": dense["encoder"],
        "embedding_dim": dense["embedding_dim"],
        "row_count": dense["row_count"],
        "index": file_identity(index_path),
        "parameters": parameters,
        "candidate_contract": "ann_candidates_require_exact_float32_rerank_v1",
    }
    write_json_atomic(destination / "ann_index.json", payload)
    artifact = ArtifactManifest(
        artifact_type="ann_index",
        producer={"action": "build_cuvs_mg_ivf_pq", "version": "1.0"},
        inputs=[
            {
                **file_identity(dense_store_manifest, role="dense_vector_store"),
                "artifact_id": dense_artifact["artifact_id"],
            }
        ],
        payload=payload,
    )
    artifact.commit(destination)
    return artifact.to_dict()


def load_ann_index(manifest_path: str | Path) -> tuple[dict, dict]:
    """Load a committed ANN index manifest and its artifact identity."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = path.with_name("artifact.json")
    marker_path = path.with_name("_SUCCESS")
    if not artifact_path.is_file() or not marker_path.is_file():
        raise ValueError(f"ANN index is not committed: {path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_id = artifact_content_id(artifact)
    if artifact.get("artifact_id") != expected_id:
        raise ValueError(f"ANN artifact identity is invalid: {path}")
    if (
        artifact.get("artifact_type") != "ann_index" or
        artifact.get("payload") != payload
    ):
        raise ValueError(f"ANN artifact does not match its manifest: {path}")
    if marker_path.read_text(encoding="utf-8").strip() != artifact.get("artifact_id"):
        raise ValueError(f"ANN success marker is inconsistent: {path}")
    index_path = _local_path(payload["index"]["uri"])
    if index_path.stat().st_size != int(payload["index"]["bytes"]):
        raise RuntimeError(f"ANN index size changed: {index_path}")
    if file_sha256(index_path) != payload["index"]["sha256"]:
        raise RuntimeError(f"ANN index content changed: {index_path}")
    return payload, artifact


def load_ann_audit(
    manifest_path: str | Path,
    *,
    index_artifact_id: str,
    n_probes: int,
    ann_candidate_count: int,
    dense_store_artifact_id: str,
    source_inventory_digest: str,
    vector_inventory_digest: str,
) -> tuple[dict, dict]:
    """Validate a committed recall audit for exact search-time settings."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = path.with_name("artifact.json")
    marker_path = path.with_name("_SUCCESS")
    if not artifact_path.is_file() or not marker_path.is_file():
        raise ValueError(f"ANN audit is not committed: {path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_id = artifact_content_id(artifact)
    if artifact.get("artifact_id") != expected_id:
        raise ValueError(f"ANN audit artifact identity is invalid: {path}")
    if (
        artifact.get("artifact_type") != "ann_recall_audit" or
        artifact.get("payload") != payload
    ):
        raise ValueError(f"ANN audit artifact does not match its manifest: {path}")
    if marker_path.read_text(encoding="utf-8").strip() != expected_id:
        raise ValueError(f"ANN audit success marker is inconsistent: {path}")
    if not payload.get("passed"):
        raise ValueError("ANN audit did not pass its declared thresholds")
    if payload.get("index_artifact_id") != index_artifact_id:
        raise ValueError("ANN audit belongs to a different index")
    if int(payload.get("n_probes", -1)) != n_probes:
        raise ValueError("ANN audit and requested n_probes differ")
    if int(payload.get("ann_candidate_count", -1)) != ann_candidate_count:
        raise ValueError("ANN audit and requested candidate depth differ")
    lineage = {
        "dense_store_artifact_id": dense_store_artifact_id,
        "source_inventory_digest": source_inventory_digest,
        "vector_inventory_digest": vector_inventory_digest,
    }
    for name, expected in lineage.items():
        if payload.get(name) != expected:
            raise ValueError(f"ANN audit belongs to a different {name}")
    return payload, artifact
