# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce reproducible candidate-recall evidence before permitting ANN mining."""

from pathlib import Path
import json

import numpy as np

from .ann_index import load_ann_index
from .ann_search import retrieve_ann_candidates
from .contracts import ArtifactManifest, atomic_path, file_identity, require_uncommitted_output, write_json_atomic
from .dense_search import exact_dense_candidates
from .dense_store import load_dense_store
from .search import read_query_table


def audit_ann_recall(*, query_path, dense_store_manifest, ann_index_manifest,
                     query_embedding_contract, output_dir, n_probes, ann_candidate_count, top_k=10,
                     sample_size=128, seed=0, minimum_recall=0.95, device="cpu"):
    """Compare ANN candidate coverage with exact top-k on a seeded query sample.

    An audit is specific to the index, query population, probe count, candidate
    depth and declared threshold. A failed audit is preserved as evidence but
    cannot pass ``load_ann_audit`` or authorize mining.
    """
    if top_k <= 0 or sample_size <= 0 or ann_candidate_count < top_k:
        raise ValueError("Audit sizes must be positive and candidate depth must cover top_k")
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    dense, dense_artifact = load_dense_store(dense_store_manifest)
    query_contract = json.loads(Path(query_embedding_contract).read_text(encoding="utf-8"))
    if any(query_contract.get(key) != dense[key] for key in ("encoder", "embedding_dim")):
        raise ValueError("Audit query embedding contract differs from the dense store")
    ann, ann_artifact = load_ann_index(ann_index_manifest)
    if ann["dense_store_artifact_id"] != dense_artifact["artifact_id"]:
        raise ValueError("ANN index and audit dense store differ")
    if top_k > int(dense["row_count"]):
        raise ValueError("Audit top_k exceeds the dense population")
    queries = read_query_table(query_path, dense)
    indices = np.sort(np.random.default_rng(seed).choice(
        len(queries), size=min(sample_size, len(queries)), replace=False,
    ))
    sampled = queries.iloc[indices].rename(columns={
        "sample_id": dense.get("id_column", "sample_id"),
        "embedding": dense.get("embedding_column", "embedding"),
    })
    destination = require_uncommitted_output(output_dir)
    sample_path = destination / "audit_queries.parquet"
    with atomic_path(sample_path) as temporary:
        sampled.to_parquet(temporary, index=False)
    _, exact_ids, _ = exact_dense_candidates(
        query_path=sample_path, dense_store_manifest=dense_store_manifest,
        candidate_limit=top_k, hard_min_similarity=-1.0, excluded_row_ids=None,
        checkpoint_path=destination / "exact_progress.npz", device=device,
    )
    _, ann_ids, _ = retrieve_ann_candidates(
        query_path=sample_path, ann_index_manifest=ann_index_manifest,
        n_probes=n_probes, ann_candidate_count=ann_candidate_count,
        _validated_index=(ann, ann_artifact),
    )
    recalls = [len(set(exact).intersection(candidate)) / top_k
               for exact, candidate in zip(exact_ids, ann_ids)]
    recall = float(np.mean(recalls))
    payload = {
        "schema_version": "1.0", "passed": recall >= minimum_recall,
        "index_artifact_id": ann_artifact["artifact_id"],
        "dense_store_artifact_id": dense_artifact["artifact_id"],
        "source_inventory_digest": dense["source_inventory_digest"],
        "vector_inventory_digest": dense["vector_inventory_digest"],
        "n_probes": n_probes, "ann_candidate_count": ann_candidate_count,
        "top_k": top_k, "sample_size": len(sampled), "seed": seed,
        "minimum_recall": minimum_recall, "mean_recall_at_k": recall,
        "per_query_recall_at_k": recalls, "sample": file_identity(sample_path),
        "exact_progress": file_identity(destination / "exact_progress.npz"),
        "method": "exact_float32_topk_candidate_coverage_v1",
        "query_embedding_contract": file_identity(query_embedding_contract),
    }
    write_json_atomic(destination / "ann_audit.json", payload)
    artifact = ArtifactManifest(
        artifact_type="ann_recall_audit", producer={"action": "audit_ann_recall", "version": "1.0"},
        inputs=[file_identity(Path(query_path), role="audit_query_population"),
                {**file_identity(ann_index_manifest), "artifact_id": ann_artifact["artifact_id"]},
                {**file_identity(dense_store_manifest), "artifact_id": dense_artifact["artifact_id"]}],
        payload=payload,
    )
    artifact.commit(destination)
    return artifact.to_dict()
