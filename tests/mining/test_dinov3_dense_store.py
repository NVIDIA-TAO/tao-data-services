# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused dense-store recovery, integrity and reranking contracts."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from nvidia_tao_ds.mining.dinov3.contracts import ArtifactManifest, file_identity
from nvidia_tao_ds.mining.dinov3.dense_store import (
    initialize_dense_store, materialize_dense_shards, finalize_dense_store,
    load_dense_store, verify_dense_store_integrity,
)
from nvidia_tao_ds.mining.dinov3.ann_search import exact_rerank_ann_candidates
from nvidia_tao_ds.mining.dinov3.internal.refinement import main as refinement_main
from nvidia_tao_ds.mining.dinov3.store import register_embedding_store


@pytest.fixture
def dense_plan(tmp_path):
    """Prepare two shards with three globally ordered rows."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "embedding": [[3.0, 4.0], [0.0, 2.0]],
            "path": [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        }
    )
    second = pd.DataFrame(
        {
            "sample_id": ["c"],
            "embedding": [[-5.0, 0.0]],
            "path": [str(tmp_path / "c.jpg")],
        }
    )
    first.to_parquet(source_root / "part-0.parquet", index=False)
    second.to_parquet(source_root / "part-1.parquet", index=False)
    registered = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
        default_storage_type="file",
    )

    dense = tmp_path / "dense"
    plan = initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    return dense, plan


def test_dense_write_resume_finalize_and_global_rows(dense_plan):
    dense, plan = dense_plan
    assert plan["row_count"] == 3
    assert plan["shards"][1]["row_start"] == 2
    result = materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0, 1]
    )
    assert result["completed"] == [0, 1]
    resumed = materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0, 1]
    )
    assert resumed["skipped"] == [0, 1]
    artifact = finalize_dense_store(plan_path=dense / "dense_store_plan.json")
    assert finalize_dense_store(
        plan_path=dense / "dense_store_plan.json"
    )["artifact_id"] == artifact["artifact_id"]
    assert artifact["payload"]["row_count"] == 3
    payload, loaded_artifact = load_dense_store(dense / "dense_store.json")
    assert loaded_artifact["artifact_id"] == artifact["artifact_id"]
    vectors = np.memmap(
        dense / "vectors.f32", mode="r", dtype="<f4", shape=(3, 2)
    )
    np.testing.assert_allclose(
        vectors,
        np.asarray([[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )
    assert payload["vector_inventory_digest"].startswith("sha256:")
    assert payload["locator_defaults"] == {"storage_type": "file"}
    integrity = verify_dense_store_integrity(dense / "dense_store.json")
    assert integrity["verified_source_shards"] == 2



@pytest.fixture
def committed_dense(dense_plan, tmp_path):
    """Publish a small valid store independently for each behavior test."""
    dense, plan = dense_plan
    materialize_dense_shards(plan_path=dense / "dense_store_plan.json", shard_indexes=[0, 1])
    finalize_dense_store(plan_path=dense / "dense_store_plan.json")
    payload, loaded_artifact = load_dense_store(dense / "dense_store.json")
    queries = tmp_path / "queries.parquet"
    pd.DataFrame(
        {"sample_id": ["query"], "embedding": [[1.0, 0.0]]}
    ).to_parquet(queries, index=False)
    return dense, plan, payload, loaded_artifact, queries


def test_dense_rerank_hydration_and_row_ids(committed_dense):
    dense, _, _, _, queries = committed_dense
    neighbors = exact_rerank_ann_candidates(
        query_path=queries,
        dense_store_manifest=dense / "dense_store.json",
        candidate_row_ids=np.asarray([[2, 0, 1]], dtype=np.int64),
        top_k=1,
        min_similarity=0.5,
        duplicate_similarity=0.99,
    )
    assert neighbors.iloc[0]["sample_id"] == "a"
    assert neighbors.iloc[0]["source_row_id"] == 0
    assert neighbors.iloc[0]["source_part"].endswith("part-0.parquet")
    assert neighbors.iloc[0]["storage_type"] == "file"
    assert neighbors.iloc[0]["cosine_similarity"] == pytest.approx(0.6)
    assert neighbors.attrs["adaptive_radius"]["underfill_exhaustion_proven"] is False

    row_ids_only = exact_rerank_ann_candidates(
        query_path=queries,
        dense_store_manifest=dense / "dense_store.json",
        candidate_row_ids=np.asarray([[2, 0, 1]], dtype=np.int64),
        top_k=1,
        min_similarity=0.5,
        duplicate_similarity=0.99,
        hydrate_locators=False,
    )
    assert row_ids_only.iloc[0]["sample_id"] == "0"
    assert "source_part" not in row_ids_only.columns



@pytest.fixture
def staged_candidates(committed_dense, tmp_path):
    """Bind candidate rows to an index, recall audit and query population."""
    dense, _, payload, loaded_artifact, queries = committed_dense
    index_dir = tmp_path / "ann_index"
    index_dir.mkdir()
    index_file = index_dir / "index.bin"
    index_file.write_bytes(b"index")
    index_payload = {
        "index": file_identity(index_file),
        "dense_store_artifact_id": loaded_artifact["artifact_id"],
        "source_inventory_digest": payload["source_inventory_digest"],
        "vector_inventory_digest": payload["vector_inventory_digest"],
    }
    (index_dir / "ann_index.json").write_text(
        json.dumps(index_payload), encoding="utf-8"
    )
    index_artifact = ArtifactManifest(
        artifact_type="ann_index",
        producer={"action": "test", "version": "1.0"},
        inputs=[],
        payload=index_payload,
    )
    index_artifact.commit(index_dir)

    audit_dir = tmp_path / "ann_audit"
    audit_dir.mkdir()
    audit_payload = {
        "passed": True,
        "index_artifact_id": index_artifact.artifact_id,
        "n_probes": 4,
        "ann_candidate_count": 3,
        "dense_store_artifact_id": loaded_artifact["artifact_id"],
        "source_inventory_digest": payload["source_inventory_digest"],
        "vector_inventory_digest": payload["vector_inventory_digest"],
    }
    (audit_dir / "ann_audit.json").write_text(
        json.dumps(audit_payload), encoding="utf-8"
    )
    audit_artifact = ArtifactManifest(
        artifact_type="ann_recall_audit",
        producer={"action": "test", "version": "1.0"},
        inputs=[],
        payload=audit_payload,
    )
    audit_artifact.commit(audit_dir)

    candidates = tmp_path / "candidate_stage"
    candidates.mkdir()
    candidate_path = candidates / "ann_candidates.npz"
    np.savez(candidate_path, candidate_ids=np.asarray([[2, 0, 1]], dtype=np.int64))
    ArtifactManifest(
        artifact_type="ann_candidates",
        producer={"action": "test", "version": "1.0"},
        inputs=[file_identity(queries, role="queries")],
        payload={
            "candidates": file_identity(candidate_path),
            "index_artifact_id": index_artifact.artifact_id,
            "dense_store_artifact_id": loaded_artifact["artifact_id"],
            "source_inventory_digest": payload["source_inventory_digest"],
            "audit_artifact_id": audit_artifact.artifact_id,
            "n_probes": 4,
            "ann_candidate_count": 3,
        },
    ).commit(candidates)
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )
    reranked = tmp_path / "reranked"
    rerank_args = [
        "ann-rerank",
        "--queries",
        str(queries),
        "--candidates",
        str(candidate_path),
        "--dense-store-manifest",
        str(dense / "dense_store.json"),
        "--ann-index-manifest",
        str(index_dir / "ann_index.json"),
        "--ann-audit-manifest",
        str(audit_dir / "ann_audit.json"),
        "--query-embedding-contract",
        str(query_contract),
        "--top-k",
        "1",
        "--min-similarity",
        "0.5",
        "--duplicate-similarity",
        "0.99",
        "--device",
        "cpu",
        "--output-dir",
        str(reranked),
    ]
    return rerank_args, reranked, loaded_artifact, payload


def test_staged_rerank_rejects_out_of_range_exclusions(staged_candidates, tmp_path):
    rerank_args, _, loaded_artifact, payload = staged_candidates
    invalid_exclusion = tmp_path / "invalid_exclusion.json"
    invalid_exclusion.write_text(
        json.dumps(
            {
                "source_row_ids": [3],
                "dense_store_artifact_id": loaded_artifact["artifact_id"],
                "source_inventory_digest": payload["source_inventory_digest"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out-of-range source_row_id"):
        refinement_main(
            [*rerank_args, "--exclude", str(invalid_exclusion)]
        )


def test_staged_rerank_publishes_lineage_and_empty_exclusion(staged_candidates, tmp_path):
    rerank_args, reranked, loaded_artifact, _ = staged_candidates
    empty_exclusion = tmp_path / "empty_exclusion.parquet"
    pd.DataFrame({"sample_id": pd.Series(dtype=str)}).to_parquet(
        empty_exclusion, index=False
    )
    assert refinement_main(
        [*rerank_args, "--exclude", str(empty_exclusion)]
    ) == 0
    staged = pd.read_parquet(reranked / "neighbors.parquet")
    assert staged.iloc[0]["sample_id"] == "a"
    assert staged.iloc[0]["search_proof"] == (
        "ann_audited_exact_float32_rerank"
    )
    assert staged.iloc[0]["dense_store_artifact_id"] == (
        loaded_artifact["artifact_id"]
    )
    summary = json.loads(
        (reranked / "search_summary.json").read_text(encoding="utf-8")
    )
    assert summary["eligible_source_rows"] == 3
    assert summary["n_probes"] == 4



def test_dense_finalize_rejects_plan_tampering(committed_dense):
    dense, _, _, _, _ = committed_dense
    dense_manifest_before = (dense / "dense_store.json").read_bytes()
    plan_path = dense / "dense_store_plan.json"
    tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered_plan["encoder"] = {"name": "wrong"}
    plan_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan identity"):
        finalize_dense_store(plan_path=plan_path)
    assert (dense / "dense_store.json").read_bytes() == dense_manifest_before



def test_dense_verifier_rejects_vector_tampering(committed_dense):
    dense, _, _, _, _ = committed_dense
    vector_path = dense / "vectors.f32"
    vector_bytes = bytearray(vector_path.read_bytes())
    vector_bytes[0] ^= 1
    vector_path.write_bytes(vector_bytes)
    # Full checksum verification must catch corruption even when the storage
    # backend coalesces timestamps. The cheap load gate checks metadata only.
    with pytest.raises(RuntimeError, match="(identity|range) changed"):
        verify_dense_store_integrity(dense / "dense_store.json")
    vector_stat = vector_path.stat()
    os.utime(vector_path, ns=(vector_stat.st_atime_ns, vector_stat.st_mtime_ns + 2_000_000_000))
    with pytest.raises(RuntimeError, match="identity changed"):
        load_dense_store(dense / "dense_store.json")
