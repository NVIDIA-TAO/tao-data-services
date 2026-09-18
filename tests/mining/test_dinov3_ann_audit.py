# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise index construction, reload, recall audits and exact reranking."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nvidia_tao_ds.mining.dinov3.ann_index import load_ann_audit, require_experimental_ann
from nvidia_tao_ds.mining.dinov3.internal.refinement import main
from nvidia_tao_ds.mining.dinov3.ann_search import ann_exact_rerank_search
from nvidia_tao_ds.mining.dinov3.dense_store import initialize_dense_store, materialize_dense_shards, finalize_dense_store
from nvidia_tao_ds.mining.dinov3.store import register_embedding_store


@pytest.fixture
def dense_inputs(tmp_path, monkeypatch):
    """Build a real small source and dense store with deterministic random data."""
    monkeypatch.setenv("TAO_DINOV3_EXPERIMENTAL_ANN", "1")
    source = tmp_path / "source"
    source.mkdir()
    vectors = np.random.default_rng(42).normal(size=(512, 8)).astype(np.float32)
    pd.DataFrame({"sample_id": [f"source-{i}" for i in range(len(vectors))],
                  "embedding": list(vectors), "path": [f"/images/{i}.png" for i in range(len(vectors))],
                  "storage_type": "file"}).to_parquet(source / "part.parquet", index=False)
    register_embedding_store(store_root=source, output_dir=tmp_path / "registered", encoder={"name": "test"})
    dense = tmp_path / "dense"
    initialize_dense_store(source_store_manifest=tmp_path / "registered/embedding_store.json", output_dir=dense)
    plan = dense / "dense_store_plan.json"
    materialize_dense_shards(plan_path=plan, shard_indexes=[0])
    artifact = finalize_dense_store(plan_path=plan)
    queries = tmp_path / "queries.parquet"
    pd.DataFrame({"sample_id": [f"query-{i}" for i in range(6)],
                  "embedding": list(np.random.default_rng(43).normal(size=(6, 8)))}).to_parquet(queries, index=False)
    contract = tmp_path / "query_contract.json"
    contract.write_text(json.dumps({"encoder": artifact["payload"]["encoder"], "embedding_dim": 8}))
    return dense / "dense_store.json", queries, contract


@pytest.fixture
def fake_cuvs(monkeypatch):
    """Inject only cuVS; exercise the actual TAO artifact and selection code."""
    calls = []

    def build(params, vectors):
        calls.append(("build", params.n_lists))
        return np.asarray(vectors).copy()

    def save(index, filename):
        with open(filename, "wb") as stream:
            np.save(stream, index)

    def load(filename):
        calls.append(("load", filename))
        return np.load(filename)

    def search(params, index, queries, k):
        calls.append(("search", params.n_probes, k))
        scores = queries @ index.T
        ids = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return 2 - 2 * np.take_along_axis(scores, ids, axis=1), ids

    fake = SimpleNamespace(IndexParams=SimpleNamespace, SearchParams=SimpleNamespace,
                           build=build, save=save, load=load, search=search)
    monkeypatch.setitem(sys.modules, "cuvs.neighbors.mg", SimpleNamespace(ivf_pq=fake))
    return calls


def _build_and_audit(tmp_path, dense_inputs, minimum_recall=1.0):
    dense, queries, contract = dense_inputs
    assert main(["build-ann-index", "--dense-store-manifest", str(dense),
                 "--output-dir", str(tmp_path / "index"), "--n-lists", "4",
                 "--pq-dim", "4", "--pq-bits", "4", "--kmeans-train-fraction", "1.0"]) == 0
    assert main(["audit-ann-recall", "--queries", str(queries), "--dense-store-manifest", str(dense),
                 "--ann-index-manifest", str(tmp_path / "index/ann_index.json"),
                 "--query-embedding-contract", str(contract), "--output-dir", str(tmp_path / "audit"),
                 "--n-probes", "4", "--ann-candidates", "16", "--top-k", "3",
                 "--sample-size", "4", "--seed", "11", "--minimum-recall", str(minimum_recall)]) == 0
    return json.loads((tmp_path / "audit/artifact.json").read_text())


def test_index_audit_and_rerank_contract(tmp_path, dense_inputs, fake_cuvs):
    """A usable recall audit is produced, bound to its settings, and consumed."""
    artifact = _build_and_audit(tmp_path, dense_inputs)
    payload = artifact["payload"]
    assert payload["passed"] is True
    assert payload["mean_recall_at_k"] == 1.0
    assert payload["sample_size"] == 4
    arguments = {name: payload[name] for name in (
        "index_artifact_id", "n_probes", "ann_candidate_count", "dense_store_artifact_id",
        "source_inventory_digest", "vector_inventory_digest",
    )}
    load_ann_audit(tmp_path / "audit/ann_audit.json", **arguments)
    with pytest.raises(ValueError, match="n_probes"):
        load_ann_audit(tmp_path / "audit/ann_audit.json", **dict(arguments, n_probes=2))
    dense, queries, _ = dense_inputs
    result = ann_exact_rerank_search(query_path=queries, dense_store_manifest=dense,
                                   ann_index_manifest=tmp_path / "index/ann_index.json",
                                   top_k=1, min_similarity=0.0, n_probes=4,
                                   ann_candidate_count=16, device="cpu")
    assert len(result) == 6
    assert result["path"].str.startswith("/images/").all()
    assert ("build", 4) in fake_cuvs
    assert ("search", 4, 16) in fake_cuvs


def test_real_cuvs_index_audit_smoke(tmp_path, dense_inputs):
    """Run the real GPU library when present; release validation requires it."""
    try:
        from cuvs.neighbors.mg import ivf_pq  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        if os.environ.get("TAO_DEFT_REQUIRE_CUVS_SMOKE") == "1":
            raise
        pytest.skip("approved runtime lacks cuVS; ANN release validation remains blocked")
    artifact = _build_and_audit(tmp_path, dense_inputs, minimum_recall=0.5)
    assert artifact["payload"]["passed"] is True


def test_ann_actions_require_explicit_experimental_opt_in(monkeypatch):
    """The stock image must not advertise an unbundled dependency as ready."""
    monkeypatch.delenv("TAO_DINOV3_EXPERIMENTAL_ANN", raising=False)
    with pytest.raises(RuntimeError, match="experimental"):
        require_experimental_ann()


def test_candidate_npz_rerank_and_search_cli(tmp_path, dense_inputs, fake_cuvs):
    """Cross every CLI publication boundary using fake cuVS and real artifacts."""
    _build_and_audit(tmp_path, dense_inputs)
    dense, queries, contract = dense_inputs
    common = ["--queries", str(queries), "--query-embedding-contract", str(contract),
              "--ann-index-manifest", str(tmp_path / "index/ann_index.json"),
              "--ann-audit-manifest", str(tmp_path / "audit/ann_audit.json")]
    assert main(["ann-candidates", *common, "--n-probes", "4", "--ann-candidates", "16",
                 "--output-dir", str(tmp_path / "candidates")]) == 0
    for action in ("ann-rerank", "ann-search"):
        options = (["--candidates", str(tmp_path / "candidates/ann_candidates.npz")]
                   if action == "ann-rerank" else ["--n-probes", "4", "--ann-candidates", "16"])
        output = tmp_path / action
        assert main([action, *common, *options, "--dense-store-manifest", str(dense),
                     "--top-k", "1", "--min-similarity", "0", "--output-dir", str(output)]) == 0
        artifact = json.loads((output / "artifact.json").read_text())
        assert artifact["payload"]["search_proof"] == "ann_audited_exact_float32_rerank"
        assert len(pd.read_parquet(output / "neighbors.parquet")) == 6
    assert ("search", 4, 16) in fake_cuvs
