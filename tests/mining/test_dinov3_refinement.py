# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusable DINOv3 refinement data operations."""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from nvidia_tao_ds.mining.dinov3.materialize import materialize_manifest
from nvidia_tao_ds.mining.dinov3 import contracts
from nvidia_tao_ds.mining.dinov3.contracts import (
    ArtifactManifest,
    artifact_audit_id,
    artifact_content_id,
    file_identity,
    require_uncommitted_output,
)
from nvidia_tao_ds.mining.dinov3.search import exact_sharded_search, source_metadata_output_names
from nvidia_tao_ds.mining.dinov3.selection import (
    allocate_multitask_budgets,
    select_grit_targets,
    select_multitask_targets,
)
from nvidia_tao_ds.mining.dinov3.store import (
    bind_store_payload_contract,
    register_embedding_store,
)


def test_refinement_parquet_reads_disable_background_prebuffer() -> None:
    """Avoid Arrow shutdown races while retaining threaded column decoding."""
    root = Path(__file__).parents[2] / "nvidia_tao_ds/mining/dinov3"
    readers = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pd"
                and node.func.attr == "read_parquet"
            ):
                options = {keyword.arg: keyword.value for keyword in node.keywords}
                assert "pre_buffer" in options, f"{path}:{node.lineno}"
                assert isinstance(options["pre_buffer"], ast.Constant)
                assert options["pre_buffer"].value is False
                readers.append((path, node.lineno))
    assert readers


def test_packaged_artifact_schema_is_canonical_yaml() -> None:
    schemas = (
        Path(__file__).parents[2]
        / "nvidia_tao_ds/mining/dinov3/schemas"
    )
    yaml_schema = yaml.safe_load(
        (schemas / "artifact.schema.yaml").read_text(encoding="utf-8")
    )
    assert yaml_schema["$id"].endswith("artifact-1.0.schema.yaml")
    assert yaml_schema["required"] == [
        "artifact_id",
        "artifact_type",
        "schema_version",
        "producer",
        "inputs",
        "payload",
        "created_at",
    ]


def test_file_identity_rejects_mutation_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"stable")
    original = contracts.os.fstat
    calls = 0

    def changed_size(file_descriptor: int):
        nonlocal calls
        stat = original(file_descriptor)
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                st_size=stat.st_size + 1,
                st_mtime_ns=stat.st_mtime_ns,
                st_ctime_ns=stat.st_ctime_ns,
            )
        return stat

    monkeypatch.setattr(contracts.os, "fstat", changed_size)
    with pytest.raises(RuntimeError, match="changed while its identity"):
        file_identity(payload)


@pytest.mark.parametrize("backend", ["sharded"])
def test_search_metadata_cannot_replace_selection_fields(tmp_path, backend):
    """Source annotations must not overwrite computed identities or scores."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "part.parquet"
    pd.DataFrame({
        "sample_id": ["source"], "embedding": [[0.8, 0.6]],
        "storage_type": ["file"], "path": ["/images/source.png"],
        "query_id": ["wrong"], "cosine_similarity": [-42.0], "rank": [99],
        "source_row_id": [99], "search_proof": ["wrong"],
        "annotation": ["preserved"],
        "image_size": [[640, 480]],
        "attributes": [{"label": "preserved", "confidence": 0.9}],
    }).to_parquet(source, index=False)
    queries = tmp_path / "queries.parquet"
    pd.DataFrame({"sample_id": ["query"], "embedding": [[1.0, 0.0]]}).to_parquet(
        queries, index=False,
    )
    if backend == "sharded":
        result = exact_sharded_search(
            query_path=queries, source_parts=[source], top_k=1, min_similarity=0.5,
        )
        assert result.iloc[0]["source_row_id"] == 0
        assert result.iloc[0]["search_proof"] == "exact_all_declared_shards"
    row = result.iloc[0]
    assert row["sample_id"] == "source"
    assert row["query_id"] == "query"
    assert row["cosine_similarity"] == pytest.approx(0.8)
    assert row["rank"] == 1
    assert row["annotation"] == "preserved"
    assert list(row["image_size"]) == [640, 480]
    assert row["attributes"] == {"label": "preserved", "confidence": 0.9}
    assert "adaptive_radius" in result.attrs
    roundtrip_path = tmp_path / "neighbors.parquet"
    result.to_parquet(roundtrip_path, index=False)
    roundtrip = pd.read_parquet(roundtrip_path, pre_buffer=False).iloc[0]
    assert list(roundtrip["image_size"]) == [640, 480]
    assert roundtrip["attributes"] == row["attributes"]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), 1e30])
@pytest.mark.parametrize("invalid_table", ["query", "source"])
def test_exact_search_rejects_nonfinite_embeddings(tmp_path, bad_value, invalid_table):
    """Invalid embeddings must fail instead of proving false corpus exhaustion."""
    queries = tmp_path / "queries.parquet"
    source = tmp_path / "source.parquet"
    query_vector = [bad_value, 0.0] if invalid_table == "query" else [1.0, 0.0]
    source_vector = [bad_value, 0.0] if invalid_table == "source" else [0.8, 0.6]
    pd.DataFrame({"sample_id": ["query"], "embedding": [query_vector]}).to_parquet(queries)
    pd.DataFrame({"sample_id": ["source"], "embedding": [source_vector]}).to_parquet(source)
    with pytest.raises(ValueError, match="non-finite"):
        exact_sharded_search(
            query_path=queries, source_parts=[source], top_k=1, min_similarity=0.5,
        )


@pytest.mark.parametrize("bad_value", [1e30, -1e30])
def test_store_registration_rejects_overflowing_embedding_norms(tmp_path, bad_value):
    """Finite components must not allow unusable cosine vectors to be committed."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame({
        "sample_id": ["source"], "embedding": [[bad_value, bad_value]],
        "storage_type": ["file"], "path": ["/images/source.png"],
    }).to_parquet(source_root / "part.parquet", index=False)
    output_dir = tmp_path / "store"
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="norms.*non-finite"):
        register_embedding_store(
            store_root=source_root, output_dir=output_dir, encoder={"name": "test"},
        )
    assert not (output_dir / "_SUCCESS").exists()


@pytest.mark.parametrize("storage_type", ["unsupported", "tar", "zip"])
def test_store_default_does_not_hide_explicit_invalid_locators(tmp_path, storage_type):
    """Defaults apply only to omitted columns, never explicit storage types."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame({
        "sample_id": ["source"], "embedding": [[1.0, 0.0]],
        "storage_type": [storage_type], "path": ["/images/source"],
    }).to_parquet(source_root / "part.parquet", index=False)
    with pytest.raises(ValueError, match="unsupported storage types|archive rows require member"):
        register_embedding_store(
            store_root=source_root, output_dir=tmp_path / "store",
            encoder={"name": "test"}, default_storage_type="file",
        )


def test_grit_is_ranked_within_each_task() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "task": ["x", "x", "y", "y"],
            "grit_score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    selected = select_grit_targets(frame, fraction=0.5)
    assert set(selected["sample_id"]) == {"a", "c"}
    assert selected.groupby("task").size().to_dict() == {"x": 1, "y": 1}


def test_grit_selection_requires_model_owned_score() -> None:
    with pytest.raises(ValueError, match="grit_score"):
        select_grit_targets(
            pd.DataFrame({"sample_id": ["a"], "task": ["x"]}),
            fraction=1.0,
        )


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_materialize_rejects_missing_sample_identity(tmp_path, previous, bad_id):
    valid = pd.DataFrame({"sample_id": ["valid"], "storage_type": ["file"], "path": ["/data/a.png"]})
    invalid = valid.assign(sample_id=bad_id)
    delta_path = tmp_path / "delta.parquet"
    previous_path = tmp_path / "previous.parquet"
    (valid if previous else invalid).to_parquet(delta_path, index=False)
    invalid.to_parquet(previous_path, index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="sample IDs"):
        materialize_manifest(
            delta_path=delta_path, output_dir=output,
            previous_path=previous_path if previous else None,
        )
    assert not (output / "_SUCCESS").exists()


def test_multitask_round_robin_deduplicates_and_fills() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["shared", "a", "b", "shared", "c", "d"],
            "task": ["x", "x", "x", "y", "y", "y"],
            "weakness_score": [3.0, 2.0, 1.0, 9.0, 8.0, 7.0],
        }
    )
    selected = select_multitask_targets(frame, per_task=2)
    assert selected["sample_id"].is_unique
    assert selected.groupby("task").size().to_dict() == {"x": 2, "y": 2}


def test_multitask_round_robin_does_not_let_first_task_monopolize_shared_ids(
) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["shared-1", "shared-2", "x-only", "shared-1", "shared-2", "y-only"],
            "task": ["x", "x", "x", "y", "y", "y"],
            "weakness_score": [9.0, 8.0, 7.0, 9.0, 8.0, 7.0],
        }
    )
    selected = select_multitask_targets(frame, per_task=2)
    assert selected[["task", "sample_id"]].to_records(index=False).tolist() == [
        ("x", "shared-1"),
        ("y", "shared-2"),
        ("x", "x-only"),
        ("y", "y-only"),
    ]


def test_multitask_weights_preserve_total_budget() -> None:
    assert allocate_multitask_budgets(
        ["resistor", "wistron", "tri"],
        total=384,
        task_weights={"resistor": 2.0},
    ) == {"resistor": 192, "tri": 96, "wistron": 96}

    frame = pd.DataFrame(
        {
            "sample_id": [
                f"{task}-{index}" for task in ("x", "y") for index in range(8)
            ],
            "task": [task for task in ("x", "y") for _ in range(8)],
            "weakness_score": list(range(8)) * 2,
        }
    )
    selected = select_multitask_targets(
        frame, total=8, task_weights={"x": 3.0, "y": 1.0}
    )
    assert selected.groupby("task").size().to_dict() == {"x": 6, "y": 2}


def test_balanced_multitask_preserves_inactive_task_budget() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"x-{index}" for index in range(12)],
            "task": ["x"] * 12,
            "weakness_score": list(range(12)),
        }
    )
    selected = select_multitask_targets(
        frame,
        total=12,
        configured_tasks=["x", "y", "z"],
        preserve_unfilled_budget=True,
    )
    assert selected.groupby("task").size().to_dict() == {"x": 4}


def test_multitask_weights_reject_unknown_or_nonpositive_tasks() -> None:
    with pytest.raises(ValueError, match="unknown tasks"):
        allocate_multitask_budgets(["x", "y"], total=4, task_weights={"z": 2.0})
    with pytest.raises(ValueError, match="finite and positive"):
        allocate_multitask_budgets(["x", "y"], total=4, task_weights={"x": 0.0})


def test_multitask_selection_rejects_unknown_weight() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "task": ["resistor", "wistron"],
            "weakness_score": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="unknown tasks"):
        select_multitask_targets(
            frame, total=2, task_weights={"resitstor": 10.0}
        )


def test_multitask_empty_active_set_remains_a_clean_stop() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": pd.Series(dtype="str"),
            "task": pd.Series(dtype="str"),
            "weakness_score": pd.Series(dtype="float64"),
        }
    )
    selected = select_multitask_targets(frame, total=8, task_weights={"x": 2.0})
    assert selected.empty
    assert {"target_rank", "strategy"}.issubset(selected.columns)


def test_exact_search_and_manifest_materialization(tmp_path: Path) -> None:
    queries = pd.DataFrame({"sample_id": ["q"], "embedding": [[1.0, 0.0]]})
    source = pd.DataFrame(
        {
            "sample_id": ["near", "far"],
            "embedding": [[0.9, 0.1], [0.0, 1.0]],
            "path": [str(tmp_path / "near.jpg"), str(tmp_path / "far.jpg")],
            "storage_type": ["file", "file"],
            "member": [None, None],
        }
    )
    query_path = tmp_path / "queries.parquet"
    source_path = tmp_path / "source.parquet"
    queries.to_parquet(query_path, index=False)
    source.to_parquet(source_path, index=False)

    neighbors = exact_sharded_search(
        query_path=query_path,
        source_parts=[source_path],
        top_k=1,
        min_similarity=0.5,
    )
    assert neighbors.iloc[0]["sample_id"] == "near"
    assert neighbors.iloc[0]["path"] == str(tmp_path / "near.jpg")
    assert neighbors.iloc[0]["search_device"] == "cpu"
    assert np.isclose(neighbors.iloc[0]["cosine_similarity"], 0.9938837)

    delta_path = tmp_path / "delta.parquet"
    source.iloc[:1].drop(columns="embedding").to_parquet(delta_path, index=False)
    result = materialize_manifest(delta_path=delta_path, output_dir=tmp_path / "round")
    assert result["payload"]["row_count"] == 1
    assert (tmp_path / "round" / "_SUCCESS").is_file()


def test_exact_search_deduplicates_embeddings_and_expands_radius(
    tmp_path: Path,
) -> None:
    queries = pd.DataFrame({"sample_id": ["q"], "embedding": [[1.0, 0.0]]})
    source = pd.DataFrame(
        {
            "sample_id": ["query-copy", "near", "near-copy", "expanded"],
            "embedding": [
                [1.0, 0.0],
                [0.90, 0.4358899],
                [0.9001, 0.4356834],
                [0.70, 0.7141428],
            ],
            "path": ["query.jpg", "near.jpg", "near-copy.jpg", "expanded.jpg"],
            "storage_type": ["file"] * 4,
            "member": [None] * 4,
        }
    )
    query_path = tmp_path / "queries.parquet"
    source_path = tmp_path / "source.parquet"
    queries.to_parquet(query_path, index=False)
    source.to_parquet(source_path, index=False)

    result = exact_sharded_search(
        query_path=query_path,
        source_parts=[source_path],
        top_k=2,
        min_similarity=0.85,
        hard_min_similarity=0.65,
        similarity_step=0.10,
        duplicate_similarity=0.995,
        candidate_multiplier=4,
    )
    assert result.iloc[1]["sample_id"] == "expanded"
    assert result.iloc[0]["sample_id"] in {"near", "near-copy"}
    assert result["accepted_at_similarity_threshold"].tolist() == pytest.approx(
        [0.85, 0.65]
    )
    stats = result.attrs["adaptive_radius"]["query_stats"][0]
    assert stats["rejected_query_duplicate"] == 1
    assert stats["rejected_selected_duplicate"] == 1
    assert stats["selected_count"] == 2
    assert stats["candidate_truncated"] is False
    assert result.attrs["adaptive_radius"]["underfill_exhaustion_proven"] is True


def test_exact_search_does_not_claim_exhaustion_after_candidate_truncation(
    tmp_path: Path,
) -> None:
    queries = pd.DataFrame({"sample_id": ["q"], "embedding": [[1.0, 0.0]]})
    source = pd.DataFrame(
        {
            "sample_id": ["copy-a", "copy-b", "valid"],
            "embedding": [[1.0, 0.0], [1.0, 0.0], [0.9, 0.4358899]],
        }
    )
    query_path = tmp_path / "queries.parquet"
    source_path = tmp_path / "source.parquet"
    queries.to_parquet(query_path, index=False)
    source.to_parquet(source_path, index=False)

    result = exact_sharded_search(
        query_path=query_path,
        source_parts=[source_path],
        top_k=2,
        min_similarity=0.8,
        duplicate_similarity=0.995,
        candidate_multiplier=1,
    )
    stats = result.attrs["adaptive_radius"]
    assert stats["candidate_truncated"] is True
    assert stats["underfill_exhaustion_proven"] is False
    assert stats["query_stats"][0]["candidate_count_above_hard_floor"] == 3


def test_register_store_audits_identity_and_applies_locator_default(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "path": ["/a.jpg", "/b.jpg"],
        }
    )
    source.to_parquet(tmp_path / "part.parquet", index=False)
    result = register_embedding_store(
        store_root=tmp_path,
        output_dir=tmp_path / "registered",
        encoder={"name": "test"},
        hash_content=False,
        default_storage_type="file",
    )
    assert result["payload"]["row_count"] == 2
    assert result["payload"]["locator_defaults"] == {"storage_type": "file"}


def test_register_store_binds_immutable_payload_contract(tmp_path: Path) -> None:
    source_root = tmp_path / "embeddings"
    source_root.mkdir()
    payload_root = tmp_path / "images"
    payload_root.mkdir()
    source = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "path": [str(payload_root / "a.jpg"), str(payload_root / "b.jpg")],
            "storage_type": ["file", "file"],
        }
    )
    source.to_parquet(source_root / "part.parquet", index=False)
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-images",
                        "version": "v1",
                        "root_uri": payload_root.resolve().as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = register_embedding_store(
        store_root=source_root,
        output_dir=tmp_path / "registered",
        encoder={"name": "test"},
        source_payload_contract=contract,
    )

    assert result["inputs"] == [
        file_identity(contract, role="source_payload_contract")
    ]
    binding = result["payload"]["source_payload_contract"]
    assert binding["sha256"] == result["inputs"][0]["sha256"]
    assert binding["locator_audit"]["row_count"] == 2
    assert binding["locator_audit"]["digest"].startswith("sha256:")
    assert result["payload"]["shards"][0]["sha256"].startswith("sha256:")

    with pytest.raises(ValueError, match="hash_content=True"):
        register_embedding_store(
            store_root=source_root,
            output_dir=tmp_path / "unhashed",
            encoder={"name": "test"},
            hash_content=False,
            source_payload_contract=contract,
        )


def test_register_store_rejects_locator_outside_payload_roots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "embeddings"
    source_root.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "embedding": [[1.0, 0.0]],
            "path": ["/outside/a.jpg"],
            "storage_type": ["file"],
        }
    ).to_parquet(source_root / "part.parquet", index=False)
    payload_root = tmp_path / "images"
    payload_root.mkdir()
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-images",
                        "version": "v1",
                        "root_uri": payload_root.resolve().as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside every contracted root"):
        register_embedding_store(
            store_root=source_root,
            output_dir=tmp_path / "registered",
            encoder={"name": "test"},
            source_payload_contract=contract,
        )


def test_bind_existing_store_to_payload_contract_without_reencoding(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "embeddings"
    payload_root = tmp_path / "images"
    store_root.mkdir()
    payload_root.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "path": [str(payload_root / "a.jpg"), str(payload_root / "b.jpg")],
            "storage_type": ["file", "file"],
        }
    ).to_parquet(store_root / "part.parquet", index=False)
    original_dir = tmp_path / "original"
    original = register_embedding_store(
        store_root=store_root,
        output_dir=original_dir,
        encoder={"name": "test"},
    )
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-images",
                        "version": "v1",
                        "root_uri": payload_root.resolve().as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bound = bind_store_payload_contract(
        source_store_manifest=original_dir / "embedding_store.json",
        source_payload_contract=contract,
        output_dir=tmp_path / "bound",
    )

    assert bound["payload"]["shards"] == original["payload"]["shards"]
    assert bound["payload"]["parent_store_artifact_id"] == original["artifact_id"]
    assert bound["payload"]["source_payload_contract"]["locator_audit"][
        "row_count"
    ] == 2
    assert bound["payload"]["source_payload_contract"]["locator_audit"][
        "algorithm"
    ] == "sealed_shard_inventory_and_canonical_root_prefix_v1"
    verification = bound["payload"]["content_verification"]
    assert verification["algorithm"] == "sha256_each_shard_with_posix_stat_v1"
    assert verification["inventory_digest"] == original["payload"][
        "inventory_digest"
    ]
    assert len(verification["shards"]) == 1
    assert bound["inputs"][0]["artifact_id"] == original["artifact_id"]
    assert bound["inputs"][1] == file_identity(
        contract, role="source_payload_contract"
    )


def test_bind_existing_store_accepts_contracted_symlink_namespace(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "embeddings"
    payload_root = tmp_path / "images"
    payload_alias = tmp_path / "images-alias"
    store_root.mkdir()
    payload_root.mkdir()
    payload_alias.symlink_to(payload_root, target_is_directory=True)
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "embedding": [[1.0, 0.0]],
            "path": [str(payload_alias / "a.jpg")],
            "storage_type": ["file"],
        }
    ).to_parquet(store_root / "part.parquet", index=False)
    original_dir = tmp_path / "original"
    register_embedding_store(
        store_root=store_root,
        output_dir=original_dir,
        encoder={"name": "test"},
    )
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-images",
                        "version": "v1",
                        "root_uri": payload_alias.as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bound = bind_store_payload_contract(
        source_store_manifest=original_dir / "embedding_store.json",
        source_payload_contract=contract,
        output_dir=tmp_path / "bound",
    )

    assert bound["payload"]["source_payload_contract"]["locator_audit"][
        "row_count"
    ] == 1


def test_bind_existing_store_rejects_changed_shard_content(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "embeddings"
    payload_root = tmp_path / "images"
    store_root.mkdir()
    payload_root.mkdir()
    shard = store_root / "part.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "embedding": [[1.0, 0.0]],
            "path": [str(payload_root / "a.jpg")],
            "storage_type": ["file"],
        }
    ).to_parquet(shard, index=False)
    original_dir = tmp_path / "original"
    register_embedding_store(
        store_root=store_root,
        output_dir=original_dir,
        encoder={"name": "test"},
    )
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "immutability": "immutable",
                "datasets": [
                    {
                        "dataset_id": "test-images",
                        "version": "v1",
                        "root_uri": payload_root.resolve().as_uri(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    changed = bytearray(shard.read_bytes())
    changed[-1] ^= 1
    shard.write_bytes(changed)

    with pytest.raises(ValueError, match="digest changed"):
        bind_store_payload_contract(
            source_store_manifest=original_dir / "embedding_store.json",
            source_payload_contract=contract,
            output_dir=tmp_path / "bound",
        )


def test_materialize_canonicalizes_only_absolute_legacy_file_paths(
    tmp_path: Path,
) -> None:
    delta = tmp_path / "legacy.parquet"
    pd.DataFrame(
        {"sample_id": ["a"], "path": ["/data/a.jpg"]}
    ).to_parquet(delta, index=False)
    result = materialize_manifest(delta_path=delta, output_dir=tmp_path / "output")
    manifest = pd.read_parquet(tmp_path / "output/training_manifest.parquet")
    assert manifest["storage_type"].tolist() == ["file"]
    assert result["payload"]["locator_inference"] == {
        "storage_type": {
            "value": "file",
            "rows": 1,
            "rule": "absolute_local_path_v1",
        }
    }
    assert result["producer"]["implementation_sha256"].startswith("sha256:")

    relative = tmp_path / "relative.parquet"
    pd.DataFrame(
        {"sample_id": ["b"], "path": ["data/b.jpg"]}
    ).to_parquet(relative, index=False)
    with pytest.raises(ValueError, match="non-absolute paths"):
        materialize_manifest(
            delta_path=relative, output_dir=tmp_path / "relative-output"
        )

    remote_uri = tmp_path / "remote-uri.parquet"
    pd.DataFrame(
        {"sample_id": ["c"], "path": ["file://remote-host/data/c.jpg"]}
    ).to_parquet(remote_uri, index=False)
    with pytest.raises(ValueError, match="non-absolute.*paths"):
        materialize_manifest(
            delta_path=remote_uri, output_dir=tmp_path / "remote-uri-output"
        )


def test_materialize_rejects_duplicate_parent_and_records_replay(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous.parquet"
    delta = tmp_path / "delta.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "storage_type": ["file", "file"],
            "path": [str(tmp_path / "a.jpg"), str(tmp_path / "a-copy.jpg")],
        }
    ).to_parquet(previous, index=False)
    pd.DataFrame(
        {
            "sample_id": ["b"],
            "storage_type": ["file"],
            "path": [str(tmp_path / "b.jpg")],
        }
    ).to_parquet(delta, index=False)
    with pytest.raises(ValueError, match="Previous manifest contains duplicate"):
        materialize_manifest(
            delta_path=delta,
            previous_path=previous,
            output_dir=tmp_path / "duplicate-parent",
        )

    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": [str(tmp_path / "a.jpg")],
        }
    ).to_parquet(previous, index=False)
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "storage_type": ["file", "file"],
            "path": [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
        }
    ).to_parquet(delta, index=False)
    result = materialize_manifest(
        delta_path=delta,
        previous_path=previous,
        output_dir=tmp_path / "replay",
        overlap_policy="drop_existing",
    )
    assert result["payload"]["replayed_rows"] == 1
    assert result["payload"]["delta_rows"] == 1
    assert pd.read_parquet(
        tmp_path / "replay" / "training_manifest.parquet"
    )["sample_id"].tolist() == ["a", "b"]


def test_materialize_publishes_task_balanced_training_view(tmp_path: Path) -> None:
    queries = tmp_path / "queries.parquet"
    delta = tmp_path / "delta.parquet"
    pd.DataFrame(
        {
            "sample_id": ["query-x", "query-y"],
            "task": ["x", "y"],
        }
    ).to_parquet(queries, index=False)
    pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "query_id": ["query-x", "query-x", "query-x", "query-y"],
            "storage_type": ["file"] * 4,
            "path": [str(tmp_path / f"{name}.jpg") for name in "abcd"],
        }
    ).to_parquet(delta, index=False)

    result = materialize_manifest(
        delta_path=delta,
        query_path=queries,
        balance_column="query_task",
        output_dir=tmp_path / "balanced",
    )

    cumulative = pd.read_parquet(
        tmp_path / "balanced" / "training_manifest.parquet"
    )
    training = pd.read_parquet(
        tmp_path / "balanced" / "balanced_training_manifest.parquet"
    )
    assert cumulative.groupby("query_task").size().to_dict() == {"x": 3, "y": 1}
    assert training.groupby("query_task").size().to_dict() == {"x": 3, "y": 3}
    assert set(training["sample_id"]) == {"a", "b", "c", "d"}
    assert result["payload"]["training_view_rows"] == 6
    assert result["payload"]["balance"]["policy"] == (
        "oversample_each_task_to_largest_group_v1"
    )


def test_register_store_rejects_duplicate_identity(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["same", "same"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "path": ["/a.jpg", "/b.jpg"],
            "storage_type": ["file", "file"],
        }
    )
    source.to_parquet(tmp_path / "part.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        register_embedding_store(
            store_root=tmp_path,
            output_dir=tmp_path / "registered",
            encoder={"name": "test"},
            hash_content=False,
        )


def test_artifact_commit_recovers_missing_success_marker(tmp_path: Path) -> None:
    artifact = ArtifactManifest(
        artifact_type="test",
        producer={"action": "test", "version": "1.0"},
        inputs=[],
        payload={"value": 1},
    )
    artifact.commit(tmp_path)
    (tmp_path / "_SUCCESS").unlink()

    assert require_uncommitted_output(tmp_path) == tmp_path
    artifact.commit(tmp_path)

    assert (tmp_path / "_SUCCESS").read_text(encoding="utf-8").strip() == (
        artifact.artifact_id
    )


def test_artifact_content_identity_is_portable_but_audit_identity_is_local(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "relocated/second.bin"
    first.write_bytes(b"same payload")
    second.parent.mkdir()
    second.write_bytes(first.read_bytes())
    manifests = []
    for path, implementation in ((first, "a" * 64), (second, "b" * 64)):
        manifests.append({
            "artifact_type": "portable_test",
            "schema_version": "1.0",
            "identity_version": "2.0",
            "producer": {
                "action": "test", "version": "1.0",
                "implementation_sha256": f"sha256:{implementation}",
            },
            "inputs": [file_identity(path)],
            "payload": {
                "source_store_manifest": str(path),
                "result_uri": path.resolve().as_uri(),
                "sha256": file_identity(path)["sha256"],
            },
        })
    assert artifact_content_id(manifests[0]) == artifact_content_id(manifests[1])
    assert artifact_audit_id(manifests[0]) != artifact_audit_id(manifests[1])


def test_registered_store_identity_survives_filesystem_relocation(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    shard = first_root / "part.parquet"
    pd.DataFrame({
        "sample_id": ["a"],
        "embedding": [[1.0, 0.0]],
        "path": ["/images/a.jpg"],
        "storage_type": ["file"],
    }).to_parquet(shard, index=False)
    (second_root / shard.name).write_bytes(shard.read_bytes())
    first = register_embedding_store(
        store_root=first_root,
        output_dir=tmp_path / "registered-first",
        encoder={"name": "test", "uri": "ngc://encoder/v1"},
    )
    second = register_embedding_store(
        store_root=second_root,
        output_dir=tmp_path / "registered-second",
        encoder={"name": "test", "uri": "ngc://encoder/v1"},
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert first["audit_id"] != second["audit_id"]


def test_materialized_manifest_identity_survives_output_relocation(
    tmp_path: Path,
) -> None:
    delta = tmp_path / "delta.parquet"
    pd.DataFrame({
        "sample_id": ["a"],
        "storage_type": ["file"],
        "path": ["/images/a.jpg"],
    }).to_parquet(delta, index=False)
    first = materialize_manifest(delta_path=delta, output_dir=tmp_path / "first")
    second = materialize_manifest(delta_path=delta, output_dir=tmp_path / "second")
    assert first["artifact_id"] == second["artifact_id"]
    assert first["audit_id"] != second["audit_id"]


def test_exact_search_cpu_cuda_decisions_match(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for decision-parity QA")
    queries = tmp_path / "queries.parquet"
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "sample_id": ["q0", "q1"],
        "embedding": [[1.0, 0.0], [0.0, 1.0]],
    }).to_parquet(queries, index=False)
    pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "embedding": [[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]],
    }).to_parquet(source, index=False)
    options = dict(
        query_path=queries,
        source_parts=[source],
        top_k=1,
        min_similarity=0.5,
        duplicate_similarity=0.999,
        query_block_rows=1,
        source_block_rows=2,
    )
    cpu = exact_sharded_search(**options, device="cpu")
    cuda = exact_sharded_search(**options, device="cuda:0")
    comparable = [
        "query_id", "sample_id", "source_row_id", "rank",
        "accepted_at_similarity_threshold",
    ]
    pd.testing.assert_frame_equal(
        cpu[comparable].reset_index(drop=True),
        cuda[comparable].reset_index(drop=True),
    )
    np.testing.assert_allclose(
        cpu["cosine_similarity"], cuda["cosine_similarity"], atol=1e-6, rtol=0,
    )


def test_source_metadata_prefix_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="collide"):
        source_metadata_output_names(["query_id", "source_query_id"])


def test_materialize_accepts_empty_delta_as_typed_noop(tmp_path: Path) -> None:
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({"sample_id": pd.Series(dtype="string")}).to_parquet(
        empty, index=False
    )
    first = materialize_manifest(
        delta_path=empty, output_dir=tmp_path / "first"
    )
    first_frame = pd.read_parquet(
        tmp_path / "first/training_manifest.parquet", pre_buffer=False
    )
    assert first["payload"]["row_count"] == 0
    assert {"sample_id", "storage_type", "path", "member"}.issubset(
        first_frame.columns
    )

    previous = tmp_path / "previous.parquet"
    pd.DataFrame({
        "sample_id": ["existing"],
        "storage_type": ["file"],
        "path": [str(tmp_path / "existing.png")],
    }).to_parquet(previous, index=False)
    resumed = materialize_manifest(
        delta_path=empty,
        previous_path=previous,
        output_dir=tmp_path / "resumed",
    )
    assert resumed["payload"]["delta_rows"] == 0
    assert pd.read_parquet(
        tmp_path / "resumed/training_manifest.parquet", pre_buffer=False
    )["sample_id"].tolist() == ["existing"]
