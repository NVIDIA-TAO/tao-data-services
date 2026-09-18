# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusable DINOv3 refinement data operations."""

import ast
from concurrent.futures import ThreadPoolExecutor
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
from nvidia_tao_ds.mining.dinov3.internal.refinement import (
    _read_excluded_row_ids,
    main as refinement_main,
)
from nvidia_tao_ds.mining.dinov3.dense_store import (
    finalize_dense_store,
    initialize_dense_store,
    load_dense_store,
    materialize_dense_shards,
    verify_dense_store_integrity,
)
from nvidia_tao_ds.mining.dinov3.ann_index import load_ann_index
from nvidia_tao_ds.mining.dinov3.ann_search import (
    exact_rerank_ann_candidates,
    retrieve_ann_candidates,
)
from nvidia_tao_ds.mining.dinov3 import dense_search
from nvidia_tao_ds.mining.dinov3 import dense_store
from nvidia_tao_ds.mining.dinov3 import contracts
from nvidia_tao_ds.mining.dinov3 import search as search_module
from nvidia_tao_ds.mining.dinov3.contracts import (
    ArtifactManifest,
    artifact_audit_id,
    artifact_content_id,
    file_identity,
    require_uncommitted_output,
)
from nvidia_tao_ds.mining.dinov3.search import (
    exact_sharded_search,
    source_metadata_output_names,
)
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


def test_artifact_identity_v2_preserves_semantic_encoder_uri_and_legacy_ids() -> None:
    legacy = {
        "artifact_type": "legacy",
        "schema_version": "1.0",
        "producer": {"action": "test", "implementation_sha256": "sha256:" + "a" * 64},
        "inputs": [],
        "payload": {"encoder": {"uri": "ngc://model/a"}},
    }
    expected_legacy = contracts.canonical_digest(legacy)
    assert artifact_content_id(legacy) == expected_legacy

    versioned = {**legacy, "identity_version": "2.0"}
    changed = {
        **versioned,
        "payload": {"encoder": {"uri": "ngc://model/b"}},
    }
    assert artifact_content_id(versioned) != artifact_content_id(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inputs", ["not-an-object"], "inputs must contain objects"),
        ("created_at", "2026-01-01", "ISO-8601"),
        ("created_at", "2026-01-01T00:00:00", "ISO-8601"),
    ],
)
def test_serialized_artifact_validation_is_schema_complete(
    field, value, message
) -> None:
    artifact = ArtifactManifest(
        artifact_type="test",
        producer={"action": "test"},
        inputs=[],
        payload={},
    ).to_dict()
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        contracts.validate_serialized_artifact(artifact)


def test_artifact_commit_serializes_conflicting_publishers(tmp_path: Path) -> None:
    artifacts = [
        ArtifactManifest(
            artifact_type="test",
            producer={"action": "test"},
            inputs=[],
            payload={"value": value},
        )
        for value in (1, 2)
    ]

    def commit(artifact):
        try:
            artifact.commit(
                tmp_path,
                json_payloads={
                    "payload.json": artifact.payload,
                },
            )
            return artifact.artifact_id
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, artifacts))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    assert json.loads((tmp_path / "artifact.json").read_text())["artifact_id"] == winners[0]
    assert (tmp_path / "_SUCCESS").read_text().strip() == winners[0]
    manifest = json.loads((tmp_path / "artifact.json").read_text())
    assert json.loads((tmp_path / "payload.json").read_text()) == manifest["payload"]


def test_artifact_commit_adopts_matching_file_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = ArtifactManifest(
        artifact_type="test",
        producer={"action": "test"},
        inputs=[],
        payload={"payload_sha256": contracts.canonical_digest("same")},
    )
    first = tmp_path / "first.tmp"
    first.write_bytes(b"same")
    original_write = contracts.write_json_atomic

    def fail_manifest(path, value):
        if Path(path).name == "artifact.json":
            raise OSError("injected manifest failure")
        return original_write(path, value)

    monkeypatch.setattr(contracts, "write_json_atomic", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        artifact.commit(
            tmp_path, file_publications=[(first, tmp_path / "payload.bin")]
        )
    assert (tmp_path / "payload.bin").read_bytes() == b"same"
    assert not (tmp_path / "artifact.json").exists()
    assert not (tmp_path / "_SUCCESS").exists()

    second = tmp_path / "second.tmp"
    second.write_bytes(b"same")
    monkeypatch.setattr(contracts, "write_json_atomic", original_write)
    artifact.commit(
        tmp_path, file_publications=[(second, tmp_path / "payload.bin")]
    )
    assert (tmp_path / "_SUCCESS").read_text().strip() == artifact.artifact_id


def test_artifact_recovery_validates_payload_before_success_marker(
    tmp_path: Path,
) -> None:
    artifact = ArtifactManifest(
        artifact_type="test",
        producer={"action": "test"},
        inputs=[],
        payload={"value": 1},
    )
    artifact.commit(tmp_path, json_payloads={"payload.json": {"value": 1}})
    (tmp_path / "_SUCCESS").unlink()
    (tmp_path / "payload.json").write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload differs"):
        artifact.commit(tmp_path, json_payloads={"payload.json": {"value": 1}})
    assert not (tmp_path / "_SUCCESS").exists()


def test_artifact_recovery_rejects_corrupt_binary_before_success_marker(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"original")
    artifact = ArtifactManifest(
        artifact_type="test",
        producer={"action": "test"},
        inputs=[],
        payload={"sha256": file_identity(staged)["sha256"]},
    )
    artifact.commit(
        tmp_path, file_publications=[(staged, tmp_path / "payload.bin")]
    )
    (tmp_path / "_SUCCESS").unlink()
    (tmp_path / "payload.bin").write_bytes(b"corrupt!")
    recovery_source = tmp_path / "recovery.bin"
    recovery_source.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="file payload differs"):
        artifact.commit(
            tmp_path,
            file_publications=[(recovery_source, tmp_path / "payload.bin")],
        )
    assert not (tmp_path / "_SUCCESS").exists()


def test_refinement_cli_serializes_payload_generation(tmp_path: Path) -> None:
    score_paths = []
    for index, winner in enumerate(("a", "b")):
        path = tmp_path / f"scores-{index}.parquet"
        pd.DataFrame({
            "sample_id": [winner],
            "task": ["domain"],
            "grit_score": [1.0],
        }).to_parquet(path, index=False)
        score_paths.append(path)
    output = tmp_path / "selection"

    def run(path):
        try:
            return refinement_main([
                "select-grit", "--scores", str(path), "--fraction", "1",
                "--output-dir", str(output),
            ])
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, score_paths))
    assert sorted(value for value in results if value is not None) == [0]
    artifact = json.loads((output / "artifact.json").read_text())
    assert file_identity(output / "selected_targets.parquet")["sha256"] == (
        artifact["payload"]["sha256"]
    )


def test_source_metadata_prefix_collision_is_rejected() -> None:
    with pytest.raises(ValueError, match="collide"):
        source_metadata_output_names(["query_id", "source_query_id"])


@pytest.mark.parametrize("backend", ["sharded", "ann_rerank", "dense"])
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
    else:
        store_dir = tmp_path / "store"
        register_embedding_store(
            store_root=source_root, output_dir=store_dir, encoder={"name": "test"},
        )
        dense_dir = tmp_path / "dense"
        initialize_dense_store(
            source_store_manifest=store_dir / "embedding_store.json", output_dir=dense_dir,
        )
        plan = dense_dir / "dense_store_plan.json"
        materialize_dense_shards(plan_path=plan, shard_indexes=[0])
        finalize_dense_store(plan_path=plan)
        search_options = dict(
            query_path=queries, dense_store_manifest=dense_dir / "dense_store.json",
            top_k=1, min_similarity=0.5, device="cpu",
        )
        if backend == "ann_rerank":
            result = exact_rerank_ann_candidates(
                **search_options, candidate_row_ids=np.array([[0]]),
            )
        else:
            result = dense_search.exact_dense_search(
                **search_options, checkpoint_path=tmp_path / "scan.npz",
            )
        assert result.iloc[0]["source_row_id"] == 0
        assert result.iloc[0]["search_proof"] == "ann_exact_float32_rerank"
    row = result.iloc[0]
    assert row["sample_id"] == "source"
    assert row["query_id"] == "query"
    assert row["cosine_similarity"] == pytest.approx(0.8)
    assert row["rank"] == 1
    assert row["source_query_id"] == "wrong"
    assert row["source_cosine_similarity"] == -42.0
    assert row["source_rank"] == 99
    assert row["source_source_row_id"] == 99
    assert row["source_search_proof"] == "wrong"
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


@pytest.mark.parametrize("fixed_size", [False, True])
@pytest.mark.parametrize("bad_value", [1e30, -1e30])
def test_dense_normalization_rejects_overflowing_embedding_norms(fixed_size, bad_value):
    """Dense conversion must reject overflow even for previously registered stores."""
    vector_type = pa.list_(pa.float32(), 2) if fixed_size else pa.list_(pa.float32())
    values = pa.array([[bad_value, bad_value]], type=vector_type)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="norms.*non-finite"):
        dense_store._vectors_from_array(values, 2)


def test_dense_normalization_matches_shared_contract_for_tiny_vectors() -> None:
    values = pa.array([[1e-30, 0.0], [0.0, -1e-30]])
    observed = dense_store._vectors_from_array(values, 2)
    expected = contracts.vector_matrix(values.to_pylist())
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        observed, np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    )


def test_json_exclusions_reject_any_conflicting_supplied_lineage(
    tmp_path: Path,
) -> None:
    exclusion = tmp_path / "excluded.json"
    exclusion.write_text(json.dumps({
        "source_row_ids": [0],
        "dense_store_artifact_id": "dense-correct",
        "source_store_artifact_id": "source-wrong",
        "source_inventory_digest": "inventory-correct",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="another source store"):
        _read_excluded_row_ids(
            str(exclusion),
            dense_store_artifact_id="dense-correct",
            source_store_artifact_id="source-correct",
            source_inventory_digest="inventory-correct",
        )


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


def test_selection_artifact_binds_scores_exclusions_and_parameters(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores.parquet"
    pd.DataFrame(
        {
            "sample_id": ["keep", "drop"],
            "task": ["x", "x"],
            "weakness_score": [2.0, 1.0],
        }
    ).to_parquet(scores, index=False)
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps({"sample_ids": ["drop"]}), encoding="utf-8")
    output = tmp_path / "selected"
    assert refinement_main(
        [
            "select-multitask",
            "--scores",
            str(scores),
            "--exclude-targets",
            str(excluded),
            "--per-task",
            "1",
            "--score-column",
            "weakness_score",
            "--output-dir",
            str(output),
        ]
    ) == 0
    artifact = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert [value["role"] for value in artifact["inputs"]] == [
        "scores",
        "excluded_targets",
    ]
    assert all(value["sha256"].startswith("sha256:") for value in artifact["inputs"])
    assert artifact["payload"]["parameters"] == {
        "per_task": 1,
        "score_column": "weakness_score",
    }


def test_weighted_selection_artifact_records_fixed_budget(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "scores.parquet"
    pd.DataFrame(
        {
            "sample_id": [
                f"{task}-{index}" for task in ("x", "y") for index in range(6)
            ],
            "task": [task for task in ("x", "y") for _ in range(6)],
            "weakness_score": list(range(6)) * 2,
        }
    ).to_parquet(scores, index=False)
    output = tmp_path / "weighted"
    assert refinement_main(
        [
            "select-multitask",
            "--scores",
            str(scores),
            "--total",
            "8",
            "--task-weights-json",
            '{"x": 3, "y": 1}',
            "--output-dir",
            str(output),
        ]
    ) == 0
    artifact = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert artifact["payload"]["parameters"] == {
        "score_column": "weakness_score",
        "selected_by_task": {"x": 6, "y": 2},
        "task_weights": {"x": 3, "y": 1},
        "total": 8,
    }


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


def test_exact_search_streams_blocks_and_preserves_custom_identity_lineage(
    tmp_path: Path,
) -> None:
    queries = tmp_path / "queries.parquet"
    pd.DataFrame({
        "asset_id": ["q0", "q1"],
        "embedding": [[1.0, 0.0], [0.0, 1.0]],
    }).to_parquet(queries, index=False)
    parts = []
    rows = [
        [("a", [0.9, 0.1]), ("b", [0.8, 0.2])],
        [("c", [0.1, 0.9]), ("d", [0.2, 0.8])],
    ]
    for index, values in enumerate(rows):
        part = tmp_path / f"part-{index}.parquet"
        pd.DataFrame({
            "asset_id": [value[0] for value in values],
            "embedding": [value[1] for value in values],
        }).to_parquet(part, index=False)
        parts.append(part)
    result = exact_sharded_search(
        query_path=queries,
        source_parts=parts,
        id_column="asset_id",
        top_k=1,
        min_similarity=0.5,
        query_block_rows=1,
        source_block_rows=1,
    ).sort_values("query_id")
    assert result["query_id"].tolist() == ["q0", "q1"]
    assert result["sample_id"].tolist() == ["a", "c"]
    assert result["source_row_id"].tolist() == [0, 2]


def test_exact_search_tie_break_is_independent_of_block_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = tmp_path / "queries.parquet"
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "sample_id": ["q"], "embedding": [[1.0, 0.0]],
    }).to_parquet(queries, index=False)
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "embedding": [[0.8, 0.6]] * 4,
    }).to_parquet(source, index=False, row_group_size=2)
    real_parquet_file = search_module.pq.ParquetFile

    class BoundedParquetFile:
        def __init__(self, *args, **kwargs):
            self.inner = real_parquet_file(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_row_group(self, *args, **kwargs):
            raise AssertionError("exact search must not hydrate an entire row group")

        def iter_batches(self, *args, **kwargs):
            for batch in self.inner.iter_batches(*args, **kwargs):
                assert len(batch) <= int(kwargs["batch_size"])
                yield batch

    monkeypatch.setattr(search_module.pq, "ParquetFile", BoundedParquetFile)
    selected = []
    for source_block_rows in (1, 2, 4):
        result = exact_sharded_search(
            query_path=queries,
            source_parts=[source],
            top_k=1,
            min_similarity=0.7,
            duplicate_similarity=0.999,
            candidate_multiplier=1,
            source_block_rows=source_block_rows,
        )
        selected.append((result.iloc[0]["sample_id"], result.iloc[0]["source_row_id"]))
    assert selected == [("a", 0)] * 3


def test_tie_break_is_identical_across_sharded_dense_and_ann_backends(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    queries = tmp_path / "queries.parquet"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "part.parquet"
    pd.DataFrame({
        "sample_id": ["query"], "embedding": [[1.0, 0.0]],
    }).to_parquet(queries, index=False)
    pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "embedding": [[0.8, 0.6]] * 4,
        "path": [f"/images/{name}.jpg" for name in "abcd"],
        "storage_type": ["file"] * 4,
    }).to_parquet(source, index=False, row_group_size=2)
    registered = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
    )
    dense = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0]
    )
    finalize_dense_store(plan_path=dense / "dense_store_plan.json")

    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        sharded = exact_sharded_search(
            query_path=queries,
            source_parts=[source],
            top_k=1,
            min_similarity=0.7,
            candidate_multiplier=1,
            source_block_rows=1,
            device=device,
        )
        dense_result = dense_search.exact_dense_search(
            query_path=queries,
            dense_store_manifest=dense / "dense_store.json",
            top_k=1,
            min_similarity=0.7,
            candidate_multiplier=1,
            checkpoint_path=tmp_path / f"dense-{device.replace(':', '-')}.npz",
            chunk_rows=1,
            checkpoint_chunks=1,
            device=device,
        )
        ann = exact_rerank_ann_candidates(
            query_path=queries,
            dense_store_manifest=dense / "dense_store.json",
            candidate_row_ids=np.asarray([[3, 2, 1, 0]], dtype=np.int64),
            top_k=1,
            min_similarity=0.7,
            candidate_multiplier=1,
            device=device,
        )
        assert sharded.iloc[0]["source_row_id"] == 0
        assert dense_result.iloc[0]["source_row_id"] == 0
        assert ann.iloc[0]["source_row_id"] == 0
        assert {sharded.iloc[0]["sample_id"], dense_result.iloc[0]["sample_id"],
                ann.iloc[0]["sample_id"]} == {"a"}


def test_materialize_custom_id_column_balances_by_custom_identity(
    tmp_path: Path,
) -> None:
    delta = tmp_path / "delta.parquet"
    queries = tmp_path / "queries.parquet"
    pd.DataFrame({
        "asset_id": ["n2", "n1"],
        "query_id": ["q2", "q1"],
        "storage_type": ["file", "file"],
        "path": ["/n2.jpg", "/n1.jpg"],
    }).to_parquet(delta, index=False)
    pd.DataFrame({
        "asset_id": ["q1", "q2"],
        "task": ["rare", "common"],
    }).to_parquet(queries, index=False)
    result = materialize_manifest(
        delta_path=delta,
        query_path=queries,
        balance_column="task",
        id_column="asset_id",
        output_dir=tmp_path / "materialized",
    )
    assert result["payload"]["id_column"] == "asset_id"
    balanced = pd.read_parquet(
        tmp_path / "materialized" / "balanced_training_manifest.parquet",
        pre_buffer=False,
    )
    assert balanced["asset_id"].tolist() == ["n2", "n1"]


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

    dense_artifacts = []
    for name in ("first", "second"):
        dense_dir = tmp_path / f"dense-{name}"
        initialize_dense_store(
            source_store_manifest=tmp_path / f"registered-{name}/embedding_store.json",
            output_dir=dense_dir,
        )
        materialize_dense_shards(
            plan_path=dense_dir / "dense_store_plan.json", shard_indexes=[0]
        )
        dense_artifacts.append(
            finalize_dense_store(plan_path=dense_dir / "dense_store_plan.json")
        )
    assert dense_artifacts[0]["artifact_id"] == dense_artifacts[1]["artifact_id"]
    assert dense_artifacts[0]["audit_id"] != dense_artifacts[1]["audit_id"]


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


def test_bind_existing_store_rejects_symlink_escape_from_contracted_root(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "embeddings"
    payload_root = tmp_path / "images"
    outside = tmp_path / "outside"
    store_root.mkdir()
    payload_root.mkdir()
    outside.mkdir()
    (payload_root / "escape").symlink_to(outside, target_is_directory=True)
    pd.DataFrame({
        "sample_id": ["a"],
        "embedding": [[0.9, 0.1]],
        "path": [str(payload_root / "escape/a.jpg")],
        "storage_type": ["file"],
    }).to_parquet(store_root / "part.parquet", index=False)
    original_dir = tmp_path / "original"
    register_embedding_store(
        store_root=store_root,
        output_dir=original_dir,
        encoder={"name": "test"},
    )
    contract = tmp_path / "source_payload_contract.json"
    contract.write_text(json.dumps({
        "schema_version": "1.0",
        "immutability": "immutable",
        "datasets": [{
            "dataset_id": "test-images",
            "version": "v1",
            "root_uri": payload_root.resolve().as_uri(),
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="outside.*contracted|contracted.*root"):
        bind_store_payload_contract(
            source_store_manifest=original_dir / "embedding_store.json",
            source_payload_contract=contract,
            output_dir=tmp_path / "bound",
        )


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


def test_dense_rerank_hydrates_configured_source_id_column(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame({
        "asset_id": ["asset-a"],
        "embedding": [[0.9, 0.1]],
        "storage_type": ["file"],
        "path": ["/images/a.png"],
    }).to_parquet(source_root / "part.parquet", index=False)
    registered = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
        id_column="asset_id",
    )
    dense = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    plan = dense / "dense_store_plan.json"
    materialize_dense_shards(plan_path=plan, shard_indexes=[0])
    finalize_dense_store(plan_path=plan)
    query = tmp_path / "query.parquet"
    pd.DataFrame({
        "sample_id": ["query"], "embedding": [[1.0, 0.0]],
    }).to_parquet(query, index=False)
    result = exact_rerank_ann_candidates(
        query_path=query,
        dense_store_manifest=dense / "dense_store.json",
        candidate_row_ids=np.asarray([[0]], dtype=np.int64),
        top_k=1,
        min_similarity=0.5,
        device="cpu",
    )
    assert result.iloc[0]["sample_id"] == "asset-a"
    assert result.iloc[0]["source_row_id"] == 0




def test_dense_exact_search_resumes_and_proves_underfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["copy", "excluded", "winner", "far"],
            "embedding": [
                [1.0, 0.0],
                [0.9, 0.4358899],
                [0.8, 0.6],
                [0.0, 1.0],
            ],
            "path": [
                str(tmp_path / "copy.jpg"),
                str(tmp_path / "excluded.jpg"),
                str(tmp_path / "winner.jpg"),
                str(tmp_path / "far.jpg"),
            ],
            "storage_type": ["file"] * 4,
        }
    ).to_parquet(source_root / "part.parquet", index=False)
    registered = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
    )
    dense = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0]
    )
    finalize_dense_store(plan_path=dense / "dense_store_plan.json")
    queries = tmp_path / "queries.parquet"
    pd.DataFrame(
        {"sample_id": ["query"], "embedding": [[1.0, 0.0]]}
    ).to_parquet(queries, index=False)
    progress = tmp_path / "progress.npz"

    original_save = dense_search._save_progress
    calls = 0

    def interrupt_after_first_checkpoint(path: Path, **arrays: np.ndarray) -> None:
        nonlocal calls
        original_save(path, **arrays)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated preemption")

    monkeypatch.setattr(dense_search, "_save_progress", interrupt_after_first_checkpoint)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        dense_search.exact_dense_candidates(
            query_path=queries,
            dense_store_manifest=dense / "dense_store.json",
            candidate_limit=4,
            hard_min_similarity=0.5,
            excluded_row_ids={1},
            checkpoint_path=progress,
            chunk_rows=1,
            checkpoint_chunks=1,
            device="cpu",
        )
    monkeypatch.setattr(dense_search, "_save_progress", original_save)
    result = dense_search.exact_dense_search(
        query_path=queries,
        dense_store_manifest=dense / "dense_store.json",
        top_k=2,
        min_similarity=0.5,
        checkpoint_path=progress,
        duplicate_similarity=0.99,
        candidate_multiplier=2,
        excluded_row_ids={1},
        chunk_rows=1,
        checkpoint_chunks=1,
        device="cpu",
    )
    assert result["sample_id"].tolist() == ["winner"]
    adaptive = result.attrs["adaptive_radius"]
    assert adaptive["scan"]["resumed_from_row"] == 1
    assert adaptive["query_stats"][0]["candidate_count_above_hard_floor"] == 2
    assert adaptive["underfill_exhaustion_proven"] is True


def test_dense_exact_search_cli_commits_lineage(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    pd.DataFrame(
        {
            "sample_id": ["near", "far"],
            "embedding": [[0.9, 0.1], [0.0, 1.0]],
            "path": [str(tmp_path / "near.jpg"), str(tmp_path / "far.jpg")],
            "storage_type": ["file", "file"],
        }
    ).to_parquet(source_root / "part.parquet", index=False)
    registered = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
    )
    dense = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0]
    )
    finalize_dense_store(plan_path=dense / "dense_store_plan.json")
    queries = tmp_path / "queries.parquet"
    pd.DataFrame(
        {"sample_id": ["query"], "embedding": [[1.0, 0.0]]}
    ).to_parquet(queries, index=False)
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )
    output = tmp_path / "search"
    assert refinement_main(
        [
            "dense-exact-search",
            "--queries",
            str(queries),
            "--source-store-manifest",
            str(registered / "embedding_store.json"),
            "--dense-store-manifest",
            str(dense / "dense_store.json"),
            "--query-embedding-contract",
            str(query_contract),
            "--top-k",
            "1",
            "--min-similarity",
            "0.5",
            "--duplicate-similarity",
            "0.999",
            "--device",
            "cpu",
            "--chunk-rows",
            "1",
            "--output-dir",
            str(output),
        ]
    ) == 0
    summary = json.loads((output / "search_summary.json").read_text())
    assert summary["search_proof"] == "exact_all_dense_rows_float32"
    assert summary["source_rows"] == 2
    artifact = json.loads((output / "artifact.json").read_text())
    source_input = next(
        item
        for item in artifact["inputs"]
        if item.get("role") == "source_store_manifest"
    )
    registered_artifact = json.loads(
        (registered / "artifact.json").read_text(encoding="utf-8")
    )
    assert source_input["artifact_id"] == registered_artifact["artifact_id"]
    assert source_input["inventory_digest"]
    assert (output / "_SUCCESS").is_file()
    first_neighbors = pd.read_parquet(output / "neighbors.parquet")
    assert first_neighbors.iloc[0]["source_store_artifact_id"] == (
        registered_artifact["artifact_id"]
    )

    continued = tmp_path / "continued-search"
    assert refinement_main(
        [
            "dense-exact-search",
            "--queries",
            str(queries),
            "--source-store-manifest",
            str(registered / "embedding_store.json"),
            "--dense-store-manifest",
            str(dense / "dense_store.json"),
            "--query-embedding-contract",
            str(query_contract),
            "--exclude",
            str(output / "neighbors.parquet"),
            "--top-k",
            "1",
            "--min-similarity",
            "0.0",
            "--duplicate-similarity",
            "0.999",
            "--device",
            "cpu",
            "--chunk-rows",
            "1",
            "--output-dir",
            str(continued),
        ]
    ) == 0
    continued_neighbors = pd.read_parquet(continued / "neighbors.parquet")
    assert continued_neighbors["sample_id"].tolist() == ["far"]


def test_dense_exact_search_rejects_unrelated_source_store(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    unrelated_root = tmp_path / "unrelated"
    source_root.mkdir()
    unrelated_root.mkdir()
    frame = pd.DataFrame(
        {
            "sample_id": ["near"],
            "embedding": [[1.0, 0.0]],
            "path": [str(tmp_path / "near.jpg")],
            "storage_type": ["file"],
        }
    )
    frame.to_parquet(source_root / "part.parquet", index=False)
    unrelated_frame = frame.copy()
    unrelated_frame["sample_id"] = ["different"]
    unrelated_frame.to_parquet(unrelated_root / "part.parquet", index=False)
    registered = tmp_path / "registered"
    unrelated = tmp_path / "unrelated_registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=registered,
        encoder={"name": "test"},
    )
    register_embedding_store(
        store_root=unrelated_root,
        output_dir=unrelated,
        encoder={"name": "test"},
    )
    dense = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=registered / "embedding_store.json",
        output_dir=dense,
    )
    materialize_dense_shards(
        plan_path=dense / "dense_store_plan.json", shard_indexes=[0]
    )
    finalize_dense_store(plan_path=dense / "dense_store_plan.json")
    queries = tmp_path / "queries.parquet"
    pd.DataFrame(
        {"sample_id": ["query"], "embedding": [[1.0, 0.0]]}
    ).to_parquet(queries, index=False)
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another source-store artifact"):
        refinement_main(
            [
                "dense-exact-search",
                "--queries",
                str(queries),
                "--source-store-manifest",
                str(unrelated / "embedding_store.json"),
                "--dense-store-manifest",
                str(dense / "dense_store.json"),
                "--query-embedding-contract",
                str(query_contract),
                "--top-k",
                "1",
                "--min-similarity",
                "0.5",
                "--device",
                "cpu",
                "--output-dir",
                str(tmp_path / "search"),
            ]
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


def test_ann_index_loader_rejects_same_size_content_mutation(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.cuvs"
    index_path.write_bytes(b"original-index")
    payload = {
        "backend": "test",
        "index": file_identity(index_path),
    }
    (tmp_path / "ann_index.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    ArtifactManifest(
        artifact_type="ann_index",
        producer={"action": "test", "version": "1.0"},
        inputs=[],
        payload=payload,
    ).commit(tmp_path)

    index_path.write_bytes(b"mutated-index!")
    assert index_path.stat().st_size == payload["index"]["bytes"]
    with pytest.raises(RuntimeError, match="content changed"):
        load_ann_index(tmp_path / "ann_index.json")


def test_ann_retrieval_rejects_unsupported_multi_gpu_candidate_depth(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at most 1024"):
        retrieve_ann_candidates(
            query_path=tmp_path / "queries.parquet",
            ann_index_manifest=tmp_path / "ann_index.json",
            n_probes=32,
            ann_candidate_count=1025,
        )


def test_exact_cli_binds_registered_store_and_refuses_committed_overwrite(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["near", "far"],
            "embedding": [[0.9, 0.1], [0.0, 1.0]],
            "path": ["/near.jpg", "/far.jpg"],
        }
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "part.parquet"
    source.to_parquet(source_path, index=False)
    store_dir = tmp_path / "registered"
    register_embedding_store(
        store_root=source_root,
        output_dir=store_dir,
        encoder={"name": "c-radio"},
        default_storage_type="file",
    )
    queries = pd.DataFrame({"sample_id": ["q"], "embedding": [[1.0, 0.0]]})
    query_path = tmp_path / "queries.parquet"
    queries.to_parquet(query_path, index=False)
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps(
            {
                "encoder": {"name": "c-radio"},
                "embedding_dim": 2,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "search"
    args = [
        "exact-search",
        "--queries",
        str(query_path),
        "--source-part",
        str(source_path),
        "--source-store-manifest",
        str(store_dir / "embedding_store.json"),
        "--query-embedding-contract",
        str(query_contract),
        "--top-k",
        "1",
        "--min-similarity",
        "0.8",
        "--output-dir",
        str(output),
    ]
    assert refinement_main(args) == 0
    artifact = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert artifact["payload"]["neighbors"]["sha256"].startswith("sha256:")
    assert artifact["inputs"][1]["artifact_id"]
    exact_neighbors = pd.read_parquet(output / "neighbors.parquet")
    assert exact_neighbors["source_store_artifact_id"].tolist() == [
        artifact["inputs"][1]["artifact_id"]
    ]

    dense_dir = tmp_path / "dense"
    initialize_dense_store(
        source_store_manifest=store_dir / "embedding_store.json",
        output_dir=dense_dir,
    )
    materialize_dense_shards(
        plan_path=dense_dir / "dense_store_plan.json", shard_indexes=[0]
    )
    finalize_dense_store(plan_path=dense_dir / "dense_store_plan.json")
    continued = tmp_path / "continued"
    assert refinement_main(
        [
            "dense-exact-search",
            "--queries",
            str(query_path),
            "--source-store-manifest",
            str(store_dir / "embedding_store.json"),
            "--dense-store-manifest",
            str(dense_dir / "dense_store.json"),
            "--query-embedding-contract",
            str(query_contract),
            "--exclude",
            str(output / "neighbors.parquet"),
            "--top-k",
            "1",
            "--min-similarity",
            "0.0",
            "--device",
            "cpu",
            "--output-dir",
            str(continued),
        ]
    ) == 0
    assert pd.read_parquet(continued / "neighbors.parquet")[
        "sample_id"
    ].tolist() == ["far"]
    with pytest.raises(RuntimeError, match="committed output"):
        refinement_main(args)


def test_exact_cli_rejects_unhashed_or_uncommitted_store(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "sample_id": ["near"],
            "embedding": [[1.0, 0.0]],
            "path": ["/near.jpg"],
        }
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "part.parquet"
    source.to_parquet(source_path, index=False)
    query_path = tmp_path / "queries.parquet"
    pd.DataFrame(
        {"sample_id": ["query"], "embedding": [[1.0, 0.0]]}
    ).to_parquet(query_path, index=False)
    query_contract = tmp_path / "query_contract.json"
    query_contract.write_text(
        json.dumps({"encoder": {"name": "test"}, "embedding_dim": 2}),
        encoding="utf-8",
    )

    def arguments(store_dir: Path, output: str) -> list[str]:
        return [
            "exact-search",
            "--queries",
            str(query_path),
            "--source-part",
            str(source_path),
            "--source-store-manifest",
            str(store_dir / "embedding_store.json"),
            "--query-embedding-contract",
            str(query_contract),
            "--top-k",
            "1",
            "--min-similarity",
            "0.8",
            "--output-dir",
            str(tmp_path / output),
        ]

    unhashed = tmp_path / "unhashed"
    register_embedding_store(
        store_root=source_root,
        output_dir=unhashed,
        encoder={"name": "test"},
        hash_content=False,
        default_storage_type="file",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        refinement_main(arguments(unhashed, "unhashed-search"))

    committed = tmp_path / "committed"
    register_embedding_store(
        store_root=source_root,
        output_dir=committed,
        encoder={"name": "test"},
        default_storage_type="file",
    )
    (committed / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="committed embedding-store"):
        refinement_main(arguments(committed, "uncommitted-search"))
