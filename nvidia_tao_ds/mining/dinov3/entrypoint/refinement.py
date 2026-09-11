# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for reusable refinement data operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from ..contracts import (
    ArtifactManifest,
    canonical_digest,
    file_identity,
    file_sha256,
    require_uncommitted_output,
    write_json_atomic,
)
from ..ann_index import (
    build_cuvs_ivf_pq_index,
    load_ann_audit,
    load_ann_index,
)
from ..ann_search import (
    ann_exact_rerank_search,
    exact_rerank_ann_candidates,
    retrieve_ann_candidates,
)
from ..dense_store import (
    finalize_dense_store,
    initialize_dense_store,
    load_dense_store,
    materialize_dense_shards,
)
from ..dense_search import exact_dense_search
from ..materialize import materialize_manifest
from ..search import exact_sharded_search
from ..selection import select_grit_targets, select_multitask_targets
from ..store import bind_store_payload_contract, register_embedding_store


def _write_selection(
    frame: pd.DataFrame,
    output_dir: str,
    strategy: str,
    *,
    scores_path: str,
    excluded_path: str | None,
    parameters: dict,
) -> None:
    destination = require_uncommitted_output(output_dir)
    output = destination / "selected_targets.parquet"
    frame.to_parquet(output, index=False)
    inputs = [file_identity(scores_path, role="scores")]
    if excluded_path:
        inputs.append(file_identity(excluded_path, role="excluded_targets"))
    artifact = ArtifactManifest(
        artifact_type="target_selection",
        producer={"action": strategy, "version": "1.0"},
        inputs=inputs,
        payload={
            **file_identity(output),
            "row_count": len(frame),
            "parameters": parameters,
        },
    )
    artifact.commit(destination)
    print(json.dumps(artifact.to_dict(), indent=2, sort_keys=True))


def _read_excluded(path: str | None) -> set[str]:
    if not path:
        return set()
    value = Path(path)
    if value.suffix == ".parquet":
        return set(pd.read_parquet(value, pre_buffer=False)["sample_id"].astype(str))
    payload = json.loads(value.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("sample_ids", [])
    return set(map(str, payload))


def _read_excluded_row_ids(
    path: str | None,
    *,
    dense_store_artifact_id: str,
    source_inventory_digest: str,
) -> set[int]:
    if not path:
        return set()
    value = Path(path)
    if value.suffix == ".parquet":
        import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

        parquet = pq.ParquetFile(value)
        if "source_row_id" not in parquet.schema_arrow.names:
            if parquet.metadata.num_rows == 0:
                return set()
            raise ValueError(
                "Indexed search exclusions must contain source_row_id"
            )
        required = {
            "source_row_id",
            "dense_store_artifact_id",
            "source_inventory_digest",
        }
        if missing := required.difference(parquet.schema_arrow.names):
            raise ValueError(
                "Nonempty indexed-search exclusions lack corpus lineage: "
                f"{sorted(missing)}"
            )
        frame = pd.read_parquet(value, columns=sorted(required), pre_buffer=False)
        if set(frame["dense_store_artifact_id"].astype(str)) != {
            dense_store_artifact_id
        }:
            raise ValueError("Indexed-search exclusions belong to another dense store")
        if set(frame["source_inventory_digest"].astype(str)) != {
            source_inventory_digest
        }:
            raise ValueError("Indexed-search exclusions belong to another source inventory")
        return set(frame["source_row_id"].astype(int))
    payload = json.loads(value.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        row_ids = payload.get("source_row_ids", [])
        if row_ids:
            if payload.get("dense_store_artifact_id") != dense_store_artifact_id:
                raise ValueError(
                    "Indexed-search exclusions belong to another dense store"
                )
            if payload.get("source_inventory_digest") != source_inventory_digest:
                raise ValueError(
                    "Indexed-search exclusions belong to another source inventory"
                )
        return set(map(int, row_ids))
    if payload:
        raise ValueError("Nonempty indexed-search exclusions require corpus lineage")
    return set()


def _validate_excluded_row_ids(row_ids: set[int], row_count: int) -> None:
    invalid = sorted(value for value in row_ids if value < 0 or value >= row_count)
    if invalid:
        raise ValueError(
            "Indexed search exclusions contain out-of-range source_row_id "
            f"values: count={len(invalid)}, examples={invalid[:10]}"
        )


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Embedding store must use local file URIs: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _validated_store_identity(
    manifest_path: str,
) -> tuple[dict, dict, dict]:
    """Load a committed embedding store and return its exact input identity."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_path = path.with_name("artifact.json")
    marker = path.with_name("_SUCCESS")
    if not artifact_path.is_file() or not marker.is_file():
        raise ValueError("Search requires a committed embedding-store artifact")
    if payload.get("fingerprint_method") != "sha256":
        raise ValueError("Search requires SHA-256 store fingerprints")
    shards = payload.get("shards", [])
    if not shards or any(not shard.get("sha256") for shard in shards):
        raise ValueError("Search requires a SHA-256 digest for every shard")
    if canonical_digest(shards) != payload.get("inventory_digest"):
        raise ValueError("Embedding-store inventory digest is invalid")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "embedding_store":
        raise ValueError("Search requires an embedding-store artifact")
    artifact_id = artifact.get("artifact_id")
    expected_artifact_id = canonical_digest(
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
    if artifact_id != expected_artifact_id:
        raise ValueError("Embedding store artifact identity is invalid")
    if artifact.get("payload") != payload:
        raise ValueError("Embedding store manifest does not match artifact payload")
    if marker.read_text(encoding="utf-8").strip() != artifact_id:
        raise ValueError("Embedding store artifact is not committed consistently")
    identity = file_identity(path, role="source_store_manifest")
    identity["artifact_id"] = artifact_id
    identity["inventory_digest"] = payload.get("inventory_digest")
    return payload, identity, artifact


def _validate_dense_source_lineage(
    source: dict, source_identity: dict, source_artifact: dict, dense: dict
) -> None:
    """Prove that a dense store indexes this store or its committed parent."""
    indexed_artifact_id = dense.get("source_store_artifact_id")
    current_artifact_id = source_identity.get("artifact_id")
    if current_artifact_id != indexed_artifact_id:
        parent_artifact_id = source.get("parent_store_artifact_id")
        expected_parent_input = ("source_embedding_store", indexed_artifact_id)
        parent_input = any(
            expected_parent_input == (item.get("role"), item.get("artifact_id"))
            for item in source_artifact.get("inputs", [])
            if isinstance(item, dict)
        )
        if not (
            parent_artifact_id == indexed_artifact_id and parent_input
        ):
            raise ValueError("Dense matrix belongs to another source-store artifact")
    if dense.get("source_inventory_digest") != source.get("inventory_digest"):
        raise ValueError("Dense matrix belongs to another source inventory")
    if dense.get("encoder") != source.get("encoder") or int(
        dense.get("embedding_dim", -1)
    ) != int(source.get("embedding_dim", -2)):
        raise ValueError("Dense matrix uses another source encoder contract")


def _validated_store(
    manifest_path: str,
    source_parts: list[str],
) -> tuple[dict, dict]:
    """Bind exact search to every immutable shard declared by a store manifest."""
    payload, identity, _ = _validated_store_identity(manifest_path)
    root = _local_path(str(payload["root_uri"]))
    shards = payload.get("shards", [])
    expected = [(root / str(shard["relative_path"])).resolve() for shard in shards]
    declared = [Path(value).resolve() for value in source_parts]
    if declared != expected:
        raise ValueError(
            "--source-part values must match the registered store shards exactly "
            "and in manifest order"
        )
    for shard, shard_path in zip(shards, expected):
        if not shard_path.is_file():
            raise FileNotFoundError(f"Registered source shard is missing: {shard_path}")
        if shard_path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"Registered source shard size changed: {shard_path}")
        observed_digest = file_sha256(shard_path)
        declared_digest = str(shard["sha256"])
        if not declared_digest.startswith("sha256:"):
            declared_digest = "sha256:" + declared_digest
        if observed_digest != declared_digest:
            raise ValueError(f"Registered source shard digest changed: {shard_path}")
    return payload, identity


def _validated_query_contract(
    contract_path: str,
    store_manifest: dict,
) -> dict:
    """Require target and source embeddings to use the same encoder contract."""
    path = Path(contract_path).resolve()
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("encoder") != store_manifest.get("encoder"):
        raise ValueError("Query and source store encoder contracts do not match")
    if int(contract.get("embedding_dim", -1)) != int(
        store_manifest.get("embedding_dim", -2)
    ):
        raise ValueError("Query and source store embedding dimensions do not match")
    return file_identity(path, role="query_embedding_contract")


def build_parser() -> argparse.ArgumentParser:
    """Build the user-facing parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    grit = commands.add_parser("select-grit")
    grit.add_argument("--scores", required=True)
    grit.add_argument("--fraction", required=True, type=float)
    grit.add_argument("--output-dir", required=True)
    grit.add_argument("--exclude-targets")

    multitask = commands.add_parser("select-multitask")
    multitask.add_argument("--scores", required=True)
    budget = multitask.add_mutually_exclusive_group(required=True)
    budget.add_argument("--per-task", type=int)
    budget.add_argument("--total", type=int)
    multitask.add_argument(
        "--task-weights-json",
        help="JSON object of positive task preference weights; omitted tasks use 1.0",
    )
    multitask.add_argument(
        "--configured-tasks-json",
        help="JSON list defining the complete task budget contract",
    )
    multitask.add_argument("--preserve-unfilled-budget", action="store_true")
    multitask.add_argument("--score-column", default="weakness_score")
    multitask.add_argument("--output-dir", required=True)
    multitask.add_argument("--exclude-targets")

    store = commands.add_parser("register-store")
    store.add_argument("--store-root", required=True)
    store.add_argument("--output-dir", required=True)
    store.add_argument("--encoder-name", required=True)
    store.add_argument("--encoder-checkpoint-digest", required=True)
    store.add_argument("--input-resolution", required=True, type=int)
    store.add_argument("--normalization", required=True)
    store.add_argument(
        "--hash-content",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    store.add_argument(
        "--default-storage-type", choices=["file", "tar", "zip"]
    )
    store.add_argument(
        "--source-payload-contract",
        help="Immutable dataset-version contract for every source locator",
    )

    bind_store = commands.add_parser("bind-store-payload")
    bind_store.add_argument("--source-store-manifest", required=True)
    bind_store.add_argument("--source-payload-contract", required=True)
    bind_store.add_argument("--output-dir", required=True)

    dense_init = commands.add_parser("dense-init")
    dense_init.add_argument("--source-store-manifest", required=True)
    dense_init.add_argument("--output-dir", required=True)

    dense_write = commands.add_parser("dense-write")
    dense_write.add_argument("--plan", required=True)
    dense_write.add_argument("--shard-index", action="append", required=True, type=int)
    dense_write.add_argument("--batch-rows", type=int, default=8192)

    dense_finalize = commands.add_parser("dense-finalize")
    dense_finalize.add_argument("--plan", required=True)

    ann_build = commands.add_parser("build-ann-index")
    ann_build.add_argument("--dense-store-manifest", required=True)
    ann_build.add_argument("--output-dir", required=True)
    ann_build.add_argument("--n-lists", type=int, default=32768)
    ann_build.add_argument("--pq-dim", type=int, default=160)
    ann_build.add_argument("--pq-bits", type=int, default=8)
    ann_build.add_argument("--kmeans-iters", type=int, default=20)
    ann_build.add_argument("--kmeans-train-fraction", type=float, default=0.03)
    ann_build.add_argument(
        "--force-random-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    search = commands.add_parser("exact-search")
    search.add_argument("--queries", required=True)
    search.add_argument("--source-part", action="append", required=True)
    search.add_argument("--source-store-manifest")
    search.add_argument("--query-embedding-contract")
    search.add_argument("--top-k", required=True, type=int)
    search.add_argument("--min-similarity", required=True, type=float)
    search.add_argument("--hard-min-similarity", type=float)
    search.add_argument("--similarity-step", type=float, default=0.02)
    search.add_argument("--duplicate-similarity", type=float, default=1.0)
    search.add_argument("--candidate-multiplier", type=int, default=10)
    search.add_argument("--exclude")
    search.add_argument("--device", default="cpu")
    search.add_argument("--output-dir", required=True)

    ann_search = commands.add_parser("ann-search")
    ann_search.add_argument("--queries", required=True)
    ann_search.add_argument("--dense-store-manifest", required=True)
    ann_search.add_argument("--ann-index-manifest", required=True)
    ann_search.add_argument("--query-embedding-contract", required=True)
    ann_search.add_argument("--ann-audit-manifest")
    ann_search.add_argument("--top-k", required=True, type=int)
    ann_search.add_argument("--min-similarity", required=True, type=float)
    ann_search.add_argument("--hard-min-similarity", type=float)
    ann_search.add_argument("--similarity-step", type=float, default=0.02)
    ann_search.add_argument("--duplicate-similarity", type=float, default=1.0)
    ann_search.add_argument("--candidate-multiplier", type=int, default=10)
    ann_search.add_argument("--n-probes", type=int, required=True)
    ann_search.add_argument("--ann-candidates", type=int, required=True)
    ann_search.add_argument("--exclude")
    ann_search.add_argument("--device", default="cuda:0")
    ann_search.add_argument("--output-dir", required=True)

    ann_candidates = commands.add_parser("ann-candidates")
    ann_candidates.add_argument("--queries", required=True)
    ann_candidates.add_argument("--ann-index-manifest", required=True)
    ann_candidates.add_argument("--query-embedding-contract", required=True)
    ann_candidates.add_argument("--ann-audit-manifest", required=True)
    ann_candidates.add_argument("--n-probes", type=int, required=True)
    ann_candidates.add_argument("--ann-candidates", type=int, required=True)
    ann_candidates.add_argument("--output-dir", required=True)

    ann_rerank = commands.add_parser("ann-rerank")
    ann_rerank.add_argument("--queries", required=True)
    ann_rerank.add_argument("--candidates", required=True)
    ann_rerank.add_argument("--dense-store-manifest", required=True)
    ann_rerank.add_argument("--ann-index-manifest", required=True)
    ann_rerank.add_argument("--ann-audit-manifest", required=True)
    ann_rerank.add_argument("--query-embedding-contract", required=True)
    ann_rerank.add_argument("--top-k", required=True, type=int)
    ann_rerank.add_argument("--min-similarity", required=True, type=float)
    ann_rerank.add_argument("--hard-min-similarity", type=float)
    ann_rerank.add_argument("--similarity-step", type=float, default=0.02)
    ann_rerank.add_argument("--duplicate-similarity", type=float, default=1.0)
    ann_rerank.add_argument("--candidate-multiplier", type=int, default=10)
    ann_rerank.add_argument("--exclude")
    ann_rerank.add_argument("--device", default="cuda:0")
    ann_rerank.add_argument("--output-dir", required=True)

    dense_exact = commands.add_parser("dense-exact-search")
    dense_exact.add_argument("--queries", required=True)
    dense_exact.add_argument("--source-store-manifest", required=True)
    dense_exact.add_argument("--dense-store-manifest", required=True)
    dense_exact.add_argument("--query-embedding-contract", required=True)
    dense_exact.add_argument("--top-k", required=True, type=int)
    dense_exact.add_argument("--min-similarity", required=True, type=float)
    dense_exact.add_argument("--hard-min-similarity", type=float)
    dense_exact.add_argument("--similarity-step", type=float, default=0.02)
    dense_exact.add_argument("--duplicate-similarity", type=float, default=1.0)
    dense_exact.add_argument("--candidate-multiplier", type=int, default=10)
    dense_exact.add_argument("--exclude")
    dense_exact.add_argument("--device", default="cuda:0")
    dense_exact.add_argument("--chunk-rows", type=int, default=32768)
    dense_exact.add_argument("--checkpoint-chunks", type=int, default=64)
    dense_exact.add_argument("--output-dir", required=True)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--delta", required=True)
    materialize.add_argument("--previous")
    materialize.add_argument("--query-manifest")
    materialize.add_argument("--balance-column")
    materialize.add_argument(
        "--overlap-policy",
        choices=("reject", "drop_existing"),
        default="reject",
    )
    materialize.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one data action."""
    args = build_parser().parse_args(argv)
    if args.command == "select-grit":
        scores = pd.read_parquet(args.scores, pre_buffer=False)
        scores = scores[
            ~scores["sample_id"].astype(str).isin(
                _read_excluded(args.exclude_targets)
            )
        ]
        frame = select_grit_targets(scores, fraction=args.fraction)
        _write_selection(
            frame,
            args.output_dir,
            "select_grit_targets",
            scores_path=args.scores,
            excluded_path=args.exclude_targets,
            parameters={"fraction": args.fraction, "score_column": "grit_score"},
        )
    elif args.command == "select-multitask":
        scores = pd.read_parquet(args.scores, pre_buffer=False)
        scores = scores[
            ~scores["sample_id"].astype(str).isin(
                _read_excluded(args.exclude_targets)
            )
        ]
        task_weights = (
            json.loads(args.task_weights_json) if args.task_weights_json else None
        )
        if task_weights is not None and not isinstance(task_weights, dict):
            raise ValueError("--task-weights-json must encode an object")
        configured_tasks = (
            json.loads(args.configured_tasks_json)
            if args.configured_tasks_json
            else None
        )
        if configured_tasks is not None and not isinstance(configured_tasks, list):
            raise ValueError("--configured-tasks-json must encode a list")
        frame = select_multitask_targets(
            scores,
            per_task=args.per_task,
            total=args.total,
            task_weights=task_weights,
            configured_tasks=configured_tasks,
            preserve_unfilled_budget=args.preserve_unfilled_budget,
            score_column=args.score_column,
        )
        parameters = {"score_column": args.score_column}
        if args.per_task is not None and not task_weights:
            parameters["per_task"] = args.per_task
        else:
            parameters.update(
                {
                    "total": args.total,
                    "task_weights": task_weights or {},
                    "selected_by_task": frame.groupby("task").size().to_dict(),
                }
            )
            if configured_tasks is not None:
                parameters["configured_tasks"] = configured_tasks
            if args.preserve_unfilled_budget:
                parameters["preserve_unfilled_budget"] = True
        _write_selection(
            frame,
            args.output_dir,
            "select_multitask_targets",
            scores_path=args.scores,
            excluded_path=args.exclude_targets,
            parameters=parameters,
        )
    elif args.command == "register-store":
        result = register_embedding_store(
            store_root=args.store_root,
            output_dir=args.output_dir,
            encoder={
                "name": args.encoder_name,
                "checkpoint_digest": args.encoder_checkpoint_digest,
                "input_resolution": args.input_resolution,
                "normalization": args.normalization,
            },
            hash_content=args.hash_content,
            default_storage_type=args.default_storage_type,
            source_payload_contract=args.source_payload_contract,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "bind-store-payload":
        result = bind_store_payload_contract(
            source_store_manifest=args.source_store_manifest,
            source_payload_contract=args.source_payload_contract,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "dense-init":
        result = initialize_dense_store(
            source_store_manifest=args.source_store_manifest,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "dense-write":
        result = materialize_dense_shards(
            plan_path=args.plan,
            shard_indexes=args.shard_index,
            batch_rows=args.batch_rows,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "dense-finalize":
        result = finalize_dense_store(plan_path=args.plan)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "build-ann-index":
        result = build_cuvs_ivf_pq_index(
            dense_store_manifest=args.dense_store_manifest,
            output_dir=args.output_dir,
            n_lists=args.n_lists,
            pq_dim=args.pq_dim,
            pq_bits=args.pq_bits,
            kmeans_n_iters=args.kmeans_iters,
            kmeans_trainset_fraction=args.kmeans_train_fraction,
            force_random_rotation=args.force_random_rotation,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "ann-candidates":
        destination = require_uncommitted_output(args.output_dir)
        ann, ann_artifact = load_ann_index(args.ann_index_manifest)
        query_contract_identity = _validated_query_contract(
            args.query_embedding_contract, ann
        )
        _, audit_artifact = load_ann_audit(
            args.ann_audit_manifest,
            index_artifact_id=ann_artifact["artifact_id"],
            n_probes=args.n_probes,
            ann_candidate_count=args.ann_candidates,
            dense_store_artifact_id=ann["dense_store_artifact_id"],
            source_inventory_digest=ann["source_inventory_digest"],
            vector_inventory_digest=ann["vector_inventory_digest"],
        )
        distances, candidate_ids, metadata = retrieve_ann_candidates(
            query_path=args.queries,
            ann_index_manifest=args.ann_index_manifest,
            n_probes=args.n_probes,
            ann_candidate_count=args.ann_candidates,
        )
        output = destination / "ann_candidates.npz"
        _write_npz_atomic(
            output, distances=distances, candidate_ids=candidate_ids
        )
        payload = {
            "candidates": file_identity(output),
            "query_count": int(candidate_ids.shape[0]),
            "candidate_count": int(candidate_ids.shape[1]),
            "index_artifact_id": ann_artifact["artifact_id"],
            "dense_store_artifact_id": ann["dense_store_artifact_id"],
            "source_inventory_digest": ann["source_inventory_digest"],
            "audit_artifact_id": audit_artifact["artifact_id"],
            **metadata,
        }
        write_json_atomic(destination / "candidate_summary.json", payload)
        ArtifactManifest(
            artifact_type="ann_candidates",
            producer={"action": "retrieve_ann_candidates", "version": "1.0"},
            inputs=[
                file_identity(args.queries, role="queries"),
                file_identity(args.ann_index_manifest, role="ann_index"),
                file_identity(args.ann_audit_manifest, role="ann_recall_audit"),
                query_contract_identity,
            ],
            payload=payload,
        ).commit(destination)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "ann-rerank":
        destination = require_uncommitted_output(args.output_dir)
        candidate_path = Path(args.candidates).resolve()
        candidate_artifact_path = candidate_path.with_name("artifact.json")
        candidate_marker = candidate_path.with_name("_SUCCESS")
        if not candidate_artifact_path.is_file() or not candidate_marker.is_file():
            raise ValueError("ANN candidates are not a committed artifact")
        candidate_artifact = json.loads(
            candidate_artifact_path.read_text(encoding="utf-8")
        )
        candidate_artifact_id = canonical_digest(
            {
                name: candidate_artifact[name]
                for name in (
                    "artifact_type",
                    "schema_version",
                    "producer",
                    "inputs",
                    "payload",
                )
            }
        )
        committed_candidate_id = candidate_marker.read_text(
            encoding="utf-8"
        ).strip()
        if (
            candidate_artifact.get("artifact_id") != candidate_artifact_id or
            candidate_artifact.get("artifact_type") != "ann_candidates" or
            committed_candidate_id != candidate_artifact_id
        ):
            raise ValueError("ANN candidate artifact identity is invalid")
        if candidate_artifact["payload"]["candidates"] != file_identity(
            candidate_path
        ):
            raise ValueError("ANN candidate payload changed after commit")
        current_query = file_identity(args.queries, role="queries")
        declared_query = next(
            value
            for value in candidate_artifact["inputs"]
            if value.get("role") == "queries"
        )
        if current_query != declared_query:
            raise ValueError("ANN candidates belong to different queries")

        dense_manifest, dense_artifact = load_dense_store(
            args.dense_store_manifest
        )
        required_candidate_fields = {
            "index_artifact_id",
            "dense_store_artifact_id",
            "source_inventory_digest",
            "audit_artifact_id",
            "n_probes",
            "ann_candidate_count",
        }
        if missing := required_candidate_fields.difference(
            candidate_artifact["payload"]
        ):
            raise ValueError(
                "ANN candidate artifact is missing lineage fields: "
                f"{sorted(missing)}"
            )
        candidate_dense_id = candidate_artifact["payload"][
            "dense_store_artifact_id"
        ]
        if candidate_dense_id != dense_artifact["artifact_id"]:
            raise ValueError("ANN candidates belong to a different dense store")
        candidate_inventory = candidate_artifact["payload"][
            "source_inventory_digest"
        ]
        if candidate_inventory != dense_manifest["source_inventory_digest"]:
            raise ValueError("ANN candidates belong to a different source inventory")
        ann_manifest, ann_artifact = load_ann_index(args.ann_index_manifest)
        if ann_manifest["dense_store_artifact_id"] != dense_artifact["artifact_id"]:
            raise ValueError("ANN index belongs to a different dense store")
        if ann_manifest["source_inventory_digest"] != dense_manifest[
            "source_inventory_digest"
        ]:
            raise ValueError("ANN index belongs to a different source inventory")
        _, audit_artifact = load_ann_audit(
            args.ann_audit_manifest,
            index_artifact_id=ann_artifact["artifact_id"],
            n_probes=int(candidate_artifact["payload"]["n_probes"]),
            ann_candidate_count=int(
                candidate_artifact["payload"]["ann_candidate_count"]
            ),
            dense_store_artifact_id=dense_artifact["artifact_id"],
            source_inventory_digest=dense_manifest["source_inventory_digest"],
            vector_inventory_digest=dense_manifest["vector_inventory_digest"],
        )
        if candidate_artifact["payload"]["index_artifact_id"] != ann_artifact[
            "artifact_id"
        ]:
            raise ValueError("ANN candidates belong to a different index")
        if candidate_artifact["payload"]["audit_artifact_id"] != audit_artifact[
            "artifact_id"
        ]:
            raise ValueError("ANN candidates belong to a different recall audit")
        query_contract_identity = _validated_query_contract(
            args.query_embedding_contract, dense_manifest
        )
        excluded_row_ids = _read_excluded_row_ids(
            args.exclude,
            dense_store_artifact_id=dense_artifact["artifact_id"],
            source_inventory_digest=dense_manifest["source_inventory_digest"],
        )
        _validate_excluded_row_ids(
            excluded_row_ids, int(dense_manifest["row_count"])
        )
        candidate_ids = np.load(candidate_path)["candidate_ids"]
        frame = exact_rerank_ann_candidates(
            query_path=args.queries,
            dense_store_manifest=args.dense_store_manifest,
            candidate_row_ids=candidate_ids,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            hard_min_similarity=args.hard_min_similarity,
            similarity_step=args.similarity_step,
            duplicate_similarity=args.duplicate_similarity,
            candidate_multiplier=args.candidate_multiplier,
            excluded_row_ids=excluded_row_ids,
            device=args.device,
        )
        search_proof = "ann_audited_exact_float32_rerank"
        frame["search_proof"] = search_proof
        frame["dense_store_artifact_id"] = dense_artifact["artifact_id"]
        frame["source_inventory_digest"] = dense_manifest[
            "source_inventory_digest"
        ]
        adaptive_radius = frame.attrs.get("adaptive_radius", {})
        output = destination / "neighbors.parquet"
        frame.to_parquet(output, index=False)
        underfilled = any(
            int(stats["selected_count"]) < args.top_k
            for stats in adaptive_radius.get("query_stats", [])
        )
        summary_path = destination / "search_summary.json"
        summary = {
            "row_count": len(frame),
            "query_count": len(adaptive_radius.get("query_stats", [])),
            "selected_query_count": (
                int(frame["query_id"].nunique()) if len(frame) else 0
            ),
            "search_proof": search_proof,
            "dense_store_artifact_id": dense_artifact["artifact_id"],
            "source_inventory_digest": dense_manifest["source_inventory_digest"],
            "candidate_artifact_id": candidate_artifact_id,
            "index_artifact_id": candidate_artifact["payload"][
                "index_artifact_id"
            ],
            "audit_artifact_id": candidate_artifact["payload"][
                "audit_artifact_id"
            ],
            "n_probes": candidate_artifact["payload"]["n_probes"],
            "ann_candidate_count": candidate_artifact["payload"][
                "ann_candidate_count"
            ],
            "eligible_source_rows": max(
                0, int(dense_manifest["row_count"]) - len(excluded_row_ids)
            ),
            "underfill_exhaustion_proven": False,
            "underfill_stop_class": (
                "search_budget_exhausted" if underfilled else None
            ),
            "excluded_source_rows": len(excluded_row_ids),
            "adaptive_radius": adaptive_radius,
        }
        write_json_atomic(summary_path, summary)
        inputs = [
            current_query,
            {
                **file_identity(candidate_artifact_path, role="ann_candidates"),
                "artifact_id": candidate_artifact_id,
            },
            {
                **file_identity(
                    args.dense_store_manifest, role="dense_vector_store"
                ),
                "artifact_id": dense_artifact["artifact_id"],
            },
            {
                **file_identity(args.ann_index_manifest, role="ann_index"),
                "artifact_id": ann_artifact["artifact_id"],
            },
            {
                **file_identity(args.ann_audit_manifest, role="ann_recall_audit"),
                "artifact_id": audit_artifact["artifact_id"],
            },
            query_contract_identity,
        ]
        if args.exclude:
            inputs.append(file_identity(args.exclude, role="excluded_samples"))
        ArtifactManifest(
            artifact_type="neighbor_selection",
            producer={"action": "exact_rerank_ann_candidates", "version": "1.0"},
            inputs=inputs,
            payload={
                "neighbors": file_identity(output),
                "summary": file_identity(summary_path),
                "row_count": len(frame),
                "search_proof": search_proof,
                "min_similarity": args.min_similarity,
                "hard_min_similarity": (
                    args.min_similarity
                    if args.hard_min_similarity is None
                    else args.hard_min_similarity
                ),
                "duplicate_similarity": args.duplicate_similarity,
                "candidate_multiplier": args.candidate_multiplier,
                "top_k": args.top_k,
                "index_artifact_id": candidate_artifact["payload"][
                    "index_artifact_id"
                ],
                "audit_artifact_id": candidate_artifact["payload"][
                    "audit_artifact_id"
                ],
                "n_probes": candidate_artifact["payload"]["n_probes"],
                "ann_candidate_count": candidate_artifact["payload"][
                    "ann_candidate_count"
                ],
                "eligible_source_rows": max(
                    0, int(dense_manifest["row_count"]) - len(excluded_row_ids)
                ),
                "excluded_source_row_ids_digest": canonical_digest(
                    sorted(excluded_row_ids)
                ),
                "underfill_exhaustion_proven": False,
                "underfill_stop_class": (
                    "search_budget_exhausted" if underfilled else None
                ),
            },
        ).commit(destination)
    elif args.command == "ann-search":
        destination = require_uncommitted_output(args.output_dir)
        dense_manifest, dense_artifact = load_dense_store(
            args.dense_store_manifest
        )
        query_contract_identity = _validated_query_contract(
            args.query_embedding_contract, dense_manifest
        )
        audit_identity = None
        search_proof = "unaudited_ann_exact_rerank"
        if args.ann_audit_manifest:
            _, ann_artifact = load_ann_index(args.ann_index_manifest)
            load_ann_audit(
                args.ann_audit_manifest,
                index_artifact_id=ann_artifact["artifact_id"],
                n_probes=args.n_probes,
                ann_candidate_count=args.ann_candidates,
                dense_store_artifact_id=dense_artifact["artifact_id"],
                source_inventory_digest=dense_manifest[
                    "source_inventory_digest"
                ],
                vector_inventory_digest=dense_manifest[
                    "vector_inventory_digest"
                ],
            )
            audit_identity = file_identity(
                args.ann_audit_manifest, role="ann_recall_audit"
            )
            search_proof = "ann_audited_exact_float32_rerank"
        excluded_row_ids = _read_excluded_row_ids(
            args.exclude,
            dense_store_artifact_id=dense_artifact["artifact_id"],
            source_inventory_digest=dense_manifest["source_inventory_digest"],
        )
        _validate_excluded_row_ids(
            excluded_row_ids, int(dense_manifest["row_count"])
        )
        frame = ann_exact_rerank_search(
            query_path=args.queries,
            dense_store_manifest=args.dense_store_manifest,
            ann_index_manifest=args.ann_index_manifest,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            n_probes=args.n_probes,
            ann_candidate_count=args.ann_candidates,
            hard_min_similarity=args.hard_min_similarity,
            similarity_step=args.similarity_step,
            duplicate_similarity=args.duplicate_similarity,
            candidate_multiplier=args.candidate_multiplier,
            excluded_row_ids=excluded_row_ids,
            device=args.device,
        )
        frame["search_proof"] = search_proof
        frame["dense_store_artifact_id"] = dense_artifact["artifact_id"]
        frame["source_inventory_digest"] = dense_manifest[
            "source_inventory_digest"
        ]
        adaptive_radius = frame.attrs.get("adaptive_radius", {})
        output = destination / "neighbors.parquet"
        frame.to_parquet(output, index=False)
        summary_path = destination / "search_summary.json"
        underfilled = any(
            int(stats["selected_count"]) < args.top_k
            for stats in adaptive_radius.get("query_stats", [])
        )
        summary = {
            "row_count": len(frame),
            "query_count": len(adaptive_radius.get("query_stats", [])),
            "selected_query_count": (
                int(frame["query_id"].nunique()) if len(frame) else 0
            ),
            "search_proof": search_proof,
            "dense_store_artifact_id": dense_artifact["artifact_id"],
            "source_inventory_digest": dense_manifest["source_inventory_digest"],
            "underfill_exhaustion_proven": False,
            "underfill_stop_class": (
                "search_budget_exhausted" if underfilled else None
            ),
            "n_probes": args.n_probes,
            "ann_candidate_count": args.ann_candidates,
            "eligible_source_rows": max(
                0, int(dense_manifest["row_count"]) - len(excluded_row_ids)
            ),
            "excluded_source_rows": len(excluded_row_ids),
            "adaptive_radius": adaptive_radius,
        }
        write_json_atomic(summary_path, summary)
        inputs = [
            file_identity(args.queries, role="queries"),
            {
                **file_identity(
                    args.dense_store_manifest, role="dense_vector_store"
                ),
                "artifact_id": dense_artifact["artifact_id"],
            },
            file_identity(args.ann_index_manifest, role="ann_index"),
            query_contract_identity,
        ]
        if audit_identity is not None:
            inputs.append(audit_identity)
        if args.exclude:
            inputs.append(file_identity(args.exclude, role="excluded_samples"))
        ArtifactManifest(
            artifact_type="neighbor_selection",
            producer={"action": "ann_exact_rerank_search", "version": "1.0"},
            inputs=inputs,
            payload={
                "neighbors": file_identity(output),
                "summary": file_identity(summary_path),
                "row_count": len(frame),
                "search_proof": search_proof,
                "min_similarity": args.min_similarity,
                "hard_min_similarity": (
                    args.min_similarity
                    if args.hard_min_similarity is None
                    else args.hard_min_similarity
                ),
                "duplicate_similarity": args.duplicate_similarity,
                "candidate_multiplier": args.candidate_multiplier,
                "top_k": args.top_k,
                "n_probes": args.n_probes,
                "ann_candidate_count": args.ann_candidates,
                "eligible_source_rows": max(
                    0, int(dense_manifest["row_count"]) - len(excluded_row_ids)
                ),
                "excluded_source_row_ids_digest": canonical_digest(
                    sorted(excluded_row_ids)
                ),
                "underfill_exhaustion_proven": False,
                "underfill_stop_class": (
                    "search_budget_exhausted" if underfilled else None
                ),
            },
        ).commit(destination)
    elif args.command == "dense-exact-search":
        destination = require_uncommitted_output(args.output_dir)
        dense_manifest, dense_artifact = load_dense_store(
            args.dense_store_manifest
        )
        source_manifest, source_identity, source_artifact = (
            _validated_store_identity(args.source_store_manifest)
        )
        _validate_dense_source_lineage(
            source_manifest,
            source_identity,
            source_artifact,
            dense_manifest,
        )
        query_contract_identity = _validated_query_contract(
            args.query_embedding_contract, source_manifest
        )
        excluded_row_ids = _read_excluded_row_ids(
            args.exclude,
            dense_store_artifact_id=dense_artifact["artifact_id"],
            source_inventory_digest=dense_manifest["source_inventory_digest"],
        )
        _validate_excluded_row_ids(
            excluded_row_ids, int(dense_manifest["row_count"])
        )
        frame = exact_dense_search(
            query_path=args.queries,
            dense_store_manifest=args.dense_store_manifest,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            checkpoint_path=destination / "exact_progress.npz",
            hard_min_similarity=args.hard_min_similarity,
            similarity_step=args.similarity_step,
            duplicate_similarity=args.duplicate_similarity,
            candidate_multiplier=args.candidate_multiplier,
            excluded_row_ids=excluded_row_ids,
            chunk_rows=args.chunk_rows,
            checkpoint_chunks=args.checkpoint_chunks,
            device=args.device,
        )
        search_proof = "exact_all_dense_rows_float32"
        frame["search_proof"] = search_proof
        frame["dense_store_artifact_id"] = dense_artifact["artifact_id"]
        frame["source_inventory_digest"] = dense_manifest[
            "source_inventory_digest"
        ]
        adaptive_radius = frame.attrs.get("adaptive_radius", {})
        output = destination / "neighbors.parquet"
        frame.to_parquet(output, index=False)
        query_stats = adaptive_radius.get("query_stats", [])
        underfilled = any(
            int(stats["selected_count"]) < args.top_k for stats in query_stats
        )
        underfill_proven = bool(
            adaptive_radius.get("underfill_exhaustion_proven", False)
        )
        stop_class = None
        if underfilled:
            stop_class = (
                "radius_exhausted"
                if underfill_proven
                else "search_budget_exhausted"
            )
        summary_path = destination / "search_summary.json"
        summary = {
            "row_count": len(frame),
            "query_count": len(query_stats),
            "selected_query_count": (
                int(frame["query_id"].nunique()) if len(frame) else 0
            ),
            "search_proof": search_proof,
            "dense_store_artifact_id": dense_artifact["artifact_id"],
            "source_inventory_digest": dense_manifest[
                "source_inventory_digest"
            ],
            "source_rows": int(dense_manifest["row_count"]),
            "eligible_source_rows": max(
                0, int(dense_manifest["row_count"]) - len(excluded_row_ids)
            ),
            "excluded_source_rows": len(excluded_row_ids),
            "candidate_truncated": bool(
                adaptive_radius.get("candidate_truncated", False)
            ),
            "underfill_exhaustion_proven": underfill_proven,
            "underfill_stop_class": stop_class,
            "adaptive_radius": adaptive_radius,
        }
        write_json_atomic(summary_path, summary)
        inputs = [
            file_identity(args.queries, role="queries"),
            source_identity,
            {
                **file_identity(
                    args.dense_store_manifest, role="dense_vector_store"
                ),
                "artifact_id": dense_artifact["artifact_id"],
            },
            query_contract_identity,
        ]
        if args.exclude:
            inputs.append(file_identity(args.exclude, role="excluded_samples"))
        ArtifactManifest(
            artifact_type="neighbor_selection",
            producer={
                "action": "exact_dense_search",
                "version": "1.0",
                "implementation_sha256": {
                    "entrypoint": file_sha256(Path(__file__)),
                    "dense_search": file_sha256(
                        Path(__file__).resolve().parents[1] / "dense_search.py"
                    ),
                },
            },
            inputs=inputs,
            payload={
                "neighbors": file_identity(output),
                "summary": file_identity(summary_path),
                "row_count": len(frame),
                "search_proof": search_proof,
                "min_similarity": args.min_similarity,
                "hard_min_similarity": (
                    args.min_similarity
                    if args.hard_min_similarity is None
                    else args.hard_min_similarity
                ),
                "duplicate_similarity": args.duplicate_similarity,
                "candidate_multiplier": args.candidate_multiplier,
                "top_k": args.top_k,
                "chunk_rows": args.chunk_rows,
                "excluded_source_row_ids_digest": canonical_digest(
                    sorted(excluded_row_ids)
                ),
                "candidate_truncated": bool(
                    adaptive_radius.get("candidate_truncated", False)
                ),
                "underfill_exhaustion_proven": underfill_proven,
                "underfill_stop_class": stop_class,
            },
        ).commit(destination)
    elif args.command == "exact-search":
        destination = require_uncommitted_output(args.output_dir)
        excluded = _read_excluded(args.exclude)
        locator_defaults = {}
        store_identity = None
        query_contract_identity = None
        if args.source_store_manifest:
            store_manifest, store_identity = _validated_store(
                args.source_store_manifest,
                args.source_part,
            )
            locator_defaults = store_manifest.get("locator_defaults", {})
            if not args.query_embedding_contract:
                raise ValueError(
                    "Registered-store search requires --query-embedding-contract"
                )
            query_contract_identity = _validated_query_contract(
                args.query_embedding_contract,
                store_manifest,
            )
        elif args.query_embedding_contract:
            raise ValueError(
                "--query-embedding-contract requires --source-store-manifest"
            )
        frame = exact_sharded_search(
            query_path=args.queries,
            source_parts=args.source_part,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            excluded_ids=excluded,
            locator_defaults=locator_defaults,
            device=args.device,
            hard_min_similarity=args.hard_min_similarity,
            similarity_step=args.similarity_step,
            duplicate_similarity=args.duplicate_similarity,
            candidate_multiplier=args.candidate_multiplier,
        )
        adaptive_radius = frame.attrs.get("adaptive_radius", {})
        output = destination / "neighbors.parquet"
        frame.to_parquet(output, index=False)
        source_ids = [
            pd.read_parquet(path, columns=["sample_id"], pre_buffer=False)["sample_id"].astype(str)
            for path in args.source_part
        ]
        source_rows = sum(len(values) for values in source_ids)
        excluded_source_rows = sum(
            values.isin(excluded).sum() for values in source_ids
        )
        eligible_rows = source_rows - int(excluded_source_rows)
        underfill_proven = bool(
            adaptive_radius.get("underfill_exhaustion_proven", False)
        )
        summary_path = destination / "search_summary.json"
        write_json_atomic(
            summary_path,
            {
                "row_count": len(frame),
                "query_count": len(adaptive_radius.get("query_stats", [])),
                "selected_query_count": (
                    int(frame["query_id"].nunique()) if len(frame) else 0
                ),
                "search_proof": "exact_all_declared_shards",
                "source_rows": source_rows,
                "excluded_ids": len(excluded),
                "excluded_source_rows": int(excluded_source_rows),
                "eligible_source_rows": eligible_rows,
                "device": args.device,
                "candidate_truncated": bool(
                    adaptive_radius.get("candidate_truncated", False)
                ),
                "underfill_exhaustion_proven": underfill_proven,
                "adaptive_radius": adaptive_radius,
            },
        )
        inputs = [file_identity(args.queries, role="queries")]
        if store_identity is not None:
            inputs.append(store_identity)
            inputs.append(query_contract_identity)
        else:
            inputs.extend(
                file_identity(path, role="source_shard")
                for path in args.source_part
            )
        if args.exclude:
            inputs.append(file_identity(args.exclude, role="excluded_samples"))
        ArtifactManifest(
            artifact_type="neighbor_selection",
            producer={
                "action": "exact_sharded_search",
                "version": "1.0",
                "implementation_sha256": {
                    "entrypoint": file_sha256(Path(__file__)),
                    "search": file_sha256(
                        Path(__file__).resolve().parents[1] / "search.py"
                    ),
                },
            },
            inputs=inputs,
            payload={
                "neighbors": file_identity(output),
                "summary": file_identity(summary_path),
                "row_count": len(frame),
                "search_proof": "exact_all_declared_shards",
                "min_similarity": args.min_similarity,
                "hard_min_similarity": (
                    args.min_similarity
                    if args.hard_min_similarity is None
                    else args.hard_min_similarity
                ),
                "duplicate_similarity": args.duplicate_similarity,
                "candidate_multiplier": args.candidate_multiplier,
                "top_k": args.top_k,
                "device": args.device,
                "excluded_sample_ids_digest": canonical_digest(sorted(excluded)),
                "candidate_truncated": bool(
                    adaptive_radius.get("candidate_truncated", False)
                ),
                "underfill_exhaustion_proven": underfill_proven,
            },
        ).commit(destination)
    elif args.command == "materialize":
        result = materialize_manifest(
            delta_path=args.delta,
            previous_path=args.previous,
            query_path=args.query_manifest,
            balance_column=args.balance_column,
            output_dir=args.output_dir,
            overlap_policy=args.overlap_policy,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
