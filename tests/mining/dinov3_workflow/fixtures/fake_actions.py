#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic leaf actions used only by workflow tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {
        "uri": path.resolve().as_uri(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _digest(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _commit(
    output_dir: Path,
    artifact_type: str,
    payload: dict,
    inputs: list[dict] | None = None,
) -> None:
    artifact = {
        "artifact_type": artifact_type,
        "schema_version": "1.0",
        "producer": {"action": "fake", "version": "1.0"},
        "inputs": inputs or [],
        "payload": payload,
        "created_at": "test",
    }
    artifact["artifact_id"] = _digest(
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
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    (output_dir / "_SUCCESS").write_text(
        artifact["artifact_id"] + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--strategy", required=True)
    train = commands.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--passes", required=True, type=int)
    train.add_argument("--num-nodes", type=int)
    train.add_argument("--gpus-per-node", type=int)
    train.add_argument("--output-dir", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--scope", required=True)
    candidates = commands.add_parser("ann-candidates")
    candidates.add_argument("--queries", required=True)
    candidates.add_argument("--ann-index-manifest", required=True)
    candidates.add_argument("--query-embedding-contract", required=True)
    candidates.add_argument("--ann-audit-manifest", required=True)
    candidates.add_argument("--n-probes", required=True, type=int)
    candidates.add_argument("--ann-candidates", required=True, type=int)
    candidates.add_argument("--output-dir", required=True)
    rerank = commands.add_parser("ann-rerank")
    rerank.add_argument("--queries", required=True)
    rerank.add_argument("--candidates", required=True)
    rerank.add_argument("--dense-store-manifest", required=True)
    rerank.add_argument("--ann-index-manifest", required=True)
    rerank.add_argument("--ann-audit-manifest", required=True)
    rerank.add_argument("--query-embedding-contract", required=True)
    rerank.add_argument("--top-k", required=True, type=int)
    rerank.add_argument("--min-similarity", required=True, type=float)
    rerank.add_argument("--hard-min-similarity", required=True, type=float)
    rerank.add_argument("--similarity-step", required=True, type=float)
    rerank.add_argument("--duplicate-similarity", required=True, type=float)
    rerank.add_argument("--candidate-multiplier", required=True, type=int)
    rerank.add_argument("--device", required=True)
    rerank.add_argument("--exclude")
    rerank.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    success_value = "ok"
    if args.command == "score":
        frame = pd.read_parquet(args.input, pre_buffer=False)
        if args.strategy == "grit_score" and "role" in frame:
            frame = frame.loc[frame["role"].astype(str) == "query"].reset_index(
                drop=True
            )
        if os.environ.get("FAKE_DROP_SCORE_ROW"):
            frame = frame.iloc[:-1].reset_index(drop=True)
        if os.environ.get("FAKE_CHANGE_SCORE_EMBEDDING") and not frame.empty:
            embeddings = frame["embedding"].tolist()
            changed = list(embeddings[0])
            changed[0] = float(changed[0]) + 0.25
            embeddings[0] = changed
            frame["embedding"] = embeddings
        name = "grit_scores.parquet" if args.strategy == "grit_score" else "task_scores.parquet"
        output_path = output_dir / name
        frame.to_parquet(output_path, index=False)
        commit_path = output_dir / "score_commit.json"
        commit_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "action": "fake_score",
                    "input_sha256": _sha256(Path(args.input)),
                    "checkpoint_sha256": _sha256(Path(args.checkpoint).resolve()),
                    "output_sha256": _sha256(output_path),
                    "implementation_sha256": os.environ.get(
                        "FAKE_IMPLEMENTATION_SHA256", _sha256(Path(__file__))
                    ),
                    "entrypoint_sha256": _sha256(Path(__file__)),
                    "request_sha256": os.environ.get(
                        "TAO_REFINEMENT_REQUEST_SHA256"
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        success_value = _sha256(commit_path)
    elif args.command == "ann-candidates":
        query_count = len(pd.read_parquet(args.queries, pre_buffer=False))
        ann_artifact = json.loads(
            Path(args.ann_index_manifest)
            .with_name("artifact.json")
            .read_text(encoding="utf-8")
        )
        ann_payload = json.loads(
            Path(args.ann_index_manifest).read_text(encoding="utf-8")
        )
        audit_artifact = json.loads(
            Path(args.ann_audit_manifest)
            .with_name("artifact.json")
            .read_text(encoding="utf-8")
        )
        np.savez(
            output_dir / "ann_candidates.npz",
            candidate_ids=np.zeros(
                (query_count, args.ann_candidates), dtype=np.int64
            ),
        )
        payload = {
            "candidates": _identity(output_dir / "ann_candidates.npz"),
            "query_count": query_count,
            "candidate_count": args.ann_candidates,
            "n_probes": args.n_probes,
            "ann_candidate_count": args.ann_candidates,
            "index_artifact_id": ann_artifact["artifact_id"],
            "audit_artifact_id": audit_artifact["artifact_id"],
            "dense_store_artifact_id": ann_payload[
                "dense_store_artifact_id"
            ],
            "source_inventory_digest": ann_payload[
                "source_inventory_digest"
            ],
        }
        (output_dir / "candidate_summary.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        _commit(output_dir, "ann_candidates", payload)
    elif args.command == "ann-rerank":
        fail_once = os.environ.get("FAKE_RERANK_FAIL_ONCE")
        if fail_once and not Path(fail_once).exists():
            Path(fail_once).write_text("failed\n", encoding="utf-8")
            return 9
        queries = pd.read_parquet(args.queries, pre_buffer=False)
        selected = queries.iloc[[0]][["sample_id"]].copy()
        selected["query_id"] = selected["sample_id"].astype(str)
        selected["sample_id"] = "ann-source"
        selected["source_row_id"] = 0
        selected["storage_type"] = "file"
        selected["path"] = "/data/ann-source.jpg"
        selected["member"] = None
        selected["cosine_similarity"] = 0.9
        selected["search_proof"] = "ann_audited_exact_float32_rerank"
        dense_artifact = json.loads(
            Path(args.dense_store_manifest)
            .with_name("artifact.json")
            .read_text(encoding="utf-8")
        )
        dense_payload = json.loads(
            Path(args.dense_store_manifest).read_text(encoding="utf-8")
        )
        selected["dense_store_artifact_id"] = dense_artifact["artifact_id"]
        selected["source_inventory_digest"] = dense_payload[
            "source_inventory_digest"
        ]
        selected.to_parquet(output_dir / "neighbors.parquet", index=False)
        summary = {
            "row_count": 1,
            "eligible_source_rows": 3,
            "search_proof": "ann_audited_exact_float32_rerank",
            "adaptive_radius": {"query_stats": []},
            "underfill_exhaustion_proven": False,
        }
        summary_path = output_dir / "search_summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        _commit(
            output_dir,
            "neighbor_selection",
            {
                "neighbors": _identity(output_dir / "neighbors.parquet"),
                "summary": _identity(summary_path),
                "search_proof": summary["search_proof"],
            },
            inputs=[
                {
                    **_identity(Path(args.dense_store_manifest)),
                    "role": "dense_vector_store",
                    "artifact_id": dense_artifact["artifact_id"],
                },
                {
                    **_identity(Path(args.query_embedding_contract)),
                    "role": "query_embedding_contract",
                },
                {
                    **_identity(Path(args.ann_index_manifest)),
                    "role": "ann_index",
                    "artifact_id": json.loads(
                        Path(args.ann_index_manifest)
                        .with_name("artifact.json")
                        .read_text(encoding="utf-8")
                    )["artifact_id"],
                },
                {
                    **_identity(Path(args.ann_audit_manifest)),
                    "role": "ann_recall_audit",
                    "artifact_id": json.loads(
                        Path(args.ann_audit_manifest)
                        .with_name("artifact.json")
                        .read_text(encoding="utf-8")
                    )["artifact_id"],
                },
            ],
        )
    elif args.command == "train":
        rows = len(pd.read_parquet(args.manifest, pre_buffer=False))
        checkpoint = output_dir / "checkpoint.pth"
        checkpoint.write_text(
            json.dumps(
                {
                    "parent": args.checkpoint,
                    "passes": args.passes,
                    "training_rows": rows,
                    "nodes": args.num_nodes,
                    "gpus_per_node": args.gpus_per_node,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        checkpoint_sha256 = _sha256(checkpoint)
        runtime_spec = output_dir / "experiment.yaml"
        runtime_spec.write_text("runtime: fake\n", encoding="utf-8")
        contract_path = output_dir / "training_contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": checkpoint_sha256,
                    "runtime_spec": str(runtime_spec.resolve()),
                    "runtime_spec_bytes": runtime_spec.stat().st_size,
                    "runtime_spec_sha256": _sha256(runtime_spec),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        commit_path = output_dir / "training_commit.json"
        commit_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "training_contract_sha256": _sha256(contract_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "runtime_spec_sha256": _sha256(runtime_spec),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        success_value = _sha256(commit_path)
    else:
        Path(args.benchmark).read_text(encoding="utf-8")
        metric_value = 1.0
        configured_values = os.environ.get("FAKE_METRIC_VALUES")
        if configured_values:
            round_index = int(output_dir.parent.name.removeprefix("round_"))
            values = [float(value) for value in configured_values.split(",")]
            metric_value = values[round_index]
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "checkpoint": args.checkpoint,
                    "evaluation_scope": args.scope,
                    "metrics": [
                        {
                            "task": "smoke",
                            "name": "score",
                            "value": (
                                float("nan")
                                if os.environ.get("FAKE_NONFINITE_METRIC")
                                else metric_value
                            ),
                            "higher_is_better": True,
                            "sample_count": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        checkpoint = Path(args.checkpoint).expanduser().absolute()
        resolved_checkpoint = checkpoint.resolve()
        evaluation_commit = output_dir / "evaluation_commit.json"
        evaluation_commit.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "evaluation_scope": args.scope,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "resolved_path": str(resolved_checkpoint),
                        "method": "sha256",
                        "sha256": _sha256(resolved_checkpoint),
                    },
                    "metrics": _identity(metrics_path),
                    "benchmark": _identity(Path(args.benchmark).resolve()),
                    "request_sha256": os.environ.get(
                        "TAO_REFINEMENT_REQUEST_SHA256"
                    ),
                    "entrypoint_sha256": _sha256(Path(__file__)),
                    "implementation_sha256": _sha256(Path(__file__)),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        success_value = _sha256(evaluation_commit)
    success_path = output_dir / "_SUCCESS"
    if not success_path.exists():
        success_path.write_text(success_value + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
