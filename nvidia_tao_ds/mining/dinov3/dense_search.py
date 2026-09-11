# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable exact search over a row-aligned dense embedding store."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from .ann_search import exact_rerank_ann_candidates
from .contracts import canonical_digest, file_identity, file_sha256
from .dense_store import load_dense_store
from .search import _matrix


def _save_progress(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def _local_vector_path(dense: dict[str, Any]) -> Path:
    parsed = urlparse(str(dense["vector_uri"]))
    if parsed.scheme != "file":
        raise ValueError("Dense exact search requires a local file:// vector store")
    return Path(unquote(parsed.path)).resolve()


def _excluded_in_chunk(
    excluded: np.ndarray, start: int, stop: int
) -> np.ndarray:
    left = int(np.searchsorted(excluded, start, side="left"))
    right = int(np.searchsorted(excluded, stop, side="left"))
    return excluded[left:right] - start


def _numpy_topk(
    scores: np.ndarray, *, limit: int, row_offset: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = min(limit, int(scores.shape[1]))
    if keep == int(scores.shape[1]):
        indexes = np.broadcast_to(
            np.arange(scores.shape[1], dtype=np.int64), scores.shape
        )
    else:
        indexes = np.argpartition(scores, -keep, axis=1)[:, -keep:]
    values = np.take_along_axis(scores, indexes, axis=1)
    return values.astype(np.float32, copy=False), indexes + row_offset


def _merge_numpy_topk(
    best_scores: np.ndarray,
    best_ids: np.ndarray,
    local_scores: np.ndarray,
    local_ids: np.ndarray,
    *,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate((best_scores, local_scores), axis=1)
    ids = np.concatenate((best_ids, local_ids), axis=1)
    keep = min(limit, int(scores.shape[1]))
    indexes = np.argpartition(scores, -keep, axis=1)[:, -keep:]
    return (
        np.take_along_axis(scores, indexes, axis=1),
        np.take_along_axis(ids, indexes, axis=1),
    )


def exact_dense_candidates(
    *,
    query_path: str | Path,
    dense_store_manifest: str | Path,
    candidate_limit: int,
    hard_min_similarity: float,
    excluded_row_ids: set[int] | None,
    checkpoint_path: str | Path,
    chunk_rows: int = 32768,
    checkpoint_chunks: int = 64,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Scan every dense row, retaining exact top candidates per query.

    The checkpoint contains the global top-k state and next dense row. A
    restarted worker validates the complete request identity before adopting it.
    """
    if candidate_limit <= 0 or chunk_rows <= 0 or checkpoint_chunks <= 0:
        raise ValueError("candidate and checkpoint sizes must be positive")
    if not -1.0 <= hard_min_similarity <= 1.0:
        raise ValueError("hard_min_similarity must be in [-1, 1]")

    dense, dense_artifact = load_dense_store(dense_store_manifest)
    queries = pd.read_parquet(query_path, pre_buffer=False)
    if missing := {"sample_id", "embedding"}.difference(queries.columns):
        raise ValueError(f"Query table is missing columns: {sorted(missing)}")
    query_vectors = _matrix(queries["embedding"])
    dimension = int(dense["embedding_dim"])
    row_count = int(dense["row_count"])
    if query_vectors.shape[1] != dimension:
        raise ValueError("Query and dense-store embedding dimensions differ")
    excluded = np.asarray(sorted(excluded_row_ids or set()), dtype=np.int64)
    if len(excluded) and (excluded[0] < 0 or excluded[-1] >= row_count):
        raise ValueError("Dense exact-search exclusions contain an invalid row ID")

    checkpoint = Path(checkpoint_path).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    implementation = file_sha256(Path(__file__))
    contract = {
        "schema_version": "1.0",
        "implementation_sha256": implementation,
        "query": file_identity(query_path),
        "dense_store_artifact_id": dense_artifact["artifact_id"],
        "source_inventory_digest": dense["source_inventory_digest"],
        "vector_inventory_digest": dense["vector_inventory_digest"],
        "row_count": row_count,
        "embedding_dim": dimension,
        "candidate_limit": int(candidate_limit),
        "hard_min_similarity": float(hard_min_similarity),
        "excluded_row_ids_digest": canonical_digest(excluded.tolist()),
        "chunk_rows": int(chunk_rows),
        "device": str(device),
    }
    contract_digest = canonical_digest(contract)
    start_row = 0
    elapsed_before = 0.0
    counts = np.zeros(len(queries), dtype=np.int64)
    best_scores = np.full(
        (len(queries), candidate_limit), -np.inf, dtype=np.float32
    )
    best_ids = np.full((len(queries), candidate_limit), -1, dtype=np.int64)
    if checkpoint.is_file():
        with np.load(checkpoint) as progress:
            if str(progress["contract_digest"].item()) != contract_digest:
                raise RuntimeError("Dense exact-search checkpoint belongs to another request")
            start_row = int(progress["next_row"].item())
            elapsed_before = float(progress["elapsed_seconds"].item())
            counts = progress["candidate_counts"].astype(np.int64, copy=True)
            best_scores = progress["scores"].astype(np.float32, copy=True)
            best_ids = progress["ids"].astype(np.int64, copy=True)
        expected_shape = (len(queries), candidate_limit)
        if best_scores.shape != expected_shape or best_ids.shape != expected_shape:
            raise RuntimeError("Dense exact-search checkpoint has the wrong shape")
        if counts.shape != (len(queries),) or not 0 <= start_row <= row_count:
            raise RuntimeError("Dense exact-search checkpoint metadata is invalid")

    vectors = np.memmap(
        _local_vector_path(dense),
        mode="r",
        dtype=np.dtype(dense["dtype"]),
        shape=(row_count, dimension),
    )
    started = time.monotonic()
    chunks_this_attempt = 0
    if device == "cpu":
        for start in range(start_row, row_count, chunk_rows):
            stop = min(row_count, start + chunk_rows)
            chunk = np.asarray(vectors[start:stop], dtype=np.float32)
            scores = query_vectors @ chunk.T
            local_excluded = _excluded_in_chunk(excluded, start, stop)
            if len(local_excluded):
                scores[:, local_excluded] = -np.inf
            counts += np.sum(scores >= hard_min_similarity, axis=1)
            local_scores, local_ids = _numpy_topk(
                scores, limit=candidate_limit, row_offset=start
            )
            best_scores, best_ids = _merge_numpy_topk(
                best_scores,
                best_ids,
                local_scores,
                local_ids,
                limit=candidate_limit,
            )
            chunks_this_attempt += 1
            if chunks_this_attempt % checkpoint_chunks == 0 or stop == row_count:
                _save_progress(
                    checkpoint,
                    contract_digest=np.asarray(contract_digest),
                    next_row=np.asarray(stop, dtype=np.int64),
                    elapsed_seconds=np.asarray(
                        elapsed_before + time.monotonic() - started,
                        dtype=np.float64,
                    ),
                    candidate_counts=counts,
                    scores=best_scores,
                    ids=best_ids,
                )
    else:
        import torch  # pylint: disable=import-outside-toplevel

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(f"CUDA exact search is unavailable: {device}")
        query_tensor = torch.from_numpy(query_vectors).to(device)
        best_score_tensor = torch.from_numpy(best_scores).to(device)
        best_id_tensor = torch.from_numpy(best_ids).to(device)
        count_tensor = torch.from_numpy(counts).to(device)
        for start in range(start_row, row_count, chunk_rows):
            stop = min(row_count, start + chunk_rows)
            chunk = torch.from_numpy(
                np.asarray(vectors[start:stop], dtype=np.float32)
            ).to(device)
            scores = query_tensor @ chunk.T
            local_excluded = _excluded_in_chunk(excluded, start, stop)
            if len(local_excluded):
                indexes = torch.from_numpy(local_excluded).to(device)
                scores[:, indexes] = float("-inf")
            count_tensor += (scores >= hard_min_similarity).sum(dim=1)
            keep = min(candidate_limit, int(scores.shape[1]))
            local_scores, local_ids = torch.topk(
                scores, k=keep, dim=1, sorted=False
            )
            local_ids += start
            combined_scores = torch.cat((best_score_tensor, local_scores), dim=1)
            combined_ids = torch.cat((best_id_tensor, local_ids), dim=1)
            keep = min(candidate_limit, int(combined_scores.shape[1]))
            best_score_tensor, indexes = torch.topk(
                combined_scores, k=keep, dim=1, sorted=False
            )
            best_id_tensor = torch.gather(combined_ids, 1, indexes)
            chunks_this_attempt += 1
            if chunks_this_attempt % checkpoint_chunks == 0 or stop == row_count:
                best_scores = best_score_tensor.cpu().numpy()
                best_ids = best_id_tensor.cpu().numpy()
                counts = count_tensor.cpu().numpy()
                _save_progress(
                    checkpoint,
                    contract_digest=np.asarray(contract_digest),
                    next_row=np.asarray(stop, dtype=np.int64),
                    elapsed_seconds=np.asarray(
                        elapsed_before + time.monotonic() - started,
                        dtype=np.float64,
                    ),
                    candidate_counts=counts,
                    scores=best_scores,
                    ids=best_ids,
                )
            del scores, chunk, local_scores, local_ids, combined_scores, combined_ids

    elapsed_this_attempt = time.monotonic() - started
    with np.load(checkpoint) as progress:
        if int(progress["next_row"].item()) != row_count:
            raise RuntimeError("Dense exact search ended without a complete checkpoint")
        best_scores = progress["scores"].astype(np.float32, copy=True)
        best_ids = progress["ids"].astype(np.int64, copy=True)
        counts = progress["candidate_counts"].astype(np.int64, copy=True)
        elapsed_total = float(progress["elapsed_seconds"].item())
    order = np.argsort(best_scores, axis=1)[:, ::-1]
    best_scores = np.take_along_axis(best_scores, order, axis=1)
    best_ids = np.take_along_axis(best_ids, order, axis=1)
    return best_scores, best_ids, {
        "contract": contract,
        "contract_digest": contract_digest,
        "rows_scanned": row_count,
        "resumed_from_row": start_row,
        "chunks_this_attempt": chunks_this_attempt,
        "elapsed_seconds_this_attempt": elapsed_this_attempt,
        "elapsed_seconds_total": elapsed_total,
        "candidate_counts_above_hard_floor": counts.tolist(),
        "checkpoint": file_identity(checkpoint),
    }


def exact_dense_search(
    *,
    query_path: str | Path,
    dense_store_manifest: str | Path,
    top_k: int,
    min_similarity: float,
    checkpoint_path: str | Path,
    hard_min_similarity: float | None = None,
    similarity_step: float = 0.02,
    duplicate_similarity: float = 1.0,
    candidate_multiplier: int = 10,
    excluded_row_ids: set[int] | None = None,
    chunk_rows: int = 32768,
    checkpoint_chunks: int = 64,
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Mine from every dense row using exact float32 scores and durable progress."""
    if top_k <= 0 or candidate_multiplier <= 0:
        raise ValueError("top_k and candidate_multiplier must be positive")
    hard_minimum = min_similarity if hard_min_similarity is None else hard_min_similarity
    candidate_limit = top_k * candidate_multiplier
    _, candidate_ids, scan = exact_dense_candidates(
        query_path=query_path,
        dense_store_manifest=dense_store_manifest,
        candidate_limit=candidate_limit,
        hard_min_similarity=hard_minimum,
        excluded_row_ids=excluded_row_ids,
        checkpoint_path=checkpoint_path,
        chunk_rows=chunk_rows,
        checkpoint_chunks=checkpoint_chunks,
        device=device,
    )
    result = exact_rerank_ann_candidates(
        query_path=query_path,
        dense_store_manifest=dense_store_manifest,
        candidate_row_ids=candidate_ids,
        top_k=top_k,
        min_similarity=min_similarity,
        hard_min_similarity=hard_minimum,
        similarity_step=similarity_step,
        duplicate_similarity=duplicate_similarity,
        candidate_multiplier=candidate_multiplier,
        excluded_row_ids=excluded_row_ids,
        device=device,
    )
    adaptive = result.attrs["adaptive_radius"]
    counts = scan["candidate_counts_above_hard_floor"]
    for stats, count in zip(adaptive["query_stats"], counts):
        stats["candidate_count_above_hard_floor"] = int(count)
        stats["candidate_truncated"] = int(count) > candidate_limit
    adaptive["candidate_truncated"] = any(
        stats["candidate_truncated"] for stats in adaptive["query_stats"]
    )
    adaptive["underfill_exhaustion_proven"] = all(
        int(stats["selected_count"]) >= top_k or not stats["candidate_truncated"]
        for stats in adaptive["query_stats"]
    )
    adaptive["candidate_limit_per_query"] = candidate_limit
    adaptive["scan"] = scan
    result.attrs["adaptive_radius"] = adaptive
    return result
