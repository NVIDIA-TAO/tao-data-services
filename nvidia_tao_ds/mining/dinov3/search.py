# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first sharded cosine search for refinement mining."""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd


SEARCH_RESULT_COLUMNS = frozenset({
    "query_id", "sample_id", "source_row_id", "rank", "cosine_similarity",
    "accepted_at_similarity_threshold", "max_similarity_to_selected",
    "source_part", "search_proof", "search_device",
})


def _matrix(values: pd.Series) -> np.ndarray:
    result = np.asarray(values.tolist(), dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(values):
        raise ValueError("Embedding column must contain equal-length vectors")
    if not np.isfinite(result).all():
        raise ValueError("Embedding column contains non-finite values")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("Embedding norms contain non-finite values")
    if np.any(norms == 0):
        raise ValueError("Embedding table contains a zero vector")
    return result / norms


def exact_sharded_search(
    *,
    query_path: str | Path,
    source_parts: list[str | Path],
    top_k: int,
    min_similarity: float,
    excluded_ids: set[str] | None = None,
    id_column: str = "sample_id",
    embedding_column: str = "embedding",
    locator_defaults: dict[str, str] | None = None,
    device: str = "cpu",
    hard_min_similarity: float | None = None,
    similarity_step: float = 0.02,
    duplicate_similarity: float = 1.0,
    candidate_multiplier: int = 10,
) -> pd.DataFrame:
    """Search every source shard and select relevant, C-RADIO-diverse neighbors.

    This implementation is deliberately streaming and correctness-first. Large
    production runs should use an audited ANN index followed by this exact
    reranker or a distributed exact terminal scan.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hard_min = min_similarity if hard_min_similarity is None else hard_min_similarity
    if not -1.0 <= hard_min <= min_similarity <= 1.0:
        raise ValueError(
            "similarity thresholds must satisfy -1 <= hard_min_similarity "
            "<= min_similarity <= 1"
        )
    if similarity_step <= 0:
        raise ValueError("similarity_step must be positive")
    if not min_similarity < duplicate_similarity <= 1.0:
        raise ValueError(
            "duplicate_similarity must be greater than min_similarity and at most 1"
        )
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")
    queries = pd.read_parquet(query_path, pre_buffer=False)
    required = {id_column, embedding_column}
    if missing := required.difference(queries.columns):
        raise ValueError(f"Query table is missing columns: {sorted(missing)}")
    query_vectors = _matrix(queries[embedding_column])
    torch_module = None
    query_tensor = None
    if device != "cpu":
        import torch  # pylint: disable=import-outside-toplevel

        if not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA search requested but unavailable: {device}")
        torch_module = torch
        query_tensor = torch.from_numpy(query_vectors).to(device)
    candidate_limit = top_k * candidate_multiplier
    heaps: list[list[tuple]] = [[] for _ in range(len(queries))]
    candidate_counts_above_floor = [0] * len(queries)
    excluded = excluded_ids or set()
    best_scores = None
    best_part_indexes = None
    best_source_indexes = None
    best_vectors = None

    resolved_parts = [Path(value) for value in source_parts]
    for part_index, part in enumerate(resolved_parts):
        columns = None if device == "cpu" else [id_column, embedding_column]
        source = pd.read_parquet(part, columns=columns, pre_buffer=False)
        if device == "cpu":
            for name, value in (locator_defaults or {}).items():
                if name not in source:
                    source[name] = value
        missing = {id_column, embedding_column}.difference(source.columns)
        if missing:
            raise ValueError(f"Source shard {part} is missing columns: {sorted(missing)}")
        source[id_column] = source[id_column].astype(str)
        source = source[~source[id_column].isin(excluded)]
        if source.empty:
            continue
        source_vectors = _matrix(source[embedding_column])
        if device == "cpu":
            similarities = query_vectors @ source_vectors.T
            source_ids = source[id_column].tolist()
            for query_index, scores in enumerate(similarities):
                candidate_indexes = np.flatnonzero(scores >= hard_min)
                candidate_counts_above_floor[query_index] += len(candidate_indexes)
                for source_index in candidate_indexes:
                    item = (
                        float(scores[source_index]),
                        source_ids[source_index],
                        str(part),
                        int(source_index),
                        {
                            str(column): source.iloc[source_index][column]
                            for column in source.columns
                            if column not in {id_column, embedding_column}
                        },
                        source_vectors[source_index].tobytes(),
                    )
                    heap = heaps[query_index]
                    if len(heap) < candidate_limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            continue

        assert torch_module is not None and query_tensor is not None
        source_tensor = torch_module.from_numpy(source_vectors).to(device)
        similarities = query_tensor @ source_tensor.T
        counts = (similarities >= hard_min).sum(dim=1).cpu().tolist()
        candidate_counts_above_floor = [
            current + int(increment)
            for current, increment in zip(candidate_counts_above_floor, counts)
        ]
        local_limit = min(candidate_limit, int(source_tensor.shape[0]))
        local_scores, local_indexes = torch_module.topk(
            similarities, k=local_limit, dim=1, sorted=False
        )
        local_scores = torch_module.where(
            local_scores >= hard_min,
            local_scores,
            torch_module.full_like(local_scores, float("-inf")),
        )
        local_parts = torch_module.full_like(local_indexes, part_index)
        local_vectors = source_tensor[local_indexes]
        if best_scores is None:
            best_scores = local_scores
            best_part_indexes = local_parts
            best_source_indexes = local_indexes
            best_vectors = local_vectors
        else:
            assert best_part_indexes is not None
            assert best_source_indexes is not None
            assert best_vectors is not None
            combined_scores = torch_module.cat((best_scores, local_scores), dim=1)
            combined_parts = torch_module.cat(
                (best_part_indexes, local_parts), dim=1
            )
            combined_indexes = torch_module.cat(
                (best_source_indexes, local_indexes), dim=1
            )
            combined_vectors = torch_module.cat(
                (best_vectors, local_vectors), dim=1
            )
            keep = min(candidate_limit, int(combined_scores.shape[1]))
            best_scores, positions = torch_module.topk(
                combined_scores, k=keep, dim=1, sorted=False
            )
            best_part_indexes = torch_module.gather(
                combined_parts, 1, positions
            )
            best_source_indexes = torch_module.gather(
                combined_indexes, 1, positions
            )
            vector_positions = positions.unsqueeze(-1).expand(
                -1, -1, int(source_tensor.shape[1])
            )
            best_vectors = torch_module.gather(
                combined_vectors, 1, vector_positions
            )
            del (
                combined_scores,
                combined_parts,
                combined_indexes,
                combined_vectors,
                positions,
                vector_positions,
            )
        del (
            similarities,
            source_tensor,
            local_scores,
            local_indexes,
            local_parts,
            local_vectors,
        )

    radii = [float(min_similarity)]
    while radii[-1] - similarity_step > hard_min:
        radii.append(float(radii[-1] - similarity_step))
    if radii[-1] > hard_min:
        radii.append(float(hard_min))

    if device == "cpu" or best_scores is None:
        candidates = [sorted(heap, reverse=True) for heap in heaps]
    else:
        assert torch_module is not None
        assert best_part_indexes is not None
        assert best_source_indexes is not None
        assert best_vectors is not None
        ordered_scores, order = torch_module.topk(
            best_scores,
            k=int(best_scores.shape[1]),
            dim=1,
            sorted=True,
        )
        ordered_parts = torch_module.gather(best_part_indexes, 1, order)
        ordered_indexes = torch_module.gather(best_source_indexes, 1, order)
        vector_order = order.unsqueeze(-1).expand(
            -1, -1, int(best_vectors.shape[2])
        )
        ordered_vectors = torch_module.gather(best_vectors, 1, vector_order)
        scores_array = ordered_scores.cpu().numpy()
        parts_array = ordered_parts.cpu().numpy()
        indexes_array = ordered_indexes.cpu().numpy()
        vectors_array = ordered_vectors.cpu().numpy()
        candidates = [[] for _ in range(len(queries))]
        references_by_part: dict[int, list[tuple[int, int, int]]] = {}
        for query_index in range(len(queries)):
            for candidate_index, score in enumerate(scores_array[query_index]):
                if not np.isfinite(score):
                    continue
                part_index = int(parts_array[query_index, candidate_index])
                source_index = int(indexes_array[query_index, candidate_index])
                references_by_part.setdefault(part_index, []).append(
                    (query_index, candidate_index, source_index)
                )
        import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

        for part_index, references in references_by_part.items():
            part = resolved_parts[part_index]
            metadata_columns = [
                name
                for name in pq.ParquetFile(part).schema_arrow.names
                if name != embedding_column
            ]
            source = pd.read_parquet(part, columns=metadata_columns, pre_buffer=False)
            for name, value in (locator_defaults or {}).items():
                if name not in source:
                    source[name] = value
            source[id_column] = source[id_column].astype(str)
            source = source[~source[id_column].isin(excluded)]
            for query_index, candidate_index, source_index in references:
                row = source.iloc[source_index]
                source_id = str(row[id_column])
                metadata = {
                    str(column): row[column]
                    for column in source.columns
                    if column != id_column
                }
                candidates[query_index].append(
                    (
                        float(scores_array[query_index, candidate_index]),
                        source_id,
                        str(part),
                        source_index,
                        metadata,
                        vectors_array[query_index, candidate_index].tobytes(),
                    )
                )
        candidates = [sorted(items, reverse=True) for items in candidates]
    pointers = [0] * len(queries)
    selected_counts = [0] * len(queries)
    query_stats = [
        {
            "query_id": str(queries.iloc[index][id_column]),
            "candidate_count": len(candidates[index]),
            "candidate_count_above_hard_floor": candidate_counts_above_floor[index],
            "candidate_truncated": (
                candidate_counts_above_floor[index] > len(candidates[index])
            ),
            "selected_count": 0,
            "rejected_query_duplicate": 0,
            "rejected_selected_duplicate": 0,
            "rejected_reused_source": 0,
            "lowest_selected_similarity": None,
            "final_radius": None,
        }
        for index in range(len(queries))
    ]
    dimension = int(query_vectors.shape[1])
    selected_shape = (top_k * max(1, len(queries)), dimension)
    if device == "cpu":
        selected_vectors = np.empty(selected_shape, dtype=np.float32)
    else:
        assert torch_module is not None
        selected_vectors = torch_module.empty(
            selected_shape, dtype=torch_module.float32, device=device
        )
    selected_vector_count = 0
    selected_source_ids: set[str] = set()
    records: list[dict] = []

    attempted_radii: list[float] = []
    for radius in radii:
        attempted_radii.append(radius)
        while True:
            made_progress = False
            for query_index, query_candidates in enumerate(candidates):
                if selected_counts[query_index] >= top_k:
                    continue
                while pointers[query_index] < len(query_candidates):
                    candidate = query_candidates[pointers[query_index]]
                    similarity, source_id, source_part, _, metadata, vector_bytes = candidate
                    if similarity < radius:
                        break
                    pointers[query_index] += 1
                    stats = query_stats[query_index]
                    if similarity >= duplicate_similarity:
                        stats["rejected_query_duplicate"] += 1
                        continue
                    if source_id in selected_source_ids:
                        stats["rejected_reused_source"] += 1
                        continue
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    max_selected_similarity = None
                    device_vector = None
                    if device != "cpu":
                        assert torch_module is not None
                        device_vector = torch_module.from_numpy(vector.copy()).to(device)
                    if selected_vector_count:
                        if device == "cpu":
                            max_selected_similarity = float(
                                np.max(selected_vectors[:selected_vector_count] @ vector)
                            )
                        else:
                            assert torch_module is not None and device_vector is not None
                            max_selected_similarity = float(
                                torch_module.max(
                                    selected_vectors[:selected_vector_count] @ device_vector
                                ).item()
                            )
                        if max_selected_similarity >= duplicate_similarity:
                            stats["rejected_selected_duplicate"] += 1
                            continue
                    rank = selected_counts[query_index] + 1
                    records.append(
                        {
                            "query_id": stats["query_id"],
                            "sample_id": source_id,
                            "rank": rank,
                            "cosine_similarity": similarity,
                            "accepted_at_similarity_threshold": radius,
                            "max_similarity_to_selected": max_selected_similarity,
                            "source_part": source_part,
                            "search_proof": "exact_all_declared_shards",
                            "search_device": device,
                            **{
                                name: value for name, value in metadata.items()
                                if name not in SEARCH_RESULT_COLUMNS
                            },
                        }
                    )
                    if device == "cpu":
                        selected_vectors[selected_vector_count] = vector
                    else:
                        assert device_vector is not None
                        selected_vectors[selected_vector_count] = device_vector
                    selected_vector_count += 1
                    selected_source_ids.add(source_id)
                    selected_counts[query_index] += 1
                    stats["selected_count"] = selected_counts[query_index]
                    stats["lowest_selected_similarity"] = similarity
                    stats["final_radius"] = radius
                    made_progress = True
                    break
            if not made_progress:
                break
        if all(count >= top_k for count in selected_counts):
            break

    columns = [
        "query_id",
        "sample_id",
        "rank",
        "cosine_similarity",
        "accepted_at_similarity_threshold",
        "max_similarity_to_selected",
        "source_part",
        "search_proof",
        "search_device",
    ]
    result = pd.DataFrame.from_records(records)
    if result.empty:
        result = pd.DataFrame(columns=columns)
    else:
        result = result.reset_index(drop=True)
    result.attrs["adaptive_radius"] = {
        "initial_min_similarity": float(min_similarity),
        "hard_min_similarity": float(hard_min),
        "similarity_step": float(similarity_step),
        "duplicate_similarity": float(duplicate_similarity),
        "candidate_multiplier": int(candidate_multiplier),
        "candidate_limit_per_query": int(candidate_limit),
        "radii_attempted": attempted_radii,
        "query_stats": query_stats,
        "candidate_truncated": any(
            stats["candidate_truncated"] for stats in query_stats
        ),
        "underfill_exhaustion_proven": all(
            stats["selected_count"] >= top_k or not stats["candidate_truncated"]
            for stats in query_stats
        ),
    }
    return result
