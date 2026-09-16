# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first sharded cosine search for refinement mining."""

from __future__ import annotations

import heapq
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import cosine_at_or_above, vector_matrix


SEARCH_RESULT_COLUMNS = frozenset({
    "query_id", "sample_id", "source_row_id", "rank", "cosine_similarity",
    "accepted_at_similarity_threshold", "max_similarity_to_selected",
    "source_part", "search_proof", "search_device",
    "source_store_artifact_id", "dense_store_artifact_id",
    "source_inventory_digest",
})


def source_metadata_output_names(
    columns: list[str] | tuple[str, ...],
    *,
    excluded: set[str] | frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Map source metadata names without permitting namespace collisions."""
    mapping: dict[str, str] = {}
    occupied = set(SEARCH_RESULT_COLUMNS)
    for column in columns:
        name = str(column)
        if name in excluded:
            continue
        output = f"source_{name}" if name in SEARCH_RESULT_COLUMNS else name
        if output in occupied:
            raise ValueError(
                "Source metadata columns collide after result namespacing: "
                f"{name!r} -> {output!r}"
            )
        occupied.add(output)
        mapping[name] = output
    return mapping


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
    query_block_rows: int = 1024,
    source_block_rows: int = 8192,
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
    if query_block_rows <= 0 or source_block_rows <= 0:
        raise ValueError("query_block_rows and source_block_rows must be positive")
    query_parquet = pq.ParquetFile(query_path)
    query_columns = query_parquet.schema_arrow.names
    required = {id_column, embedding_column}
    if missing := required.difference(query_columns):
        raise ValueError(f"Query table is missing columns: {sorted(missing)}")
    queries = pd.read_parquet(
        query_path,
        columns=[name for name in query_columns if name != embedding_column],
        pre_buffer=False,
    )
    if queries.empty:
        raise ValueError("Query table must contain at least one row")
    first_query_batch = next(
        query_parquet.iter_batches(batch_size=1, columns=[embedding_column])
    )
    embedding_dim = int(vector_matrix(
        first_query_batch.column(0).to_pylist(), label="Query embedding"
    ).shape[1])
    import torch  # pylint: disable=import-outside-toplevel

    compute_device = torch.device(device)
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or a CUDA device")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA search requested but unavailable: {device}")
    candidate_limit = top_k * candidate_multiplier
    heaps: list[list[tuple]] = [[] for _ in range(len(queries))]
    candidate_counts_above_floor = [0] * len(queries)
    excluded = excluded_ids or set()
    resolved_parts = [Path(value) for value in source_parts]
    global_part_offset = 0
    for part_index, part in enumerate(resolved_parts):
        source_parquet = pq.ParquetFile(part)
        missing = {id_column, embedding_column}.difference(
            source_parquet.schema_arrow.names
        )
        if missing:
            raise ValueError(f"Source shard {part} is missing columns: {sorted(missing)}")
        source_offset = 0
        for source_batch in source_parquet.iter_batches(
            batch_size=source_block_rows, columns=[id_column, embedding_column]
        ):
            source = source_batch.to_pandas()
            source_ids_all = source[id_column].astype(str).to_numpy()
            keep_mask = ~pd.Series(source_ids_all).isin(excluded).to_numpy()
            original_indexes = np.arange(
                source_offset, source_offset + len(source), dtype=np.int64
            )[keep_mask]
            source_ids = source_ids_all[keep_mask]
            if len(source_ids):
                source_vectors = vector_matrix(
                    source.loc[keep_mask, embedding_column],
                    label=f"Source shard {part} embedding",
                )
                source_tensor = torch.from_numpy(source_vectors).to(compute_device)
                query_offset = 0
                for query_batch in query_parquet.iter_batches(
                    batch_size=query_block_rows, columns=[embedding_column]
                ):
                    query_vectors = vector_matrix(
                        query_batch.column(0).to_pylist(), label="Query embedding"
                    )
                    if query_vectors.shape[1] != embedding_dim or source_vectors.shape[1] != embedding_dim:
                        raise ValueError("Query and source embedding dimensions differ")
                    query_tensor = torch.from_numpy(query_vectors).to(compute_device)
                    similarities = query_tensor @ source_tensor.T
                    score_values = similarities.cpu().numpy()
                    for relative_query, scores in enumerate(score_values):
                        eligible = np.flatnonzero(
                            cosine_at_or_above(scores, hard_min)
                        )
                        query_index = query_offset + relative_query
                        candidate_counts_above_floor[query_index] += len(eligible)
                        heap = heaps[query_index]
                        for position in eligible:
                            score = float(scores[position])
                            local_index = int(original_indexes[position])
                            global_row_id = global_part_offset + local_index
                            item = (
                                score,
                                -global_row_id,
                                part_index,
                                local_index,
                                str(source_ids[position]),
                                global_row_id,
                            )
                            if len(heap) < candidate_limit:
                                heapq.heappush(heap, item)
                            elif item[:2] > heap[0][:2]:
                                heapq.heapreplace(heap, item)
                    query_offset += len(query_vectors)
                    del query_tensor, similarities
                del source_tensor
            source_offset += len(source)
        global_part_offset += int(source_parquet.metadata.num_rows)

    radii = [float(min_similarity)]
    while radii[-1] - similarity_step > hard_min:
        radii.append(float(radii[-1] - similarity_step))
    if radii[-1] > hard_min:
        radii.append(float(hard_min))

    candidates = [[] for _ in range(len(queries))]
    references_by_part: dict[int, set[int]] = {}
    for heap in heaps:
        for reference in heap:
            references_by_part.setdefault(reference[2], set()).add(reference[3])

    # Hydrate retained candidates in bounded record batches. Candidate vectors
    # are spooled once to disk even when referenced by several queries.
    candidate_spool = tempfile.TemporaryFile()
    hydrated_rows: dict[
        tuple[int, int], tuple[str, dict[str, object], int]
    ] = {}
    metadata_names_by_part: dict[int, dict[str, str]] = {}
    for part_index, local_indexes in references_by_part.items():
        part = resolved_parts[part_index]
        parquet = pq.ParquetFile(part)
        columns = list(parquet.schema_arrow.names)
        metadata_names = source_metadata_output_names(
            list(dict.fromkeys(columns + list((locator_defaults or {}).keys()))),
            excluded={id_column, embedding_column},
        )
        metadata_names_by_part[part_index] = metadata_names
        indexes = sorted(local_indexes)
        index_pointer = 0
        batch_start = 0
        for batch in parquet.iter_batches(
            batch_size=source_block_rows, columns=columns
        ):
            batch_stop = batch_start + len(batch)
            if index_pointer >= len(indexes):
                break
            if indexes[index_pointer] >= batch_stop:
                batch_start = batch_stop
                continue
            frame = batch.to_pandas()
            while (
                index_pointer < len(indexes) and
                indexes[index_pointer] < batch_stop
            ):
                local_index = indexes[index_pointer]
                if local_index < batch_start:
                    raise RuntimeError("Candidate row hydration order is invalid")
                row = frame.iloc[local_index - batch_start]
                metadata: dict[str, object] = dict(locator_defaults or {})
                metadata.update({
                    name: row[name]
                    for name in metadata_names
                    if name in frame.columns
                })
                vector = vector_matrix(
                    [row[embedding_column]],
                    label=f"Source shard {part} embedding",
                )[0]
                vector_offset = candidate_spool.tell()
                candidate_spool.write(vector.tobytes())
                hydrated_rows[(part_index, local_index)] = (
                    str(row[id_column]),
                    metadata,
                    vector_offset,
                )
                index_pointer += 1
            batch_start = batch_stop
        if index_pointer != len(indexes):
            raise RuntimeError(f"Candidate row is missing from source shard: {part}")

    for query_index, heap in enumerate(heaps):
        for reference in heap:
            (
                score, _, part_index, local_index, source_id, global_row_id,
            ) = reference
            hydrated_id, metadata, _ = hydrated_rows[(part_index, local_index)]
            if hydrated_id != source_id:
                raise RuntimeError("Source row identity changed during exact search")
            candidates[query_index].append((
                score, source_id, str(resolved_parts[part_index]), local_index,
                global_row_id, metadata, (part_index, local_index),
            ))
    candidates = [
        sorted(items, key=lambda item: (-item[0], item[4], item[1]))
        for items in candidates
    ]
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
    dimension = embedding_dim
    selected_shape = (top_k * max(1, len(queries)), dimension)
    selected_spool = tempfile.TemporaryFile()
    selected_vectors = np.memmap(
        selected_spool, mode="w+", dtype=np.float32, shape=selected_shape
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
                    (
                        similarity,
                        source_id,
                        source_part,
                        _,
                        source_row_id,
                        metadata,
                        vector_key,
                    ) = candidate
                    if not cosine_at_or_above(similarity, radius):
                        break
                    pointers[query_index] += 1
                    stats = query_stats[query_index]
                    if cosine_at_or_above(similarity, duplicate_similarity):
                        stats["rejected_query_duplicate"] += 1
                        continue
                    if source_id in selected_source_ids:
                        stats["rejected_reused_source"] += 1
                        continue
                    vector_offset = hydrated_rows[vector_key][2]
                    candidate_spool.seek(vector_offset)
                    vector = np.frombuffer(
                        candidate_spool.read(dimension * np.dtype(np.float32).itemsize),
                        dtype=np.float32,
                    )
                    if vector.shape != (dimension,):
                        raise RuntimeError("Candidate vector spool is truncated")
                    max_selected_similarity = None
                    if selected_vector_count:
                        block_maxima = []
                        for block_start in range(
                            0, selected_vector_count, source_block_rows
                        ):
                            block_stop = min(
                                selected_vector_count,
                                block_start + source_block_rows,
                            )
                            block = np.asarray(
                                selected_vectors[block_start:block_stop]
                            )
                            if compute_device.type == "cpu":
                                block_maxima.append(float(np.max(block @ vector)))
                            else:
                                block_tensor = torch.from_numpy(block).to(compute_device)
                                vector_tensor = torch.from_numpy(vector.copy()).to(
                                    compute_device
                                )
                                block_maxima.append(float(
                                    torch.max(block_tensor @ vector_tensor).item()
                                ))
                        max_selected_similarity = max(block_maxima)
                        if cosine_at_or_above(
                            max_selected_similarity, duplicate_similarity
                        ):
                            stats["rejected_selected_duplicate"] += 1
                            continue
                    rank = selected_counts[query_index] + 1
                    records.append(
                        {
                            "query_id": stats["query_id"],
                            "sample_id": source_id,
                            "source_row_id": source_row_id,
                            "rank": rank,
                            "cosine_similarity": similarity,
                            "accepted_at_similarity_threshold": radius,
                            "max_similarity_to_selected": max_selected_similarity,
                            "source_part": source_part,
                            "search_proof": "exact_all_declared_shards",
                            "search_device": device,
                            **{
                                metadata_names_by_part[vector_key[0]][name]: value
                                for name, value in metadata.items()
                            },
                        }
                    )
                    selected_vectors[selected_vector_count] = vector
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
        "source_row_id",
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
    selected_vectors.flush()
    del selected_vectors
    selected_spool.close()
    candidate_spool.close()
    return result
