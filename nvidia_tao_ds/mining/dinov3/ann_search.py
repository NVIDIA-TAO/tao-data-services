# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ANN candidate retrieval with exact float32 cosine reranking."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .ann_index import load_ann_index
from .contracts import cosine_at_or_above, file_posix_identity, vector_matrix
from .dense_store import load_dense_store
from .search import source_metadata_output_names


CUVS_MG_MAX_CANDIDATES = 1024


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Refinement search requires local file URIs: {uri}")
    return Path(unquote(parsed.path)).resolve()


def _radii(initial: float, hard_minimum: float, step: float) -> list[float]:
    values = [float(initial)]
    while values[-1] - step > hard_minimum:
        values.append(float(values[-1] - step))
    if values[-1] > hard_minimum:
        values.append(float(hard_minimum))
    return values


def _validate_parameters(
    *,
    top_k: int,
    min_similarity: float,
    hard_min_similarity: float,
    similarity_step: float,
    duplicate_similarity: float,
    candidate_multiplier: int,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not -1.0 <= hard_min_similarity <= min_similarity <= 1.0:
        raise ValueError(
            "similarity thresholds must satisfy -1 <= hard minimum <= minimum <= 1"
        )
    if similarity_step <= 0:
        raise ValueError("similarity_step must be positive")
    if not min_similarity < duplicate_similarity <= 1.0:
        raise ValueError(
            "duplicate_similarity must be greater than min_similarity and at most 1"
        )
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")


def _candidate_lists(
    *,
    query_vectors: np.ndarray,
    candidate_row_ids: np.ndarray,
    dense_vectors: np.memmap,
    hard_min_similarity: float,
    candidate_limit: int,
    excluded_row_ids: set[int],
) -> tuple[list[list[tuple[float, int, bytes]]], list[dict[str, Any]]]:
    if candidate_row_ids.ndim != 2 or candidate_row_ids.shape[0] != len(query_vectors):
        raise ValueError("ANN candidate IDs must have shape [query_count, candidate_count]")
    row_count = int(dense_vectors.shape[0])
    candidates = []
    stats = []
    for query_index, raw_ids in enumerate(candidate_row_ids):
        valid = []
        observed = set()
        for raw_id in raw_ids:
            row_id = int(raw_id)
            if row_id < 0 or row_id >= row_count:
                continue
            if row_id in excluded_row_ids or row_id in observed:
                continue
            observed.add(row_id)
            valid.append(row_id)
        ids = np.asarray(valid, dtype=np.int64)
        if len(ids):
            vectors = np.asarray(dense_vectors[ids], dtype=np.float32)
            scores = vectors @ query_vectors[query_index]
            above = np.flatnonzero(cosine_at_or_above(scores, hard_min_similarity))
            ordered = above[
                np.lexsort((ids[above], -scores[above]))[:candidate_limit]
            ]
            query_candidates = [
                (float(scores[index]), int(ids[index]), vectors[index].tobytes())
                for index in ordered
            ]
            above_floor = int(len(above))
        else:
            query_candidates = []
            above_floor = 0
        candidates.append(query_candidates)
        stats.append(
            {
                "candidate_count": len(query_candidates),
                "candidate_count_above_hard_floor": above_floor,
                "ann_unique_candidate_count": len(ids),
                "candidate_truncated": True,
            }
        )
    return candidates, stats


def _select_diverse(
    *,
    queries: pd.DataFrame,
    query_vectors: np.ndarray,
    candidates: list[list[tuple[float, int, bytes]]],
    query_stats: list[dict[str, Any]],
    top_k: int,
    min_similarity: float,
    hard_min_similarity: float,
    similarity_step: float,
    duplicate_similarity: float,
    device: str,
) -> pd.DataFrame:
    torch_module = None
    if device != "cpu":
        import torch  # pylint: disable=import-outside-toplevel

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(f"CUDA deduplication is unavailable: {device}")
        torch_module = torch
    dimension = int(query_vectors.shape[1])
    selected_shape = (top_k * max(1, len(queries)), dimension)
    if torch_module is None:
        selected_vectors: Any = np.empty(selected_shape, dtype=np.float32)
    else:
        selected_vectors = torch_module.empty(
            selected_shape, dtype=torch_module.float32, device=device
        )

    pointers = [0] * len(queries)
    selected_counts = [0] * len(queries)
    selected_row_ids: set[int] = set()
    selected_vector_count = 0
    records = []
    for index, stats in enumerate(query_stats):
        stats.update(
            {
                "query_id": str(queries.iloc[index]["sample_id"]),
                "selected_count": 0,
                "rejected_query_duplicate": 0,
                "rejected_selected_duplicate": 0,
                "rejected_reused_source": 0,
                "lowest_selected_similarity": None,
                "final_radius": None,
            }
        )

    attempted_radii = []
    for radius in _radii(min_similarity, hard_min_similarity, similarity_step):
        attempted_radii.append(radius)
        while True:
            made_progress = False
            for query_index, query_candidates in enumerate(candidates):
                if selected_counts[query_index] >= top_k:
                    continue
                while pointers[query_index] < len(query_candidates):
                    similarity, row_id, vector_bytes = query_candidates[
                        pointers[query_index]
                    ]
                    if similarity < radius:
                        break
                    pointers[query_index] += 1
                    stats = query_stats[query_index]
                    if cosine_at_or_above(similarity, duplicate_similarity):
                        stats["rejected_query_duplicate"] += 1
                        continue
                    if row_id in selected_row_ids:
                        stats["rejected_reused_source"] += 1
                        continue
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    device_vector = None
                    max_selected_similarity = None
                    if torch_module is not None:
                        device_vector = torch_module.from_numpy(vector.copy()).to(device)
                    if selected_vector_count:
                        if torch_module is None:
                            max_selected_similarity = float(
                                np.max(selected_vectors[:selected_vector_count] @ vector)
                            )
                        else:
                            max_selected_similarity = float(
                                torch_module.max(
                                    selected_vectors[:selected_vector_count]
                                    @ device_vector
                                ).item()
                            )
                        if cosine_at_or_above(
                            max_selected_similarity, duplicate_similarity
                        ):
                            stats["rejected_selected_duplicate"] += 1
                            continue
                    records.append(
                        {
                            "query_id": stats["query_id"],
                            "sample_id": str(row_id),
                            "source_row_id": row_id,
                            "rank": selected_counts[query_index] + 1,
                            "cosine_similarity": similarity,
                            "accepted_at_similarity_threshold": radius,
                            "max_similarity_to_selected": max_selected_similarity,
                            "search_proof": "ann_exact_float32_rerank",
                            "search_device": device,
                        }
                    )
                    if torch_module is None:
                        selected_vectors[selected_vector_count] = vector
                    else:
                        selected_vectors[selected_vector_count] = device_vector
                    selected_vector_count += 1
                    selected_row_ids.add(row_id)
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

    result = pd.DataFrame.from_records(records)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "query_id",
                "sample_id",
                "source_row_id",
                "rank",
                "cosine_similarity",
                "accepted_at_similarity_threshold",
                "max_similarity_to_selected",
                "search_proof",
                "search_device",
            ]
        )
    result.attrs["adaptive_radius"] = {
        "initial_min_similarity": float(min_similarity),
        "hard_min_similarity": float(hard_min_similarity),
        "similarity_step": float(similarity_step),
        "duplicate_similarity": float(duplicate_similarity),
        "radii_attempted": attempted_radii,
        "query_stats": query_stats,
        "candidate_truncated": True,
        "underfill_exhaustion_proven": False,
    }
    return result


def _hydrate_locators(frame: pd.DataFrame, dense: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    source_root = _local_path(dense["source_root_uri"])
    stops = [int(shard["row_stop"]) for shard in dense["shards"]]
    requested: dict[int, list[tuple[int, int]]] = {}
    for output_index, row_id in enumerate(frame["source_row_id"].astype(int)):
        shard_index = bisect_right(stops, row_id)
        shard = dense["shards"][shard_index]
        requested.setdefault(shard_index, []).append(
            (output_index, row_id - int(shard["row_start"]))
        )

    hydrated: dict[int, dict[str, Any]] = {}
    for shard_index, references in requested.items():
        shard = dense["shards"][shard_index]
        part = source_root / shard["relative_path"]
        current_identity = file_posix_identity(part)
        expected_identity = shard.get("source_posix_identity")
        if expected_identity is not None and current_identity != expected_identity:
            raise RuntimeError(f"Source locator shard changed after commit: {part}")
        if current_identity["bytes"] != int(shard["source_bytes"]):
            raise RuntimeError(f"Source locator shard size changed: {part}")
        parquet = pq.ParquetFile(part)
        metadata_columns = [
            name
            for name in parquet.schema_arrow.names
            if name != dense["embedding_column"]
        ]
        metadata_names = source_metadata_output_names(
            list(dict.fromkeys(
                metadata_columns + list(dense.get("locator_defaults", {}).keys())
            )),
            excluded={dense["id_column"]},
        )
        starts = []
        current = 0
        for row_group in range(parquet.num_row_groups):
            starts.append(current)
            current += parquet.metadata.row_group(row_group).num_rows
        by_group: dict[int, list[tuple[int, int]]] = {}
        for output_index, local_row in references:
            row_group = bisect_right(starts, local_row) - 1
            by_group.setdefault(row_group, []).append(
                (output_index, local_row - starts[row_group])
            )
        for row_group, group_references in by_group.items():
            table = parquet.read_row_group(row_group, columns=metadata_columns)
            group = table.to_pandas()
            for output_index, local_row in group_references:
                row = group.iloc[local_row]
                raw_metadata = {
                    **dense.get("locator_defaults", {}),
                    **{str(column): row[column] for column in group.columns},
                }
                source_id = raw_metadata.pop(dense["id_column"], None)
                metadata = {
                    metadata_names[name]: value
                    for name, value in raw_metadata.items()
                }
                metadata["sample_id"] = source_id
                metadata["source_part"] = str(part)
                hydrated[output_index] = metadata

    result = frame.copy()
    for output_index in range(len(result)):
        metadata = hydrated[output_index]
        if metadata.get("sample_id") is None:
            raise RuntimeError(
                f"Source locator row has no {dense['id_column']!r} identity column"
            )
        result.at[output_index, "sample_id"] = str(metadata.pop("sample_id"))
        for name, value in metadata.items():
            if name not in result:
                # Preserve list/struct annotations as scalar metadata cells.
                result[name] = pd.Series(index=result.index, dtype="object")
            result.at[output_index, name] = value
    return result


def exact_rerank_ann_candidates(
    *,
    query_path: str | Path,
    dense_store_manifest: str | Path,
    candidate_row_ids: np.ndarray,
    top_k: int,
    min_similarity: float,
    hard_min_similarity: float | None = None,
    similarity_step: float = 0.02,
    duplicate_similarity: float = 1.0,
    candidate_multiplier: int = 10,
    excluded_row_ids: set[int] | None = None,
    device: str = "cpu",
    hydrate_locators: bool = True,
) -> pd.DataFrame:
    """Exactly rerank ANN candidates and apply radius and diversity policy."""
    hard_minimum = min_similarity if hard_min_similarity is None else hard_min_similarity
    _validate_parameters(
        top_k=top_k,
        min_similarity=min_similarity,
        hard_min_similarity=hard_minimum,
        similarity_step=similarity_step,
        duplicate_similarity=duplicate_similarity,
        candidate_multiplier=candidate_multiplier,
    )
    dense, _ = load_dense_store(dense_store_manifest)
    queries = pd.read_parquet(query_path, pre_buffer=False)
    required = {"sample_id", "embedding"}
    if missing := required.difference(queries.columns):
        raise ValueError(f"Query table is missing columns: {sorted(missing)}")
    query_vectors = vector_matrix(queries["embedding"], label="Query embedding")
    if int(dense["embedding_dim"]) != int(query_vectors.shape[1]):
        raise ValueError("Query and dense-store embedding dimensions differ")
    vector_path = _local_path(dense["vector_uri"])
    dense_vectors = np.memmap(
        vector_path,
        mode="r",
        dtype=np.dtype(dense["dtype"]),
        shape=(int(dense["row_count"]), int(dense["embedding_dim"])),
    )
    candidates, stats = _candidate_lists(
        query_vectors=query_vectors,
        candidate_row_ids=np.asarray(candidate_row_ids),
        dense_vectors=dense_vectors,
        hard_min_similarity=hard_minimum,
        candidate_limit=top_k * candidate_multiplier,
        excluded_row_ids=excluded_row_ids or set(),
    )
    result = _select_diverse(
        queries=queries,
        query_vectors=query_vectors,
        candidates=candidates,
        query_stats=stats,
        top_k=top_k,
        min_similarity=min_similarity,
        hard_min_similarity=hard_minimum,
        similarity_step=similarity_step,
        duplicate_similarity=duplicate_similarity,
        device=device,
    )
    if hydrate_locators:
        result = _hydrate_locators(result, dense)
    result.attrs["adaptive_radius"]["candidate_multiplier"] = candidate_multiplier
    result.attrs["adaptive_radius"]["candidate_limit_per_query"] = (
        top_k * candidate_multiplier
    )
    return result


def retrieve_ann_candidates(
    *,
    query_path: str | Path,
    ann_index_manifest: str | Path,
    n_probes: int,
    ann_candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Retrieve candidates without importing the exact-rerank runtime."""
    if n_probes <= 0 or ann_candidate_count <= 0:
        raise ValueError("n_probes and ann_candidate_count must be positive")
    if ann_candidate_count > CUVS_MG_MAX_CANDIDATES:
        raise ValueError(
            "cuVS multi-GPU IVF-PQ supports at most "
            f"{CUVS_MG_MAX_CANDIDATES} merged candidates per query"
        )
    ann, ann_artifact = load_ann_index(ann_index_manifest)
    queries = pd.read_parquet(query_path, pre_buffer=False)
    query_vectors = vector_matrix(queries["embedding"], label="Query embedding")
    if query_vectors.shape[1] != int(ann["embedding_dim"]):
        raise ValueError("Query and ANN index embedding dimensions differ")
    try:
        from cuvs.neighbors.mg import ivf_pq  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("cuVS is required for ANN retrieval") from error
    index_path = _local_path(ann["index"]["uri"])
    index = ivf_pq.load(str(index_path))
    distances, candidate_ids = ivf_pq.search(
        ivf_pq.SearchParams(
            n_probes=n_probes,
            lut_dtype=np.float32,
            internal_distance_dtype=np.float32,
            coarse_search_dtype=np.float32,
        ),
        index,
        np.ascontiguousarray(query_vectors),
        k=ann_candidate_count,
    )
    metadata = {
        "backend": ann["backend"],
        "backend_version": ann["backend_version"],
        "index_artifact_id": ann_artifact["artifact_id"],
        "n_probes": n_probes,
        "ann_candidate_count": ann_candidate_count,
    }
    return distances, candidate_ids, metadata


def ann_exact_rerank_search(
    *,
    query_path: str | Path,
    dense_store_manifest: str | Path,
    ann_index_manifest: str | Path,
    top_k: int,
    min_similarity: float,
    n_probes: int,
    ann_candidate_count: int,
    hard_min_similarity: float | None = None,
    similarity_step: float = 0.02,
    duplicate_similarity: float = 1.0,
    candidate_multiplier: int = 10,
    excluded_row_ids: set[int] | None = None,
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Retrieve cuVS candidates and make exact float32 mining decisions."""
    if n_probes <= 0 or ann_candidate_count <= 0:
        raise ValueError("n_probes and ann_candidate_count must be positive")
    _, dense_artifact = load_dense_store(dense_store_manifest)
    ann, _ = load_ann_index(ann_index_manifest)
    if ann["dense_store_artifact_id"] != dense_artifact["artifact_id"]:
        raise ValueError("ANN index and dense vector store do not match")
    _, candidate_ids, metadata = retrieve_ann_candidates(
        query_path=query_path,
        ann_index_manifest=ann_index_manifest,
        n_probes=n_probes,
        ann_candidate_count=ann_candidate_count,
    )
    result = exact_rerank_ann_candidates(
        query_path=query_path,
        dense_store_manifest=dense_store_manifest,
        candidate_row_ids=candidate_ids,
        top_k=top_k,
        min_similarity=min_similarity,
        hard_min_similarity=hard_min_similarity,
        similarity_step=similarity_step,
        duplicate_similarity=duplicate_similarity,
        candidate_multiplier=candidate_multiplier,
        excluded_row_ids=excluded_row_ids,
        device=device,
    )
    result.attrs["ann"] = metadata
    return result
