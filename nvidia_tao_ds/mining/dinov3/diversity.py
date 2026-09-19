# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deterministic radius and diversity selection for DINOv3 mining."""

from typing import Any

import numpy as np
import pandas as pd

from .contracts import cosine_at_or_above


def _radii(initial: float, hard_minimum: float, step: float) -> list[float]:
    values = [float(initial)]
    while values[-1] - step > hard_minimum:
        values.append(float(values[-1] - step))
    if values[-1] > hard_minimum:
        values.append(float(hard_minimum))
    return values


def select_diverse(
    *,
    queries: pd.DataFrame,
    embedding_dim: int,
    candidates: list[list[tuple[float, int, bytes]]],
    query_stats: list[dict[str, Any]],
    top_k: int,
    min_similarity: float,
    hard_min_similarity: float,
    similarity_step: float,
    duplicate_similarity: float,
    device: str,
    vector_loader=None,
    source_id_getter=None,
) -> pd.DataFrame:
    """Apply one round-robin radius/diversity policy to any candidate producer."""
    torch_module = None
    if device != "cpu":
        import torch  # pylint: disable=import-outside-toplevel

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(f"CUDA deduplication is unavailable: {device}")
        torch_module = torch
    dimension = embedding_dim
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
                    if not cosine_at_or_above(similarity, radius):
                        break
                    pointers[query_index] += 1
                    stats = query_stats[query_index]
                    if cosine_at_or_above(similarity, duplicate_similarity):
                        stats["rejected_query_duplicate"] += 1
                        continue
                    source_identity = source_id_getter(row_id) if source_id_getter else row_id
                    if source_identity in selected_row_ids:
                        stats["rejected_reused_source"] += 1
                        continue
                    vector = (vector_loader(vector_bytes) if vector_loader else
                              np.frombuffer(vector_bytes, dtype=np.float32))
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
                    selected_row_ids.add(source_identity)
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
