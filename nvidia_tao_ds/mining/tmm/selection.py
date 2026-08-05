# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assignment policy helpers for the mining commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import cuml.neighbors
import numpy as np
import pandas as pd

from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.mining.tmm.dataset_selection import identify_images_by_class, identify_rare_images
from nvidia_tao_ds.mining.tmm.vector_preparation import _embeddings_to_lists, l2_normalize

if TYPE_CHECKING:
    import cudf

    from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat


def _merge_capped(retrieved_filepaths: Set[str], result_df: pd.DataFrame, desired_count: int) -> int:
    """Add a pass's matched filepaths to ``retrieved_filepaths``, nearest-first, at the cap.

    Never lets the set exceed ``desired_count``. Files are added in
    ascending best-distance order with filepath as a deterministic tie-break, so
    when the cap truncates, the nearest matches are the ones kept. Returns how many
    new unique files were added.
    """
    source_cols = [c for c in result_df.columns if c.startswith("top") and c.endswith("_source_filepath")]
    if not source_cols:
        return 0
    frames = []
    for source_col in source_cols:
        dist_col = source_col.replace("_source_filepath", "_distance")
        frame = result_df[[source_col, dist_col]].dropna(subset=[source_col])
        frames.append(frame.rename(columns={source_col: "filepath", dist_col: "distance"}))
    best = (
        pd.concat(frames, ignore_index=True)
        .groupby("filepath")["distance"].min()
        .reset_index()
        .sort_values(["distance", "filepath"])
    )
    before = len(retrieved_filepaths)
    for filepath in best["filepath"]:
        if len(retrieved_filepaths) >= desired_count:
            break
        retrieved_filepaths.add(filepath)
    return len(retrieved_filepaths) - before


def split_datasets_by_class(
    df_source: "cudf.DataFrame", df_target: "cudf.DataFrame",
    source_detection_file: Optional[str], target_detection_file: Optional[str],
    rare_class_list: Set[str], source_filepath_column_name: str = "filepath",
    target_filepath_column_name: str = "filepath",
    *, fmt: AnnotationFormat,
) -> Dict[str, Tuple["cudf.DataFrame", "cudf.DataFrame"]]:
    """
    Split source and target into per-class subsets for each rare class, plus a "non_rare" subset.
    Returns {class_name: (class_source_df, class_target_df)}.
    """
    logger.info("Splitting datasets into per-class subsets...")

    class_subsets = {}
    for class_name in rare_class_list:
        source_has_class = identify_images_by_class(
            df_source, source_detection_file, class_name, source_filepath_column_name, fmt=fmt
        )
        target_has_class = identify_images_by_class(
            df_target, target_detection_file, class_name, target_filepath_column_name, fmt=fmt
        )
        class_source = df_source[source_has_class].reset_index(drop=True)
        class_target = df_target[target_has_class].reset_index(drop=True)
        class_subsets[class_name] = (class_source, class_target)
        logger.info(f"  • {class_name}: source={len(class_source)}, target={len(class_target)}")

    # Common = images that contain no rare class
    logger.info("\nCreating non_rare subset (images without any rare classes)...")
    source_is_rare = identify_rare_images(
        df_source, source_detection_file, rare_class_list, source_filepath_column_name, fmt=fmt
    )
    target_is_rare = identify_rare_images(
        df_target, target_detection_file, rare_class_list, target_filepath_column_name, fmt=fmt
    )
    common_source = df_source[~source_is_rare].reset_index(drop=True)
    common_target = df_target[~target_is_rare].reset_index(drop=True)
    class_subsets["non_rare"] = (common_source, common_target)
    logger.info(f"  • non_rare: source={len(common_source)}, target={len(common_target)}")

    return class_subsets


def perform_tmm_retrieval(
    target_embeddings: np.ndarray, source_embeddings: np.ndarray,
    source_df: "cudf.DataFrame", target_df: "cudf.DataFrame",
    topn: int, knn_metric: str,
    source_filepath_column_name: str = "filepath",
    target_filepath_column_name: str = "filepath",
    excluded_filepaths: Optional[Set[str]] = None,
    save_embeddings: bool = False,
    candidate_expansion_factor: int = 5,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Single-pass TMM matching: for each target, find topn unique source files.
    Row-by-row assignment ensures no source file is assigned to multiple targets.

    A target that cannot collect topn unique sources rolls back its tentative
    picks — they are never locked, so they stay available to later targets — and
    is reported for retry.

    Returns (result_df, targets_with_duplicates) where targets_with_duplicates
    lists target indices that couldn't be satisfied (fewer than topn unique matches).
    """
    if len(target_embeddings) == 0 or len(source_embeddings) == 0:
        return pd.DataFrame(), []

    if excluded_filepaths:
        source_filepaths_pd = source_df[source_filepath_column_name].to_pandas()
        mask = ~source_filepaths_pd.isin(excluded_filepaths)
        filtered_indices = mask[mask].index.tolist()
        if not filtered_indices:
            return pd.DataFrame(), []
        source_df_filtered = source_df.iloc[filtered_indices].reset_index(drop=True)
        source_embeddings_filtered = source_embeddings[filtered_indices]
    else:
        source_df_filtered = source_df
        source_embeddings_filtered = source_embeddings

    if len(source_embeddings_filtered) == 0:
        return pd.DataFrame(), []

    target_emb = l2_normalize(target_embeddings)
    source_emb = l2_normalize(source_embeddings_filtered)

    # Search with candidate_expansion_factor*topn candidates so enough remain after
    # uniqueness filtering.
    k_for_search = min(candidate_expansion_factor * topn, len(source_embeddings_filtered))
    if k_for_search == 0:
        return pd.DataFrame(), []

    knn = cuml.neighbors.NearestNeighbors(n_neighbors=k_for_search, metric=knn_metric)
    knn.fit(source_emb)
    distances, indices = knn.kneighbors(target_emb)

    target_filepaths = target_df[target_filepath_column_name].to_pandas()
    source_filepaths = source_df_filtered[source_filepath_column_name].to_pandas()

    indices_host = indices.get() if hasattr(indices, "get") else indices
    distances_host = distances.get() if hasattr(distances, "get") else distances
    target_emb_list = _embeddings_to_lists(target_embeddings) if save_embeddings else None

    # Row-by-row assignment: each source file goes to at most one target.
    assigned_source_files: set = set()
    num_targets = len(target_embeddings)

    output_data: dict = {"target_filepath": []}
    if save_embeddings:
        output_data["target_embedding"] = []
    for i in range(topn):
        output_data[f"top{i + 1}_source_filepath"] = []
        output_data[f"top{i + 1}_distance"] = []
        if save_embeddings:
            output_data[f"top{i + 1}_embed"] = []

    targets_with_duplicates = []
    for target_idx in range(num_targets):
        candidate_indices = indices_host[target_idx, :]
        candidate_distances = distances_host[target_idx, :]

        # Tentatively gather topn unique, unassigned sources without locking them.
        tentative: list = []
        for candidate_idx in range(k_for_search):
            if len(tentative) >= topn:
                break
            source_idx = int(candidate_indices[candidate_idx])
            source_filepath = source_filepaths.iloc[source_idx]
            if source_filepath in assigned_source_files:
                continue
            if any(source_filepath == picked[0] for picked in tentative):
                continue
            tentative.append((source_filepath, float(candidate_distances[candidate_idx]), source_idx))

        if len(tentative) < topn:
            # Rollback: leave the tentative sources unlocked for later targets.
            targets_with_duplicates.append(target_idx)
            continue

        output_data["target_filepath"].append(target_filepaths.iloc[target_idx])
        if save_embeddings:
            output_data["target_embedding"].append(target_emb_list[target_idx])
        for i, (source_filepath, distance, source_idx) in enumerate(tentative):
            assigned_source_files.add(source_filepath)
            output_data[f"top{i + 1}_source_filepath"].append(source_filepath)
            output_data[f"top{i + 1}_distance"].append(distance)
            if save_embeddings:
                output_data[f"top{i + 1}_embed"].append(_embeddings_to_lists(source_embeddings_filtered[source_idx]))

    return pd.DataFrame(output_data), targets_with_duplicates


def iterative_tmm_retrieval(
    target_embeddings: np.ndarray, source_embeddings: np.ndarray,
    source_df: "cudf.DataFrame", target_df: "cudf.DataFrame",
    desired_count: int, knn_metric: str,
    source_filepath_column_name: str = "filepath",
    target_filepath_column_name: str = "filepath",
    max_iterations: int = 8, initial_topn: Optional[int] = None,
    candidate_expansion_factor: int = 5, excluded_filepaths: Optional[Set[str]] = None,
    output_dir: Optional[str] = None,
    subset_name: str = "", save_embeddings: bool = False,
) -> Set[str]:
    """
    Iteratively call perform_tmm_retrieval, growing the candidate pool each round,
    until desired_count unique files are matched or max_iterations is reached.

    topn is fixed at ceil(desired_count / num_targets); the candidate pool grows by
    candidate_expansion_factor each iteration. Targets satisfied in a previous
    iteration are dropped from subsequent ones. The result never exceeds
    desired_count; an empty target set returns an empty set. Returns the
    set of unique matched filepaths.
    """
    num_targets = len(target_embeddings)
    if num_targets == 0 or desired_count <= 0:
        return set()
    if initial_topn is None:
        initial_topn = max(1, int(np.ceil(desired_count / num_targets)))

    topn = initial_topn  # kept constant; only the candidate pool grows each iteration
    retrieved_filepaths: set = set()
    current_excluded = set(excluded_filepaths) if excluded_filepaths else set()
    remaining_target_indices = list(range(num_targets))
    current_target_embeddings = target_embeddings
    current_target_df = target_df

    logger.info(f"  Iterative TMM matching (target: {desired_count} unique files, {num_targets} targets, "
                f"topn={initial_topn}, candidate_expansion_factor={candidate_expansion_factor}x/iter)")
    if current_excluded:
        logger.info(f"  Excluding {len(current_excluded)} already-matched files...")

    for iteration in range(max_iterations):
        if len(retrieved_filepaths) >= desired_count or not remaining_target_indices:
            break
        expansion = candidate_expansion_factor * (iteration + 1)

        result_df, targets_with_duplicates = perform_tmm_retrieval(
            current_target_embeddings, source_embeddings, source_df, current_target_df,
            topn, knn_metric,
            source_filepath_column_name, target_filepath_column_name,
            excluded_filepaths=current_excluded | retrieved_filepaths,
            save_embeddings=save_embeddings,
            candidate_expansion_factor=expansion,
        )
        if result_df.empty:
            break
        if output_dir and subset_name:
            result_df.to_parquet(
                Path(output_dir) / f"{subset_name}_iteration_{iteration + 1}_topn_{topn}.parquet", index=False,
            )

        new_unique = _merge_capped(retrieved_filepaths, result_df, desired_count)
        logger.info(f"    Iteration {iteration + 1}: unique files={len(retrieved_filepaths)} (+{new_unique}), "
                    f"targets needing retry: {len(targets_with_duplicates)}")

        if new_unique == 0 or not targets_with_duplicates:
            break
        # targets_with_duplicates index into the current subset; map back to originals.
        remaining_target_indices = [remaining_target_indices[i] for i in targets_with_duplicates]
        current_target_embeddings = target_embeddings[remaining_target_indices]
        current_target_df = target_df.iloc[remaining_target_indices].reset_index(drop=True)

    return retrieved_filepaths
