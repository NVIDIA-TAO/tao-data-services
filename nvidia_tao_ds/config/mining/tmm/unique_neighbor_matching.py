# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for iterative unique-assignment k-NN mining (unique_neighbor_matching)."""

from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from nvidia_tao_ds.config.utils.types import BOOL_FIELD, INT_FIELD, STR_FIELD


@dataclass
class UniqueNeighborMatchingConfig:
    """Configuration for unique_neighbor_matching mining.

    Iterative unique-assignment matching in two modes: ``global`` (match
    ``desired_unique_count`` source images that look like the target set) and
    ``class_stratified`` (split into per-rare-class pools via COCO/KITTI
    annotations, allocate budget by target class ratio, match per class with
    non-rare-pool fallback). Writes a directory of outputs.

    Required fields:
        source_path: Path to source parquet file OR directory of parquet files.
        target_path: Path to target parquet file OR directory of parquet files.
        output_dir: Output DIRECTORY. Writes final_unique_files.parquet,
            summary.json, per-iteration parquets, and (with visualize) viz PNGs.
        desired_unique_count: Total number of unique source files to match.

    Optional fields:
        allocation_policy: 'global' or 'class_stratified'.
        distance_metric: 'euclidean', 'cosine', or 'manhattan'. Embeddings are
            L2-normalized before search.
        candidate_expansion_factor: candidate-pool multiplier over the per-target
            top-n; grows each iteration until the desired count is reached.
        source_embedding_column / target_embedding_column: embedding columns.
        source_filepath_column / target_filepath_column: filepath columns. The
            source one is also the column of final_unique_files.parquet.
        exclude_path: optional parquet with a 'filepath' column; those images are
            removed from the source pool before matching (null disables).
        source_detection_file / target_detection_file: COCO .json OR KITTI label
            directory. Required by logic in class_stratified mode (null disables).
        detection_format: 'coco' or 'kitti' — declared explicitly (not inferred from
            the path); required whenever a detection file is provided.
        rare_class_list: comma-separated rare class names (e.g. 'person,bicycle').
            Required by logic in class_stratified mode.
        save_embeddings: include embeddings in per-iteration parquet outputs.
        visualize: save per-class visualization grids (needs Pillow + matplotlib).
    """

    source_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to source parquet or directory>",
        description="Path to source parquet file or directory of parquet files",
    )
    target_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to target parquet or directory>",
        description="Path to target parquet file or directory of parquet files",
    )
    output_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description=(
            "Output DIRECTORY; writes final_unique_files.parquet, summary.json, "
            "per-iteration parquets, and optional viz PNGs"
        ),
    )
    desired_unique_count: int = INT_FIELD(
        value=MISSING,
        default_value="<total unique files to retrieve>",
        description="Total number of unique source files to return",
    )
    allocation_policy: str = STR_FIELD(
        value="global",
        default_value="global",
        valid_options="global,class_stratified",
        description="Allocation policy",
    )
    distance_metric: str = STR_FIELD(
        value="euclidean",
        default_value="euclidean",
        valid_options="euclidean,cosine,manhattan",
        description="Distance metric for k-NN search (embeddings are L2-normalized first)",
    )
    candidate_expansion_factor: int = INT_FIELD(
        value=5,
        default_value=5,
        description=(
            "Candidate-pool multiplier over the per-target top-n. The k-NN search "
            "fetches candidate_expansion_factor x top-n candidates so enough remain "
            "after unique assignment; the pool grows by this factor each iteration."
        ),
    )
    source_embedding_column: str = STR_FIELD(
        value="embedding",
        default_value="embedding",
        description="Embedding column name in source data",
    )
    target_embedding_column: str = STR_FIELD(
        value="embedding",
        default_value="embedding",
        description="Embedding column name in target data",
    )
    source_filepath_column: str = STR_FIELD(
        value="filepath",
        default_value="filepath",
        description="Filepath column in source; also the column of final_unique_files.parquet",
    )
    target_filepath_column: str = STR_FIELD(
        value="filepath",
        default_value="filepath",
        description="Filepath column name in target data",
    )
    exclude_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Optional parquet with a 'filepath' column; those images are excluded from source (null disables)",
    )
    source_detection_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="COCO .json or KITTI label dir for source; required by logic in class_stratified (null disables)",
    )
    target_detection_file: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="COCO .json or KITTI label dir for target; required by logic in class_stratified (null disables)",
    )
    detection_format: Optional[str] = STR_FIELD(
        value=None,
        default_value="coco",
        valid_options="coco,kitti",
        description=(
            "Format of the source/target detection files: 'coco' (.json) or 'kitti' (label directory). "
            "Declared explicitly (not inferred from the path); required whenever a detection file is provided."
        ),
    )
    rare_class_list: str = STR_FIELD(
        value="",
        default_value="",
        description="Comma-separated rare class names, e.g. 'person,bicycle'; required by logic in class_stratified",
    )
    save_embeddings: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Include embeddings in the per-iteration parquet outputs (off by default)",
    )
    visualize: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Save per-class visualization grids (requires Pillow and matplotlib)",
    )
