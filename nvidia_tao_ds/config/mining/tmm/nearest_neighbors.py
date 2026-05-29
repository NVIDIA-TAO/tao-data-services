# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for GPU-accelerated k-NN nearest-neighbor mining."""

from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import INT_FIELD, STR_FIELD


@dataclass
class NearestNeighborsConfig:
    """Configuration for k-NN mining.

    Required fields:
        source_parquet: Path to source parquet file.
        target_parquet: Path to target parquet file.
        output_parquet: Path to save output parquet file.

    Optional fields:
        topn: Number of similar items to find per target.
        knn_metric: Distance metric for k-NN search
            (one of 'euclidean', 'cosine', 'manhattan').
        source_embed_column_name: Column name for embeddings in source data.
        target_embed_column_name: Column name for embeddings in target data.
        filter_by_label: When 'true', drop mined pairs whose source label
            differs from the target label.
    """

    source_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to source parquet>",
        description="Path to source parquet file"
    )
    target_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to target parquet>",
        description="Path to target parquet file"
    )
    output_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to output parquet>",
        description="Path to save output parquet file"
    )
    topn: int = INT_FIELD(
        value=5,
        default_value=5,
        description="Number of similar items to find per target"
    )
    knn_metric: str = STR_FIELD(
        value="euclidean",
        default_value="euclidean",
        valid_options="euclidean,cosine,manhattan",
        description="Distance metric for k-NN search"
    )
    source_embed_column_name: str = STR_FIELD(
        value="embedding",
        default_value="embedding",
        description="Column name for embeddings in source data"
    )
    target_embed_column_name: str = STR_FIELD(
        value="embedding",
        default_value="embedding",
        description="Column name for embeddings in target data"
    )
    filter_by_label: str = STR_FIELD(
        value="false",
        default_value="false",
        description="When 'true', drop mined pairs whose source label differs from the target label"
    )
