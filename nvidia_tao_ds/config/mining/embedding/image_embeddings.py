# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for image embeddings (CLIP / SigLIP)."""

from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import INT_FIELD, STR_FIELD


@dataclass
class ImageEmbeddingsConfig:
    """Configuration for image embeddings.

    Required fields:
        input_parquet: Input parquet containing filepaths.
        output_parquet: Output parquet for embeddings.
        model: Choice of embeddings (e.g. 'CLIP', 'SigLIP').
        model_path: Hugging Face model path or TAO checkpoint path.

    Optional fields:
        model_config_path: TAO experiment spec path. Required when model_path is a TAO checkpoint.
        batch_size: Number of files to process in parallel.
    """

    input_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to input parquet>",
        description="Input parquet containing filepaths"
    )
    output_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to output parquet>",
        description="Output parquet for embeddings"
    )
    model: str = STR_FIELD(
        value=MISSING,
        default_value="<embedding model name, e.g. CLIP or SigLIP>",
        description="Choice of embeddings",
        valid_options="CLIP,SigLIP"
    )
    model_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to Hugging Face model or TAO checkpoint>",
        description="Hugging Face model path or TAO checkpoint path"
    )
    model_config_path: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to TAO experiment spec. Required when model_path is a TAO checkpoint"
    )
    batch_size: int = INT_FIELD(
        value=64,
        default_value=64,
        description="Number of files to process in parallel"
    )
