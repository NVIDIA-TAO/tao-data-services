# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for text embeddings (CLIP / SigLIP / SigLIP2)."""

from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import INT_FIELD, STR_FIELD


@dataclass
class TextEmbeddingsConfig:
    """Configuration for text embeddings.

    Required fields:
        input_parquet: Input parquet containing a ``text`` column.
        output_parquet: Output parquet for embeddings.
        model: Choice of embeddings (e.g. 'CLIP', 'SigLIP', 'SigLIP2').
        model_path: Hugging Face model path.

    Optional fields:
        batch_size: Number of texts to process in parallel.
    """

    input_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to input parquet>",
        description="Input parquet containing a text column"
    )
    output_parquet: str = STR_FIELD(
        value=MISSING,
        default_value="<path to output parquet>",
        description="Output parquet for embeddings"
    )
    model: str = STR_FIELD(
        value=MISSING,
        default_value="<embedding model name, e.g. CLIP, SigLIP, or SigLIP2>",
        description="Choice of embeddings",
        valid_options="CLIP,SigLIP,SigLIP2"
    )
    model_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to Hugging Face model>",
        description="Hugging Face model path"
    )
    batch_size: int = INT_FIELD(
        value=64,
        default_value=64,
        description="Number of texts to process in parallel"
    )
