# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for RCCA gap analysis using VCN AOI classification inference."""

from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import FLOAT_FIELD, INT_FIELD, STR_FIELD


@dataclass
class GapAnalysisConfig:
    """Configuration for VCN AOI KPI gap analysis.

    Required fields:
        inference_results_dir: Directory containing the TAO VCN
            inference CSV.
        train_config: Path to the VCN train config YAML. Used to read
            ``dataset.classify.input_map`` (list of lighting conditions)
            and ``dataset.classify.image_ext`` (file extension with dot).
        results_dir: Output directory for gap analysis results.

    Optional fields:
        kpi_media_path: Root directory prepended to relative image paths
            in the CSV. If empty, ``input_path`` values in the CSV are
            treated as absolute paths.
        threshold: Classification threshold. A sample is predicted
            NO_PASS when ``siamese_score > threshold``. If negative, the
            threshold is auto-computed from the inference results using
            ``min_recall``.
        min_recall: Minimum NO_PASS recall a candidate threshold must
            achieve when auto-computing (0.0 - 1.0).
        top_k_per_label: Maximum number of weakest samples to keep per
            ground-truth label. Acts as a per-label augmentation budget.
    """

    inference_results_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<path to TAO VCN inference results directory>",
        description="Directory containing the TAO VCN inference CSV (searched recursively)."
    )
    train_config: str = STR_FIELD(
        value=MISSING,
        default_value="<path to VCN train config YAML>",
        description="Path to the VCN train config YAML providing 'dataset.classify.input_map' and 'dataset.classify.image_ext'."
    )
    kpi_media_path: str = STR_FIELD(
        value="",
        default_value="",
        description="Root directory prepended to relative image paths in the CSV. If empty, input_path values are treated as absolute paths."
    )
    threshold: float = FLOAT_FIELD(
        value=-1.0,
        default_value=-1.0,
        description="Classification threshold; predict NO_PASS when siamese_score > threshold. Negative triggers auto-compute."
    )
    min_recall: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Minimum NO_PASS recall for auto-computing the threshold."
    )
    top_k_per_label: int = INT_FIELD(
        value=50,
        default_value=50,
        valid_min=1,
        description="Maximum number of weakest samples to keep per ground-truth label."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description="Output directory for gap analysis results."
    )
