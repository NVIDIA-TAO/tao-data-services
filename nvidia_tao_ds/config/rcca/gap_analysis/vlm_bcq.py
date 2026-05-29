# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for RCCA gap analysis using a VLM binary classification question (BCQ)."""

from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import STR_FIELD


@dataclass
class GapAnalysisConfig:
    """Configuration for KPI gap analysis.

    Required fields:
        predictions_json: Path to predictions JSON file.
        results_dir: Output directory for gap analysis results.

    Optional fields:
        videos_dir: Base directory for resolving relative video_id paths.
            If empty, video_id values in predictions are treated as absolute paths.
    """

    predictions_json: str = STR_FIELD(
        value=MISSING,
        default_value="<path to predictions JSON>",
        description="Path to predictions JSON file. Each item must have 'video_id', 'response', and 'gt' fields."
    )
    videos_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory containing videos. If empty, video_id in predictions are treated as absolute paths."
    )
    results_dir: Optional[str] = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description="Output directory for gap analysis results."
    )
