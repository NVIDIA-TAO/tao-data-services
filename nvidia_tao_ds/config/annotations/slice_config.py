# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional, List, Any

from nvidia_tao_ds.config.utils.types import DATACLASS_FIELD


@dataclass
class DataConfig:
    """Dataset configuration template."""

    format: str = 'COCO'
    annotation_file: str = MISSING


@dataclass
class FilterConfig:
    """Dataset configuration template."""

    mode: str = "random"  # category, number
    reuse_categories: bool = True
    dump_remainder: bool = False
    split: Any = 0.25
    num_samples: int = 100
    included_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    re_patterns: List[str] = field(default_factory=lambda: [])


@dataclass
class SliceConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    filter: FilterConfig = DATACLASS_FIELD(FilterConfig())
    results_dir: Optional[str] = None
