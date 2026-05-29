# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass
from typing import Optional, List, Dict
import multiprocessing
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_format: str = STR_FIELD(value=MISSING, default_value="<specify input format>")
    kpi_sources: Optional[List[Dict[str, str]]] = LIST_FIELD([])
    image_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    ann_path: str = STR_FIELD(
        value=MISSING, default_value="<specify path to annotations>"
    )
    mapping: str = STR_FIELD(value=MISSING)


@dataclass
class GraphConfig:
    """Graph configuration template."""

    height: int = INT_FIELD(value=15)
    width: int = INT_FIELD(value=15)
    show_all: bool = BOOL_FIELD(value=False)
    generate_summary_and_graph: bool = BOOL_FIELD(value=True)


@dataclass
class WandbConfig:
    """Wandb configuration template."""

    project: Optional[str] = STR_FIELD(value=None, default_value="TAO data analytics")
    entity: Optional[str] = STR_FIELD(value=None)
    save_code: bool = BOOL_FIELD(value=False)
    name: Optional[str] = STR_FIELD(value=None)
    notes: Optional[str] = STR_FIELD(value=None)
    tags: Optional[List] = LIST_FIELD(None)
    visualize: bool = BOOL_FIELD(value=False)


@dataclass
class VisualizeConfig:
    """Visualize configuration template."""

    platform: str = STR_FIELD(value="local")
    tag: Optional[str] = STR_FIELD(value=None)


@dataclass
class KpiConfig:
    """KPI configuration template."""

    iou_threshold: float = FLOAT_FIELD(value=0.5)
    filter: bool = BOOL_FIELD(value=False)
    num_recall_points: int = INT_FIELD(value=11)
    ignore_sqwidth: int = INT_FIELD(value=0)
    conf_threshold: float = FLOAT_FIELD(value=0.5)
    is_internal: bool = BOOL_FIELD(value=False)


@dataclass
class ImageConfig:
    """Image configuration template."""

    generate_image_with_bounding_box: bool = BOOL_FIELD(value=False)
    sample_size: int = INT_FIELD(value=100)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    workers: int = INT_FIELD(value=multiprocessing.cpu_count())
    data: DataConfig = DATACLASS_FIELD(DataConfig())
    image: ImageConfig = DATACLASS_FIELD(ImageConfig())
    apply_correction: bool = BOOL_FIELD(value=False)
    graph: GraphConfig = DATACLASS_FIELD(GraphConfig())
    wandb: WandbConfig = DATACLASS_FIELD(WandbConfig())
    visualize: VisualizeConfig = DATACLASS_FIELD(VisualizeConfig())
    kpi: KpiConfig = DATACLASS_FIELD(KpiConfig())
    results_dir: Optional[str] = STR_FIELD(
        value="", default_value=""
    )
