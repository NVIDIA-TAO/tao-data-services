# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING

from nvidia_tao_ds.config.utils.types import (
    STR_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
)


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_format: str = STR_FIELD(value="KITTI")
    output_format: str = STR_FIELD(value="COCO")


@dataclass
class KITTIConfig:
    """Dataset configuration template."""

    image_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    label_dir: str = STR_FIELD(
        value=MISSING, default_value="<specify labels directory>"
    )
    project: Optional[str] = STR_FIELD(None, default_value="annotations")
    mapping: Optional[str] = STR_FIELD(None)
    no_skip: bool = BOOL_FIELD(value=False)
    preserve_hierarchy: bool = BOOL_FIELD(value=False)


@dataclass
class COCOConfig:
    """Dataset configuration template."""

    ann_file: str = STR_FIELD(
        value=MISSING, default_value="<specify path to annotation file>"
    )
    refine_box: bool = BOOL_FIELD(value=False)
    use_all_categories: bool = BOOL_FIELD(value=False)
    add_background: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to add background to the class list, so as to make other classes, 1-indexed.",
    )


@dataclass
class ODVGConfig:
    """Dataset configuration template."""

    ann_file: str = STR_FIELD(
        value=MISSING, default_value="<specify path to annotation file>"
    )
    labelmap_file: Optional[str] = STR_FIELD(
        value=None, default_value="<specify path to labelmap file>"
    )


@dataclass
class ClassConfig:
    """Class configuration template."""

    CLASS_LIST: list = LIST_FIELD(arrList=[], default_value=["Person", "FourierGR1T2", "AgilityDigit", "Transporter"])
    SUB_CLASS_DICT: dict = DICT_FIELD(hashMap={}, default_value={})
    MAP_CLASS_NAMES: dict = DICT_FIELD(hashMap={}, default_value={
        "Person": "Person",
        "FourierGR1T2": "FourierGR1T2",
        "AgilityDigit": "AgilityDigit",
        "Transporter": "Transporter"
    })
    ATTRIBUTE_DICT: dict = DICT_FIELD(hashMap={}, default_value={
        "Person": "person.moving",
        "FourierGR1T2": "fourier_gr1_t2.moving",
        "AgilityDigit": "agility_digit.moving",
        "Transporter": "transporter.moving"
    })
    CLASS_RANGE_DICT: dict = DICT_FIELD(hashMap={}, default_value={
        "Person": 40,
        "FourierGR1T2": 40,
        "AgilityDigit": 40,
        "Transporter": 40
    })


@dataclass
class AnchorInitConfig:
    """Anchor initialization configuration template."""

    num_anchor: int = INT_FIELD(value=900, default_value=900)
    detection_range: float = FLOAT_FIELD(value=-1, default_value=-1)
    sample_ratio: int = INT_FIELD(value=-1, default_value=-1)
    output_file_name: str = STR_FIELD(value="anchor_init.npy", default_value="anchor_init.npy")


@dataclass
class AICityConfig:
    """Dataset configuration template."""

    root: str = STR_FIELD(value=MISSING, default_value="<specify data root>")
    version: str = STR_FIELD(value=MISSING, default_value="2025")
    split: str = STR_FIELD(value=MISSING, default_value="train")
    class_config: ClassConfig = DATACLASS_FIELD(ClassConfig())
    recentering: bool = BOOL_FIELD(value=MISSING, default_value=True)
    rgb_format: str = STR_FIELD(value=MISSING, default_value="mp4")
    depth_format: str = STR_FIELD(value=MISSING, default_value="h5")
    camera_grouping_mode: str = STR_FIELD(value=MISSING, default_value="random")
    anchor_init_config: AnchorInitConfig = DATACLASS_FIELD(AnchorInitConfig())
    num_frames: int = INT_FIELD(value=9000, default_value=9000)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    kitti: KITTIConfig = DATACLASS_FIELD(KITTIConfig())
    coco: COCOConfig = DATACLASS_FIELD(COCOConfig())
    odvg: ODVGConfig = DATACLASS_FIELD(ODVGConfig())
    aicity: AICityConfig = DATACLASS_FIELD(AICityConfig())
    results_dir: Optional[str] = STR_FIELD(
        value="", default_value=""
    )
    verbose: bool = BOOL_FIELD(value=False)
