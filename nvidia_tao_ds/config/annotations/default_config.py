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
class NVIDIAPASConfig:
    """NVIDIA PAS to TAO CLIP conversion configuration template."""

    root: str = STR_FIELD(value=MISSING, default_value="<specify NVIDIA_PAS root>")
    use_symlinks: bool = BOOL_FIELD(value=True)
    only_natural_and_original: bool = BOOL_FIELD(value=False)
    exclude_natural_from_aug: bool = BOOL_FIELD(value=False)
    val_sample_fraction: float = FLOAT_FIELD(value=0.1, default_value=0.1)
    include_test: bool = BOOL_FIELD(value=False)
    clean_output: bool = BOOL_FIELD(value=False)


@dataclass
class NVIDIAPAIDFPASConfig:
    """Experimental NVIDIA PAIDF PAS to TAO CLIP conversion configuration."""

    raw_output_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<specify PAIDF PAS SDG output root>",
    )
    attribute_vocab_path: str = STR_FIELD(
        value=MISSING,
        default_value="<specify TAO-FT attribute vocabulary>",
    )
    input_layout: str = STR_FIELD(
        value="split_dataset",
        default_value="split_dataset",
        valid_options="split_dataset,single_run",
        description=(
            "split_dataset uses V3.1 train/ with optional val and eval/test "
            "inputs; single_run is the legacy experimental layout."
        ),
    )
    caption_policy: str = STR_FIELD(
        value="all",
        default_value="all",
        valid_options="all,easy,medium,hard",
        description=(
            "all emits every PAS query; a named difficulty emits only that "
            "difficulty."
        ),
    )
    overwrite: bool = BOOL_FIELD(value=False)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(DataConfig())
    kitti: KITTIConfig = DATACLASS_FIELD(KITTIConfig())
    coco: COCOConfig = DATACLASS_FIELD(COCOConfig())
    odvg: ODVGConfig = DATACLASS_FIELD(ODVGConfig())
    aicity: AICityConfig = DATACLASS_FIELD(AICityConfig())
    nvidia_pas: NVIDIAPASConfig = DATACLASS_FIELD(NVIDIAPASConfig())
    nvidia_paidf_pas: NVIDIAPAIDFPASConfig = DATACLASS_FIELD(
        NVIDIAPAIDFPASConfig()
    )
    results_dir: Optional[str] = STR_FIELD(
        value="", default_value=""
    )
    verbose: bool = BOOL_FIELD(value=False)
