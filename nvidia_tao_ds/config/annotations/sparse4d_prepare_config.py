# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for TAO Sparse4D data preparation."""

from dataclasses import dataclass
from typing import Dict, List

from omegaconf import MISSING

from nvidia_tao_ds.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


SPARSE4D_DEFAULT_CLASSES = [
    "person",
    "gr1_t2",
    "agility_digit",
    "nova_carter",
]
SPARSE4D_DEFAULT_SUBCLASS_MAP = {
    "person": ["Person", "Human"],
    "gr1_t2": ["FourierGR1T2", "Fourier_GR1_T2_Humanoid"],
    "agility_digit": ["AgilityDigit", "Agility_Digit_Humanoid"],
    "nova_carter": ["NovaCarter", "Nova_Carter"],
}


@dataclass
class LazyIndexConfig:
    """Options for the trusted Sparse4D annotation-PKL index."""

    annotation_source: str = STR_FIELD(
        value="",
        description="Annotation directory or split text file.",
    )
    force: bool = BOOL_FIELD(
        value=False,
        description="Re-index every source PKL instead of reusing unchanged entries.",
    )
    workers: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Worker count; zero selects a bounded CPU-dependent default.",
    )
    write_camera_counts: bool = BOOL_FIELD(
        value=True,
        description="Also write the legacy _pkl_cam_counts.pkl sidecar.",
    )
    camera_counts_path: str = STR_FIELD(
        value="",
        description="Optional override for the camera-count sidecar path.",
    )


@dataclass
class SceneSelectionConfig:
    """Portable ways to select raw Sparse4D scene directories."""

    data_root: str = STR_FIELD(
        value="",
        description="Root containing raw scene directories.",
    )
    scenes: List[str] = LIST_FIELD(
        arrList=[],
        description="Optional raw scene names below data_root.",
    )
    scenes_file: str = STR_FIELD(
        value="",
        description="Optional text file containing raw scene names.",
    )
    scene_dirs: List[str] = LIST_FIELD(
        arrList=[],
        description="Optional explicit raw scene directories.",
    )
    train_split: str = STR_FIELD(
        value="",
        description="Optional Sparse4D PKL split used to derive raw scene names.",
    )
    dedup_regex: str = STR_FIELD(
        value=r"^CT[\w.]+?__",
        description="Regex removing generated prefixes from split scene names.",
    )


@dataclass
class LTT2DGTConfig:
    """Options for visible 2D box sidecars extracted from raw scenes."""

    selection: SceneSelectionConfig = DATACLASS_FIELD(SceneSelectionConfig())
    annotation_version: str = STR_FIELD(
        value="v0.1",
        valid_options="v0.0,v0.1",
    )
    output_dir: str = STR_FIELD(
        value="",
        description="Directory for per-scene ltt_2dgt/v1 NPZ files.",
    )
    frame_stride: int = INT_FIELD(value=1, valid_min=1)
    max_frames_per_scene: int = INT_FIELD(value=0, valid_min=0)
    drop_names: List[str] = LIST_FIELD(
        arrList=["pallet"],
        description="Source object types to omit explicitly.",
    )
    overwrite: bool = BOOL_FIELD(
        value=False,
        description="Allow replacement of existing scene sidecars.",
    )


@dataclass
class LTTDataConfig:
    """Options for model-independent ltt_data/v2 geometry extraction."""

    selection: SceneSelectionConfig = DATACLASS_FIELD(SceneSelectionConfig())
    output_path: str = STR_FIELD(
        value="",
        description="Output ltt_data/v2 NPZ path.",
    )
    calibration_mode: str = STR_FIELD(
        value="aic25",
        valid_options="aic24,aic25",
    )
    calibration_file: str = STR_FIELD(
        value="",
        description="Optional calibration filename or path override.",
    )
    annotation_version: str = STR_FIELD(
        value="v0.1",
        valid_options="v0.0,v0.1",
    )
    frame_stride: int = INT_FIELD(value=10, valid_min=1)
    max_frames_per_scene: int = INT_FIELD(value=0, valid_min=0)
    min_visibility: float = FLOAT_FIELD(
        value=0.0,
        valid_min=0.0,
        valid_max=1.0,
    )
    max_per_class: int = INT_FIELD(value=50000, valid_min=0)
    image_width: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Optional image-width override; zero uses calibration.",
    )
    image_height: int = INT_FIELD(
        value=0,
        valid_min=0,
        description="Optional image-height override; zero uses calibration.",
    )
    seed: int = INT_FIELD(value=0)


@dataclass
class RTDETR2DConfig:
    """Options for normalizing RT-DETR KITTI archives."""

    input_dir: str = STR_FIELD(
        value="",
        description="Scene directory containing <camera>/labels.tar.gz.",
    )
    output_path: str = STR_FIELD(
        value="",
        description="Output ltt_rtdetr2d/v1 NPZ path.",
    )
    scene_name: str = STR_FIELD(
        value="",
        description="Optional scene-name override.",
    )
    camera_map: Dict[str, str] = DICT_FIELD(
        hashMap={},
        description="Map label archive directory names to calibration camera names.",
    )
    class_map: Dict[str, str] = DICT_FIELD(
        hashMap={},
        description="Map detector labels to names in class_names.",
    )
    confidence_threshold: float = FLOAT_FIELD(
        value=0.4,
        valid_min=0.0,
        valid_max=1.0,
    )
    frame_stride: int = INT_FIELD(value=1, valid_min=1)
    max_frames_per_camera: int = INT_FIELD(value=0, valid_min=0)


@dataclass
class SV2DConfig:
    """Options for calibration-free COCO-to-Sparse4D conversion."""

    manifest_path: str = STR_FIELD(
        value="",
        description="JSON manifest describing one or more COCO image sources.",
    )
    cache_dir: str = STR_FIELD(
        value="",
        description="Directory for ltt_rtdetr2d/v1 NPZ caches.",
    )
    pkl_dir: str = STR_FIELD(
        value="",
        description="Directory for GT-less Sparse4D annotation PKLs.",
    )
    dataset: str = STR_FIELD(
        value="all",
        description="Manifest dataset name, or all.",
    )
    split_output: str = STR_FIELD(
        value="",
        description="Optional split-file output path.",
    )
    suffix: str = STR_FIELD(
        value="",
        description="Safe filename suffix applied to generated scene artifacts.",
    )
    max_images: int = INT_FIELD(value=0, valid_min=0)
    keep_empty: bool = BOOL_FIELD(
        value=False,
        description="Retain background-only images and mark them as valid frames.",
    )
    canonical_width: int = INT_FIELD(value=1920, valid_min=1)
    canonical_height: int = INT_FIELD(value=1080, valid_min=1)
    drop_names: List[str] = LIST_FIELD(
        arrList=["pallet"],
        description="Source category names to omit rather than remap.",
    )
    class_map: Dict[str, str] = DICT_FIELD(
        hashMap={},
        description="Optional source-category to class_names mapping.",
    )


@dataclass
class Sparse4DPrepareConfig:
    """Top-level Sparse4D artifact-preparation configuration."""

    operation: str = STR_FIELD(
        value="lazy_index",
        valid_options="lazy_index,ltt_2dgt,ltt_data,rtdetr_2d,sv2d",
        description="Artifact preparation operation to execute.",
    )
    class_names: List[str] = LIST_FIELD(
        arrList=SPARSE4D_DEFAULT_CLASSES,
        description=(
            "Ordered taxonomy shared verbatim with Sparse4D dataset.classes; "
            "defaults to the released four-class experiment."
        ),
    )
    subclass_map: Dict[str, List[str]] = DICT_FIELD(
        hashMap=SPARSE4D_DEFAULT_SUBCLASS_MAP,
        description=(
            "Parent-class to raw source-name aliases; keys must also appear "
            "in class_names."
        ),
    )
    lazy_index: LazyIndexConfig = DATACLASS_FIELD(LazyIndexConfig())
    ltt_2dgt: LTT2DGTConfig = DATACLASS_FIELD(LTT2DGTConfig())
    ltt_data: LTTDataConfig = DATACLASS_FIELD(LTTDataConfig())
    rtdetr_2d: RTDETR2DConfig = DATACLASS_FIELD(RTDETR2DConfig())
    sv2d: SV2DConfig = DATACLASS_FIELD(SV2DConfig())
    results_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<specify results directory>",
    )
