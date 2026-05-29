# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import List, Optional, Union
from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class RotationAIConfig:
    """Rotation configuration template."""

    enabled: bool = BOOL_FIELD(value=False, default_value=True)
    gt_cache: str = STR_FIELD(value="", default_value="label.json")


@dataclass
class RotationConfig:
    """Rotation configuration template."""

    angle: List[float] = LIST_FIELD(arrList=[0], default_value=[0, 10, 30])
    units: str = STR_FIELD(value="degrees", default_value="degrees")
    refine_box: RotationAIConfig = DATACLASS_FIELD(RotationAIConfig())


@dataclass
class ShearConfig:
    """Rotation configuration template."""

    shear_ratio_x: List[float] = LIST_FIELD(arrList=[0], default_value=[0])
    shear_ratio_y: List[float] = LIST_FIELD(arrList=[0], default_value=[0])


@dataclass
class FlipConfig:
    """Flip configuration template."""

    flip_horizontal: bool = BOOL_FIELD(value=False, default_value=False)
    flip_vertical: bool = BOOL_FIELD(value=False, default_value=False)


@dataclass
class TranslationConfig:
    """Translation configuration template."""

    translate_x: List[int] = LIST_FIELD(arrList=[0], default_value=[0])
    translate_y: List[int] = LIST_FIELD(arrList=[0], default_value=[0])


@dataclass
class HueConfig:
    """Hue configuration template."""

    hue_rotation_angle: List[float] = LIST_FIELD(arrList=[0], default_value=[0])


@dataclass
class SaturationConfig:
    """Saturation configuration template."""

    saturation_shift: List[float] = LIST_FIELD(arrList=[1], default_value=[1])


@dataclass
class ContrastConfig:
    """Contrast configuration template."""

    contrast: List[float] = LIST_FIELD(arrList=[0], default_value=[0])
    center: List[float] = LIST_FIELD(arrList=[127], default_value=[127])


@dataclass
class BrightnessConfig:
    """Contrast configuration template."""

    offset: List[float] = LIST_FIELD(arrList=[0], default_value=[0])


@dataclass
class SpatialAugmentationConfig:
    """Spatial augmentation configuration template."""

    rotation: RotationConfig = DATACLASS_FIELD(RotationConfig())
    shear: ShearConfig = DATACLASS_FIELD(ShearConfig())
    flip: FlipConfig = DATACLASS_FIELD(FlipConfig())
    translation: TranslationConfig = DATACLASS_FIELD(TranslationConfig())


@dataclass
class ColorAugmentationConfig:
    """Color augmentation configuration template."""

    hue: HueConfig = DATACLASS_FIELD(HueConfig())
    saturation: SaturationConfig = DATACLASS_FIELD(SaturationConfig())
    contrast: ContrastConfig = DATACLASS_FIELD(ContrastConfig())
    brightness: BrightnessConfig = DATACLASS_FIELD(BrightnessConfig())


@dataclass
class KernelFilterConfig:
    """Blur configuration template."""

    std: List[float] = LIST_FIELD(arrList=[0.1], default_value=[0.1])
    size: List[float] = LIST_FIELD(arrList=[3], default_value=[3])


@dataclass
class DataConfig:
    """Dataset configuration template."""

    dataset_type: str = STR_FIELD(value="coco")
    output_image_width: Union[int, None] = INT_FIELD(value=None, default_value=1000)
    output_image_height: Union[int, None] = INT_FIELD(value=None, default_value=300)
    image_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    ann_path: str = STR_FIELD(value=MISSING, default_value="<specify annotation path>")
    output_dataset: str = STR_FIELD(
        value=MISSING, default_value="<specify output dataset path>"
    )
    batch_size: int = INT_FIELD(value=8, default_value=2)
    include_masks: bool = BOOL_FIELD(value=False)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    random_seed: int = INT_FIELD(value=42)
    num_gpus: int = INT_FIELD(value=1)
    gpu_ids: List[int] = LIST_FIELD(arrList=[0])
    data: DataConfig = DATACLASS_FIELD(DataConfig())
    color_aug: ColorAugmentationConfig = DATACLASS_FIELD(ColorAugmentationConfig())
    spatial_aug: SpatialAugmentationConfig = DATACLASS_FIELD(
        SpatialAugmentationConfig()
    )
    blur_aug: KernelFilterConfig = DATACLASS_FIELD(KernelFilterConfig())
    results_dir: Optional[str] = STR_FIELD(
        "", default_value=""
    )
    cuda_blocking: bool = BOOL_FIELD(
        value=False,
        description="Debug flag to add CUDA_LAUNCH_BLOCKING=1 to the command calls.",
    )
