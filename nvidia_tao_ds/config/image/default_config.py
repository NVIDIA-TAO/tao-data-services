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
)


@dataclass
class DataConfig:
    """Dataset configuration template."""

    image_dir: str = STR_FIELD(
        value=MISSING, default_value="images", description="Output image path"
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    data: DataConfig = DATACLASS_FIELD(
        DataConfig(), description="Input data parameters"
    )
    in_place: Optional[bool] = BOOL_FIELD(
        True, default_value=False, description="If correction needs to be done inplace"
    )
    results_dir: str = STR_FIELD(value="", default_value="")
