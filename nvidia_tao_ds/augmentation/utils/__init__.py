# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for TAO augment."""

from nvidia_tao_ds.augmentation.dataloader.coco_callable import CocoInputCallable
from nvidia_tao_ds.augmentation.dataloader.kitti_callable import KittiInputCallable
from nvidia_tao_ds.augmentation.pipeline.sharded_pipeline import (
    build_coco_pipeline,
    build_kitti_pipeline,
)

callable_dict = {
    'kitti': KittiInputCallable,
    'coco': CocoInputCallable
}

pipeline_dict = {
    'kitti': build_kitti_pipeline,
    'coco': build_coco_pipeline
}
