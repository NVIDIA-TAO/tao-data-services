# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
import os
import subprocess
import sys
from omegaconf import OmegaConf
from PIL import Image

import pytest

from nvidia_tao_core.config.augmentation.default_config import ExperimentConfig


def check_image_size(output_dir, input_dir, height=None, width=None):
    """"
    Utility function to read images in a given directory

    Args:
    images: List of all images.
    """
    is_fixed_size = height and width
    expected_size = (width, height)
    for filename in os.listdir(output_dir):
        img = Image.open(os.path.join(output_dir, filename))
        if is_fixed_size:
            assert img.size == expected_size
        else:
            img_original = Image.open(os.path.join(input_dir, filename))
            assert img.size == img_original.size


@pytest.fixture(scope='session')
def cfg_kitti():
    """CI Hydra config."""
    cfg = ExperimentConfig()
    cfg.data.dataset_type = 'kitti'
    cfg.data.image_dir = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/images'
    cfg.data.ann_path = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/labels'
    cfg.spatial_aug.rotation.refine_box.gt_cache = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/mal.json'
    return cfg


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("height, width", [(1111, 311), (None, None)])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("refine_box", [False, True])
@pytest.mark.parametrize("num_gpus", [1])
def test_kitti(tmp_path_factory, cfg_kitti, height, width, batch_size, refine_box, num_gpus):
    """"
    Test function to validate data augmentation
    """
    cfg_kitti.spatial_aug.rotation.refine_box.enabled = refine_box
    cfg_kitti.data.batch_size = batch_size
    cfg_kitti.data.output_image_height = height
    cfg_kitti.data.output_image_width = width
    time_str = datetime.now().strftime("%y_%m_%d_%H:%M:%S")
    cfg_kitti.results_dir = tmp_path_factory.mktemp(f"output_kitti_{time_str}")
    tmp_yaml = os.path.join(cfg_kitti.results_dir, 'kitti.yaml')
    OmegaConf.save(cfg_kitti, tmp_yaml)
    # Create system call.
    os.environ['TAO_VISIBLE_DEVICES'] = str(list(range(num_gpus)))[1:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
    call = f"python nvidia_tao_ds/augmentation/entrypoint/augment.py generate -e {tmp_yaml} num_gpus={num_gpus}"
    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(cfg_kitti.results_dir)

    # Reading all images in a given directory
    check_image_size(
        os.path.join(cfg_kitti.results_dir, 'images'),
        cfg_kitti.data.image_dir,
        height, width)


@pytest.fixture(scope='session')
def cfg_coco():
    """CI Hydra config."""
    cfg = ExperimentConfig()
    cfg.data.dataset_type = 'coco'
    cfg.data.image_dir = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/images'
    cfg.data.ann_path = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/mal.json'
    cfg.spatial_aug.rotation.refine_box.gt_cache = '/media/scratch_metropolis2/tao_ci/data_services/data/kitti/mal.json'
    return cfg


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("height, width", [(1222, 322), (None, None)])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("refine_box", [False, True])
@pytest.mark.parametrize("num_gpus", [1])
def test_coco(tmp_path_factory, cfg_coco, height, width, batch_size, refine_box, num_gpus):
    """"
    Test function to validate data augmentation
    """
    cfg_coco.spatial_aug.rotation.refine_box.enabled = refine_box
    cfg_coco.data.batch_size = batch_size
    cfg_coco.data.output_image_height = height
    cfg_coco.data.output_image_width = width
    time_str = datetime.now().strftime("%y_%m_%d_%H:%M:%S")
    cfg_coco.results_dir = tmp_path_factory.mktemp(f"output_coco_{time_str}")
    tmp_yaml = os.path.join(cfg_coco.results_dir, 'coco.yaml')
    OmegaConf.save(cfg_coco, tmp_yaml)
    # Create system call.
    os.environ['TAO_VISIBLE_DEVICES'] = str(list(range(num_gpus)))[1:-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
    call = f"python nvidia_tao_ds/augmentation/entrypoint/augment.py generate -e {tmp_yaml} num_gpus={num_gpus}"
    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(cfg_coco.results_dir)

    # Reading all images in a given directory
    check_image_size(
        os.path.join(cfg_coco.results_dir, 'images'),
        cfg_coco.data.image_dir,
        height, width)
