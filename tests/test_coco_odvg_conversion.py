# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess
import sys
from pycocotools.coco import COCO


TEST_COCO_ANNOT = "/media/scratch_metropolis2/tao_ci/data_services/data/coco/instances_val2017.json"

@pytest.fixture(scope='session')
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('ds_annot')


@pytest.mark.order(1)
def test_coco_to_odvg_conversion(results_dir):
    """"
    Test function to validate KITTI to COCO data format conversion
    """
    # Create system call.
    call = [
        "python",
        "nvidia_tao_ds/annotations/scripts/convert.py",
        f"coco.ann_file={TEST_COCO_ANNOT}",
        "data.input_format=COCO",
        "data.output_format=ODVG",
        f"results_dir={results_dir}"
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(os.path.join(results_dir, "instances_val2017_odvg.jsonl"))
    assert os.path.exists(os.path.join(results_dir, "instances_val2017_odvg_labelmap.json"))


@pytest.mark.order(2)
def test_coco_to_kitti_conversion(results_dir):
    """"
    Test function to validate COCO to KITTI data format conversion
    """
   # Create system call.
    call = [
        "python",
        "nvidia_tao_ds/annotations/scripts/convert.py",
        f"odvg.ann_file={os.path.join(results_dir, 'instances_val2017_odvg.jsonl')}",
        f"odvg.labelmap_file={os.path.join(results_dir, 'instances_val2017_odvg_labelmap.json')}",
        "data.input_format=ODVG",
        "data.output_format=COCO",
        f"results_dir={results_dir}"
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)

    coco_json_path = os.path.join(results_dir, "instances_val2017_odvg.json")
    # Check whether data in the required format was created.
    assert os.path.exists(coco_json_path)

    coco = COCO(coco_json_path)

    for img_id in coco.imgs.keys():
        img = coco.loadImgs(img_id)
        assert len(img) > 0, "image not found"
            
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        assert len(anns) > 0, "annotation not found"
