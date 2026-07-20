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
@pytest.mark.parametrize("num_samples", [10])
@pytest.mark.parametrize("split", [5, 0.2])
@pytest.mark.parametrize("mode", ['number', 'random'])
def test_coco_slicer(results_dir, split, mode, num_samples):
    """Test function to validate COCO annotation slicer."""
    # Create system call.
    call = [
        'python',
        'nvidia_tao_ds/annotations/scripts/slice.py',
        f'data.annotation_file={TEST_COCO_ANNOT}',
        f'filter.mode={mode}',
        f'filter.num_samples={num_samples}',
        f'filter.split={split}',
        f'results_dir={results_dir}',
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)
    # Check whether data in the required format was created.
    if mode == 'random':
        if isinstance(split, int):
            for i in range(split):
                assert os.path.exists(os.path.join(results_dir, f"part_{i}.json"))
        elif isinstance(split, float):
            assert os.path.exists(os.path.join(results_dir, f"kept.json"))
    elif mode == 'num_samples':
        assert os.path.exists(os.path.join(results_dir, f"kept.json"))
        coco = COCO(os.path.join(results_dir, "kept.json"))
        assert num_samples == len(coco.getImgIds()), "Number of samples not matched."


@pytest.mark.order(2)
def test_coco_merger(results_dir):
    """Test function to validate COCO annotation merger."""
    # Create system call.
    call = [
        "python",
        "nvidia_tao_ds/annotations/scripts/merge.py",
        "data.annotations="
        f"['{results_dir}/part_0.json','{results_dir}/part_1.json', '{results_dir}/part_2.json', '{results_dir}/part_3.json', '{results_dir}/part_4.json']",
        f"results_dir={results_dir}"
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    coco_merged = COCO(os.path.join(results_dir, "output.json"))
    coco = COCO(TEST_COCO_ANNOT)
    assert len(coco_merged.getImgIds()) == len(coco.getImgIds()), "Number of images are not equal."
    assert len(coco_merged.getAnnIds()) == len(coco.getAnnIds()), "Number of annotations are not equal."
    assert len(coco_merged.getCatIds()) == len(coco.getCatIds()), "Number of categories are not equal."
