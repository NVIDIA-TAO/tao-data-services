# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess
import sys


def get_diff(kitt_file_1, kitt_file_2):
    """
    Function to identify is two KITTI files are the same

    Args:
    kitt_file_1 (string): Path to KITTI file 1
    kitt_file_2 (string): Path to KITTI file 1

    Returns: True if the two KITTI files are an exact match
    """
    with open(kitt_file_1) as file_1, open(kitt_file_2) as file_2:
        for l1, l2 in zip(sorted(file_1), sorted(file_2)):
            l1_contents = l1.strip().split()
            l2_contents = l2.strip().split()

            if l1_contents[0] != l2_contents[0]:
                print("get_diff :: Labels in {kitt_file_1} and {kitt_file_2} are not the same")
                return False
            
            for i in range(1, 14):
                if int(float(l1_contents[i])) != int(float(l2_contents[i])): # Note: Setting int to tolerate delicmal differences.
                    print(f"get_diff :: Bounding boxes in {kitt_file_1} and {kitt_file_2} are not the same")
                    return False
        
    return True


def iterate_and_compare(dir_1, dir_2):
    """
    Fuction to compare the KITTI files under two directories

    Args:
    dir_1 (string): Path to directory 1
    dir_2 (string): Path to directory 2
    
    Returns: True if the files are the two directories are the same
    """
    dir_1_files = (os.listdir(dir_1))
    dir_2_files = (os.listdir(dir_2))

    for file in dir_1_files:
        if file in dir_2_files:
            return get_diff(os.path.join(dir_1, file), os.path.join(dir_2, file))
        else:
            print(f"iterate_and_compare :: {file} not in {dir_2}")
            return False
    
    return True


@pytest.mark.order(1)
def test_kitti_to_coco_conversion():
    """"
    Test function to validate KITTI to COCO data format conversion
    """
    # Create system call.
    call = (
        "python nvidia_tao_ds/annotations/scripts/convert.py 'kitti.image_dir=/media/scratch_metropolis2/tao_ci/data_services/data/Auto_Label/images' 'kitti.label_dir=/media/scratch_metropolis2/tao_ci/data_services/data/Auto_Label/labels' 'data.input_format=KITTI' 'data.output_format=COCO' 'results_dir='Output_COCO_Dir"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists("Output_COCO_Dir/Auto_Label.json")

@pytest.mark.order(2)
def test_coco_to_kitti_conversion():
    """"
    Test function to validate COCO to KITTI data format conversion
    """
    # Create system call.
    call = (
        "python nvidia_tao_ds/annotations/scripts/convert.py 'coco.ann_file=Output_COCO_Dir/Auto_Label.json' 'data.input_format=COCO' 'data.output_format=KITTI' 'results_dir='Output_KITTI_Dir"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert iterate_and_compare("/media/scratch_metropolis2/tao_ci/data_services/data/Auto_Label/labels", "Output_KITTI_Dir")
