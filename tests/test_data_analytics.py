# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""test script to run TAO analyze and validate test cases."""

import os
import pytest
import subprocess
import sys
import tarfile
import shutil
from glob import glob
import json

__TEST_DATA_FILENAME = "/media/scratch_metropolis2/tao_ci/data_services/data/Data_Analytics/data_analytics_2.tar.gz"
__TEST_DATA_SUBDIR = "data"


def clear_result_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


@pytest.fixture
def test_data_dir():
    """"
    function to validate untar the data file data_analytics.tar.gz
    into data folder.
    """
    test_archive = os.path.join(os.path.dirname(__file__),
                                __TEST_DATA_FILENAME)
    test_dir = os.path.join(os.path.dirname(__file__), __TEST_DATA_SUBDIR)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    # Create a new data folder.
    os.mkdir(test_dir)
    # Extract tar
    print(f"INFO: Extracting the `{test_archive}` archive , please wait...")
    tar = tarfile.open(test_archive)
    tar.extractall(path=test_dir)
    tar.close()
    assert os.path.exists(test_dir)
    return test_dir


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_kitti_dataset_analyz(test_data_dir):
    """"
    Test function to validate analyze process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result' "
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    graphs_glob_string = f"{test_data_dir}/result/graphs/*.pdf"
    assert len(glob(graphs_glob_string, recursive=False)) > 0, (
            "pdf files not created inside result directory."
        )


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_kitti_dataset_analyz_small_data(test_data_dir):
    """"
    Test function to validate analyze process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    call = (
        f" mkdir {test_data_dir}/data/imagedata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" mkdir {test_data_dir}/data/labeldata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" cp -r {test_data_dir}/data/images/00000* {test_data_dir}/data/imagedata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" cp -r {test_data_dir}/data/labels/00000* {test_data_dir}/data/labeldata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.image_dir={test_data_dir}/data/imagedata' \
            'data.ann_path={test_data_dir}/data/labeldata' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result' "
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    graphs_glob_string = f"{test_data_dir}/result/graphs/*.pdf"
    assert len(glob(graphs_glob_string, recursive=False)) > 0, (
            "pdf files not created inside result directory."
        )


def test_kitti_dataset_validate(test_data_dir):
    """"
    Test function to validate  dataset validate process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.input_format=KITTI ' 'results_dir={test_data_dir}/result' \
            apply_correction=False"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."


def test_kitti_dataset_validate_and_correct_small_data(test_data_dir):
    """"
    Test function to validate dataset validate and correct process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    call = (
        f" mkdir {test_data_dir}/data/imagedata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" mkdir {test_data_dir}/data/labeldata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" cp -r {test_data_dir}/data/images/00000* {test_data_dir}/data/imagedata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    call = (
        f" cp -r {test_data_dir}/data/labels/00000* {test_data_dir}/data/labeldata"
    )
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.image_dir={test_data_dir}/data/imagedata' \
            'data.ann_path={test_data_dir}/data/labeldata' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result' \
            apply_correction=True"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    assert os.path.exists(f"{test_data_dir}/result/corrected_kitti_files"), "Result/corrected_kitti_files Directory not created."


def test_kitti_dataset_validate_and_correct(test_data_dir):
    """"
    Test function to validate dataset validate and correct process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result' \
            apply_correction=True"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    assert os.path.exists(f"{test_data_dir}/result/corrected_kitti_files"), "Result/corrected_kitti_files Directory not created."


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_kitti_no_images(test_data_dir):
    """"
    Test function to validate data-analytics with only kitti label files.
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.image_dir={test_data_dir}/data/1234' apply_correction=True \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    assert os.path.exists(f"{test_data_dir}/result/corrected_kitti_files"), "Result/corrected_kitti_files Directory not created."
    clear_result_directory(f"{test_data_dir}/result/")   
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.image_dir={test_data_dir}/data/1234' apply_correction=False \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    assert (not os.path.exists(f"{test_data_dir}/result/coco.json")), "Result/corrected_kitti_files Directory has been created."
    clear_result_directory(f"{test_data_dir}/result/")

    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.image_dir={test_data_dir}/data/1234' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists("intermediate_kitti_files"), "Intermediate Directory not created."
    graphs_glob_string = f"{test_data_dir}/result/graphs/*.pdf"
    assert len(glob(graphs_glob_string, recursive=False)) > 0, (
            "pdf files not created inside result directory."
        )


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_coco_dataset_analyz(test_data_dir):
    """"
    Test function to validate analyze process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    graphs_glob_string = f"{test_data_dir}/result/graphs/*.pdf"
    assert len(glob(graphs_glob_string, recursive=False)) > 0, (
            "pdf files not created inside result directory."
        )


def test_coco_dataset_validate(test_data_dir):
    """"
    Test function to validate  dataset validate process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result' \
            apply_correction=False"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."


def test_coco_dataset_validate_and_correct(test_data_dir):
    """"
    Test function to validate dataset validate and correct process
    """
    clear_result_directory(f"{test_data_dir}/result/")

    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result' \
            apply_correction=True"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists(f"{test_data_dir}/result/coco.json"), "coco.json file not created."


@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
def test_coco_no_images(test_data_dir):
    """"
    Test function to validate data-analytics with only coco label files.
    """
    clear_result_directory(f"{test_data_dir}/result/")

    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.image_dir={test_data_dir}/data/1234' apply_correction=True \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert os.path.exists(f"{test_data_dir}/result/coco.json"), "coco.json file not created."
    clear_result_directory(f"{test_data_dir}/result/")

    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/validate.py \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.image_dir={test_data_dir}/data/1234' apply_correction=False \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    assert (not os.path.exists(f"{test_data_dir}/result/coco.json")), "coco.json file has been created."
    clear_result_directory(f"{test_data_dir}/result/")

    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.image_dir={test_data_dir}/data/1234' \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result"), "Result Directory not created."
    graphs_glob_string = f"{test_data_dir}/result/graphs/*.pdf"
    assert len(glob(graphs_glob_string, recursive=False)) > 0, (
            "pdf files not created inside result directory."
        )


@pytest.mark.skipif(os.getenv("WANDB_API_KEY", None) is None, reason="require to set up WANDB_API_KEY")
def test_kitti_dataset_wandb_analyz(test_data_dir):
    """"
    Test function to validate wandb analyze process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/labels' \
            'data.input_format=KITTI' 'results_dir={test_data_dir}/result' \
            'wandb.visualize=True' 'image.generate_image_with_bounding_box=True' 'image.sample_size=2'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result/wandb"), "result/wandb Directory not created."
    json_file_path = f"{test_data_dir}/result/wandb/latest-run/files/wandb-summary.json"
    assert os.path.exists(json_file_path), "wandb summary file not created."

    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    json_keys = contents.keys()
    table_list = ["object_count_chart1_table", "object_count_chart2_table", "Count statistics",
                  "Occlusion_chart1_table", "bbox_area_chart1_table", "Area statistics",
                  "Area statistics per object class", "truncation_chart1_table",
                  "coordinates_chart1_table", "image_chart1_table", "image_chart2_table",               
                  "image_chart3_table", "Image Statistics"]
    for item in table_list:
        assert item in json_keys, f"{item} table not found in wandb summary."


@pytest.mark.skipif(os.getenv("WANDB_API_KEY", None) is None, reason="require to set up WANDB_API_KEY")
def test_coco_dataset_wandb_analyz(test_data_dir):
    """"
    Test function to validate wandb analyze process
    """
    clear_result_directory(f"{test_data_dir}/result/")
    # Create system call.
    call = (
        f"python nvidia_tao_ds/data_analytics/scripts/analyze.py \
            'data.image_dir={test_data_dir}/data/images' \
            'data.ann_path={test_data_dir}/data/coco.json' \
            'data.input_format=COCO' 'results_dir={test_data_dir}/result' \
            'wandb.visualize=True' 'image.generate_image_with_bounding_box=True' 'image.sample_size=2'"
    )

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout,
                          stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(f"{test_data_dir}/result/wandb"), "result/wandb Directory not created."
    json_file_path = f"{test_data_dir}/result/wandb/latest-run/files/wandb-summary.json"
    assert os.path.exists(json_file_path), "wandb summary file not created."

    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())

    json_keys = contents.keys()
    table_list = ["object_count_chart1_table", "object_count_chart2_table", "Count statistics",
                  "bbox_area_chart1_table", "Area statistics", "Area statistics per object class",
                  "coordinates_chart1_table", "image_chart1_table", "image_chart2_table",
                  "image_chart3_table", "Image Statistics"]
    for item in table_list:
        assert item in json_keys, f"{item} table not found in wandb summary."
