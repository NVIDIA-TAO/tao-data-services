# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import sys
import types
from unittest import mock

import numpy as np
import pytest
import subprocess

# nvidia_tao_core.microservices is not available in the test environment.
# Mock it before any tao_ds imports trigger the import chain.
_cloud_mod = types.ModuleType("nvidia_tao_core.microservices.handlers.cloud_handlers.utils")
_cloud_mod.status_callback = mock.MagicMock()
sys.modules.setdefault("nvidia_tao_core.microservices", types.ModuleType("nvidia_tao_core.microservices"))
sys.modules.setdefault("nvidia_tao_core.microservices.handlers", types.ModuleType("nvidia_tao_core.microservices.handlers"))
sys.modules.setdefault("nvidia_tao_core.microservices.handlers.cloud_handlers", types.ModuleType("nvidia_tao_core.microservices.handlers.cloud_handlers"))
sys.modules["nvidia_tao_core.microservices.handlers.cloud_handlers.utils"] = _cloud_mod

TEST_SDG_PATH = "/media/scratch_metropolis2/tao_ci/data_services/data/aicity_ovpkl/SDG_data/SURF_Booth_031325/full_data/"
TEST_HUGGINGFACE_PATH = "/media/scratch_metropolis2/tao_ci/data_services/data/aicity_ovpkl/MTMC_Tracking_2025/"


@pytest.fixture(scope='session')
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('anno_pkls')


def test_anchor_initialization_blas_thread_limit(tmp_path):
    """anchor_initialization must cap BLAS threads to <= 64 via threadpool_limits during KMeans."""
    from nvidia_tao_ds.annotations.conversion.aicity_to_ovpkl import anchor_initialization

    rng = np.random.default_rng(0)
    n_boxes = 20
    info = {"gt_boxes": rng.uniform(-10, 10, size=(n_boxes, 9)).astype(np.float32)}
    pkl_dir = tmp_path / "train"
    pkl_dir.mkdir()
    with open(pkl_dir / "scene_infos_train.pkl", "wb") as f:
        pickle.dump({"infos": [info], "metadata": {}}, f)

    output_file = str(tmp_path / "anchors.npy")
    anchor_initialization(ann_file=str(pkl_dir), num_anchor=5, output_file_name=output_file)

    assert os.path.exists(output_file)
    # Confirm KMeans completed without crashing and produced the expected number of anchors
    assert np.load(output_file).shape[0] == 5


def test_video_decode_uses_release_ffmpeg_and_limits_frames(tmp_path, monkeypatch):
    from nvidia_tao_ds.annotations.conversion import aicity_to_ovpkl

    video_path = tmp_path / "videos" / "Camera.mp4"
    image_dir = tmp_path / "Camera" / "rgb"
    video_path.parent.mkdir()
    video_path.touch()
    monkeypatch.setattr(aicity_to_ovpkl.shutil, "which", lambda _name: "/usr/local/bin/ffmpeg")

    def fake_run(command, check):
        assert check is True
        assert command[command.index("-frames:v") + 1] == "3"
        assert command[-1].endswith("%09d.jpg")
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / "000000000.jpg").touch()

    monkeypatch.setattr(aicity_to_ovpkl.subprocess, "run", fake_run)

    aicity_to_ovpkl._video_to_frames(str(video_path), str(image_dir), num_frames=3)


def test_video_decode_rejects_silent_empty_opencv_fallback(tmp_path, monkeypatch):
    from nvidia_tao_ds.annotations.conversion import aicity_to_ovpkl

    video_path = tmp_path / "videos" / "Camera.mp4"
    image_dir = tmp_path / "Camera" / "rgb"
    video_path.parent.mkdir()
    video_path.touch()
    monkeypatch.setattr(aicity_to_ovpkl.shutil, "which", lambda _name: None)
    monkeypatch.setattr(aicity_to_ovpkl, "video2frame_multi_cameras_syn", lambda _root: None)

    with pytest.raises(RuntimeError, match="produced no frames"):
        aicity_to_ovpkl._video_to_frames(str(video_path), str(image_dir), num_frames=3)


@pytest.mark.order(1)
def test_aicity_to_ovpkl_conversion(results_dir):
    """"
    Test function to validate AICity to OVPKL data format conversion
    """
    # Create system call.
    call = [
        "python",
        "nvidia_tao_ds/annotations/scripts/convert.py",
        f"aicity.root={TEST_SDG_PATH}",
        "aicity.version='2025'",
        "aicity.split=''",
        "aicity.camera_grouping_mode='random'",
        "aicity.recentering=True",
        "aicity.rgb_format='h5'",
        "aicity.depth_format='h5'",
        "aicity.num_frames=100",
        "data.input_format=AICity",
        "data.output_format=OVPKL",
        f"results_dir={results_dir}"
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(os.path.join(results_dir, "SURF_Booth_031325+bev-sensor-buffer-zone-c6_infos_.pkl"))


@pytest.mark.order(2)
def test_ovpkl_to_aicity_conversion(results_dir):
    """"
    Test function to validate AICity to OVPKL data format conversion
    """
    # Create system call.
    call = [
        "python",
        "nvidia_tao_ds/annotations/scripts/convert.py",
        f"aicity.root={TEST_HUGGINGFACE_PATH}",
        "aicity.version='2025'",
        "aicity.split='train'",
        "aicity.camera_grouping_mode=''",
        "aicity.rgb_format='mp4'",
        "aicity.depth_format='h5'",
        "aicity.recentering=True",
        "aicity.num_frames=200",
        "aicity.anchor_init_config.num_anchor=10",
        "data.input_format=AICity",
        "data.output_format=OVPKL",
        f"results_dir={results_dir}"
    ]

    # Run the call as subprocess.
    subprocess.check_call(call, shell=False, stdout=sys.stdout, stderr=sys.stdout)

    # Check whether data in the required format was created.
    assert os.path.exists(os.path.join(results_dir, "train", "Warehouse_014_infos_train.pkl"))
