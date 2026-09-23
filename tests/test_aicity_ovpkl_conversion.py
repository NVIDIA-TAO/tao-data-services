# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AICity-to-OVPKL conversion using generated local fixtures."""

import json
import os
import pickle
import sys
import types
from unittest import mock

import h5py
import numpy as np
import pytest

# nvidia_tao_core.microservices is not available in the test environment.
# Mock it before any tao_ds imports trigger the import chain.
_cloud_mod = types.ModuleType(
    "nvidia_tao_core.microservices.handlers.cloud_handlers.utils"
)
_cloud_mod.status_callback = mock.MagicMock()
sys.modules.setdefault(
    "nvidia_tao_core.microservices",
    types.ModuleType("nvidia_tao_core.microservices"),
)
sys.modules.setdefault(
    "nvidia_tao_core.microservices.handlers",
    types.ModuleType("nvidia_tao_core.microservices.handlers"),
)
sys.modules.setdefault(
    "nvidia_tao_core.microservices.handlers.cloud_handlers",
    types.ModuleType("nvidia_tao_core.microservices.handlers.cloud_handlers"),
)
sys.modules[
    "nvidia_tao_core.microservices.handlers.cloud_handlers.utils"
] = _cloud_mod

from nvidia_tao_ds.annotations.conversion import aicity_to_ovpkl  # noqa: E402


_CAMERA_STORAGE_NAME = "Camera.front.h5"
_CAMERA_NAME = "Camera.front"
_CLASS_CONFIG = {
    "CLASS_LIST": ["Person", "Transporter"],
    "SUB_CLASS_DICT": {"Person": ["Human"]},
}


@pytest.mark.parametrize("h5_file", [False, True])
def test_camera_discovery_uses_real_spatialai_api(tmp_path, h5_file):
    """Exercise the installed 1.x/2.x API, not a camera-discovery mock."""
    names = ["Camera.0002", "Camera.0001"]
    for name in names:
        path = tmp_path / (name + ".h5" if h5_file else name)
        if h5_file:
            with h5py.File(path, "w"):
                pass
        else:
            path.mkdir()
    (tmp_path / "ignore.txt").write_text("fixture", encoding="utf-8")
    expected = [name + ".h5" if h5_file else name for name in sorted(names)]
    assert aicity_to_ovpkl.get_cam_names_in_scene(str(tmp_path), h5_file=h5_file) == expected


def _write_h5_scene(split_root, scene_name, separate_depth):
    """Create a minimal two-frame AICity scene using real HDF5 containers."""
    scene_dir = split_root / scene_name
    scene_dir.mkdir(parents=True)
    camera_path = scene_dir / _CAMERA_STORAGE_NAME
    with h5py.File(camera_path, "w") as camera_file:
        rgb_group = camera_file.create_group("rgb")
        for frame_id in range(2):
            rgb_group.create_dataset(
                f"rgb_{frame_id:05}.jpg",
                data=np.zeros((1,), dtype=np.uint8),
            )
        if not separate_depth:
            depth_group = camera_file.create_group("distance_to_image_plane_png")
            for frame_id in range(2):
                depth_group.create_dataset(
                    f"distance_to_image_plane_{frame_id:05}.png",
                    data=np.ones((2, 2), dtype=np.uint16),
                )

    if separate_depth:
        depth_dir = scene_dir / "depth_maps"
        depth_dir.mkdir()
        with h5py.File(depth_dir / _CAMERA_STORAGE_NAME, "w") as depth_file:
            for frame_id in range(2):
                depth_file.create_dataset(
                    f"distance_to_image_plane_{frame_id:05}.png",
                    data=np.ones((2, 2), dtype=np.uint16),
                )

    annotation = {
        "object id": 7,
        "object name": f"{scene_name}_person",
        "object type": "Human",
        "3d location": [1.0, 2.0, 0.5],
        "3d bounding box scale": [0.5, 0.6, 1.7],
        "3d bounding box rotation": [0.0, 0.0, 0.25],
        "2d bounding box": {_CAMERA_NAME: [0.0, 0.0, 10.0, 10.0]},
        "2d bounding box visible": {_CAMERA_NAME: [0.0, 0.0, 5.0, 10.0]},
    }
    with open(scene_dir / "ground_truth.json", "w", encoding="utf-8") as stream:
        json.dump({"0": [], "1": [annotation]}, stream)
    return scene_dir


def _patch_scene_metadata(monkeypatch):
    """Replace calibration discovery while retaining real on-disk scene data."""
    calibration = {
        _CAMERA_NAME: {
            "intrinsic matrix": np.eye(3, dtype=np.float64),
            "projection matrix w2c": np.eye(4, dtype=np.float64),
        }
    }

    def fake_load_calib(*_args, **_kwargs):
        return calibration, None

    def fake_camera_names(*_args, **_kwargs):
        return [_CAMERA_STORAGE_NAME]

    monkeypatch.setattr(aicity_to_ovpkl, "load_calib", fake_load_calib)
    monkeypatch.setattr(
        aicity_to_ovpkl,
        "get_cam_names_in_scene",
        fake_camera_names,
    )


def _load_scene(output_dir, scene_name):
    """Load one generated scene output."""
    with open(
        output_dir / "train" / f"{scene_name}_infos_train.pkl",
        "rb",
    ) as stream:
        return pickle.load(stream)


def test_aicity_conversion_is_deterministic_and_handles_empty_frames(
    tmp_path,
    monkeypatch,
):
    """Conversion should sort scenes and emit shape-stable empty annotations."""
    split_root = tmp_path / "dataset" / "train"
    # Deliberately create reverse lexical order to exercise deterministic traversal.
    _write_h5_scene(split_root, "scene_b", separate_depth=True)
    _write_h5_scene(split_root, "scene_a", separate_depth=False)
    (split_root / "not_a_scene.txt").write_text("ignored", encoding="utf-8")
    _patch_scene_metadata(monkeypatch)

    output_dir = tmp_path / "output"
    aicity_to_ovpkl.create_ov_infos_aicity2025(
        str(tmp_path / "dataset"),
        str(output_dir),
        rgb_format="h5",
        depth_format="h5",
        version="2025",
        split="train",
        class_config=aicity_to_ovpkl.update_class_config(_CLASS_CONFIG),
        camera_group_config=None,
        recentering=False,
        num_frames=-1,
    )

    scene_a = _load_scene(output_dir, "scene_a")
    scene_b = _load_scene(output_dir, "scene_b")
    expected_metadata = {
        "version": "2025",
        "split_type": "train",
        "schema_version": "aicity_ovpkl/v1",
        "class_names": ["Person", "Transporter"],
    }
    assert scene_a["metadata"] == expected_metadata
    assert scene_b["metadata"] == expected_metadata

    empty_info = scene_a["infos"][0]
    assert empty_info["gt_boxes"].shape == (0, 7)
    assert empty_info["gt_velocity"].shape == (0, 3)
    assert empty_info["instance_inds"].shape == (0,)
    assert empty_info["asset_inds"].shape == (0,)
    assert empty_info["valid_flag"].shape == (0,)
    assert np.issubdtype(empty_info["instance_inds"].dtype, np.integer)
    assert np.issubdtype(empty_info["asset_inds"].dtype, np.integer)
    assert np.issubdtype(empty_info["gt_names"].dtype, np.str_)

    annotated_a = scene_a["infos"][1]
    annotated_b = scene_b["infos"][1]
    assert annotated_a["gt_names"].tolist() == ["Person"]
    assert annotated_a["gt_visibility"][0][_CAMERA_NAME] == pytest.approx(0.5)
    # Stable scene ordering also makes globally assigned asset IDs repeatable.
    assert annotated_a["asset_inds"].tolist() == [1]
    assert annotated_b["asset_inds"].tolist() == [2]
    assert list(annotated_a["cams"]) == [_CAMERA_NAME]

    depth_key = "distance_to_image_plane_00001.png"
    assert annotated_a["cams"][_CAMERA_NAME]["depth_map_path"] == (
        os.path.join("scene_a", _CAMERA_STORAGE_NAME),
        os.path.join("distance_to_image_plane_png", depth_key),
    )
    assert annotated_b["cams"][_CAMERA_NAME]["depth_map_path"] == (
        os.path.join("scene_b", "depth_maps", _CAMERA_STORAGE_NAME),
        depth_key,
    )
    assert ".h5.h5" not in str(annotated_b["cams"][_CAMERA_NAME]["depth_map_path"])
    for scene_info in (annotated_a, annotated_b):
        depth_path, dataset_key = scene_info["cams"][_CAMERA_NAME][
            "depth_map_path"
        ]
        with h5py.File(split_root / depth_path, "r") as depth_file:
            assert dataset_key in depth_file


def test_anchor_initialization_filters_generated_index_pkls(tmp_path):
    """Anchor initialization must ignore lazy-index and camera-count pickles."""
    rng = np.random.default_rng(0)
    info = {"gt_boxes": rng.uniform(-10, 10, size=(20, 9)).astype(np.float32)}
    pkl_dir = tmp_path / "train"
    pkl_dir.mkdir()
    with open(pkl_dir / "scene_infos_train.pkl", "wb") as stream:
        pickle.dump({"infos": [info], "metadata": {}}, stream)
    with open(pkl_dir / "_lazy_index.pkl", "wb") as stream:
        pickle.dump({"frame_index": []}, stream)
    with open(pkl_dir / "train_lazy_index.pkl", "wb") as stream:
        pickle.dump({"frame_index": []}, stream)
    with open(pkl_dir / "_pkl_cam_counts.pkl", "wb") as stream:
        pickle.dump({"scene_infos_train.pkl": 1}, stream)
    (pkl_dir / "directory.pkl").mkdir()

    output_file = tmp_path / "anchors.npy"
    aicity_to_ovpkl.anchor_initialization(
        ann_file=str(pkl_dir),
        num_anchor=5,
        output_file_name=str(output_file),
    )

    assert output_file.exists()
    assert np.load(output_file).shape == (5, 11)


def test_anchor_initialization_skips_empty_3d_data(tmp_path):
    """Anchor generation should be a no-op when every frame is annotation-free."""
    pkl_dir = tmp_path / "train"
    pkl_dir.mkdir()
    with open(pkl_dir / "empty_infos_train.pkl", "wb") as stream:
        pickle.dump(
            {
                "infos": [{"gt_boxes": np.zeros((0, 7), dtype=np.float32)}],
                "metadata": {},
            },
            stream,
        )

    output_file = tmp_path / "anchors.npy"
    aicity_to_ovpkl.anchor_initialization(
        ann_file=str(pkl_dir),
        num_anchor=5,
        output_file_name=str(output_file),
    )
    assert not output_file.exists()


def test_video_decode_uses_release_ffmpeg_and_limits_frames(tmp_path, monkeypatch):
    """The preferred decoder should use ffmpeg and honor the frame limit."""
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

    aicity_to_ovpkl._video_to_frames(
        str(video_path),
        str(image_dir),
        num_frames=3,
    )


def test_video_decode_rejects_silent_empty_opencv_fallback(
    tmp_path,
    monkeypatch,
):
    """The compatibility decoder must fail if it produces no image frames."""
    video_path = tmp_path / "videos" / "Camera.mp4"
    image_dir = tmp_path / "Camera" / "rgb"
    video_path.parent.mkdir()
    video_path.touch()
    monkeypatch.setattr(aicity_to_ovpkl.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        aicity_to_ovpkl,
        "video2frame_multi_cameras_syn",
        lambda _root: None,
    )

    with pytest.raises(RuntimeError, match="produced no frames"):
        aicity_to_ovpkl._video_to_frames(
            str(video_path),
            str(image_dir),
            num_frames=3,
        )
