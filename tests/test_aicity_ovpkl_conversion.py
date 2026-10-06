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


def _unlabeled_config(tmp_path, rgb_format="jpg"):
    """Build real image/calibration inputs and merge the shipped CLI schema."""
    from pathlib import Path
    from omegaconf import OmegaConf
    from PIL import Image
    from nvidia_tao_ds.config.annotations.default_config import ExperimentConfig

    scene = tmp_path / "dataset" / "train" / "RealWarehouse"
    scene.mkdir(parents=True)
    sensors = []
    for camera_index, camera in enumerate(["Camera1", "Camera2"]):
        transform = np.eye(4)
        transform[0, 3] = camera_index + 1.0
        sensors.append({
            "id": camera,
            "type": "camera",
            "intrinsicMatrix": [[20, 0, 8], [0, 20, 6], [0, 0, 1]],
            "extrinsicMatrix": transform[:3].tolist(),
            "attributes": [
                {"name": "frameWidth", "value": "16"},
                {"name": "frameHeight", "value": "12"},
            ],
        })
        if rgb_format == "h5":
            with h5py.File(scene / f"{camera}.h5", "w") as stream:
                for frame_id in range(3):
                    stream.create_dataset(
                        f"rgb/rgb_{frame_id:05}.jpg",
                        data=np.full((12, 16, 3), frame_id, dtype=np.uint8),
                    )
        else:
            image_dir = scene / camera / "rgb"
            image_dir.mkdir(parents=True)
            for frame_id in range(3):
                Image.new("RGB", (16, 12), color=(frame_id, 0, 0)).save(
                    image_dir / f"{frame_id:09d}.jpg"
                )
    (scene / "calibration.json").write_text(json.dumps({
        "version": "1.0", "osmURL": "", "calibrationType": "cartesian",
        "sensors": sensors,
    }), encoding="utf-8")
    spec = Path(aicity_to_ovpkl.__file__).parents[1] / "experiment_specs/aicity2ovpkl.yaml"
    cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.load(spec))
    cfg.results_dir = str(tmp_path / "output")
    cfg.aicity.root = str(tmp_path / "dataset")
    cfg.aicity.rgb_format = rgb_format
    cfg.aicity.camera_grouping_mode = ""
    cfg.aicity.recentering = False
    cfg.aicity.class_config.CLASS_LIST = ["person"]
    cfg.aicity.load_annotations = False
    cfg.aicity.fps = 12.5
    return cfg, scene


@pytest.mark.parametrize("rgb_format", ["jpg", "h5"])
def test_public_converter_prepares_unlabeled_calibrated_scene(tmp_path, monkeypatch, rgb_format):
    """Missing GT/depth is explicit, calibration is real, and anchors are untouched."""
    from pathlib import Path

    cfg, scene = _unlabeled_config(tmp_path, rgb_format)
    output = Path(cfg.results_dir)
    output.mkdir()
    anchor = output / cfg.aicity.anchor_init_config.output_file_name
    anchor.write_bytes(b"existing pretrained anchors")
    anchor_init = mock.Mock(side_effect=AssertionError("must not fit anchors on unlabeled data"))
    monkeypatch.setattr(aicity_to_ovpkl, "anchor_initialization", anchor_init)

    aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)

    assert not (scene / "ground_truth.json").exists()
    assert anchor.read_bytes() == b"existing pretrained anchors"
    anchor_init.assert_not_called()
    payload = _load_scene(output, "RealWarehouse")
    assert payload["metadata"]["class_names"] == ["person"]
    assert len(payload["infos"]) == 3
    for frame_id, info in enumerate(payload["infos"]):
        assert info["gt_boxes"] is None
        assert info["timestamp"] == pytest.approx(frame_id / 12.5)
        assert info["frame_idx"] == frame_id
        assert info["token"] == f"RealWarehouse__{frame_id:09d}"
        assert list(info["cams"]) == ["Camera1", "Camera2"]
        for camera_index, camera in enumerate(info["cams"].values()):
            assert "depth_map_path" not in camera
            assert camera["cam_intrinsic"][0, 0] == 20
            assert camera["sensor2world_transform"][0, 3] == camera_index + 1.0


def test_public_converter_still_requires_gt_by_default(tmp_path):
    """An absent GT file must not silently change a labeled conversion's route."""
    cfg, _ = _unlabeled_config(tmp_path)
    from nvidia_tao_ds.config.annotations.default_config import AICityConfig
    cfg.aicity.load_annotations = AICityConfig().load_annotations
    assert cfg.aicity.load_annotations is True
    with pytest.raises(FileNotFoundError, match="ground_truth.json"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_converter_rejects_invalid_capture_rate(tmp_path, monkeypatch, fps):
    """Reject invalid timestamps before decoding videos or creating outputs."""
    cfg, _ = _unlabeled_config(tmp_path)
    cfg.aicity.fps = fps
    cfg.aicity.rgb_format = "mp4"
    decode = mock.Mock(side_effect=AssertionError("must validate before decoding"))
    monkeypatch.setattr(aicity_to_ovpkl, "video_to_frame", decode)
    with pytest.raises(ValueError, match="fps must be finite and positive"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)
    decode.assert_not_called()
    assert not os.path.exists(cfg.results_dir)


@pytest.mark.parametrize("num_frames", [0, -2])
def test_converter_rejects_invalid_frame_limit(tmp_path, monkeypatch, num_frames):
    """An invalid frame limit must not start an expensive MP4 conversion."""
    cfg, _ = _unlabeled_config(tmp_path)
    cfg.aicity.num_frames = num_frames
    cfg.aicity.rgb_format = "mp4"
    decode = mock.Mock(side_effect=AssertionError("must validate before decoding"))
    monkeypatch.setattr(aicity_to_ovpkl, "video_to_frame", decode)
    with pytest.raises(ValueError, match="num_frames must be -1 or positive"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)
    decode.assert_not_called()
    assert not os.path.exists(cfg.results_dir)


@pytest.mark.parametrize("rgb_format", ["jpg", "h5"])
def test_unlabeled_conversion_reports_missing_camera_storage(tmp_path, rgb_format):
    """Identify the calibration camera and expected path for missing images."""
    cfg, scene = _unlabeled_config(tmp_path, rgb_format)
    missing = scene / ("Camera2.h5" if rgb_format == "h5" else "Camera2/rgb")
    missing.rename(missing.with_name(missing.name + ".unavailable"))
    with pytest.raises(FileNotFoundError, match="Calibration camera Camera2: expected RGB") as error:
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)
    assert str(missing) in str(error.value)
    if rgb_format == "h5":
        assert str(scene / "Camera2.hdf5") in str(error.value)
    assert not list(tmp_path.rglob("*.pkl"))


def test_unlabeled_conversion_rejects_missing_camera_frame(tmp_path):
    """Do not publish image references that fail later in the training loader."""
    cfg, scene = _unlabeled_config(tmp_path)
    (scene / "Camera2/rgb/000000001.jpg").unlink()
    with pytest.raises(ValueError, match="contiguous camera frames"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)
    assert not list(tmp_path.rglob("*.pkl"))


def test_velocity_uses_configured_frame_rate():
    """Ground-truth velocities use the same time base as frame timestamps."""
    velocity = aicity_to_ovpkl._get_object_velocity(
        np.array([[0.1, 0, 0]]), np.array([7]),
        [{"object id": 7, "3d location": [0, 0, 0]}], fps=12.5,
    )
    np.testing.assert_allclose(velocity, [[1.25, 0, 0]])


@pytest.mark.parametrize("rgb_format", ["jpg", "h5"])
def test_unlabeled_artifacts_load_in_tao_pytorch(tmp_path, rgb_format, sparse4d_runtime):
    """Exercise real DS PKLs, lazy index, KITTI cache, image/depth and TAO joins.

    Run with the companion TAO PyTorch checkout on PYTHONPATH. Minimal Data
    Services images without co-training support skip this integration test;
    --require-sparse4d-runtime makes missing capabilities a hard failure.
    """
    import io
    from pathlib import Path
    import tarfile
    dataset_module, transforms = sparse4d_runtime
    from nvidia_tao_ds.annotations.sparse4d.lazy_index import build_lazy_index
    from nvidia_tao_ds.annotations.sparse4d.sidecars import build_rtdetr_archive_sidecar

    cfg, scene = _unlabeled_config(tmp_path, rgb_format)
    aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)
    output = Path(cfg.results_dir)
    pkl_path = output / "train/RealWarehouse_infos_train.pkl"
    split = output / "mixed.txt"
    split.write_text(str(pkl_path) + "\n", encoding="utf-8")
    build_lazy_index(split, num_workers=1)
    for camera in ["Camera1", "Camera2"]:
        label_dir = scene / "rt-detr" / camera
        label_dir.mkdir(parents=True)
        with tarfile.open(label_dir / "labels.tar.gz", "w:gz") as archive:
            for frame_id in range(3):
                # Preserve an explicitly processed frame with no detections.
                labels = b"person 0 0 0 1 1 6 9 0 0 0 0 0 0 0 0.9\n" if frame_id < 2 else b""
                member = tarfile.TarInfo(f"{frame_id:09d}.txt")
                member.size = len(labels)
                archive.addfile(member, io.BytesIO(labels))
    cache_dir = output / "rtdetr"
    build_rtdetr_archive_sidecar(
        scene / "rt-detr", cache_dir / "RealWarehouse__rtdetr2d.npz",
        class_names=["person"], scene_name="RealWarehouse",
    )
    for lazy in (False, True):
        dataset = dataset_module.Omniverse3DDetTrackDataset(
            data_root=str(scene.parent), anno_file=str(split), classes=["person"],
            lazy_load=lazy, with_seq_flag=True,
        )
        assert len(dataset) == 3
        load_cache = transforms.LoadRTDETR2D(cache_dir=cache_dir, class_names=["person"])
        for frame_id in range(3):
            sample = dataset.get_data_info(frame_id)
            assert sample["has_3d_gt"] is False
            assert sample["gt_bboxes_3d"].shape == (0, 9)
            assert sample["timestamp"] == pytest.approx(frame_id / 12.5)
            assert sample["depth_map_filename"] == [None, None]
            sample = transforms.LoadMultiViewImageFromFiles(h5_file=rgb_format == "h5")(sample)
            sample = transforms.LoadDepthMap(default_shape=(12, 16), h5_file=True)(sample)
            assert all(image.shape == (12, 16, 3) for image in sample["img"])
            assert all(np.all(depth == -1) for depth in sample["gt_depth"])
            sample = load_cache(sample)
            assert sample["has_2d_pseudo"] is True
            assert sample["has_3d_gt"] is False
            for camera_index in range(2):
                intrinsic = np.array([[20, 0, 8], [0, 20, 6], [0, 0, 1]])
                extrinsic = np.eye(4)
                extrinsic[0, 3] = camera_index + 1.0
                np.testing.assert_allclose(sample["lidar2img"][camera_index][:3], intrinsic @ extrinsic[:3])
                assert sample["det_boxes_2d"][camera_index].shape == ((1 if frame_id < 2 else 0), 4)


@pytest.mark.parametrize("rgb_format", ["jpg", "h5"])
def test_unlabeled_scene_rejects_unequal_camera_lengths(tmp_path, rgb_format):
    """Unequal clips must not silently truncate the camera with extra frames."""
    cfg, scene = _unlabeled_config(tmp_path, rgb_format)
    if rgb_format == "h5":
        with h5py.File(scene / "Camera2.h5", "a") as stream:
            del stream["rgb/rgb_00002.jpg"]
    else:
        (scene / "Camera2/rgb/000000002.jpg").unlink()
    with pytest.raises(ValueError, match="same number of synchronized frames"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)


def test_unlabeled_h5_rejects_noncontiguous_keys(tmp_path):
    """Do not emit HDF5 references to missing source frame IDs."""
    cfg, scene = _unlabeled_config(tmp_path, "h5")
    with h5py.File(scene / "Camera2.h5", "a") as stream:
        stream.move("rgb/rgb_00001.jpg", "rgb/rgb_00003.jpg")
    with pytest.raises(ValueError, match="contiguous camera frames"):
        aicity_to_ovpkl.convert_aicity_to_ovpkl(cfg)


def test_convert_cli_accepts_unlabeled_spec_and_timing_overrides(tmp_path, monkeypatch):
    """Run the actual Hydra command used by annotations convert, with its schema."""
    from pathlib import Path
    from omegaconf import OmegaConf
    from nvidia_tao_ds.annotations.scripts import convert

    cfg, _ = _unlabeled_config(tmp_path)
    spec_dir = Path(aicity_to_ovpkl.__file__).parents[1] / "experiment_specs"
    spec = OmegaConf.load(spec_dir / "aicity2ovpkl_unlabeled.yaml")
    spec.aicity.root = cfg.aicity.root
    spec.results_dir = cfg.results_dir
    OmegaConf.save(spec, tmp_path / "unlabeled.yaml")
    monkeypatch.setattr(sys, "argv", [
        "convert", "--config-path", str(tmp_path), "--config-name", "unlabeled",
        "aicity.load_annotations=false", "aicity.fps=12.5", "aicity.num_frames=2",
    ])
    convert.main()

    payload = _load_scene(Path(cfg.results_dir), "RealWarehouse")
    assert len(payload["infos"]) == 2
    assert payload["infos"][1]["timestamp"] == pytest.approx(1 / 12.5)
    assert all(info["gt_boxes"] is None for info in payload["infos"])
    assert not list(Path(cfg.results_dir).glob("*.npy"))
