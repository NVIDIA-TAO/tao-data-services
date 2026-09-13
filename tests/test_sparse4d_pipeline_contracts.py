# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-stage contract tests for Sparse4D data preparation."""

import io
import json
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace

from omegaconf import OmegaConf

from nvidia_tao_ds.annotations.scripts.sparse4d_prepare import run_operation
from nvidia_tao_ds.annotations.sparse4d import ltt_geometry
from nvidia_tao_ds.annotations.sparse4d.contracts import (
    LTT_DATA_SCHEMA_VERSION,
    RTDETR_2D_SCHEMA_VERSION,
    load_contract,
)
from nvidia_tao_ds.annotations.sparse4d.ltt_geometry import iter_gt_frames
from nvidia_tao_ds.config.annotations.sparse4d_prepare_config import (
    Sparse4DPrepareConfig,
)


def _config(tmp_path: Path):
    """Return a mutable structured config rooted in the test directory."""
    cfg = OmegaConf.structured(Sparse4DPrepareConfig)
    cfg.results_dir = str(tmp_path / "results")
    return cfg


def _write_projectable_scene(path: Path):
    """Write one raw scene with a visible 3D object and calibrated camera."""
    path.mkdir()
    (path / "calibration.json").write_text(
        json.dumps(
            {
                "sensors": [
                    {
                        "id": "cam0",
                        "type": "camera",
                        "intrinsicMatrix": [
                            [100.0, 0.0, 50.0],
                            [0.0, 100.0, 50.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "extrinsicMatrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "width": 100,
                        "height": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (path / "ground_truth.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "object id": 7,
                        "object type": "Human",
                        "3d location": [0.0, 0.0, 10.0],
                        "3d bounding box scale": [2.0, 2.0, 2.0],
                        "3d bounding box rotation": [0.0, 0.0, 0.2],
                        "2d bounding box": {"cam0": [35.0, 35.0, 65.0, 65.0]},
                        "2d bounding box visible": {
                            "cam0": [40.0, 40.0, 60.0, 60.0]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_label_archive(camera_dir: Path):
    """Write a tiny KITTI archive with one detection and one empty frame."""
    camera_dir.mkdir(parents=True)
    with tarfile.open(camera_dir / "labels.tar.gz", "w:gz") as archive:
        members = {
            "labels/frame_000042.txt": (
                "Human 0 0 0 1 2 20 30 0 0 0 0 0 0 0 0.9\n"
            ),
            "labels/frame_000043.txt": "",
        }
        for name, value in members.items():
            payload = value.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_ltt_data_operation_writes_runtime_contract(tmp_path):
    """Exercise config dispatch, taxonomy aliases, geometry, and safe output."""
    scene = tmp_path / "scene"
    _write_projectable_scene(scene)
    cfg = _config(tmp_path)
    cfg.operation = "ltt_data"
    cfg.ltt_data.selection.scene_dirs = [str(scene)]
    cfg.ltt_data.output_path = str(tmp_path / "ltt_data.npz")
    cfg.ltt_data.frame_stride = 1
    cfg.ltt_data.max_per_class = 10

    summary = run_operation(cfg)
    artifact = load_contract(
        summary["artifact_path"],
        LTT_DATA_SCHEMA_VERSION,
    )

    assert summary["num_samples"] == 1
    assert summary["seen_per_class"]["person"] == 1
    assert artifact["packed"].shape == (1, 18)
    assert artifact["class_id"].tolist() == [0]
    assert artifact["group_id"].tolist() == [0]
    assert artifact["metadata"]["scenes"] == ["scene"]


def test_rtdetr_operation_preserves_empty_frame_validity(tmp_path):
    """Exercise archive dispatch with aliases and background-only frames."""
    input_dir = tmp_path / "scene" / "rt-detr"
    _write_label_archive(input_dir / "cam_a")
    cfg = _config(tmp_path)
    cfg.operation = "rtdetr_2d"
    cfg.rtdetr_2d.input_dir = str(input_dir)
    cfg.rtdetr_2d.output_path = str(tmp_path / "scene__rtdetr2d.npz")
    cfg.rtdetr_2d.scene_name = "scene"
    cfg.rtdetr_2d.camera_map = {"cam_a": "CameraA"}

    summary = run_operation(cfg)
    artifact = load_contract(
        summary["output_path"],
        RTDETR_2D_SCHEMA_VERSION,
    )

    assert summary["cam_names"] == ["CameraA"]
    assert artifact["frame_id"].tolist() == [42]
    assert artifact["class_id"].tolist() == [0]
    assert artifact["valid_frame_id"].tolist() == [42, 43]
    assert artifact["valid_cam"].tolist() == [0, 0]
    assert artifact["metadata"]["class_map"] == {"Human": "person"}


def test_sv2d_output_split_is_lazy_index_compatible(tmp_path):
    """Feed generated GT-less PKLs directly into the lazy-index operation."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    coco = tmp_path / "coco.json"
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "frame_1.jpg",
                        "width": 100,
                        "height": 50,
                    },
                    {
                        "id": 2,
                        "file_name": "frame_2.jpg",
                        "width": 100,
                        "height": 50,
                    },
                ],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 5, 20, 10],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "fixture",
                        "scene_name": "SV2D__fixture",
                        "kind": "file",
                        "coco": str(coco),
                        "image_root": str(image_root),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = _config(tmp_path)
    cfg.operation = "sv2d"
    cfg.sv2d.manifest_path = str(manifest)
    cfg.sv2d.cache_dir = str(tmp_path / "cache")
    cfg.sv2d.pkl_dir = str(tmp_path / "pkls")
    cfg.sv2d.keep_empty = True

    sv2d_summary = run_operation(cfg)
    split_path = sv2d_summary["split_artifacts"]["split_path"]
    cfg.operation = "lazy_index"
    cfg.lazy_index.annotation_source = split_path
    cfg.lazy_index.workers = 1
    index_summary = run_operation(cfg)

    assert index_summary["num_pkls"] == 1
    assert index_summary["num_frames"] == 2
    with open(index_summary["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["metadata"] == {"version": "sv2d_2d_only"}
    assert [entry["local_idx"] for entry in index["frame_index"]] == [0, 1]
    assert {entry["scene_name"] for entry in index["frame_index"]} == {
        "SV2D__fixture"
    }
    assert list(index["pkl_cam_counts"].values()) == [1]


def test_monolithic_sampling_is_independent_of_optional_parser(
    tmp_path,
    monkeypatch,
):
    """Keep source-order stride and limit semantics identical with ijson."""
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "ground_truth.json").write_text(
        '{"5": [], "4": [{"source": "first"}], '
        '"2": [{"source": "second"}], "6": []}',
        encoding="utf-8",
    )

    monkeypatch.setattr(ltt_geometry, "_ijson", None)
    fallback = list(iter_gt_frames(scene, frame_stride=2, max_frames=2))

    def kvitems(stream, prefix):
        assert prefix == ""
        return json.load(stream).items()

    monkeypatch.setattr(
        ltt_geometry,
        "_ijson",
        SimpleNamespace(kvitems=kvitems),
    )
    streamed = list(iter_gt_frames(scene, frame_stride=2, max_frames=2))

    expected = [
        (4, [{"source": "first"}]),
        (2, [{"source": "second"}]),
    ]
    assert fallback == expected
    assert streamed == expected
