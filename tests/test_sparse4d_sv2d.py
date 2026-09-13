# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tests for Sparse4D SV2D artifact production."""

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    RTDETR_2D_SCHEMA_VERSION,
    load_contract,
)
from nvidia_tao_ds.annotations.sparse4d.sv2d import (
    build_sv2d_artifacts,
    load_dataset_manifest,
    strip_h5_uri,
    write_split_artifacts,
)


CLASSES = ["forklift", "person"]


def _write_coco(path):
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 2,
                        "file_name": "h5://fixture:image_2.jpg",
                        "width": 100,
                        "height": 50,
                    },
                    {
                        "id": 1,
                        "file_name": "h5://fixture:image_1.jpg",
                        "width": 100,
                        "height": 50,
                    },
                ],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "pallet"},
                    {"id": 3, "name": "forklift"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5]},
                    {"image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
                    {"image_id": 1, "category_id": 3, "bbox": [50, 5, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sv2d_bundle_matches_runtime_pkl_and_safe_cache_contracts(tmp_path):
    """The bundle retains empty frames, HDF5 keys and exact class ordering."""
    coco_path = tmp_path / "annotations.json"
    _write_coco(coco_path)
    dataset = {
        "name": "fixture",
        "scene_name": "SV2D__fixture",
        "coco": str(coco_path),
        "kind": "h5",
        "h5_path": "images.h5",
        "weight": 2,
    }

    result = build_sv2d_artifacts(
        dataset,
        tmp_path / "cache",
        tmp_path / "pkls",
        class_names=CLASSES,
        keep_empty=True,
        suffix="_smoke",
    )

    assert result["scene"] == "SV2D__fixture_smoke"
    assert result["num_frames"] == 2
    assert result["num_detections"] == 2
    assert result["per_class"] == {"forklift": 1, "person": 1}
    assert result["npz_path"].endswith(
        "SV2D__fixture_smoke__rtdetr2d.npz"
    )
    assert result["pkl_path"].endswith(
        "SV2D__fixture_smoke_infos_train.pkl"
    )

    cache = load_contract(result["npz_path"], RTDETR_2D_SCHEMA_VERSION)
    assert cache["class_id"].tolist() == [0, 1]
    np.testing.assert_allclose(
        cache["box"],
        [[960, 108, 1152, 324], [192, 108, 576, 324]],
    )
    assert cache["valid_frame_id"].tolist() == [0, 1]
    assert cache["valid_cam"].tolist() == [0, 0]
    assert cache["metadata"]["class_names"] == CLASSES
    assert cache["metadata"]["virtual_camera"]["width"] == 1920
    assert cache["metadata"]["virtual_camera"]["height"] == 1080
    assert "pallet" not in cache["metadata"]["class_map"]

    with open(result["pkl_path"], "rb") as stream:
        document = pickle.load(stream)
    assert document["metadata"] == {"version": "sv2d_2d_only"}
    assert len(document["infos"]) == 2
    first = document["infos"][0]
    assert first["frame_idx"] == 0
    assert first["gt_boxes"] is None
    assert first["scene_name"] == "SV2D__fixture_smoke"
    assert first["cams"]["cam0"]["data_path"] == (
        str((tmp_path / "images.h5").resolve()),
        "rgb/image_1.jpg",
    )
    assert "depth_map_path" not in first["cams"]["cam0"]
    assert first["cams"]["cam0"]["cam_intrinsic"].shape == (3, 3)
    assert first["cams"]["cam0"]["sensor2world_transform"].shape == (4, 4)


def test_sv2d_bundle_supports_custom_resolution_and_empty_output(tmp_path):
    """Canonical dimensions are configurable and zero-frame output is stable."""
    coco = {
        "images": [
            {"id": "b", "file_name": "b.jpg", "width": 100, "height": 50}
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [],
    }
    dataset = {
        "name": "empty",
        "scene_name": "SV2D__empty",
        "coco": coco,
        "kind": "file",
        "image_root": str(tmp_path / "images"),
    }

    result = build_sv2d_artifacts(
        dataset,
        tmp_path / "cache",
        tmp_path / "pkls",
        class_names=CLASSES,
        canonical_width=200,
        canonical_height=100,
    )
    cache = load_contract(result["npz_path"], RTDETR_2D_SCHEMA_VERSION)

    assert result["num_frames"] == 0
    assert cache["box"].shape == (0, 4)
    assert cache["valid_frame_id"].shape == (0,)
    assert cache["metadata"]["virtual_camera"]["width"] == 200
    assert cache["metadata"]["virtual_camera"]["height"] == 100
    with open(result["pkl_path"], "rb") as stream:
        assert pickle.load(stream)["infos"] == []


def test_sv2d_bundle_requires_explicit_taxonomy_mapping(tmp_path):
    """An unknown source class cannot silently acquire an integer label."""
    dataset = {
        "name": "mapped",
        "scene_name": "SV2D__mapped",
        "coco": {
            "images": [
                {"id": 1, "file_name": "1.jpg", "width": 100, "height": 50}
            ],
            "categories": [{"id": 1, "name": "worker"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]}
            ],
        },
        "kind": "file",
        "image_root": str(tmp_path),
    }
    with pytest.raises(ValueError, match="ordered class_names"):
        build_sv2d_artifacts(
            dataset,
            tmp_path / "cache",
            tmp_path / "pkls",
            class_names=CLASSES,
        )

    result = build_sv2d_artifacts(
        dataset,
        tmp_path / "cache",
        tmp_path / "pkls",
        class_names=CLASSES,
        category_map={"worker": "person"},
    )
    cache = load_contract(result["npz_path"], RTDETR_2D_SCHEMA_VERSION)
    assert cache["class_id"].tolist() == [1]
    assert cache["metadata"]["class_map"] == {"worker": "person"}


def test_sv2d_manifest_resolves_relative_sources(tmp_path):
    """Dataset manifests resolve portable paths relative to their own file."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "fixture",
                        "scene_name": "SV2D__fixture",
                        "coco": "labels/coco.json",
                        "kind": "file",
                        "image_root": "images",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    datasets = load_dataset_manifest(manifest)

    assert datasets == [
        {
            "name": "fixture",
            "scene_name": "SV2D__fixture",
            "coco": str((tmp_path / "labels/coco.json").resolve()),
            "kind": "file",
            "image_root": str((tmp_path / "images").resolve()),
            "weight": 1.0,
        }
    ]


def test_sv2d_split_and_weights_are_unique_and_atomic(tmp_path):
    """The split references each PKL once and preserves advisory weights."""
    results = [
        {
            "dataset": "b",
            "scene": "SceneB",
            "pkl_path": str(tmp_path / "SceneB.pkl"),
        },
        {
            "dataset": "a",
            "scene": "SceneA",
            "pkl_path": str(tmp_path / "SceneA.pkl"),
        },
    ]
    datasets = [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 2}]

    outputs = write_split_artifacts(
        results, datasets, tmp_path / "sv2d_train_split_smoke.txt"
    )

    assert Path(outputs["split_path"]).read_text(encoding="utf-8").splitlines() == [
        str((tmp_path / "SceneB.pkl").resolve()),
        str((tmp_path / "SceneA.pkl").resolve()),
    ]
    assert json.loads(Path(outputs["weights_path"]).read_text(encoding="utf-8")) == {
        "SceneA": 0.5,
        "SceneB": 2.0,
    }
    assert outputs["weights_path"].endswith(
        "sv2d_train_split_smoke.sv2d_weights.json"
    )

    with pytest.raises(ValueError, match="Duplicate SV2D result scene"):
        write_split_artifacts(
            [results[0], dict(results[0])],
            datasets,
            tmp_path / "duplicate.txt",
        )


def test_h5_uri_rejects_parent_traversal():
    """HDF5 keys cannot escape their expected rgb namespace."""
    assert strip_h5_uri("h5://tag:folder/image.jpg") == "folder/image.jpg"
    with pytest.raises(ValueError, match="traverse"):
        strip_h5_uri("h5://tag:../secret")
