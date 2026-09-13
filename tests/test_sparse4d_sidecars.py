# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tests for Sparse4D 2D sidecar producers."""

import io
import json
import tarfile

import numpy as np
import pytest

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    LTT_2DGT_SCHEMA_VERSION,
    RTDETR_2D_SCHEMA_VERSION,
    load_contract,
)
from nvidia_tao_ds.annotations.sparse4d.sidecars import (
    build_ltt_2dgt_scene,
    build_rtdetr_archive_sidecar,
    write_ltt_2dgt_sidecar,
    write_rtdetr_2d_sidecar,
)


CLASSES = ["person", "forklift"]
CAMERAS = ["CameraA", "CameraB"]


def test_ltt_sidecar_is_deterministic_and_preserves_visibility_semantics(tmp_path):
    """Rows sort by their joins while visible-area weights match legacy TAO."""
    frames = {
        "9": [],
        "4": [
            {
                "object type": "forklift",
                "object id": 8,
                "2d bounding box": {"CameraB": [10, 20, 30, 50]},
            },
            {
                "object type": "person",
                "object id": 7,
                "2d bounding box": {
                    "CameraB": [0, 0, 20, 20],
                    "CameraA": [0, 0, 10, 10],
                },
                "2d bounding box visible": {"CameraA": [0, 0, 5, 5]},
            },
            {
                "object type": "ignore_me",
                "object id": 99,
                "2d bounding box": {"CameraA": [0, 0, 10, 10]},
            },
        ],
    }

    output = write_ltt_2dgt_sidecar(
        tmp_path / "Warehouse__ltt2dgt",
        scene="Warehouse",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frames=frames,
        class_name_map={"ignore_me": None},
        metadata={"anno_version": "v0.1"},
    )
    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert artifact["frame_id"].tolist() == [4, 4, 4]
    assert artifact["instance_id"].tolist() == [7, 7, 8]
    assert artifact["class_id"].tolist() == [0, 0, 1]
    assert artifact["cam"].tolist() == [0, 1, 1]
    np.testing.assert_allclose(
        artifact["box3"],
        [[0, 0, 5, 5], [0, 0, 20, 20], [10, 20, 30, 50]],
    )
    np.testing.assert_allclose(artifact["occ"], [0.25, 0.0, 0.0])
    assert artifact["metadata"]["class_names"] == CLASSES
    assert artifact["metadata"]["cam_names"] == CAMERAS
    assert artifact["metadata"]["num_frames"] == 2
    assert artifact["metadata"]["anno_version"] == "v0.1"


def test_ltt_sidecar_accepts_a_generator_and_empty_annotations(tmp_path):
    """A streamed, annotation-free scene still emits shape-stable arrays."""
    frames = ({"frame_id": frame_id, "annotations": []} for frame_id in [2, 1])

    output = write_ltt_2dgt_sidecar(
        tmp_path / "empty.npz",
        scene="empty",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frames=frames,
    )
    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert artifact["metadata"]["num_frames"] == 2
    assert artifact["metadata"]["num_rows"] == 0
    assert artifact["box2"].shape == (0, 4)
    assert artifact["instance_id"].dtype == np.int64


def test_ltt_sidecar_skips_invalid_full_boxes_and_falls_back_visible_boxes(
    tmp_path,
):
    """Clipped source boxes do not abort production sidecar generation."""
    frames = {
        3: [
            {
                "object type": "person",
                "object id": 1,
                "2d bounding box": {
                    "CameraA": [0, 0, 10, 20],
                    "CameraB": [2, 4, 8, 4],
                },
                "2d bounding box visible": {
                    "CameraA": [0, 0, 5, 0],
                },
            },
            {
                "object type": "forklift",
                "object id": 2,
                "2d bounding box": {"CameraA": [0, 0, np.nan, 20]},
            },
            {
                "object type": "forklift",
                "object id": 3,
                "2d bounding box": {"CameraB": [1, 2, 11, 22]},
                "2d bounding box visible": {
                    "CameraB": [1, 2, 11, np.inf],
                },
            },
        ]
    }

    output = write_ltt_2dgt_sidecar(
        tmp_path / "robust.npz",
        scene="robust",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frames=frames,
    )
    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert artifact["instance_id"].tolist() == [1, 3]
    np.testing.assert_allclose(
        artifact["box2"], [[0, 0, 10, 20], [1, 2, 11, 22]]
    )
    np.testing.assert_allclose(artifact["box3"], artifact["box2"])
    np.testing.assert_allclose(artifact["occ"], [0.0, 0.0])
    assert artifact["metadata"]["num_skipped_invalid_full_boxes"] == 2
    assert artifact["metadata"]["num_visible_box_fallbacks"] == 2


def test_ltt_scene_adapter_reads_raw_json_without_external_data(tmp_path):
    """The file adapter derives cameras and writes the consumer filename."""
    scene = tmp_path / "SceneA"
    ground_truth = scene / "ground_truth_final"
    ground_truth.mkdir(parents=True)
    (scene / "calibration.json").write_text(
        json.dumps(
            {"sensors": [{"type": "camera", "id": "CameraA"}]}
        ),
        encoding="utf-8",
    )
    (ground_truth / "ground_truth_000003.json").write_text(
        json.dumps(
            [
                {
                    "object type": "Human",
                    "object id": 4,
                    "2d bounding box": {"CameraA": [0, 0, 10, 20]},
                }
            ]
        ),
        encoding="utf-8",
    )

    output = build_ltt_2dgt_scene(
        scene,
        tmp_path / "sidecars",
        class_names=CLASSES,
        class_name_map={"Human": "person"},
    )
    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert output.endswith("sidecars/SceneA__ltt2dgt.npz")
    assert artifact["frame_id"].tolist() == [3]
    assert artifact["class_id"].tolist() == [0]
    assert artifact["metadata"]["cam_names"] == ["CameraA"]


@pytest.mark.parametrize(
    "frames,message",
    [
        (
            {0: [{"object type": "robot", "object id": 1, "box2": {}}]},
            "ordered class_names",
        ),
        (
            {
                0: [
                    {
                        "object type": "person",
                        "object id": 1,
                        "box2": {"UnknownCamera": [0, 0, 10, 10]},
                    }
                ]
            },
            "not in camera_names",
        ),
    ],
)
def test_ltt_sidecar_rejects_taxonomy_and_camera_mismatches(
    tmp_path, frames, message
):
    """Unmapped names cannot silently change class or camera indices."""
    with pytest.raises(ValueError, match=message):
        write_ltt_2dgt_sidecar(
            tmp_path / "bad.npz",
            scene="bad",
            class_names=CLASSES,
            camera_names=CAMERAS,
            frames=frames,
        )


def test_rtdetr_sidecar_records_valid_empty_cameras_and_ordered_taxonomy(tmp_path):
    """Empty camera frames are explicit and detection rows sort deterministically."""
    frames = [
        {
            "frame_id": 5,
            "cameras": {
                "CameraB": [],
                "CameraA": [
                    {"class_name": "forklift", "box": [20, 2, 40, 30], "score": 0.8},
                    {"class_name": "person", "box": [1, 2, 10, 20], "score": 0.9},
                ],
            },
        },
        {"frame_id": 3, "cameras": {"CameraA": []}},
    ]

    output = write_rtdetr_2d_sidecar(
        tmp_path / "Warehouse__rtdetr2d.npz",
        scene="Warehouse",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frames=frames,
    )
    artifact = load_contract(output, RTDETR_2D_SCHEMA_VERSION)

    assert artifact["frame_id"].tolist() == [5, 5]
    assert artifact["class_id"].tolist() == [0, 1]
    assert artifact["cam"].tolist() == [0, 0]
    assert artifact["valid_frame_id"].tolist() == [3, 5, 5]
    assert artifact["valid_cam"].tolist() == [0, 0, 1]
    assert artifact["metadata"]["num_valid_frame_cameras"] == 3
    assert artifact["metadata"]["class_names"] == CLASSES
    assert artifact["box"].dtype == np.float32


def test_rtdetr_sidecar_requires_explicit_valid_records(tmp_path):
    """Duplicate frames and invalid detector records fail instead of aliasing."""
    values = {
        "path": tmp_path / "bad.npz",
        "scene": "bad",
        "class_names": CLASSES,
        "camera_names": CAMERAS,
    }
    with pytest.raises(ValueError, match="Duplicate frame_id"):
        write_rtdetr_2d_sidecar(
            **values,
            frames=[
                {"frame_id": 1, "cameras": {}},
                {"frame_id": 1, "cameras": {}},
            ],
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        write_rtdetr_2d_sidecar(
            **values,
            frames={
                1: {
                    "CameraA": [
                        {"class_name": "person", "box": [0, 0, 1, 1], "score": 2}
                    ]
                }
            },
        )


def test_rtdetr_archive_adapter_keeps_empty_label_frames(tmp_path):
    """KITTI tar conversion filters classes but records explicit empty joins."""
    camera_dir = tmp_path / "SceneA" / "rt-detr" / "cam_a"
    camera_dir.mkdir(parents=True)
    with tarfile.open(camera_dir / "labels.tar.gz", "w:gz") as archive:
        for name, value in [
            (
                "labels/frame_000042.txt",
                "Human 0 0 0 1 2 20 30 0 0 0 0 0 0 0 0.9\n"
                "pallet 0 0 0 3 4 40 50 0 0 0 0 0 0 0 0.99\n",
            ),
            ("labels/frame_000043.txt", ""),
        ]:
            payload = value.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    result = build_rtdetr_archive_sidecar(
        camera_dir.parent,
        tmp_path / "SceneA__rtdetr2d.npz",
        class_names=CLASSES,
        class_name_map={"Human": "person", "pallet": None},
        camera_map={"cam_a": "CameraA"},
    )
    artifact = load_contract(result["output_path"], RTDETR_2D_SCHEMA_VERSION)

    assert artifact["frame_id"].tolist() == [42]
    assert artifact["class_id"].tolist() == [0]
    assert artifact["valid_frame_id"].tolist() == [42, 43]
    assert artifact["valid_cam"].tolist() == [0, 0]
    assert artifact["metadata"]["raw_class_counts"] == {"Human": 1, "pallet": 1}
