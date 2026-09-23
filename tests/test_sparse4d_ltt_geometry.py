# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tests for Sparse4D Loose-to-Tight geometry extraction."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nvidia_tao_ds.annotations.sparse4d import ltt_geometry
from nvidia_tao_ds.annotations.sparse4d.contracts import (
    LTT_DATA_SCHEMA_VERSION,
    load_contract,
)
from nvidia_tao_ds.annotations.sparse4d.ltt_geometry import (
    I_VISIBILITY,
    SL_IMAGE_WH,
    build_name_to_id,
    extract_ltt_data,
    iter_gt_frames,
    resolve_scene_paths,
    scene_from_split_line,
)


CLASSES = ["person", "forklift"]


def _annotation(x_position=0.0, object_type="person"):
    """Return one projected 3D annotation in front of an identity camera."""
    return {
        "object id": 7,
        "object type": object_type,
        "3d location": [x_position, 0.0, 10.0],
        "3d bounding box scale": [2.0, 2.0, 2.0],
        "3d bounding box rotation": [0.0, 0.0, 0.2],
        "2d bounding box": {"cam0": [35.0, 35.0, 65.0, 65.0]},
        "2d bounding box visible": {"cam0": [40.0, 40.0, 60.0, 60.0]},
    }


def _write_scene(root: Path, name="scene_a", per_frame=False):
    """Create a minimal raw scene with calibration and two frames."""
    scene = root / name
    scene.mkdir(parents=True)
    calibration = {
        "sensors": [
            {
                "id": "cam0",
                "type": "camera",
                "intrinsicMatrix": [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
                "extrinsicMatrix": np.eye(4).tolist(),
                "width": 100,
                "height": 100,
            }
        ]
    }
    (scene / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    frames = {"1": [_annotation(0.2)], "0": [_annotation(0.0)]}
    if per_frame:
        output = scene / "ground_truth_final"
        output.mkdir()
        for frame_id, annotations in frames.items():
            (output / f"ground_truth_{int(frame_id):05d}.json").write_text(
                json.dumps(annotations), encoding="utf-8"
            )
    else:
        (scene / "ground_truth.json").write_text(
            json.dumps(frames), encoding="utf-8"
        )
    return scene


@pytest.mark.parametrize("per_frame", [False, True])
def test_extract_ltt_data_matches_runtime_contract(tmp_path, per_frame):
    """Generate an LTT cache without importing torch or external datasets."""
    scene = _write_scene(tmp_path, per_frame=per_frame)
    name_to_id = build_name_to_id(CLASSES)

    metadata = extract_ltt_data(
        [scene],
        tmp_path / "ltt",
        CLASSES,
        name_to_id,
        frame_stride=1,
        max_per_class=10,
        seed=17,
    )
    artifact = load_contract(
        metadata["artifact_path"], LTT_DATA_SCHEMA_VERSION
    )

    assert artifact["packed"].shape == (2, 18)
    assert artifact["packed"].dtype == np.float32
    assert artifact["class_id"].dtype == np.int16
    assert artifact["class_id"].tolist() == [0, 0]
    assert artifact["group_id"].dtype == np.int64
    assert sorted(artifact["group_id"].tolist()) == [0, 1]
    assert artifact["packed"][:, SL_IMAGE_WH].tolist() == [
        [100.0, 100.0],
        [100.0, 100.0],
    ]
    assert artifact["packed"][:, I_VISIBILITY].tolist() == pytest.approx(
        [4.0 / 9.0, 4.0 / 9.0]
    )
    assert artifact["metadata"]["class_names"] == CLASSES
    assert artifact["metadata"]["num_frames_used"] == 2
    assert artifact["metadata"]["num_samples"] == 2


def test_scene_resolution_from_split_is_deduplicated_and_strict(tmp_path):
    """Resolve prefixed/grouped training PKLs back to one raw scene."""
    scene = _write_scene(tmp_path / "raw")
    split = tmp_path / "train.txt"
    split.write_text(
        "/annotations/CTv4.0__scene_a+group0_infos_train.pkl\n"
        "/annotations/CTv4.0__scene_a+group1_infos_train.pkl\n",
        encoding="utf-8",
    )

    assert scene_from_split_line(
        "/annotations/CTv4.0__scene_a+group0_infos_train.pkl"
    ) == "scene_a"
    assert resolve_scene_paths(data_root=tmp_path / "raw", train_split=split) == [
        scene.resolve()
    ]

    split.write_text("/annotations/missing_infos_train.pkl\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_scene_paths(data_root=tmp_path / "raw", train_split=split)


def test_taxonomy_aliases_preserve_order_and_reject_conflicts():
    """Keep class IDs fixed while accepting declared source subclasses."""
    mapping = build_name_to_id(
        CLASSES,
        {"person": ["adult"], "forklift": ["fork_truck"]},
    )
    assert mapping == {
        "person": 0,
        "forklift": 1,
        "adult": 0,
        "fork_truck": 1,
    }
    with pytest.raises(ValueError, match="parent"):
        build_name_to_id(CLASSES, {"unknown": ["alias"]})
    with pytest.raises(ValueError, match="more than one"):
        build_name_to_id(
            CLASSES,
            {"person": ["forklift"]},
        )


def test_frame_iteration_fallback_preserves_source_order(tmp_path, monkeypatch):
    """Match streaming order in the standard-library fallback."""
    scene = _write_scene(tmp_path)
    monkeypatch.setattr(ltt_geometry, "_ijson", None)

    frames = list(iter_gt_frames(scene, frame_stride=1, max_frames=1))

    assert len(frames) == 1
    assert frames[0][0] == 1


def test_frame_iteration_rejects_per_frame_numeric_aliases(tmp_path):
    """Reject filenames that normalize to the same numeric frame ID."""
    frame_dir = tmp_path / "scene" / "ground_truth_final"
    frame_dir.mkdir(parents=True)
    for filename in ("ground_truth_1.json", "ground_truth_00001.json"):
        (frame_dir / filename).write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate normalized frame ID 1"):
        list(iter_gt_frames(frame_dir.parent, max_frames=1))


def test_frame_iteration_streams_monolithic_json(tmp_path, monkeypatch):
    """Use optional ijson without materializing the whole scene document."""
    scene = _write_scene(tmp_path)
    streamed = []

    def kvitems(stream, prefix):
        assert prefix == ""
        streamed.append(stream.name)
        return json.load(stream).items()

    monkeypatch.setattr(
        ltt_geometry,
        "_ijson",
        SimpleNamespace(kvitems=kvitems),
    )

    frames = list(iter_gt_frames(scene, frame_stride=1, max_frames=1))

    assert streamed == [str(scene / "ground_truth.json")]
    assert frames[0][0] == 1


@pytest.mark.parametrize("streamed", [False, True])
def test_frame_iteration_rejects_monolithic_numeric_aliases(
    tmp_path,
    monkeypatch,
    streamed,
):
    """Reject colliding numeric keys in stdlib and streamed readers."""
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "ground_truth.json").write_text(
        '{"1": [], "01": []}',
        encoding="utf-8",
    )

    def kvitems(stream, prefix):
        assert prefix == ""
        return json.load(stream).items()

    parser = SimpleNamespace(kvitems=kvitems) if streamed else None
    monkeypatch.setattr(ltt_geometry, "_ijson", parser)

    with pytest.raises(ValueError, match="Duplicate normalized frame ID 1"):
        list(iter_gt_frames(scene, max_frames=1))


def test_extraction_rejects_empty_result(tmp_path):
    """Fail loudly when no configured class can produce a training row."""
    scene = _write_scene(tmp_path)

    with pytest.raises(ValueError, match="No Loose-to-Tight samples"):
        extract_ltt_data(
            [scene],
            tmp_path / "empty.npz",
            CLASSES,
            build_name_to_id(CLASSES),
            metadata={"source": "fixture"},
            min_visibility=0.9,
        )


def test_geometry_values_for_axis_aligned_box(tmp_path):
    """Pin projection and camera distance numerically, not just array shapes."""
    scene = _write_scene(tmp_path)
    annotation = _annotation()
    annotation["3d bounding box rotation"] = [0.0, 0.0, 0.0]
    (scene / "ground_truth.json").write_text(
        json.dumps({"0": [annotation]}), encoding="utf-8"
    )
    summary = extract_ltt_data(
        [scene], tmp_path / "numeric.npz", CLASSES,
        build_name_to_id(CLASSES), frame_stride=1,
    )
    row = load_contract(summary["artifact_path"], LTT_DATA_SCHEMA_VERSION)["packed"][0]
    np.testing.assert_allclose(row[ltt_geometry.SL_EXTENT], [2, 2, 2])
    np.testing.assert_allclose(
        row[ltt_geometry.SL_LOOSE], [38.888889, 38.888889, 61.111111, 61.111111], atol=1e-5
    )
    np.testing.assert_allclose(row[ltt_geometry.SL_TIGHT], [35, 35, 65, 65])
    assert row[ltt_geometry.I_DISTANCE] == pytest.approx(10.0)
    assert row[I_VISIBILITY] == pytest.approx(4 / 9)


def test_projection_only_calibration_is_rejected_for_ltt_geometry(tmp_path):
    """A projection matrix cannot stand in for a metric rigid transform."""
    scene = _write_scene(tmp_path)
    calibration = {"sensors": [{
        "id": "cam0", "type": "camera",
        "cameraMatrix": [[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]],
    }]}
    (scene / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(ValueError, match="separate intrinsics"):
        ltt_geometry.load_scene_calibration(scene)
