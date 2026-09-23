# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for versioned Sparse4D data-service artifacts."""

import numpy as np
import pytest

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    LTT_2DGT_SCHEMA_VERSION,
    LTT_DATA_SCHEMA_VERSION,
    RTDETR_2D_SCHEMA_VERSION,
    decode_metadata,
    encode_metadata,
    load_contract,
    validate_class_names,
    validate_scene_name,
    write_ltt_2dgt,
    write_ltt_data,
    write_rtdetr_2d,
)


CLASSES = ["person", "gr1_t2"]
CAMERAS = ["Camera1", "Camera2"]


def _valid_packed(row_count=1):
    """Return valid fixed-width Loose-to-Tight geometry rows."""
    row = np.asarray(
        [
            2.0, 4.0, 1.5,
            0.1, 0.2, 0.3, 10.0,
            0.0, 0.0, 100.0, 80.0,
            10.0, 12.0, 90.0, 70.0,
            1920.0, 1080.0, 0.75,
        ],
        dtype=np.float32,
    )
    return np.repeat(row[None, :], row_count, axis=0)


def _read_npz(path):
    """Read all arrays from a trusted test fixture without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_npz(path, arrays, metadata=None):
    """Write a deliberately mutable NPZ fixture for negative load tests."""
    values = dict(arrays)
    if metadata is not None:
        values["_meta"] = encode_metadata(metadata)
    np.savez_compressed(path, **values)


def _valid_ltt_2dgt(tmp_path, name="valid_ltt_2dgt.npz"):
    """Write one valid visible-2D supervision fixture."""
    return write_ltt_2dgt(
        tmp_path / name,
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[4, 4],
        instance_id=[10, 11],
        class_id=[0, 1],
        cam=[0, 1],
        box2=[[1, 2, 9, 12], [2, 3, 8, 10]],
        box3=[[2, 3, 8, 11], [2, 3, 8, 10]],
        occ=[0.75, 1.0],
    )


def _valid_ltt_data(tmp_path, name="valid_ltt_data.npz"):
    """Write one valid packed-geometry fixture."""
    return write_ltt_data(
        tmp_path / name,
        class_names=CLASSES,
        packed=_valid_packed(2),
        class_id=[0, 1],
        group_id=[7, 8],
    )


def _valid_rtdetr(tmp_path, name="valid_rtdetr.npz"):
    """Write one valid pseudo-label fixture with explicit validity rows."""
    return write_rtdetr_2d(
        tmp_path / name,
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[8],
        cam=[0],
        class_id=[1],
        box=[[10, 20, 30, 40]],
        score=[0.8],
        valid_frame_id=[8, 9],
        valid_cam=[0, 1],
        metadata={"num_valid_frame_cameras": 2},
    )


def test_ltt_2dgt_round_trip_is_pickle_free(tmp_path):
    """Write the exact safe numeric schema consumed by Sparse4D."""
    output = write_ltt_2dgt(
        tmp_path / "scene__ltt2dgt",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[4, 4],
        instance_id=[10, 11],
        class_id=[0, 1],
        cam=[0, 1],
        box2=[[1, 2, 9, 12], [2, 3, 8, 10]],
        box3=[[2, 3, 8, 11], [2, 3, 8, 10]],
        occ=[0.75, 1.0],
        metadata={"anno_version": "v0.1"},
    )

    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert output.endswith(".npz")
    assert artifact["metadata"] == {
        "anno_version": "v0.1",
        "cam_names": CAMERAS,
        "class_names": CLASSES,
        "num_rows": 2,
        "scene": "scene",
        "schema_version": LTT_2DGT_SCHEMA_VERSION,
    }
    assert artifact["frame_id"].dtype == np.int32
    assert artifact["instance_id"].dtype == np.int64
    assert artifact["box2"].shape == (2, 4)
    assert artifact["occ"].tolist() == pytest.approx([0.75, 1.0])
    with np.load(output, allow_pickle=False) as archive:
        assert archive["_meta"].dtype == np.uint8


def test_ltt_2dgt_allows_an_explicit_empty_scene(tmp_path):
    """Keep empty scenes representable without object arrays."""
    output = write_ltt_2dgt(
        tmp_path / "empty.npz",
        scene="empty",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[],
        instance_id=[],
        class_id=[],
        cam=[],
        box2=[],
        box3=[],
        occ=[],
    )

    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert artifact["metadata"]["num_rows"] == 0
    assert artifact["box2"].shape == (0, 4)
    assert artifact["class_id"].dtype == np.int16


def test_ltt_data_requires_grouped_finite_geometry(tmp_path):
    """Persist source-frame groups so train/validation rows cannot leak."""
    output = write_ltt_data(
        tmp_path / "ltt_training.npz",
        class_names=CLASSES,
        packed=_valid_packed(2),
        class_id=[0, 1],
        group_id=[7, 8],
        metadata={"source": "fixture"},
    )

    artifact = load_contract(output, LTT_DATA_SCHEMA_VERSION)

    assert artifact["packed"].shape == (2, 18)
    assert artifact["class_id"].dtype == np.int16
    assert artifact["group_id"].dtype == np.int64
    assert artifact["group_id"].tolist() == [7, 8]
    assert artifact["metadata"]["source"] == "fixture"

    with pytest.raises(ValueError, match="equal row counts"):
        write_ltt_data(
            tmp_path / "bad.npz",
            class_names=CLASSES,
            packed=np.zeros((2, 18)),
            class_id=[0],
            group_id=[0, 1],
        )


def test_rtdetr_cache_records_valid_empty_frame_camera_pairs(tmp_path):
    """Differentiate an empty detector result from missing inference."""
    output = write_rtdetr_2d(
        tmp_path / "scene__rtdetr2d.npz",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[8],
        cam=[0],
        class_id=[1],
        box=[[10, 20, 30, 40]],
        score=[0.8],
        valid_frame_id=[8, 8, 9, 9],
        valid_cam=[0, 1, 0, 1],
    )

    artifact = load_contract(output, RTDETR_2D_SCHEMA_VERSION)

    assert artifact["valid_frame_id"].tolist() == [8, 8, 9, 9]
    assert artifact["valid_cam"].tolist() == [0, 1, 0, 1]
    assert artifact["metadata"]["num_rows"] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"class_names": ["person", "person"]}, "unique"),
        ({"scene": "../escape"}, "filename component"),
        ({"class_id": [2]}, "outside class_names"),
        ({"occ": [1.1]}, r"in \[0, 1\]"),
    ],
)
def test_ltt_2dgt_rejects_ambiguous_contracts(tmp_path, kwargs, message):
    """Reject values that would be silently misinterpreted at training time."""
    values = {
        "scene": "scene",
        "class_names": CLASSES,
        "camera_names": CAMERAS,
        "frame_id": [1],
        "instance_id": [2],
        "class_id": [0],
        "cam": [0],
        "box2": [[1, 2, 3, 4]],
        "box3": [[1, 2, 3, 4]],
        "occ": [1.0],
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        write_ltt_2dgt(tmp_path / "bad.npz", **values)


def test_rtdetr_validity_columns_are_atomic(tmp_path):
    """Require both validity columns and validate camera indices."""
    values = {
        "scene": "scene",
        "class_names": CLASSES,
        "camera_names": CAMERAS,
        "frame_id": [],
        "cam": [],
        "class_id": [],
        "box": [],
        "score": [],
    }
    with pytest.raises(ValueError, match="supplied together"):
        write_rtdetr_2d(
            tmp_path / "one-column.npz", valid_frame_id=[1], **values
        )
    with pytest.raises(ValueError, match="outside cam_names"):
        write_rtdetr_2d(
            tmp_path / "bad-camera.npz",
            valid_frame_id=[1],
            valid_cam=[2],
            **values,
        )


def test_rtdetr_writer_requires_detections_to_be_marked_valid(tmp_path):
    """Reject inconsistent validity declarations before writing an artifact."""
    with pytest.raises(ValueError, match="must be marked valid"):
        write_rtdetr_2d(
            tmp_path / "missing-valid-pair.npz",
            scene="scene",
            class_names=CLASSES,
            camera_names=CAMERAS,
            frame_id=[8],
            cam=[0],
            class_id=[1],
            box=[[10, 20, 30, 40]],
            score=[0.8],
            valid_frame_id=[9],
            valid_cam=[0],
        )


def test_rtdetr_writer_records_validity_count(tmp_path):
    """Record the validity row count from the arrays, not caller metadata."""
    output = write_rtdetr_2d(
        tmp_path / "validity-count.npz",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[],
        cam=[],
        class_id=[],
        box=[],
        score=[],
        valid_frame_id=[8, 9],
        valid_cam=[0, 1],
        metadata={"num_valid_frame_cameras": 99},
    )

    artifact = load_contract(output, RTDETR_2D_SCHEMA_VERSION)

    assert artifact["metadata"]["num_valid_frame_cameras"] == 2

    legacy_output = write_rtdetr_2d(
        tmp_path / "legacy-metadata.npz",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[],
        cam=[],
        class_id=[],
        box=[],
        score=[],
        metadata={"num_valid_frame_cameras": 99},
    )
    legacy_artifact = load_contract(
        legacy_output, RTDETR_2D_SCHEMA_VERSION
    )
    assert "num_valid_frame_cameras" not in legacy_artifact["metadata"]


def test_rtdetr_writer_requires_unique_validity_pairs(tmp_path):
    """Keep the recorded validity count equal to distinct processed pairs."""
    with pytest.raises(ValueError, match="pairs must be unique"):
        write_rtdetr_2d(
            tmp_path / "duplicate-validity.npz",
            scene="scene",
            class_names=CLASSES,
            camera_names=CAMERAS,
            frame_id=[],
            cam=[],
            class_id=[],
            box=[],
            score=[],
            valid_frame_id=[8, 8],
            valid_cam=[0, 0],
        )


def test_taxonomy_and_scene_validation_are_strict():
    """Keep ordered IDs and artifact filenames unambiguous."""
    assert validate_class_names(CLASSES) == tuple(CLASSES)
    assert validate_scene_name("Warehouse_A") == "Warehouse_A"
    with pytest.raises(ValueError, match="at least one"):
        validate_class_names([])
    with pytest.raises(ValueError, match="non-empty"):
        validate_class_names(["person", ""])
    with pytest.raises(ValueError, match="filename component"):
        validate_scene_name("a\\b")
    with pytest.raises(ValueError, match="sequence of names"):
        validate_class_names("person")


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("frame_id", [True], "boolean"),
        ("instance_id", [1.25], "integer-like"),
        ("frame_id", [np.inf], "finite"),
        ("class_id", [32768], "int16 range"),
    ],
)
def test_integer_vectors_reject_lossy_values_before_conversion(
    tmp_path, field, value, message
):
    """Reject booleans, fractional/non-finite values, and dtype overflow."""
    values = {
        "scene": "scene",
        "class_names": CLASSES,
        "camera_names": CAMERAS,
        "frame_id": [1],
        "instance_id": [2],
        "class_id": [0],
        "cam": [0],
        "box2": [[1, 2, 3, 4]],
        "box3": [[1, 2, 3, 4]],
        "occ": [1.0],
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        write_ltt_2dgt(tmp_path / "bad-integer.npz", **values)


def test_integer_vectors_accept_exact_integer_like_values(tmp_path):
    """Accept real numeric values that represent exact in-range integers."""
    output = write_ltt_2dgt(
        tmp_path / "integer-like.npz",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[4.0],
        instance_id=[10.0],
        class_id=[0.0],
        cam=[1.0],
        box2=[[1, 2, 9, 12]],
        box3=[[2, 3, 8, 11]],
        occ=[0.75],
    )

    artifact = load_contract(output, LTT_2DGT_SCHEMA_VERSION)

    assert artifact["frame_id"].tolist() == [4]
    assert artifact["instance_id"].tolist() == [10]
    assert artifact["class_id"].tolist() == [0]
    assert artifact["cam"].tolist() == [1]


def test_load_contract_rejects_missing_and_unexpected_keys(tmp_path):
    """Require the schema key set and avoid loading an unexpected object array."""
    output = _valid_ltt_2dgt(tmp_path)
    arrays = _read_npz(output)

    missing = dict(arrays)
    missing.pop("occ")
    missing_path = tmp_path / "missing.npz"
    _write_npz(missing_path, missing)
    with pytest.raises(ValueError, match="missing NPZ keys.*occ"):
        load_contract(missing_path, LTT_2DGT_SCHEMA_VERSION)

    unexpected = dict(arrays)
    unexpected["unsafe"] = np.asarray([{"payload": "object"}], dtype=object)
    unexpected_path = tmp_path / "unexpected.npz"
    _write_npz(unexpected_path, unexpected)
    with pytest.raises(ValueError, match="unexpected NPZ keys.*unsafe"):
        load_contract(unexpected_path, LTT_2DGT_SCHEMA_VERSION)


def test_load_contract_rejects_object_data_in_a_contract_key(tmp_path):
    """Keep allow_pickle disabled even when an allowed key stores objects."""
    output = _valid_ltt_2dgt(tmp_path)
    arrays = _read_npz(output)
    arrays["frame_id"] = np.asarray([object(), object()], dtype=object)
    bad_path = tmp_path / "object-frame-id.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match="frame_id.*unsafe or unreadable"):
        load_contract(bad_path, LTT_2DGT_SCHEMA_VERSION)


def test_load_contract_requires_exact_dtype_shape_and_row_counts(tmp_path):
    """Reject coercible but non-canonical arrays and structurally bad rows."""
    output = _valid_ltt_2dgt(tmp_path)
    original = _read_npz(output)

    wrong_dtype = dict(original)
    wrong_dtype["frame_id"] = original["frame_id"].astype(np.int64)
    dtype_path = tmp_path / "dtype.npz"
    _write_npz(dtype_path, wrong_dtype)
    with pytest.raises(ValueError, match="frame_id must have dtype int32"):
        load_contract(dtype_path, LTT_2DGT_SCHEMA_VERSION)

    wrong_shape = dict(original)
    wrong_shape["box2"] = original["box2"].reshape(2, 2, 2)
    shape_path = tmp_path / "shape.npz"
    _write_npz(shape_path, wrong_shape)
    with pytest.raises(ValueError, match=r"box2 must have shape \(N, 4\)"):
        load_contract(shape_path, LTT_2DGT_SCHEMA_VERSION)

    wrong_rows = dict(original)
    wrong_rows["occ"] = original["occ"][:1]
    rows_path = tmp_path / "rows.npz"
    _write_npz(rows_path, wrong_rows)
    with pytest.raises(ValueError, match="equal row counts"):
        load_contract(rows_path, LTT_2DGT_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"num_rows": 3}, "does not match artifact rows"),
        ({"num_rows": True}, "non-negative integer"),
        ({"class_names": "person"}, "ordered JSON list"),
        ({"cam_names": ["Camera1", "Camera1"]}, "unique"),
        ({"scene": "../escape"}, "metadata scene is invalid"),
    ],
)
def test_load_contract_validates_required_metadata(
    tmp_path, updates, message
):
    """Validate metadata row count, taxonomy, cameras, and scene identity."""
    output = _valid_ltt_2dgt(tmp_path)
    arrays = _read_npz(output)
    metadata = decode_metadata(arrays.pop("_meta"))
    metadata.update(updates)
    bad_path = tmp_path / "bad-metadata.npz"
    _write_npz(bad_path, arrays, metadata)

    with pytest.raises(ValueError, match=message):
        load_contract(bad_path, LTT_2DGT_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "column, index, value, message",
    [
        ("class_id", 0, 2, "outside class_names"),
        ("cam", 0, 2, "outside cam_names"),
        ("box2", (0, 2), 0.0, "ordered"),
        ("box3", (0, 0), np.nan, "finite"),
        ("occ", 0, 1.1, r"in \[0, 1\]"),
    ],
)
def test_load_contract_validates_ltt_2dgt_values(
    tmp_path, column, index, value, message
):
    """Validate taxonomy indices, cameras, boxes, and occurrence weights."""
    output = _valid_ltt_2dgt(tmp_path)
    arrays = _read_npz(output)
    arrays[column][index] = value
    bad_path = tmp_path / "bad-ltt-value.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match=message):
        load_contract(bad_path, LTT_2DGT_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "column, index, value, message",
    [
        ("class_id", 0, 2, "outside class_names"),
        ("group_id", 0, -1, "non-negative"),
        ("packed", (0, 0), np.nan, "finite"),
        ("packed", (0, 9), -1.0, "ordered"),
        ("packed", (0, 17), 1.1, r"visibility must be in \[0, 1\]"),
    ],
)
def test_load_contract_validates_ltt_data_values(
    tmp_path, column, index, value, message
):
    """Validate grouped geometry taxonomy, groups, boxes, and ranges."""
    output = _valid_ltt_data(tmp_path)
    arrays = _read_npz(output)
    arrays[column][index] = value
    bad_path = tmp_path / "bad-packed-value.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match=message):
        load_contract(bad_path, LTT_DATA_SCHEMA_VERSION)


@pytest.mark.parametrize(
    "column, index, value, message",
    [
        ("class_id", 0, 2, "outside class_names"),
        ("cam", 0, 2, "outside cam_names"),
        ("box", (0, 2), 0.0, "ordered"),
        ("score", 0, np.inf, r"in \[0, 1\]"),
        ("valid_cam", 0, 2, "outside cam_names"),
    ],
)
def test_load_contract_validates_rtdetr_values(
    tmp_path, column, index, value, message
):
    """Validate pseudo-label taxonomy, cameras, boxes, scores, and validity."""
    output = _valid_rtdetr(tmp_path)
    arrays = _read_npz(output)
    arrays[column][index] = value
    bad_path = tmp_path / "bad-rtdetr-value.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match=message):
        load_contract(bad_path, RTDETR_2D_SCHEMA_VERSION)


def test_load_contract_validates_rtdetr_optional_columns(tmp_path):
    """Require paired, aligned validity columns and consistent metadata."""
    output = _valid_rtdetr(tmp_path)
    original = _read_npz(output)

    unpaired = dict(original)
    unpaired.pop("valid_cam")
    unpaired_path = tmp_path / "unpaired-validity.npz"
    _write_npz(unpaired_path, unpaired)
    with pytest.raises(ValueError, match="supplied together"):
        load_contract(unpaired_path, RTDETR_2D_SCHEMA_VERSION)

    unequal = dict(original)
    unequal["valid_cam"] = original["valid_cam"][:1]
    unequal_path = tmp_path / "unequal-validity.npz"
    _write_npz(unequal_path, unequal)
    with pytest.raises(ValueError, match="equal row counts"):
        load_contract(unequal_path, RTDETR_2D_SCHEMA_VERSION)

    bad_count = dict(original)
    metadata = decode_metadata(bad_count.pop("_meta"))
    metadata["num_valid_frame_cameras"] = 1
    count_path = tmp_path / "bad-validity-count.npz"
    _write_npz(count_path, bad_count, metadata)
    with pytest.raises(ValueError, match="num_valid_frame_cameras=1"):
        load_contract(count_path, RTDETR_2D_SCHEMA_VERSION)


def test_load_contract_requires_detection_pairs_to_be_marked_valid(tmp_path):
    """Reject detections whose frame/camera join is declared unprocessed."""
    output = _valid_rtdetr(tmp_path)
    arrays = _read_npz(output)
    arrays["valid_frame_id"][0] = 7
    bad_path = tmp_path / "missing-valid-pair.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match="must be marked valid"):
        load_contract(bad_path, RTDETR_2D_SCHEMA_VERSION)


def test_load_contract_requires_unique_rtdetr_validity_pairs(tmp_path):
    """Reject duplicate processed-pair rows in externally supplied caches."""
    output = _valid_rtdetr(tmp_path)
    arrays = _read_npz(output)
    arrays["valid_frame_id"][1] = arrays["valid_frame_id"][0]
    arrays["valid_cam"][1] = arrays["valid_cam"][0]
    bad_path = tmp_path / "duplicate-valid-pair.npz"
    _write_npz(bad_path, arrays)

    with pytest.raises(ValueError, match="pairs must be unique"):
        load_contract(bad_path, RTDETR_2D_SCHEMA_VERSION)


def test_load_contract_allows_rtdetr_without_optional_validity(tmp_path):
    """Keep legacy caches valid when neither optional validity column exists."""
    output = write_rtdetr_2d(
        tmp_path / "legacy.npz",
        scene="scene",
        class_names=CLASSES,
        camera_names=CAMERAS,
        frame_id=[8],
        cam=[0],
        class_id=[1],
        box=[[10, 20, 30, 40]],
        score=[0.8],
    )

    artifact = load_contract(output, RTDETR_2D_SCHEMA_VERSION)

    assert "valid_frame_id" not in artifact
    assert "valid_cam" not in artifact


def test_load_contract_rejects_unknown_and_mismatched_schemas(tmp_path):
    """Reject unknown requested/declared schemas and known schema mismatches."""
    output = _valid_ltt_2dgt(tmp_path)
    with pytest.raises(ValueError, match="Unsupported Sparse4D schema"):
        load_contract(output, "unknown/v1")
    with pytest.raises(ValueError, match="Expected schema"):
        load_contract(output, LTT_DATA_SCHEMA_VERSION)

    arrays = _read_npz(output)
    metadata = decode_metadata(arrays.pop("_meta"))
    metadata["schema_version"] = "unknown/v1"
    unknown_path = tmp_path / "unknown-schema.npz"
    _write_npz(unknown_path, arrays, metadata)
    with pytest.raises(ValueError, match="Unsupported Sparse4D schema"):
        load_contract(unknown_path, LTT_2DGT_SCHEMA_VERSION)
