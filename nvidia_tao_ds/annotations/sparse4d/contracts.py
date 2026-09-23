# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe, versioned array contracts shared with TAO Sparse4D consumers."""

from __future__ import annotations

import json
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np


LTT_2DGT_SCHEMA_VERSION = "ltt_2dgt/v1"
LTT_DATA_SCHEMA_VERSION = "ltt_data/v2"
RTDETR_2D_SCHEMA_VERSION = "ltt_rtdetr2d/v1"
PACKED_GEOMETRY_WIDTH = 18

_SUPPORTED_SCHEMAS = frozenset(
    {
        LTT_2DGT_SCHEMA_VERSION,
        LTT_DATA_SCHEMA_VERSION,
        RTDETR_2D_SCHEMA_VERSION,
    }
)
_LTT_2DGT_SPECS = {
    "frame_id": (np.dtype(np.int32), None),
    "instance_id": (np.dtype(np.int64), None),
    "class_id": (np.dtype(np.int16), None),
    "cam": (np.dtype(np.int16), None),
    "box2": (np.dtype(np.float32), 4),
    "box3": (np.dtype(np.float32), 4),
    "occ": (np.dtype(np.float32), None),
}
_LTT_DATA_SPECS = {
    "packed": (np.dtype(np.float32), PACKED_GEOMETRY_WIDTH),
    "class_id": (np.dtype(np.int16), None),
    "group_id": (np.dtype(np.int64), None),
}
_RTDETR_2D_SPECS = {
    "frame_id": (np.dtype(np.int32), None),
    "cam": (np.dtype(np.int16), None),
    "class_id": (np.dtype(np.int16), None),
    "box": (np.dtype(np.float32), 4),
    "score": (np.dtype(np.float32), None),
}
_RTDETR_VALIDITY_SPECS = {
    "valid_frame_id": (np.dtype(np.int32), None),
    "valid_cam": (np.dtype(np.int16), None),
}


def validate_scene_name(scene_name: str) -> str:
    """Return a safe scene filename component or raise ``ValueError``."""
    if not isinstance(scene_name, str) or not scene_name.strip():
        raise ValueError("scene_name must be a non-empty string")
    if scene_name in {".", ".."} or any(
        separator in scene_name for separator in ("/", "\\", "\x00")
    ):
        raise ValueError(
            "scene_name must be a plain filename component without traversal"
        )
    if "+" in scene_name:
        raise ValueError("scene_name must not contain '+', the runtime BEV-group separator")
    return scene_name


def validate_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an ordered Sparse4D taxonomy."""
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be a sequence of names")
    names = tuple(class_names)
    if not names:
        raise ValueError("class_names must contain at least one class")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("class_names must be unique and ordered")
    return names


def encode_metadata(metadata: Mapping) -> np.ndarray:
    """Encode JSON metadata as a one-dimensional ``uint8`` NPZ array."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    encoded = json.dumps(dict(metadata), sort_keys=True).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


def decode_metadata(value: np.ndarray) -> dict:
    """Decode and validate metadata produced by :func:`encode_metadata`."""
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError("NPZ _meta must be a one-dimensional uint8 JSON buffer")
    metadata = json.loads(array.tobytes().decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("NPZ _meta JSON must decode to an object")
    return metadata


def normalize_npz_path(path: os.PathLike | str) -> Path:
    """Return ``path`` with an explicit ``.npz`` suffix."""
    output = Path(path).expanduser()
    return output if output.suffix == ".npz" else Path(f"{output}.npz")


def _atomic_npz_dump(path: os.PathLike | str, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically write a compressed NPZ next to its final destination."""
    output = normalize_npz_path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return str(output)


def _integer_vector(value, dtype, name: str) -> np.ndarray:
    """Validate integer-like values before converting to a bounded dtype."""
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be one-dimensional") from error
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    target_dtype = np.dtype(dtype)
    limits = np.iinfo(target_dtype)
    converted = []
    for item in source:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must not contain boolean values")
        if isinstance(item, Integral):
            integer = int(item)
        elif isinstance(item, Real):
            if not np.isfinite(item):
                raise ValueError(f"{name} must contain only finite values")
            integer = int(item)
            if item != integer:
                raise ValueError(f"{name} must contain only integer-like values")
        else:
            raise ValueError(f"{name} must contain only integer-like values")
        if integer < limits.min or integer > limits.max:
            raise ValueError(
                f"{name} contains a value outside the {target_dtype.name} range"
            )
        converted.append(integer)
    return np.asarray(converted, dtype=target_dtype)


def _vector(value, dtype, name: str) -> np.ndarray:
    """Return a one-dimensional array with the requested dtype."""
    target_dtype = np.dtype(dtype)
    if np.issubdtype(target_dtype, np.integer):
        return _integer_vector(value, target_dtype, name)
    array = np.asarray(value, dtype=target_dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _matrix(value, width: int, dtype, name: str) -> np.ndarray:
    """Return an ``N x width`` array with the requested dtype."""
    array = np.asarray(value, dtype=dtype)
    if array.size == 0:
        return array.reshape(0, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    return array


def _require_equal_rows(arrays: Mapping[str, np.ndarray]) -> int:
    """Require all arrays to have the same leading dimension."""
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Artifact arrays must have equal row counts: {lengths}")
    return next(iter(lengths.values()), 0)


def _validate_class_ids(class_ids: np.ndarray, class_names: Sequence[str]) -> None:
    """Require every class ID to index the ordered taxonomy."""
    if len(class_ids) and (
        int(class_ids.min()) < 0 or int(class_ids.max()) >= len(class_names)
    ):
        raise ValueError("class_id contains a value outside class_names")


def _validate_camera_ids(camera_ids: np.ndarray, camera_names: Sequence[str]) -> None:
    """Require every camera ID to index the ordered camera list."""
    if len(camera_ids) and (
        int(camera_ids.min()) < 0 or int(camera_ids.max()) >= len(camera_names)
    ):
        raise ValueError("cam contains a value outside cam_names")


def _validate_ordered_boxes(boxes: np.ndarray, name: str) -> None:
    """Require each xyxy box to have non-decreasing corner coordinates."""
    if len(boxes) and np.any(
        (boxes[:, 2] < boxes[:, 0]) | (boxes[:, 3] < boxes[:, 1])
    ):
        raise ValueError(f"{name} must use ordered x1,y1,x2,y2 corners")


def _validate_array_specs(arrays: Mapping[str, np.ndarray], specs: Mapping) -> None:
    """Validate exact safe dtypes and vector or fixed-width matrix shapes."""
    for name, (expected_dtype, width) in specs.items():
        array = arrays[name]
        if array.dtype.hasobject:
            raise ValueError(f"{name} must not contain Python objects")
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{name} must have dtype {expected_dtype.name}, got "
                f"{array.dtype.name}"
            )
        if width is None:
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
        elif array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must have shape (N, {width})")


def _metadata_names(metadata: Mapping, key: str) -> tuple[str, ...]:
    """Return one required ordered list of names from JSON metadata."""
    value = metadata.get(key)
    if not isinstance(value, list):
        raise ValueError(f"metadata {key} must be an ordered JSON list")
    try:
        return validate_class_names(value)
    except ValueError as error:
        raise ValueError(f"metadata {key} is invalid: {error}") from error


def _metadata_row_count(metadata: Mapping, actual: int) -> None:
    """Require metadata ``num_rows`` to be a non-negative exact row count."""
    value = metadata.get("num_rows")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("metadata num_rows must be a non-negative integer")
    if value != actual:
        raise ValueError(
            f"metadata num_rows={value} does not match artifact rows={actual}"
        )


def _validate_scene_metadata(metadata: Mapping) -> None:
    """Require safe scene metadata for per-scene artifacts."""
    try:
        validate_scene_name(metadata.get("scene"))
    except ValueError as error:
        raise ValueError(f"metadata scene is invalid: {error}") from error


def _validate_ltt_2dgt(arrays: Mapping[str, np.ndarray], metadata: Mapping) -> None:
    """Validate a loaded ``ltt_2dgt/v1`` artifact."""
    _validate_array_specs(arrays, _LTT_2DGT_SPECS)
    row_count = _require_equal_rows(arrays)
    classes = _metadata_names(metadata, "class_names")
    cameras = _metadata_names(metadata, "cam_names")
    _metadata_row_count(metadata, row_count)
    _validate_scene_metadata(metadata)
    _validate_class_ids(arrays["class_id"], classes)
    _validate_camera_ids(arrays["cam"], cameras)
    if not np.isfinite(arrays["box2"]).all() or not np.isfinite(
        arrays["box3"]
    ).all():
        raise ValueError("LTT boxes must contain only finite values")
    _validate_ordered_boxes(arrays["box2"], "box2")
    _validate_ordered_boxes(arrays["box3"], "box3")
    occurrence = arrays["occ"]
    if not np.isfinite(occurrence).all() or np.any(
        (occurrence < 0.0) | (occurrence > 1.0)
    ):
        raise ValueError("occ must contain finite values in [0, 1]")


def _validate_packed_geometry(packed: np.ndarray) -> None:
    """Validate the fixed semantic fields within packed LTT geometry."""
    if not np.isfinite(packed).all():
        raise ValueError("packed geometry must contain only finite values")
    if len(packed) and np.any(packed[:, 0:3] <= 0.0):
        raise ValueError("packed geometry extents must be positive")
    if len(packed) and np.any(packed[:, 6] < 0.0):
        raise ValueError("packed geometry distance must be non-negative")
    _validate_ordered_boxes(packed[:, 7:11], "packed loose boxes")
    _validate_ordered_boxes(packed[:, 11:15], "packed tight boxes")
    if len(packed) and np.any(packed[:, 15:17] <= 0.0):
        raise ValueError("packed image width and height must be positive")
    visibility = packed[:, 17]
    if np.any((visibility < 0.0) | (visibility > 1.0)):
        raise ValueError("packed visibility must be in [0, 1]")


def _validate_ltt_data(arrays: Mapping[str, np.ndarray], metadata: Mapping) -> None:
    """Validate a loaded ``ltt_data/v2`` artifact."""
    _validate_array_specs(arrays, _LTT_DATA_SPECS)
    row_count = _require_equal_rows(arrays)
    classes = _metadata_names(metadata, "class_names")
    _metadata_row_count(metadata, row_count)
    _validate_class_ids(arrays["class_id"], classes)
    if len(arrays["group_id"]) and int(arrays["group_id"].min()) < 0:
        raise ValueError("group_id values must be non-negative")
    _validate_packed_geometry(arrays["packed"])


def _metadata_optional_count(metadata: Mapping, key: str, actual: int) -> None:
    """Validate an optional non-negative metadata count when present."""
    if key not in metadata:
        return
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"metadata {key} must be a non-negative integer")
    if value != actual:
        raise ValueError(
            f"metadata {key}={value} does not match artifact rows={actual}"
        )


def _validate_rtdetr_2d(arrays: Mapping[str, np.ndarray], metadata: Mapping) -> None:
    """Validate a loaded ``ltt_rtdetr2d/v1`` artifact."""
    detection_arrays = {name: arrays[name] for name in _RTDETR_2D_SPECS}
    _validate_array_specs(detection_arrays, _RTDETR_2D_SPECS)
    row_count = _require_equal_rows(detection_arrays)
    classes = _metadata_names(metadata, "class_names")
    cameras = _metadata_names(metadata, "cam_names")
    _metadata_row_count(metadata, row_count)
    _validate_scene_metadata(metadata)
    _validate_class_ids(arrays["class_id"], classes)
    _validate_camera_ids(arrays["cam"], cameras)
    if not np.isfinite(arrays["box"]).all():
        raise ValueError("RT-DETR boxes must contain only finite values")
    _validate_ordered_boxes(arrays["box"], "RT-DETR boxes")
    scores = arrays["score"]
    if not np.isfinite(scores).all() or np.any(
        (scores < 0.0) | (scores > 1.0)
    ):
        raise ValueError("score must contain finite values in [0, 1]")

    has_validity = "valid_frame_id" in arrays
    validity_count = 0
    if has_validity:
        validity_arrays = {name: arrays[name] for name in _RTDETR_VALIDITY_SPECS}
        _validate_array_specs(validity_arrays, _RTDETR_VALIDITY_SPECS)
        validity_count = _require_equal_rows(validity_arrays)
        _validate_camera_ids(arrays["valid_cam"], cameras)
        valid_pairs = set(
            zip(arrays["valid_frame_id"].tolist(), arrays["valid_cam"].tolist())
        )
        if len(valid_pairs) != validity_count:
            raise ValueError(
                "RT-DETR valid frame/camera pairs must be unique"
            )
        detection_pairs = set(
            zip(arrays["frame_id"].tolist(), arrays["cam"].tolist())
        )
        if not detection_pairs.issubset(valid_pairs):
            raise ValueError(
                "Every RT-DETR detection frame/camera pair must be marked valid"
            )
    _metadata_optional_count(metadata, "num_valid_frame_cameras", validity_count)


def write_ltt_2dgt(
    path: os.PathLike | str,
    *,
    scene: str,
    class_names: Sequence[str],
    camera_names: Sequence[str],
    frame_id,
    instance_id,
    class_id,
    cam,
    box2,
    box3,
    occ,
    metadata: Mapping | None = None,
) -> str:
    """Write a safe ``ltt_2dgt/v1`` visible-2D supervision sidecar."""
    scene = validate_scene_name(scene)
    classes = validate_class_names(class_names)
    cameras = validate_class_names(camera_names)
    arrays = {
        "frame_id": _vector(frame_id, np.int32, "frame_id"),
        "instance_id": _vector(instance_id, np.int64, "instance_id"),
        "class_id": _vector(class_id, np.int16, "class_id"),
        "cam": _vector(cam, np.int16, "cam"),
        "box2": _matrix(box2, 4, np.float32, "box2"),
        "box3": _matrix(box3, 4, np.float32, "box3"),
        "occ": _vector(occ, np.float32, "occ"),
    }
    row_count = _require_equal_rows(arrays)
    _validate_class_ids(arrays["class_id"], classes)
    _validate_camera_ids(arrays["cam"], cameras)
    if not np.isfinite(arrays["box2"]).all() or not np.isfinite(
        arrays["box3"]
    ).all():
        raise ValueError("LTT boxes must contain only finite values")
    _validate_ordered_boxes(arrays["box2"], "box2")
    _validate_ordered_boxes(arrays["box3"], "box3")
    if not np.isfinite(arrays["occ"]).all() or np.any(
        (arrays["occ"] < 0.0) | (arrays["occ"] > 1.0)
    ):
        raise ValueError("occ must contain finite values in [0, 1]")
    document = dict(metadata or {})
    document.update(
        {
            "schema_version": LTT_2DGT_SCHEMA_VERSION,
            "scene": scene,
            "class_names": list(classes),
            "cam_names": list(cameras),
            "num_rows": row_count,
        }
    )
    return _atomic_npz_dump(path, {"_meta": encode_metadata(document), **arrays})


def write_ltt_data(
    path: os.PathLike | str,
    *,
    class_names: Sequence[str],
    packed,
    class_id,
    group_id,
    metadata: Mapping | None = None,
) -> str:
    """Write a grouped ``ltt_data/v2`` raw-geometry training cache."""
    classes = validate_class_names(class_names)
    arrays = {
        "packed": _matrix(
            packed, PACKED_GEOMETRY_WIDTH, np.float32, "packed"
        ),
        "class_id": _vector(class_id, np.int16, "class_id"),
        "group_id": _vector(group_id, np.int64, "group_id"),
    }
    row_count = _require_equal_rows(arrays)
    _validate_class_ids(arrays["class_id"], classes)
    _validate_packed_geometry(arrays["packed"])
    if len(arrays["group_id"]) and int(arrays["group_id"].min()) < 0:
        raise ValueError("group_id values must be non-negative")
    document = dict(metadata or {})
    document.update(
        {
            "schema_version": LTT_DATA_SCHEMA_VERSION,
            "class_names": list(classes),
            "num_rows": row_count,
        }
    )
    return _atomic_npz_dump(path, {"_meta": encode_metadata(document), **arrays})


def write_rtdetr_2d(
    path: os.PathLike | str,
    *,
    scene: str,
    class_names: Sequence[str],
    camera_names: Sequence[str],
    frame_id,
    cam,
    class_id,
    box,
    score,
    valid_frame_id=None,
    valid_cam=None,
    metadata: Mapping | None = None,
) -> str:
    """Write a safe ``ltt_rtdetr2d/v1`` pseudo-label cache."""
    scene = validate_scene_name(scene)
    classes = validate_class_names(class_names)
    cameras = validate_class_names(camera_names)
    arrays = {
        "frame_id": _vector(frame_id, np.int32, "frame_id"),
        "cam": _vector(cam, np.int16, "cam"),
        "class_id": _vector(class_id, np.int16, "class_id"),
        "box": _matrix(box, 4, np.float32, "box"),
        "score": _vector(score, np.float32, "score"),
    }
    row_count = _require_equal_rows(arrays)
    _validate_class_ids(arrays["class_id"], classes)
    _validate_camera_ids(arrays["cam"], cameras)
    if not np.isfinite(arrays["box"]).all():
        raise ValueError("RT-DETR boxes must contain only finite values")
    _validate_ordered_boxes(arrays["box"], "RT-DETR boxes")
    if not np.isfinite(arrays["score"]).all() or np.any(
        (arrays["score"] < 0.0) | (arrays["score"] > 1.0)
    ):
        raise ValueError("score must contain finite values in [0, 1]")
    if (valid_frame_id is None) != (valid_cam is None):
        raise ValueError("valid_frame_id and valid_cam must be supplied together")
    validity_count = 0
    if valid_frame_id is not None:
        valid_arrays = {
            "valid_frame_id": _vector(
                valid_frame_id, np.int32, "valid_frame_id"
            ),
            "valid_cam": _vector(valid_cam, np.int16, "valid_cam"),
        }
        validity_count = _require_equal_rows(valid_arrays)
        _validate_camera_ids(valid_arrays["valid_cam"], cameras)
        valid_pairs = set(
            zip(
                valid_arrays["valid_frame_id"].tolist(),
                valid_arrays["valid_cam"].tolist(),
            )
        )
        if len(valid_pairs) != validity_count:
            raise ValueError(
                "RT-DETR valid frame/camera pairs must be unique"
            )
        detection_pairs = set(
            zip(arrays["frame_id"].tolist(), arrays["cam"].tolist())
        )
        if not detection_pairs.issubset(valid_pairs):
            raise ValueError(
                "Every RT-DETR detection frame/camera pair must be marked valid"
            )
        arrays.update(valid_arrays)
    document = dict(metadata or {})
    document.update(
        {
            "schema_version": RTDETR_2D_SCHEMA_VERSION,
            "scene": scene,
            "class_names": list(classes),
            "cam_names": list(cameras),
            "num_rows": row_count,
        }
    )
    if valid_frame_id is not None:
        document["num_valid_frame_cameras"] = validity_count
    else:
        document.pop("num_valid_frame_cameras", None)
    return _atomic_npz_dump(path, {"_meta": encode_metadata(document), **arrays})


def load_contract(path: os.PathLike | str, expected_schema: str) -> dict:
    """Load and fully validate one supported, numeric-only Sparse4D NPZ."""
    if (
        not isinstance(expected_schema, str) or
        expected_schema not in _SUPPORTED_SCHEMAS
    ):
        raise ValueError(f"Unsupported Sparse4D schema {expected_schema!r}")

    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        names = tuple(archive.files)
        if len(names) != len(set(names)):
            raise ValueError("Sparse4D artifact contains duplicate NPZ keys")
        present = set(names)
        if "_meta" not in present:
            raise ValueError("Sparse4D artifact is missing _meta")
        try:
            metadata = decode_metadata(np.asarray(archive["_meta"]))
        except ValueError as error:
            raise ValueError("Sparse4D artifact has unsafe or invalid _meta") from error

        declared_schema = metadata.get("schema_version")
        if (
            not isinstance(declared_schema, str) or
            declared_schema not in _SUPPORTED_SCHEMAS
        ):
            raise ValueError(
                f"Unsupported Sparse4D schema {declared_schema!r}"
            )
        if declared_schema != expected_schema:
            raise ValueError(
                f"Expected schema {expected_schema!r}, got {declared_schema!r}"
            )

        required = {
            LTT_2DGT_SCHEMA_VERSION: set(_LTT_2DGT_SPECS) | {"_meta"},
            LTT_DATA_SCHEMA_VERSION: set(_LTT_DATA_SPECS) | {"_meta"},
            RTDETR_2D_SCHEMA_VERSION: set(_RTDETR_2D_SPECS) | {"_meta"},
        }[declared_schema]
        optional = (
            set(_RTDETR_VALIDITY_SPECS)
            if declared_schema == RTDETR_2D_SCHEMA_VERSION
            else set()
        )
        missing = required - present
        unexpected = present - required - optional
        if missing:
            raise ValueError(
                f"Sparse4D artifact is missing NPZ keys: {sorted(missing)}"
            )
        if unexpected:
            raise ValueError(
                f"Sparse4D artifact has unexpected NPZ keys: {sorted(unexpected)}"
            )
        if declared_schema == RTDETR_2D_SCHEMA_VERSION and (
            ("valid_frame_id" in present) != ("valid_cam" in present)
        ):
            raise ValueError(
                "valid_frame_id and valid_cam must be supplied together"
            )

        arrays = {}
        for name in sorted(present - {"_meta"}):
            try:
                array = np.asarray(archive[name])
            except ValueError as error:
                raise ValueError(
                    f"Sparse4D artifact array {name!r} is unsafe or unreadable"
                ) from error
            if array.dtype.hasobject:
                raise ValueError(
                    f"Sparse4D artifact array {name!r} must not contain objects"
                )
            arrays[name] = array

    validators = {
        LTT_2DGT_SCHEMA_VERSION: _validate_ltt_2dgt,
        LTT_DATA_SCHEMA_VERSION: _validate_ltt_data,
        RTDETR_2D_SCHEMA_VERSION: _validate_rtdetr_2d,
    }
    validators[declared_schema](arrays, metadata)
    return {"metadata": metadata, **arrays}
