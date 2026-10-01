# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic producers for Sparse4D 2D supervision sidecars.

The functions in this module accept in-memory annotation records. Dataset-
specific readers can therefore stay separate from the stable, pickle-free NPZ
contracts consumed by TAO Sparse4D.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tarfile
from typing import Optional

import numpy as np

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    validate_class_names,
    validate_scene_name,
    write_ltt_2dgt,
    write_rtdetr_2d,
)
from nvidia_tao_ds.annotations.sparse4d.ltt_geometry import iter_gt_frames


_FRAME_NUMBER = re.compile(r"(\d+)")


def _frame_id(value) -> int:
    """Return a non-negative integral frame identifier."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frame_id must be a non-negative integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("frame_id must be a non-negative integer") from error
    try:
        matches = float(value) == converted
    except (TypeError, ValueError):
        matches = str(value).strip() == str(converted)
    if converted < 0 or not matches:
        raise ValueError("frame_id must be a non-negative integer")
    return converted


def _instance_id(value) -> int:
    """Return an integral instance identifier without lossy coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("instance_id must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("instance_id must be an integer") from error
    try:
        matches = float(value) == converted
    except (TypeError, ValueError):
        matches = str(value).strip() == str(converted)
    if not matches:
        raise ValueError("instance_id must be an integer")
    return converted


def _xyxy(value, field: str) -> np.ndarray:
    """Validate one finite, positive-area ``x1,y1,x2,y2`` box."""
    try:
        box = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain four numeric values") from error
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{field} must contain four finite numeric values")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{field} must use positive-area x1,y1,x2,y2 corners")
    return box


def _class_id(
    annotation: Mapping,
    name_to_id: Mapping[str, int],
    class_name_map: Mapping[str, Optional[str]],
) -> Optional[int]:
    """Resolve a source class through an explicit ordered taxonomy."""
    source_name = annotation.get("class_name", annotation.get("object type"))
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("Every annotation must provide class_name or object type")
    target_name = class_name_map.get(source_name, source_name)
    if target_name is None:
        return None
    if target_name not in name_to_id:
        raise ValueError(
            f"Source class {source_name!r} resolves to {target_name!r}, which is "
            "not in the ordered class_names taxonomy"
        )
    return name_to_id[target_name]


def _validate_class_name_map(
    value: Optional[Mapping[str, Optional[str]]],
    class_names: Sequence[str],
) -> dict[str, Optional[str]]:
    """Validate every explicit source-to-target taxonomy mapping."""
    mapping = dict(value or {})
    for source_name, target_name in mapping.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("class_name_map keys must be non-empty strings")
        if target_name is not None and target_name not in class_names:
            raise ValueError(
                f"class_name_map target {target_name!r} is not in class_names"
            )
    return mapping


def _annotation_frames(frames) -> list[tuple[int, Sequence]]:
    """Normalize keyed or record-style annotation frames and sort by ID."""
    if isinstance(frames, Mapping):
        candidates = list(frames.items())
    elif isinstance(frames, Iterable) and not isinstance(frames, (str, bytes)):
        candidates = []
        for record in frames:
            if not isinstance(record, Mapping) or "frame_id" not in record:
                raise ValueError("Each frame must be a mapping containing frame_id")
            candidates.append((record["frame_id"], record.get("annotations", [])))
    else:
        raise TypeError("frames must be a mapping or iterable of frame records")

    normalized = []
    seen = set()
    for raw_frame_id, frame_annotations in candidates:
        current_frame_id = _frame_id(raw_frame_id)
        if current_frame_id in seen:
            raise ValueError(f"Duplicate frame_id {current_frame_id}")
        seen.add(current_frame_id)
        if frame_annotations is None:
            frame_annotations = []
        if not isinstance(frame_annotations, Sequence) or isinstance(
            frame_annotations, (str, bytes)
        ):
            raise ValueError("Frame annotations must be a sequence")
        normalized.append((current_frame_id, frame_annotations))
    return sorted(normalized, key=lambda item: item[0])


def write_ltt_2dgt_sidecar(
    path: os.PathLike | str,
    *,
    scene: str,
    class_names: Sequence[str],
    camera_names: Sequence[str],
    frames,
    class_name_map: Optional[Mapping[str, Optional[str]]] = None,
    metadata: Optional[Mapping] = None,
) -> str:
    """Build one ``ltt_2dgt/v1`` sidecar from AICity-style frame records.

    ``frames`` may be ``{frame_id: annotations}`` or an iterable of mappings
    with ``frame_id`` and ``annotations``. Each annotation uses ``object type``,
    ``object id``, ``2d bounding box`` and optional ``2d bounding box visible``
    keys. The normalized aliases ``class_name``, ``instance_id``, ``box2`` and
    ``box3`` are also accepted.

    A missing or invalid visible box preserves TAO's legacy semantics: the
    full box is copied to ``box3`` and its visibility/occlusion weight is zero.
    Invalid full boxes are skipped per camera because clipped projections can
    legitimately have zero area. Source classes can only be dropped by mapping
    them explicitly to ``None``.
    """
    classes = validate_class_names(class_names)
    cameras = validate_class_names(camera_names)
    name_to_id = {name: index for index, name in enumerate(classes)}
    camera_to_id = {name: index for index, name in enumerate(cameras)}
    resolved_class_map = _validate_class_name_map(class_name_map, classes)
    rows = []
    skipped_invalid_full_boxes = 0
    visible_box_fallbacks = 0
    normalized_frames = _annotation_frames(frames)

    for current_frame_id, frame_annotations in normalized_frames:
        for annotation in frame_annotations:
            if not isinstance(annotation, Mapping):
                raise ValueError("Every annotation must be a mapping")
            current_class_id = _class_id(
                annotation, name_to_id, resolved_class_map
            )
            if current_class_id is None:
                continue
            current_instance_id = _instance_id(
                annotation.get("instance_id", annotation.get("object id", -1))
            )
            full_boxes = annotation.get("box2", annotation.get("2d bounding box"))
            visible_boxes = annotation.get(
                "box3", annotation.get("2d bounding box visible")
            )
            if full_boxes is None:
                continue
            if not isinstance(full_boxes, Mapping):
                raise ValueError("2D full boxes must be a camera-to-box mapping")
            if visible_boxes is not None and not isinstance(visible_boxes, Mapping):
                raise ValueError("2D visible boxes must be a camera-to-box mapping")

            for camera_name, full_value in full_boxes.items():
                if camera_name not in camera_to_id:
                    raise ValueError(
                        f"Annotation camera {camera_name!r} is not in camera_names"
                    )
                try:
                    full_box = _xyxy(full_value, "2D full box")
                except ValueError:
                    skipped_invalid_full_boxes += 1
                    continue
                if visible_boxes is not None and camera_name in visible_boxes:
                    try:
                        visible_box = _xyxy(
                            visible_boxes[camera_name], "2D visible box"
                        )
                    except ValueError:
                        visible_box = full_box
                        weight = 0.0
                        visible_box_fallbacks += 1
                    else:
                        full_area = (full_box[2] - full_box[0]) * (
                            full_box[3] - full_box[1]
                        )
                        visible_area = (visible_box[2] - visible_box[0]) * (
                            visible_box[3] - visible_box[1]
                        )
                        weight = float(
                            np.clip(visible_area / full_area, 0.0, 1.0)
                        )
                else:
                    visible_box = full_box
                    weight = 0.0
                rows.append(
                    (
                        current_frame_id,
                        current_instance_id,
                        current_class_id,
                        camera_to_id[camera_name],
                        full_box,
                        visible_box,
                        weight,
                    )
                )

    rows.sort(
        key=lambda row: (
            row[0], row[1], row[2], row[3], *row[4].tolist(), *row[5].tolist()
        )
    )
    document = dict(metadata or {})
    document.setdefault("num_frames", len(normalized_frames))
    document["num_skipped_invalid_full_boxes"] = skipped_invalid_full_boxes
    document["num_visible_box_fallbacks"] = visible_box_fallbacks
    return write_ltt_2dgt(
        path,
        scene=scene,
        class_names=classes,
        camera_names=cameras,
        frame_id=[row[0] for row in rows],
        instance_id=[row[1] for row in rows],
        class_id=[row[2] for row in rows],
        cam=[row[3] for row in rows],
        box2=[row[4] for row in rows],
        box3=[row[5] for row in rows],
        occ=[row[6] for row in rows],
        metadata=document,
    )


def iter_aicity_annotation_frames(
    scene_dir: os.PathLike | str,
    *,
    frame_stride: int = 1,
    max_frames: int = 0,
):
    """Adapt the shared GT iterator to visible-sidecar frame records.

    Both LTT producers use numeric per-frame ordering, source order for streamed
    monolithic JSON, and ignore non-frame metadata keys.
    """
    if isinstance(frame_stride, bool) or int(frame_stride) < 1:
        raise ValueError("frame_stride must be a positive integer")
    if isinstance(max_frames, bool) or int(max_frames) < 0:
        raise ValueError("max_frames must be a non-negative integer")
    scene_path = Path(scene_dir).expanduser()
    if not ((scene_path / "ground_truth_final").is_dir() or
            (scene_path / "ground_truth.json").is_file()):
        raise FileNotFoundError(f"No ground truth under {scene_path}")
    for frame_id, frame_annotations in iter_gt_frames(scene_path, frame_stride, max_frames):
        yield {"frame_id": frame_id, "annotations": frame_annotations}


def _scene_camera_names(scene_path: Path, frames: Sequence[Mapping]) -> list[str]:
    """Discover stable camera names from calibration and annotation boxes."""
    camera_names = set()
    calibration_path = scene_path / "calibration.json"
    if calibration_path.is_file():
        with open(calibration_path, "r", encoding="utf-8") as stream:
            calibration = json.load(stream)
        if isinstance(calibration, Mapping):
            for sensor in calibration.get("sensors", []):
                if (
                    isinstance(sensor, Mapping) and
                    sensor.get("type") == "camera" and
                    isinstance(sensor.get("id"), str) and
                    sensor["id"]
                ):
                    camera_names.add(sensor["id"])
    for frame in frames:
        for annotation in frame.get("annotations", []):
            if not isinstance(annotation, Mapping):
                continue
            full_boxes = annotation.get("box2", annotation.get("2d bounding box"))
            if isinstance(full_boxes, Mapping):
                camera_names.update(str(name) for name in full_boxes)
    if not camera_names:
        raise ValueError(
            "Could not discover camera names; pass camera_names for an empty scene"
        )
    return sorted(camera_names)


def build_ltt_2dgt_scene(
    scene_dir: os.PathLike | str,
    output_dir: os.PathLike | str,
    *,
    class_names: Sequence[str],
    camera_names: Optional[Sequence[str]] = None,
    class_name_map: Optional[Mapping[str, Optional[str]]] = None,
    anno_version: str = "v0.1",
    frame_stride: int = 1,
    max_frames: int = 0,
) -> str:
    """Build ``<scene>__ltt2dgt.npz`` directly from raw AICity JSON."""
    scene_path = Path(scene_dir).expanduser().resolve()
    scene = validate_scene_name(scene_path.name)
    frames = list(
        iter_aicity_annotation_frames(
            scene_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
    )
    resolved_cameras = (
        _scene_camera_names(scene_path, frames)
        if camera_names is None
        else list(camera_names)
    )
    output = Path(output_dir).expanduser().resolve() / f"{scene}__ltt2dgt.npz"
    return write_ltt_2dgt_sidecar(
        output,
        scene=scene,
        class_names=class_names,
        camera_names=resolved_cameras,
        frames=frames,
        class_name_map=class_name_map,
        metadata={
            "anno_version": anno_version,
            "frame_stride": int(frame_stride),
        },
    )


def _detection_frames(frames) -> list[tuple[int, Mapping]]:
    """Normalize frame/camera detection mappings and sort by frame ID."""
    if isinstance(frames, Mapping):
        candidates = list(frames.items())
    elif isinstance(frames, Iterable) and not isinstance(frames, (str, bytes)):
        candidates = []
        for record in frames:
            if not isinstance(record, Mapping) or "frame_id" not in record:
                raise ValueError("Each frame must be a mapping containing frame_id")
            candidates.append((record["frame_id"], record.get("cameras", {})))
    else:
        raise TypeError("frames must be a mapping or iterable of frame records")

    normalized = []
    seen = set()
    for raw_frame_id, cameras in candidates:
        current_frame_id = _frame_id(raw_frame_id)
        if current_frame_id in seen:
            raise ValueError(f"Duplicate frame_id {current_frame_id}")
        seen.add(current_frame_id)
        if cameras is None:
            cameras = {}
        if not isinstance(cameras, Mapping):
            raise ValueError("Frame cameras must be a camera-to-detections mapping")
        normalized.append((current_frame_id, cameras))
    return sorted(normalized, key=lambda item: item[0])


def write_rtdetr_2d_sidecar(
    path: os.PathLike | str,
    *,
    scene: str,
    class_names: Sequence[str],
    camera_names: Sequence[str],
    frames,
    class_name_map: Optional[Mapping[str, Optional[str]]] = None,
    metadata: Optional[Mapping] = None,
) -> str:
    """Build one ``ltt_rtdetr2d/v1`` cache from normalized detections.

    Each frame contains a ``cameras`` mapping. Merely including a camera key
    records a valid frame/camera join, so an empty detection list remains an
    explicit background-only sample rather than being mistaken for missing
    inference. Detections contain ``class_name``, an xyxy ``box`` and ``score``.
    """
    classes = validate_class_names(class_names)
    cameras = validate_class_names(camera_names)
    name_to_id = {name: index for index, name in enumerate(classes)}
    camera_to_id = {name: index for index, name in enumerate(cameras)}
    resolved_class_map = _validate_class_name_map(class_name_map, classes)
    rows = []
    valid_pairs = []

    normalized_frames = _detection_frames(frames)
    for current_frame_id, detections_by_camera in normalized_frames:
        for camera_name, detections in detections_by_camera.items():
            if camera_name not in camera_to_id:
                raise ValueError(
                    f"Detection camera {camera_name!r} is not in camera_names"
                )
            camera_id = camera_to_id[camera_name]
            valid_pairs.append((current_frame_id, camera_id))
            if detections is None:
                detections = []
            if not isinstance(detections, Sequence) or isinstance(
                detections, (str, bytes)
            ):
                raise ValueError("Camera detections must be a sequence")
            for detection in detections:
                if not isinstance(detection, Mapping):
                    raise ValueError("Every detection must be a mapping")
                current_class_id = _class_id(
                    detection, name_to_id, resolved_class_map
                )
                if current_class_id is None:
                    continue
                box = _xyxy(
                    detection.get("box", detection.get("bbox")), "Detection box"
                )
                try:
                    score = float(detection["score"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "Every detection must provide a numeric score"
                    ) from error
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("Detection score must be finite and in [0, 1]")
                rows.append(
                    (
                        current_frame_id,
                        camera_id,
                        current_class_id,
                        box,
                        score,
                    )
                )

    rows.sort(
        key=lambda row: (
            row[0], row[1], row[2], *row[3].tolist(), -row[4]
        )
    )
    valid_pairs.sort()
    document = dict(metadata or {})
    document["num_valid_frame_cameras"] = len(valid_pairs)
    return write_rtdetr_2d(
        path,
        scene=scene,
        class_names=classes,
        camera_names=cameras,
        frame_id=[row[0] for row in rows],
        cam=[row[1] for row in rows],
        class_id=[row[2] for row in rows],
        box=[row[3] for row in rows],
        score=[row[4] for row in rows],
        valid_frame_id=[pair[0] for pair in valid_pairs],
        valid_cam=[pair[1] for pair in valid_pairs],
        metadata=document,
    )


def frame_id_from_name(member_name: str) -> Optional[int]:
    """Return the last integer from a KITTI label filename, if present."""
    numbers = _FRAME_NUMBER.findall(Path(member_name).stem)
    return int(numbers[-1]) if numbers else None


def build_rtdetr_archive_sidecar(
    input_dir: os.PathLike | str,
    output_path: os.PathLike | str,
    *,
    class_names: Sequence[str],
    class_name_map: Optional[Mapping[str, Optional[str]]] = None,
    camera_map: Optional[Mapping[str, str]] = None,
    confidence_threshold: float = 0.4,
    frame_stride: int = 1,
    max_frames_per_camera: int = 0,
    scene_name: Optional[str] = None,
) -> dict:
    """Normalize ``<camera>/labels.tar.gz`` archives into one safe cache."""
    root = Path(input_dir).expanduser().resolve()
    label_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "labels.tar.gz").is_file()
    )
    if not label_dirs:
        raise ValueError(f"No <camera>/labels.tar.gz found under {root}")
    classes = validate_class_names(class_names)
    resolved_camera_map = dict(camera_map or {})
    camera_names = [
        resolved_camera_map.get(path.name, path.name) for path in label_dirs
    ]
    validate_class_names(camera_names)
    resolved_class_map = _validate_class_name_map(class_name_map, classes)
    confidence_threshold = float(confidence_threshold)
    if not np.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    if isinstance(frame_stride, bool) or int(frame_stride) < 1:
        raise ValueError("frame_stride must be a positive integer")
    if isinstance(max_frames_per_camera, bool) or int(max_frames_per_camera) < 0:
        raise ValueError("max_frames_per_camera must be a non-negative integer")
    frame_stride = int(frame_stride)
    max_frames_per_camera = int(max_frames_per_camera)

    frames = {}
    raw_class_counts = {}
    frames_per_camera = {}
    effective_class_map = {}
    for camera_name, label_dir in zip(camera_names, label_dirs):
        seen_frames = set()
        with tarfile.open(label_dir / "labels.tar.gz", "r:gz") as archive:
            members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith(".txt")
                ),
                key=lambda member: (
                    frame_id_from_name(member.name) is None,
                    frame_id_from_name(member.name) or 0,
                    member.name,
                ),
            )
            for member in members:
                current_frame_id = frame_id_from_name(member.name)
                if current_frame_id is None or current_frame_id % frame_stride:
                    continue
                if (
                    max_frames_per_camera and
                    len(seen_frames) >= max_frames_per_camera and
                    current_frame_id not in seen_frames
                ):
                    continue
                seen_frames.add(current_frame_id)
                detections = frames.setdefault(current_frame_id, {}).setdefault(
                    camera_name, []
                )
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                for line in stream.read().decode("utf-8", "ignore").splitlines():
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    source_name = parts[0]
                    raw_class_counts[source_name] = (
                        raw_class_counts.get(source_name, 0) + 1
                    )
                    target_name = resolved_class_map.get(source_name, source_name)
                    if target_name is None:
                        continue
                    if target_name not in classes:
                        raise ValueError(
                            f"Unmapped detector class {source_name!r}; add a class_map "
                            "alias or explicitly map it to null to drop it"
                        )
                    effective_class_map[source_name] = target_name
                    try:
                        box = [float(parts[index]) for index in range(4, 8)]
                        score = float(parts[-1])
                    except ValueError:
                        continue
                    if not np.isfinite([*box, score]).all():
                        continue
                    if (
                        score < confidence_threshold or
                        box[2] - box[0] <= 1.0 or
                        box[3] - box[1] <= 1.0
                    ):
                        continue
                    detections.append(
                        {"class_name": target_name, "box": box, "score": score}
                    )
        frames_per_camera[camera_name] = len(seen_frames)

    scene = validate_scene_name(scene_name or root.parent.name)
    metadata = {
        "conf_thr": confidence_threshold,
        "frame_stride": frame_stride,
        "source": "RT-DETR KITTI labels.tar.gz",
        "class_map": effective_class_map,
        "raw_class_counts": raw_class_counts,
        "frames_per_camera": frames_per_camera,
        "num_rows": sum(
            len(detections)
            for cameras in frames.values()
            for detections in cameras.values()
        ),
        "num_valid_frame_cameras": sum(
            len(cameras) for cameras in frames.values()
        ),
    }
    output = write_rtdetr_2d_sidecar(
        output_path,
        scene=scene,
        class_names=classes,
        camera_names=camera_names,
        frames=frames,
        metadata=metadata,
    )
    return {
        "output_path": output,
        "scene": scene,
        "cam_names": camera_names,
        **metadata,
    }
