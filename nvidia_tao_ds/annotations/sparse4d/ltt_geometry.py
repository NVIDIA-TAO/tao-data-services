# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract model-independent Loose-to-Tight geometry from raw Sparse4D scenes."""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
import os
from pathlib import Path
import re
from typing import Dict, Iterator, Optional, Sequence, Tuple

try:
    import ijson as _ijson  # type: ignore
except ImportError:
    _ijson = None
import numpy as np

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    PACKED_GEOMETRY_WIDTH,
    validate_class_names,
    write_ltt_data,
)


SL_EXTENT = slice(0, 3)
I_THETA = 3
I_PHI = 4
I_PSI = 5
I_DISTANCE = 6
SL_LOOSE = slice(7, 11)
SL_TIGHT = slice(11, 15)
SL_IMAGE_WH = slice(15, 17)
I_VISIBILITY = 17


def build_name_to_id(
    class_names: Sequence[str],
    subclass_map: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, int]:
    """Build a source-name map without changing the ordered output taxonomy."""
    names = validate_class_names(class_names)
    mapping = {name: index for index, name in enumerate(names)}
    for parent, subclasses in (subclass_map or {}).items():
        if parent not in mapping:
            raise ValueError(f"subclass_map parent is not in class_names: {parent!r}")
        if isinstance(subclasses, (str, bytes)):
            raise ValueError(f"subclass_map[{parent!r}] must be a sequence of names")
        for subclass in subclasses:
            value = str(subclass)
            if not value:
                raise ValueError("subclass aliases must be non-empty strings")
            prior = mapping.setdefault(value, mapping[parent])
            if prior != mapping[parent]:
                raise ValueError(
                    f"Subclass alias {value!r} maps to more than one class"
                )
    return mapping


def bbox_area(box: Sequence[float]) -> float:
    """Return the non-negative area of an xyxy box."""
    array = np.asarray(box, dtype=np.float64).reshape(4)
    if not np.isfinite(array).all():
        return 0.0
    return max(0.0, float(array[2] - array[0])) * max(
        0.0, float(array[3] - array[1])
    )


def iter_gt_frames(
    scene_dir: os.PathLike | str,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> Iterator[Tuple[int, list]]:
    """Yield frames from per-frame or monolithic ground truth.

    Per-frame files are sorted numerically. Monolithic JSON preserves source
    order with either the optional streaming parser or the standard-library
    fallback, so installed dependencies cannot change sampling results.
    """
    scene = Path(scene_dir).expanduser()
    stride = max(1, int(frame_stride))
    limit = max(0, int(max_frames))
    per_frame_dir = scene / "ground_truth_final"
    emitted = 0
    if per_frame_dir.is_dir():
        indexed = {}
        for path in sorted(per_frame_dir.glob("ground_truth_*.json")):
            try:
                frame_id = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            prior_path = indexed.get(frame_id)
            if prior_path is not None:
                raise ValueError(
                    f"Duplicate normalized frame ID {frame_id} in "
                    f"{per_frame_dir}: {prior_path.name!r} and {path.name!r}"
                )
            indexed[frame_id] = path
        for frame_id, path in sorted(indexed.items()):
            if frame_id % stride:
                continue
            with path.open("r", encoding="utf-8") as stream:
                annotations = json.load(stream)
            if not isinstance(annotations, list):
                raise ValueError(f"Ground-truth frame must contain a list: {path}")
            yield frame_id, annotations
            emitted += 1
            if limit and emitted >= limit:
                return
        return

    ground_truth_path = scene / "ground_truth.json"
    if not ground_truth_path.is_file():
        return

    def selected_frames(items):
        """Yield validated, strided frames from key/annotation pairs."""
        emitted_count = 0
        normalized_keys = {}
        for key, annotations in items:
            try:
                frame_id = int(key)
            except (TypeError, ValueError):
                continue
            key_text = str(key)
            prior_key = normalized_keys.get(frame_id)
            if prior_key is not None:
                raise ValueError(
                    f"Duplicate normalized frame ID {frame_id} in "
                    f"{ground_truth_path}: {prior_key!r} and {key_text!r}"
                )
            normalized_keys[frame_id] = key_text
            if frame_id % stride:
                continue
            if limit and emitted_count >= limit:
                continue
            if not isinstance(annotations, list):
                raise ValueError(
                    f"Ground-truth frame {frame_id!r} must contain a list"
                )
            yield frame_id, annotations
            emitted_count += 1

    if _ijson is None:
        with ground_truth_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(
                "Monolithic ground truth must contain an object: "
                f"{ground_truth_path}"
            )
        yield from selected_frames(document.items())
    else:
        with ground_truth_path.open("rb") as stream:
            yield from selected_frames(_ijson.kvitems(stream, ""))


def has_ground_truth(scene_dir: os.PathLike | str) -> bool:
    """Return whether a directory has a supported ground-truth layout."""
    scene = Path(scene_dir).expanduser()
    return scene.is_dir() and (
        (scene / "ground_truth_final").is_dir() or
        (scene / "ground_truth.json").is_file()
    )


def scene_from_split_line(
    line: str,
    dedup_regex: str = r"^CT[\w.]+?__",
) -> str:
    """Extract the original raw-scene name from a Sparse4D split row."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        raise ValueError("split line must identify an annotation PKL")
    scene = Path(stripped.split()[0]).name.split("+")[0]
    for suffix in (
        "_infos_train.pkl",
        "_infos_test.pkl",
        "_infos_val.pkl",
        "_infos.pkl",
        ".pkl",
    ):
        if scene.endswith(suffix):
            scene = scene[: -len(suffix)]
            break
    return re.sub(dedup_regex, "", scene) if dedup_regex else scene


def _resolve_named_scene(
    data_root: Optional[os.PathLike | str],
    scene_name: os.PathLike | str,
) -> Optional[Path]:
    value = Path(scene_name).expanduser()
    if value.is_absolute() or data_root is None:
        return value.resolve() if has_ground_truth(value) else None
    root = Path(data_root).expanduser()
    direct = root / value
    if has_ground_truth(direct):
        return direct.resolve()
    for candidate in sorted(root.glob(f"*/{value}")):
        if has_ground_truth(candidate):
            return candidate.resolve()
    return None


def resolve_scene_paths(
    *,
    data_root: Optional[os.PathLike | str] = None,
    scenes: Optional[Sequence[str]] = None,
    scenes_file: Optional[os.PathLike | str] = None,
    scene_dirs: Optional[Sequence[os.PathLike | str]] = None,
    train_split: Optional[os.PathLike | str] = None,
    dedup_regex: str = r"^CT[\w.]+?__",
) -> list[Path]:
    """Resolve explicit or split-derived scenes to unique absolute paths."""
    resolved = []
    for value in scene_dirs or ():
        path = Path(value).expanduser().resolve()
        if not has_ground_truth(path):
            raise ValueError(f"Scene has no supported ground truth: {path}")
        resolved.append(path)

    names = [str(name) for name in (scenes or ())]
    if scenes_file:
        with Path(scenes_file).expanduser().open("r", encoding="utf-8") as stream:
            names.extend(
                line.strip()
                for line in stream
                if line.strip() and not line.lstrip().startswith("#")
            )
    if train_split:
        split_names = set()
        with Path(train_split).expanduser().open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and not line.lstrip().startswith("#"):
                    split_names.add(scene_from_split_line(line, dedup_regex))
        names.extend(sorted(split_names))

    missing = []
    for name in names:
        path = _resolve_named_scene(data_root, name)
        if path is None:
            missing.append(name)
        else:
            resolved.append(path)
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} raw scene(s); first: {missing[0]}"
        )
    if not resolved and data_root:
        root = Path(data_root).expanduser()
        resolved.extend(
            child.resolve()
            for child in sorted(root.iterdir())
            if has_ground_truth(child)
        )
    unique = list(dict.fromkeys(resolved))
    if not unique:
        raise ValueError("No scenes with ground truth were resolved")
    return unique


def _attribute(sensor: dict, name: str):
    for attribute in sensor.get("attributes", []) or []:
        if attribute.get("name") == name:
            return attribute.get("value")
    return None


def _reshape_world2cam(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 12:
        matrix = np.vstack(
            [matrix.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]]
        )
    else:
        matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("Camera extrinsic matrix contains non-finite values")
    return matrix


def _decode_camera(name: str, value: dict) -> Optional[Tuple[str, dict]]:
    intrinsic = None
    world2cam = None
    for key in (
        "intrinsicMatrix",
        "intrinsic_matrix",
        "intrinsic matrix",
        "cam_intrinsic",
        "K",
    ):
        if key in value:
            intrinsic = np.asarray(value[key], dtype=np.float64).reshape(3, 3)
            break
    for key in (
        "extrinsicMatrix",
        "w2c_matrix",
        "projection matrix w2c",
        "sensor2world_transform",
        "world2cam",
    ):
        if key in value:
            world2cam = _reshape_world2cam(value[key])
            break
    if intrinsic is None or world2cam is None:
        camera_matrix = value.get("cameraMatrix")
        if camera_matrix is not None:
            intrinsic = np.eye(3, dtype=np.float64)
            world2cam = _reshape_world2cam(camera_matrix)
    if intrinsic is None or world2cam is None:
        return None
    if not np.isfinite(intrinsic).all():
        raise ValueError(f"Camera {name!r} intrinsic contains non-finite values")
    image_size = value.get("image_size") or value.get("imageSize")
    if image_size is not None:
        image_width, image_height = (float(item) for item in image_size)
    else:
        width = value.get("width", _attribute(value, "frameWidth"))
        height = value.get("height", _attribute(value, "frameHeight"))
        image_width = float(width) if width else 2.0 * float(intrinsic[0, 2])
        image_height = float(height) if height else 2.0 * float(intrinsic[1, 2])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Camera {name!r} has an invalid image size")
    return name, {
        "K": intrinsic,
        "w2c": world2cam,
        "wh": (image_width, image_height),
    }


def load_scene_calibration(
    scene_dir: os.PathLike | str,
    *,
    calibration_file: Optional[os.PathLike | str] = None,
    calibration_mode: str = "aic25",
) -> Dict[str, dict]:
    """Load global cameras from NVSchema or legacy calibration JSON."""
    if calibration_mode not in {"aic24", "aic25"}:
        raise ValueError("calibration_mode must be 'aic24' or 'aic25'")
    scene = Path(scene_dir).expanduser()
    if calibration_file is None:
        filename = (
            "calibration_bevformer.json"
            if calibration_mode == "aic24"
            else "calibration.json"
        )
        path = scene / filename
    else:
        path = Path(calibration_file).expanduser()
        if not path.is_absolute():
            path = scene / path
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    cameras: Dict[str, dict] = {}
    if isinstance(document, dict) and isinstance(document.get("sensors"), list):
        for sensor in document["sensors"]:
            if not isinstance(sensor, dict) or sensor.get("type", "camera") != "camera":
                continue
            name = str(sensor.get("id", ""))
            decoded = _decode_camera(name, sensor)
            if name and decoded:
                cameras[decoded[0]] = decoded[1]
    else:

        def visit(mapping: dict) -> None:
            for name, value in mapping.items():
                if not isinstance(value, dict):
                    continue
                decoded = _decode_camera(str(name), value)
                if decoded:
                    cameras.setdefault(decoded[0], decoded[1])
                else:
                    visit(value)

        if isinstance(document, dict):
            visit(document)
    if not cameras:
        raise ValueError(f"No supported camera calibration found in {path}")
    return dict(sorted(cameras.items()))


def parse_objects(
    annotations: Sequence[dict],
    name_to_id: Dict[str, int],
    annotation_version: str = "v0.1",
) -> Tuple[list[int], np.ndarray, np.ndarray]:
    """Convert valid 3D annotations to class IDs and box7 arrays."""
    if annotation_version not in {"v0.0", "v0.1"}:
        raise ValueError("annotation_version must be 'v0.0' or 'v0.1'")
    indices, classes, boxes = [], [], []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            continue
        class_id = name_to_id.get(str(annotation.get("object type", "")))
        if class_id is None:
            continue
        try:
            location = np.asarray(
                annotation["3d location"], dtype=np.float64
            ).reshape(3)
            dimensions = np.asarray(
                annotation["3d bounding box scale"], dtype=np.float64
            ).reshape(3)
            yaw = float(annotation["3d bounding box rotation"][2])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not np.isfinite(location).all() or
            not np.isfinite(dimensions).all() or
            not np.isfinite(yaw) or
            np.any(dimensions <= 0)
        ):
            continue
        if annotation_version == "v0.0":
            yaw = -yaw
        indices.append(index)
        classes.append(class_id)
        boxes.append([*location, *dimensions, yaw])
    return (
        indices,
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float64).reshape(-1, 7),
    )


def camera_view_geometry(
    boxes3d_world: np.ndarray,
    world2cam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute relative yaw, elevation, bearing, and distance."""
    boxes = np.asarray(boxes3d_world, dtype=np.float64).reshape(-1, 7)
    transform = _reshape_world2cam(world2cam)
    homogeneous = np.concatenate(
        [boxes[:, :3], np.ones((len(boxes), 1), dtype=np.float64)],
        axis=1,
    )
    camera_centers = (homogeneous @ transform.T)[:, :3]
    x_camera, y_camera, z_camera = camera_centers.T
    distance = np.linalg.norm(camera_centers, axis=1)
    horizontal_distance = np.hypot(x_camera, z_camera)
    bearing = np.arctan2(x_camera, z_camera)
    elevation = np.arctan2(-y_camera, horizontal_distance)
    forward_world = transform[2, :3]
    camera_yaw_world = np.arctan2(forward_world[1], forward_world[0])
    relative_yaw = boxes[:, 6] - camera_yaw_world
    return relative_yaw, elevation, bearing, distance


def project_cuboid_aabb(
    box7: Sequence[float],
    world2cam: np.ndarray,
    intrinsic: np.ndarray,
    *,
    image_wh: Optional[Sequence[float]] = None,
    origin_z: float = 0.5,
    near_plane: float = 0.1,
) -> Optional[np.ndarray]:
    """Project one 3D cuboid to an image-clipped xyxy AABB."""
    x, y, z, width, length, height, yaw = (float(value) for value in box7)
    if not np.isfinite([x, y, z, width, length, height, yaw]).all():
        return None
    if width <= 0 or length <= 0 or height <= 0:
        return None
    z_min, z_max = (
        (-height / 2.0, height / 2.0)
        if origin_z == 0.5
        else (0.0, height)
    )
    corners = np.asarray(
        [
            (x_sign, y_sign, z_offset)
            for x_sign in (-width / 2.0, width / 2.0)
            for y_sign in (-length / 2.0, length / 2.0)
            for z_offset in (z_min, z_max)
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world = corners @ rotation.T + np.asarray([x, y, z])
    transform = _reshape_world2cam(world2cam)
    camera = (
        np.concatenate([world, np.ones((8, 1), dtype=np.float64)], axis=1)
        @ transform.T
    )[:, :3]
    if np.any(camera[:, 2] <= near_plane):
        return None
    pixels = camera @ np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).T
    uv = pixels[:, :2] / pixels[:, 2:3]
    box = np.asarray(
        [
            uv[:, 0].min(),
            uv[:, 1].min(),
            uv[:, 0].max(),
            uv[:, 1].max(),
        ],
        dtype=np.float32,
    )
    if image_wh is not None:
        image_width, image_height = (float(value) for value in image_wh)
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_wh values must be positive")
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, image_width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, image_height)
    return box


def pack_geometry(
    *,
    extent_wlh,
    theta,
    phi,
    psi,
    distance,
    loose_xyxy,
    tight_xyxy,
    image_wh,
    visibility,
) -> np.ndarray:
    """Pack raw values into the runtime's fixed N by 18 layout."""
    theta_array = np.asarray(theta).reshape(-1)
    count = len(theta_array)
    values = {
        "extent_wlh": (np.asarray(extent_wlh), (count, 3)),
        "phi": (np.asarray(phi), (count,)),
        "psi": (np.asarray(psi), (count,)),
        "distance": (np.asarray(distance), (count,)),
        "loose_xyxy": (np.asarray(loose_xyxy), (count, 4)),
        "tight_xyxy": (np.asarray(tight_xyxy), (count, 4)),
        "image_wh": (np.asarray(image_wh), (count, 2)),
        "visibility": (np.asarray(visibility), (count,)),
    }
    for name, (array, expected_shape) in values.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
    output = np.empty((count, PACKED_GEOMETRY_WIDTH), dtype=np.float32)
    output[:, SL_EXTENT] = values["extent_wlh"][0]
    output[:, I_THETA] = theta_array
    output[:, I_PHI] = values["phi"][0]
    output[:, I_PSI] = values["psi"][0]
    output[:, I_DISTANCE] = values["distance"][0]
    output[:, SL_LOOSE] = values["loose_xyxy"][0]
    output[:, SL_TIGHT] = values["tight_xyxy"][0]
    output[:, SL_IMAGE_WH] = values["image_wh"][0]
    output[:, I_VISIBILITY] = values["visibility"][0]
    if not np.isfinite(output).all():
        raise ValueError("packed Loose-to-Tight geometry contains non-finite values")
    return output


class ClassBalancedReservoir:
    """Memory-bounded uniform reservoir with an independent cap per class."""

    def __init__(
        self,
        num_classes: int,
        max_per_class: Optional[int],
        *,
        seed: int = 0,
    ) -> None:
        """Initialize independent deterministic reservoirs for each class."""
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.max_per_class = (
            int(max_per_class) if max_per_class and max_per_class > 0 else None
        )
        self.rng = np.random.default_rng(seed)
        self.heaps: list[list] = [[] for _ in range(self.num_classes)]
        self.counts = [0] * self.num_classes
        self._tie = 0

    def add_batch(self, class_ids, packed, group_ids) -> None:
        """Add samples and their source-frame group IDs."""
        classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
        rows = np.asarray(packed, dtype=np.float32)
        groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
        if rows.shape != (len(classes), PACKED_GEOMETRY_WIDTH):
            raise ValueError(
                "packed must have shape "
                f"{(len(classes), PACKED_GEOMETRY_WIDTH)}, got {rows.shape}"
            )
        if groups.shape != classes.shape:
            raise ValueError("group_ids must contain one value per packed row")
        keys = self.rng.random(len(classes))
        for index, class_id in enumerate(classes):
            if class_id < 0 or class_id >= self.num_classes:
                continue
            self.counts[class_id] += 1
            item = (
                float(keys[index]),
                self._tie,
                rows[index].copy(),
                int(groups[index]),
            )
            self._tie += 1
            heap = self.heaps[class_id]
            if self.max_per_class is None:
                heap.append(item)
            elif len(heap) < self.max_per_class:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled geometry, class IDs, and frame-group IDs."""
        rows, classes, groups = [], [], []
        for class_id, heap in enumerate(self.heaps):
            for _, _, row, group_id in heap:
                rows.append(row)
                classes.append(class_id)
                groups.append(group_id)
        return (
            np.asarray(rows, dtype=np.float32).reshape(
                -1, PACKED_GEOMETRY_WIDTH
            ),
            np.asarray(classes, dtype=np.int16),
            np.asarray(groups, dtype=np.int64),
        )

    def kept_per_class(self) -> list[int]:
        """Return retained sample counts in taxonomy order."""
        return [len(heap) for heap in self.heaps]

    def seen_per_class(self) -> list[int]:
        """Return observed sample counts in taxonomy order."""
        return list(self.counts)


def extract_ltt_data(
    scene_paths: Sequence[os.PathLike | str],
    output_path: os.PathLike | str,
    class_names: Sequence[str],
    name_to_id: Dict[str, int],
    *,
    calibration_file: Optional[os.PathLike | str] = None,
    calibration_mode: str = "aic25",
    annotation_version: str = "v0.1",
    frame_stride: int = 10,
    max_frames_per_scene: int = 0,
    min_visibility: float = 0.0,
    max_per_class: int = 50000,
    image_size_override: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    metadata: Optional[dict] = None,
) -> dict:
    """Extract a deterministic, frame-grouped ltt_data/v2 cache."""
    names = validate_class_names(class_names)
    if set(name_to_id.values()).difference(range(len(names))):
        raise ValueError("name_to_id contains an ID outside class_names")
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0, 1]")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if max_frames_per_scene < 0 or max_per_class < 0:
        raise ValueError("frame and per-class limits must be non-negative")
    if image_size_override is not None and (
        len(image_size_override) != 2 or
        min(float(value) for value in image_size_override) <= 0
    ):
        raise ValueError("image_size_override must contain positive width/height")

    reservoir = ClassBalancedReservoir(
        len(names), max_per_class, seed=seed
    )
    frames_used = 0
    resolved_scenes = []
    for scene_value in scene_paths:
        scene = Path(scene_value).expanduser().resolve()
        if not has_ground_truth(scene):
            raise ValueError(f"Scene has no supported ground truth: {scene}")
        cameras = load_scene_calibration(
            scene,
            calibration_file=calibration_file,
            calibration_mode=calibration_mode,
        )
        resolved_scenes.append(scene.name)
        for _, frame_annotations in iter_gt_frames(
            scene,
            frame_stride=frame_stride,
            max_frames=max_frames_per_scene,
        ):
            if not frame_annotations:
                continue
            annotation_indices, class_ids, boxes = parse_objects(
                frame_annotations, name_to_id, annotation_version
            )
            if not annotation_indices:
                continue
            group_id = frames_used
            frames_used += 1
            per_camera = defaultdict(list)
            for local_index, annotation_index in enumerate(annotation_indices):
                annotation = frame_annotations[annotation_index]
                amodal_by_camera = annotation.get("2d bounding box", {}) or {}
                visible_by_camera = (
                    annotation.get("2d bounding box visible", {}) or {}
                )
                if not isinstance(amodal_by_camera, dict):
                    continue
                for camera_name, amodal_value in amodal_by_camera.items():
                    if camera_name not in cameras:
                        continue
                    try:
                        amodal = np.asarray(
                            amodal_value, dtype=np.float64
                        ).reshape(4)
                    except (TypeError, ValueError):
                        continue
                    amodal_area = bbox_area(amodal)
                    if amodal_area <= 1.0:
                        continue
                    visible_value = (
                        visible_by_camera.get(camera_name)
                        if isinstance(visible_by_camera, dict)
                        else None
                    )
                    if visible_value is None:
                        visibility = 0.0
                    else:
                        try:
                            visibility = float(
                                np.clip(
                                    bbox_area(visible_value) / amodal_area,
                                    0.0,
                                    1.0,
                                )
                            )
                        except (TypeError, ValueError):
                            continue
                    if visibility >= min_visibility:
                        per_camera[camera_name].append(
                            (local_index, amodal, visibility)
                        )

            for camera_name in sorted(per_camera):
                items = per_camera[camera_name]
                camera = cameras[camera_name]
                image_wh = image_size_override or camera["wh"]
                local_indices = np.asarray(
                    [item[0] for item in items], dtype=np.int64
                )
                subset = boxes[local_indices]
                theta, phi, psi, distance = camera_view_geometry(
                    subset, camera["w2c"]
                )
                valid_indices, loose_boxes = [], []
                for item_index, box in enumerate(subset):
                    loose = project_cuboid_aabb(
                        box,
                        camera["w2c"],
                        camera["K"],
                        image_wh=image_wh,
                    )
                    if loose is not None and bbox_area(loose) > 1.0:
                        valid_indices.append(item_index)
                        loose_boxes.append(loose)
                if not valid_indices:
                    continue
                valid = np.asarray(valid_indices, dtype=np.int64)
                packed = pack_geometry(
                    extent_wlh=subset[valid, 3:6],
                    theta=theta[valid],
                    phi=phi[valid],
                    psi=psi[valid],
                    distance=distance[valid],
                    loose_xyxy=np.stack(loose_boxes),
                    tight_xyxy=np.stack([items[index][1] for index in valid]),
                    image_wh=np.broadcast_to(
                        np.asarray(image_wh, dtype=np.float32),
                        (len(valid), 2),
                    ),
                    visibility=np.asarray(
                        [items[index][2] for index in valid], dtype=np.float32
                    ),
                )
                reservoir.add_batch(
                    class_ids[local_indices][valid],
                    packed,
                    np.full(len(packed), group_id, dtype=np.int64),
                )

    packed, class_ids, group_ids = reservoir.to_arrays()
    if not len(class_ids):
        raise ValueError("No Loose-to-Tight samples were extracted")
    document = dict(metadata or {})
    document.update(
        {
            "class_names": list(names),
            "scenes": resolved_scenes,
            "box_wiring": (
                "input=cuboid_box1,target=amodal_box2,weight=visible/amodal"
            ),
            "frame_stride": int(frame_stride),
            "min_visibility": float(min_visibility),
            "max_per_class": int(max_per_class),
            "anno_version": annotation_version,
            "calib_mode": calibration_mode,
            "img_wh_override": image_size_override,
            "seen_per_class": dict(zip(names, reservoir.seen_per_class())),
            "kept_per_class": dict(zip(names, reservoir.kept_per_class())),
            "num_samples": int(len(class_ids)),
            "num_frames_used": int(frames_used),
        }
    )
    artifact_path = write_ltt_data(
        output_path,
        class_names=names,
        packed=packed,
        class_id=class_ids,
        group_id=group_ids,
        metadata=document,
    )
    document["artifact_path"] = artifact_path
    return document
