# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build calibration-free SV2D bundles for Sparse4D 2D co-training."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import tempfile
from typing import Optional

import numpy as np

from nvidia_tao_ds.annotations.sparse4d.contracts import (
    validate_class_names,
    validate_scene_name,
    write_rtdetr_2d,
)


CANONICAL_WIDTH = 1920
CANONICAL_HEIGHT = 1080
DEFAULT_CAMERA_NAME = "cam0"
DEFAULT_DROP_CLASS_NAMES = frozenset({"pallet"})
DEFAULT_VIRTUAL_CAMERA = {
    "width": CANONICAL_WIDTH,
    "height": CANONICAL_HEIGHT,
    "fov_h_deg": 70.0,
    "cam_xyz": (-12.0, 0.0, 6.0),
    "target_xyz": (18.0, 0.0, 0.87),
    "world_up": (0.0, 0.0, 1.0),
}


def validate_suffix(suffix: str) -> str:
    """Return a suffix that cannot redirect generated artifact paths."""
    if not isinstance(suffix, str):
        raise ValueError("suffix must be a string")
    if any(separator in suffix for separator in ("/", "\\", "\x00")):
        raise ValueError("suffix must not contain path separators or NUL")
    return suffix


def build_intrinsic(width: int, height: int, fov_h_deg: float) -> np.ndarray:
    """Build the centred pinhole intrinsic used by the Sparse4D SV2D route."""
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("Virtual camera width and height must be positive integers")
    width = int(width)
    height = int(height)
    fov_h_deg = float(fov_h_deg)
    if width <= 0 or height <= 0 or not 0.0 < fov_h_deg < 180.0:
        raise ValueError(
            "Virtual camera dimensions must be positive and fov_h_deg in (0, 180)"
        )
    focal = (width / 2.0) / np.tan(np.deg2rad(fov_h_deg) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_world_to_camera(
    camera_xyz: Sequence[float],
    target_xyz: Sequence[float],
    world_up: Sequence[float],
) -> np.ndarray:
    """Build an OpenCV-style world-to-camera matrix (+Z forward, +Y down)."""
    try:
        camera = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        up = np.asarray(world_up, dtype=np.float64).reshape(3)
    except (TypeError, ValueError) as error:
        raise ValueError("Virtual camera vectors must each contain three values") from error
    if not np.isfinite(np.concatenate([camera, target, up])).all():
        raise ValueError("Virtual camera vectors must contain finite values")
    forward = target - camera
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError("Virtual camera target must differ from camera position")
    forward /= forward_norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        raise ValueError("world_up must not be parallel to the camera view")
    right /= right_norm
    down = np.cross(forward, right)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.stack([right, down, forward], axis=1)
    camera_to_world[:3, 3] = camera
    return np.linalg.inv(camera_to_world)


def build_virtual_camera(
    *,
    width: int = CANONICAL_WIDTH,
    height: int = CANONICAL_HEIGHT,
    values: Optional[Mapping] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return intrinsic, world-to-camera and JSON-safe camera metadata."""
    config = dict(DEFAULT_VIRTUAL_CAMERA)
    config.update(dict(values or {}))
    config["width"] = int(width)
    config["height"] = int(height)
    intrinsic = build_intrinsic(
        config["width"], config["height"], config["fov_h_deg"]
    )
    world_to_camera = build_world_to_camera(
        config["cam_xyz"], config["target_xyz"], config["world_up"]
    )
    metadata = {
        "width": config["width"],
        "height": config["height"],
        "fov_h_deg": float(config["fov_h_deg"]),
        "cam_xyz": [float(value) for value in config["cam_xyz"]],
        "target_xyz": [float(value) for value in config["target_xyz"]],
        "world_up": [float(value) for value in config["world_up"]],
    }
    return intrinsic, world_to_camera, metadata


def strip_h5_uri(file_name: str) -> str:
    """Convert ``h5://<tag>:<name>`` to a normalized dataset-relative key."""
    name = str(file_name)
    if name.startswith("h5://"):
        name = name[len("h5://"):]
        if ":" in name:
            name = name.split(":", 1)[1]
    name = name.lstrip("/")
    parts = PurePosixPath(name).parts
    if (
        not name or
        name == "." or
        ".." in parts or
        "\\" in name or
        "\x00" in name
    ):
        raise ValueError("SV2D image names must not be empty or traverse parents")
    return str(PurePosixPath(*parts))


def resolve_image_reference(dataset: Mapping, file_name: str, base_dir: Path):
    """Resolve a COCO image reference to a file path or ``(HDF5, key)`` tuple."""
    name = strip_h5_uri(file_name)
    kind = dataset.get("kind")
    if kind == "h5":
        if "h5_path" not in dataset:
            raise ValueError("An h5 SV2D dataset must provide h5_path")
        h5_path = Path(dataset["h5_path"]).expanduser()
        if not h5_path.is_absolute():
            h5_path = base_dir / h5_path
        return (str(h5_path.resolve()), f"rgb/{name}")
    if kind != "file":
        raise ValueError(f"Unsupported SV2D dataset kind {kind!r}")
    if "image_root" not in dataset:
        raise ValueError("A file SV2D dataset must provide image_root")
    image_root = Path(dataset["image_root"]).expanduser()
    if not image_root.is_absolute():
        image_root = base_dir / image_root
    image_root = image_root.resolve()
    image_path = (image_root / name).resolve()
    try:
        image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError("SV2D image names must remain under image_root") from error
    return str(image_path)


def _load_coco(value) -> tuple[dict, Path]:
    """Load a COCO mapping or JSON file and return its reference directory."""
    if isinstance(value, Mapping):
        return dict(value), Path.cwd()
    path = Path(value).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("SV2D COCO input must be a JSON object")
    return document, path.parent


def load_dataset_manifest(path: os.PathLike | str) -> list[dict]:
    """Load and resolve a portable list of file/HDF5 COCO datasets."""
    manifest_path = Path(path).expanduser().resolve()
    with open(manifest_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    datasets = document.get("datasets") if isinstance(document, Mapping) else document
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(
            "SV2D manifest must be a non-empty list or {'datasets': [...]} object"
        )
    resolved = []
    names = set()
    scenes = set()
    for raw_dataset in datasets:
        if not isinstance(raw_dataset, Mapping):
            raise ValueError("Every SV2D manifest entry must be an object")
        dataset = dict(raw_dataset)
        missing = {"name", "scene_name", "coco", "kind"}.difference(dataset)
        if missing:
            raise ValueError(f"SV2D manifest entry is missing {sorted(missing)}")
        name = dataset["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("SV2D manifest dataset names must be non-empty and unique")
        names.add(name)
        scene = validate_scene_name(dataset["scene_name"])
        if scene in scenes:
            raise ValueError("SV2D manifest scene_name values must be unique")
        scenes.add(scene)
        kind = dataset["kind"]
        if kind not in {"file", "h5"}:
            raise ValueError(f"Unsupported SV2D dataset kind {kind!r}")
        path_keys = ["coco", "h5_path" if kind == "h5" else "image_root"]
        for key in path_keys:
            if key not in dataset:
                raise ValueError(f"SV2D dataset {name!r} is missing {key}")
            value = Path(dataset[key]).expanduser()
            if not value.is_absolute():
                value = manifest_path.parent / value
            dataset[key] = str(value.resolve())
        dataset.setdefault("weight", 1.0)
        resolved.append(dataset)
    return resolved


def _stable_id(value) -> tuple[str, object]:
    """Return a deterministic sortable key for a JSON identifier."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("COCO IDs must be integers or strings")
    return type(value).__name__, value


def _positive_dimension(value, name: str) -> int:
    """Return a positive, losslessly converted integer dimension."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        converted = int(value)
        exact = float(value) == converted
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if converted <= 0 or not exact:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _scaled_xywh(
    value,
    source_width: float,
    source_height: float,
    target_width: int,
    target_height: int,
) -> Optional[list[float]]:
    """Scale and clip a COCO xywh box, dropping degenerate geometry."""
    try:
        x_min, y_min, box_width, box_height = (
            float(component) for component in value
        )
    except (TypeError, ValueError) as error:
        raise ValueError("COCO bbox must contain four numeric xywh values") from error
    dimensions = [source_width, source_height, x_min, y_min, box_width, box_height]
    if not np.isfinite(dimensions).all() or source_width <= 0 or source_height <= 0:
        raise ValueError("COCO image dimensions and bbox values must be finite")
    if box_width <= 0 or box_height <= 0:
        return None
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    box = [
        float(np.clip(x_min * scale_x, 0.0, target_width)),
        float(np.clip(y_min * scale_y, 0.0, target_height)),
        float(np.clip((x_min + box_width) * scale_x, 0.0, target_width)),
        float(np.clip((y_min + box_height) * scale_y, 0.0, target_height)),
    ]
    if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
        return None
    return box


def _atomic_pickle(path: Path, value) -> None:
    """Write one trusted runtime pickle atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_text(path: Path, value: str) -> None:
    """Write UTF-8 text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_sv2d_artifacts(
    dataset: Mapping,
    cache_dir: os.PathLike | str,
    pkl_dir: os.PathLike | str,
    *,
    class_names: Sequence[str],
    category_map: Optional[Mapping[str, Optional[str]]] = None,
    drop_names: Optional[Sequence[str]] = None,
    canonical_width: int = CANONICAL_WIDTH,
    canonical_height: int = CANONICAL_HEIGHT,
    virtual_camera: Optional[Mapping] = None,
    camera_name: str = DEFAULT_CAMERA_NAME,
    frames_per_second: float = 30.0,
    max_images: int = 0,
    keep_empty: bool = False,
    suffix: str = "",
) -> dict:
    """Build one GT-less PKL and paired ``ltt_rtdetr2d/v1`` cache from COCO."""
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping")
    if not isinstance(dataset.get("name"), str) or not dataset["name"]:
        raise ValueError("SV2D dataset name must be a non-empty string")
    scene = validate_scene_name(dataset.get("scene_name"))
    suffix = validate_suffix(suffix)
    artifact_scene = validate_scene_name(f"{scene}{suffix}")
    classes = validate_class_names(class_names)
    camera_names = validate_class_names([camera_name])
    canonical_width = _positive_dimension(canonical_width, "canonical_width")
    canonical_height = _positive_dimension(canonical_height, "canonical_height")
    if isinstance(max_images, bool) or int(max_images) != max_images or max_images < 0:
        raise ValueError("max_images must be a non-negative integer")
    max_images = int(max_images)
    frames_per_second = float(frames_per_second)
    if not np.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("frames_per_second must be finite and positive")

    coco, base_dir = _load_coco(dataset.get("coco"))
    images = coco.get("images", [])
    categories = coco.get("categories", [])
    coco_annotations = coco.get("annotations", [])
    if not all(
        isinstance(values, list)
        for values in (images, categories, coco_annotations)
    ):
        raise ValueError("COCO images, categories and annotations must be lists")

    class_to_id = {name: index for index, name in enumerate(classes)}
    requested_map = dict(category_map or {})
    for source_name, target_name in requested_map.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("category_map keys must be non-empty strings")
        if target_name is not None and target_name not in class_to_id:
            raise ValueError(
                f"category_map target {target_name!r} is not in class_names"
            )
    dropped_names = set(DEFAULT_DROP_CLASS_NAMES if drop_names is None else drop_names)
    if any(not isinstance(name, str) or not name for name in dropped_names):
        raise ValueError("drop_names must contain non-empty strings")
    category_targets = {}
    category_names = {}
    for category in categories:
        if not isinstance(category, Mapping) or "id" not in category:
            raise ValueError("Every COCO category must be an object containing id")
        category_id = _stable_id(category["id"])
        if category_id in category_targets:
            raise ValueError(f"Duplicate COCO category ID {category['id']!r}")
        source_name = category.get("name")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("Every COCO category must have a non-empty name")
        target_name = requested_map.get(source_name, source_name)
        if source_name in dropped_names and source_name not in requested_map:
            target_name = None
        if target_name is not None and target_name not in class_to_id:
            raise ValueError(
                f"COCO category {source_name!r} resolves to {target_name!r}, which "
                "is not in the ordered class_names taxonomy"
            )
        category_targets[category_id] = (
            None if target_name is None else class_to_id[target_name]
        )
        category_names[category_id] = source_name

    image_records = {}
    for image in images:
        if not isinstance(image, Mapping) or "id" not in image:
            raise ValueError("Every COCO image must be an object containing id")
        image_id = _stable_id(image["id"])
        if image_id in image_records:
            raise ValueError(f"Duplicate COCO image ID {image['id']!r}")
        image_records[image_id] = image

    annotations_by_image = {}
    for annotation in coco_annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("Every COCO annotation must be an object")
        image_id = _stable_id(annotation.get("image_id"))
        category_id = _stable_id(annotation.get("category_id"))
        if image_id not in image_records:
            raise ValueError(
                f"COCO annotation references unknown image ID {annotation.get('image_id')!r}"
            )
        if category_id not in category_targets:
            raise ValueError(
                "COCO annotation references unknown category ID "
                f"{annotation.get('category_id')!r}"
            )
        class_id = category_targets[category_id]
        if class_id is None:
            continue
        image = image_records[image_id]
        try:
            source_width = float(image["width"])
            source_height = float(image["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every annotated COCO image needs numeric dimensions") from error
        box = _scaled_xywh(
            annotation.get("bbox"),
            source_width,
            source_height,
            int(canonical_width),
            int(canonical_height),
        )
        if box is not None:
            annotations_by_image.setdefault(image_id, []).append((class_id, box))

    intrinsic, world_to_camera, virtual_camera_metadata = build_virtual_camera(
        width=canonical_width,
        height=canonical_height,
        values=virtual_camera,
    )
    infos = []
    detection_rows = []
    for image_id in sorted(image_records):
        image = image_records[image_id]
        try:
            source_width = float(image["width"])
            source_height = float(image["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every COCO image needs numeric dimensions") from error
        if (
            not np.isfinite([source_width, source_height]).all() or
            source_width <= 0 or
            source_height <= 0
        ):
            raise ValueError("COCO image dimensions must be finite and positive")
        mapped_boxes = sorted(
            annotations_by_image.get(image_id, []),
            key=lambda item: (item[0], *item[1]),
        )
        if not mapped_boxes and not keep_empty:
            continue
        if "file_name" not in image:
            raise ValueError("Every retained COCO image must provide file_name")

        frame_id = len(infos)
        for class_id, box in mapped_boxes:
            detection_rows.append((frame_id, class_id, box))
        token = f"{artifact_scene}__{frame_id:09d}"
        infos.append(
            {
                "frame_idx": frame_id,
                "cams": {
                    camera_names[0]: {
                        "data_path": resolve_image_reference(
                            dataset, image["file_name"], base_dir
                        ),
                        "sample_data_token": f"{token}+{camera_names[0]}",
                        "cam_intrinsic": intrinsic.copy(),
                        # Sparse4D retains this historical field name but uses
                        # it as world-to-camera while building projection_mat.
                        "sensor2world_transform": world_to_camera.copy(),
                        "group_info_dict": {
                            "origin": np.asarray([0.0, 0.0], dtype=np.float32),
                            "dimensions": np.asarray(
                                [-50.0, -50.0, 50.0, 50.0], dtype=np.float32
                            ),
                        },
                    }
                },
                "scene_name": artifact_scene,
                "timestamp": float(frame_id) / frames_per_second,
                "token": token,
                "group_name": artifact_scene,
                "gt_boxes": None,
            }
        )
        if max_images and len(infos) >= max_images:
            break

    metadata = {
        "conf_thr": 0.0,
        "frame_stride": 1,
        "num_valid_frame_cameras": len(infos),
        "source": (
            "SV2D COCO GT; canonical "
            f"{int(canonical_width)}x{int(canonical_height)}"
        ),
        "class_map": {
            category_names[category_id]: classes[target_id]
            for category_id, target_id in sorted(category_targets.items())
            if target_id is not None
        },
        "virtual_camera": virtual_camera_metadata,
    }
    cache_path = Path(cache_dir).expanduser().resolve() / (
        f"{artifact_scene}__rtdetr2d.npz"
    )
    pkl_path = Path(pkl_dir).expanduser().resolve() / (
        f"{artifact_scene}_infos_train.pkl"
    )
    written_cache = write_rtdetr_2d(
        cache_path,
        scene=artifact_scene,
        class_names=classes,
        camera_names=camera_names,
        frame_id=[row[0] for row in detection_rows],
        cam=[0] * len(detection_rows),
        class_id=[row[1] for row in detection_rows],
        box=[row[2] for row in detection_rows],
        score=[1.0] * len(detection_rows),
        valid_frame_id=list(range(len(infos))),
        valid_cam=[0] * len(infos),
        metadata=metadata,
    )
    _atomic_pickle(
        pkl_path,
        {"infos": infos, "metadata": {"version": "sv2d_2d_only"}},
    )
    class_counts = Counter(row[1] for row in detection_rows)
    return {
        "dataset": dataset["name"],
        "scene": artifact_scene,
        "num_frames": len(infos),
        "num_detections": len(detection_rows),
        "per_class": {
            classes[class_id]: count
            for class_id, count in sorted(class_counts.items())
        },
        "npz_path": written_cache,
        "pkl_path": str(pkl_path),
    }


def write_split_artifacts(
    results: Sequence[Mapping],
    datasets: Sequence[Mapping],
    output_path: os.PathLike | str,
) -> dict:
    """Atomically write a sequence-safe split and advisory scene weights."""
    dataset_by_name = {}
    for dataset in datasets:
        name = dataset.get("name")
        if not isinstance(name, str) or not name or name in dataset_by_name:
            raise ValueError("SV2D manifest dataset names must be non-empty and unique")
        dataset_by_name[name] = dataset

    lines = []
    scene_weights = {}
    seen_paths = set()
    for result in results:
        dataset_name = result.get("dataset")
        if dataset_name not in dataset_by_name:
            raise ValueError(f"No SV2D manifest entry for result {dataset_name!r}")
        scene = validate_scene_name(result.get("scene"))
        if scene in scene_weights:
            raise ValueError(f"Duplicate SV2D result scene {scene!r}")
        try:
            weight = float(dataset_by_name[dataset_name].get("weight", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("SV2D weights must be finite and positive") from error
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("SV2D weights must be finite and positive")
        pkl_path = str(Path(result["pkl_path"]).expanduser().resolve())
        if pkl_path in seen_paths:
            raise ValueError(f"Duplicate SV2D PKL path {pkl_path!r}")
        seen_paths.add(pkl_path)
        lines.append(pkl_path)
        scene_weights[scene] = weight

    output = Path(output_path).expanduser().resolve()
    weights_path = output.with_suffix(".sv2d_weights.json")
    _atomic_text(output, "".join(f"{path}\n" for path in lines))
    _atomic_text(
        weights_path,
        json.dumps(scene_weights, indent=2, sort_keys=True) + "\n",
    )
    return {"split_path": str(output), "weights_path": str(weights_path)}
