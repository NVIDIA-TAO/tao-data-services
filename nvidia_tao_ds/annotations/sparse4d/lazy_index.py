# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the trusted-pickle index consumed by Sparse4D lazy loading."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import math
from numbers import Integral, Real
import os
from pathlib import Path
import pickle
import tempfile
from typing import Optional


GENERATED_FILENAMES = {"_lazy_index.pkl", "_pkl_cam_counts.pkl"}


def _is_annotation_pkl(path: Path) -> bool:
    """Return whether a path is a source annotation rather than a cache."""
    return (
        path.is_file() and
        path.suffix == ".pkl" and
        path.name not in GENERATED_FILENAMES and
        not path.name.endswith("_lazy_index.pkl")
    )


def get_lazy_index_path(annotation_source: os.PathLike | str) -> Path:
    """Return the cache path expected by the TAO Sparse4D runtime."""
    source = Path(annotation_source).expanduser()
    if source.is_dir():
        return source / "_lazy_index.pkl"
    if source.suffix == ".txt":
        return source.with_name(f"{source.stem}_lazy_index.pkl")
    raise ValueError(
        f"Expected an annotation directory or split .txt file: {source}"
    )


def get_camera_counts_path(annotation_source: os.PathLike | str) -> Path:
    """Return the default camera-count sidecar path."""
    source = Path(annotation_source).expanduser()
    base_dir = source if source.is_dir() else source.parent
    return base_dir / "_pkl_cam_counts.pkl"


def resolve_annotation_paths(annotation_source: os.PathLike | str) -> list[Path]:
    """Resolve deterministic absolute PKL paths from a directory or split file."""
    source = Path(annotation_source).expanduser()
    if source.is_dir():
        paths = [
            path
            for path in sorted(source.iterdir())
            if _is_annotation_pkl(path)
        ]
    elif source.suffix == ".txt":
        if not source.is_file():
            raise FileNotFoundError(f"Annotation split not found: {source}")
        paths = []
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                value = Path(stripped.split()[0]).expanduser()
                if not value.is_absolute():
                    value = source.parent / value
                paths.append(value)
    else:
        raise ValueError(
            f"Expected an annotation directory or split .txt file: {source}"
        )

    paths = [path.resolve() for path in paths]
    if not paths:
        raise ValueError(f"No annotation PKLs found in: {source}")
    generated = [
        path
        for path in paths
        if path.name in GENERATED_FILENAMES or
        path.name.endswith("_lazy_index.pkl")
    ]
    if generated:
        raise ValueError(f"Split file references a generated cache: {generated[0]}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} annotation PKL(s) do not exist; first: {missing[0]}"
        )
    return paths


def _load_annotation_document(path: Path) -> tuple[dict, dict]:
    """Load and validate the top-level structure of an annotation PKL."""
    with path.open("rb") as stream:
        document = pickle.load(stream)
    if not isinstance(document, dict) or not isinstance(
        document.get("infos"), list
    ):
        raise ValueError("expected a dictionary containing an 'infos' list")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("expected 'metadata' to be a dictionary")
    return document, metadata


def _validate_scene_name(value, local_idx: int) -> str:
    """Validate a scene name used by the runtime's global frame sort."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"infos[{local_idx}].scene_name must be a nonempty string"
        )
    return value


def _validate_timestamp(value, local_idx: int):
    """Validate a timestamp used by the runtime's global frame sort."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"infos[{local_idx}].timestamp must be a finite real number"
        )
    if not isinstance(value, Integral):
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                f"infos[{local_idx}].timestamp must be a finite real number"
            )
    return value


def _validate_frame_idx(value, local_idx: int) -> int:
    """Return a losslessly normalized integer frame index."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"infos[{local_idx}].frame_idx must be a lossless integer"
        )
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"infos[{local_idx}].frame_idx must be a lossless integer"
        ) from error
    if value != normalized:
        raise ValueError(
            f"infos[{local_idx}].frame_idx must be a lossless integer"
        )
    return normalized


def _validate_camera_names(cameras, local_idx: int) -> frozenset[str]:
    """Validate and return the camera-name set for one frame."""
    if not isinstance(cameras, Mapping):
        raise ValueError(f"infos[{local_idx}].cams must be a mapping")
    for camera_name in cameras:
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError(
                f"infos[{local_idx}].cams must use nonempty string names"
            )
    return frozenset(cameras)


def _signature_from_stat(stat_result: os.stat_result) -> dict[str, int]:
    """Return the deterministic file attributes used for cache reuse."""
    return {
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _index_one_pkl(pkl_path: Path | str) -> tuple:
    """Return validated entries and cache bookkeeping for one PKL."""
    path = Path(pkl_path).expanduser().resolve()
    try:
        stat_before = path.stat()
        document, metadata = _load_annotation_document(path)
        infos = document["infos"]
        entries = []
        expected_camera_names = None
        for local_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                raise ValueError(f"infos[{local_idx}] must be a dictionary")
            missing = {"scene_name", "timestamp", "cams"}.difference(info)
            if missing:
                raise ValueError(
                    f"infos[{local_idx}] is missing required keys {sorted(missing)}"
                )
            scene_name = _validate_scene_name(info["scene_name"], local_idx)
            timestamp = _validate_timestamp(info["timestamp"], local_idx)
            frame_idx = _validate_frame_idx(
                info.get("frame_idx", local_idx), local_idx
            )
            camera_names = _validate_camera_names(info["cams"], local_idx)
            if expected_camera_names is None:
                expected_camera_names = camera_names
            elif camera_names != expected_camera_names:
                raise ValueError(
                    f"infos[{local_idx}].cams has camera names "
                    f"{sorted(camera_names)}, expected "
                    f"{sorted(expected_camera_names)} from infos[0]"
                )
            entries.append(
                {
                    "pkl_path": str(path),
                    "local_idx": local_idx,
                    "scene_name": scene_name,
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                }
            )
        camera_count = len(expected_camera_names) if expected_camera_names else 0
        stat_after = path.stat()
        signature = _signature_from_stat(stat_after)
        if signature != _signature_from_stat(stat_before):
            raise ValueError("annotation PKL changed while it was being indexed")
        return (
            str(path),
            entries,
            metadata,
            stat_after.st_mtime,
            signature,
            camera_count,
            len(entries),
        )
    except Exception as error:
        raise RuntimeError(f"Failed to index annotation PKL: {path}") from error


def _load_pickle_mapping(path: Path) -> dict:
    """Load a mapping produced by this trusted dataset-generation workflow."""
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a dictionary in cache: {path}")
    return value


def _safe_load_pickle_mapping(path: Path) -> dict:
    """Load cache bookkeeping, returning an empty cache when it is malformed."""
    try:
        return _load_pickle_mapping(path)
    except Exception:
        return {}


def _group_cached_entries(frame_index) -> dict[str, list[dict]]:
    """Validate and group cached frame entries by their absolute source path."""
    if not isinstance(frame_index, list):
        return {}
    grouped = {}
    for entry in frame_index:
        if not isinstance(entry, dict):
            return {}
        required = {
            "pkl_path", "local_idx", "scene_name", "timestamp", "frame_idx"
        }
        if not required.issubset(entry):
            return {}
        pkl_path = entry["pkl_path"]
        local_idx = entry["local_idx"]
        valid_path = (
            isinstance(pkl_path, str) and
            bool(pkl_path) and
            Path(pkl_path).is_absolute()
        )
        valid_local_idx = not (
            isinstance(local_idx, bool) or
            not isinstance(local_idx, Integral) or
            local_idx < 0
        )
        if not valid_path or not valid_local_idx:
            return {}
        try:
            _validate_scene_name(entry["scene_name"], int(local_idx))
            _validate_timestamp(entry["timestamp"], int(local_idx))
            _validate_frame_idx(entry["frame_idx"], int(local_idx))
        except ValueError:
            return {}
        grouped.setdefault(pkl_path, []).append(entry)
    return grouped


def _cached_nonnegative_integer(values, key: str) -> Optional[int]:
    """Return one valid cached nonnegative integer, otherwise ``None``."""
    if not isinstance(values, Mapping) or key not in values:
        return None
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        return None
    return int(value)


def _normalize_signature(value) -> Optional[dict[str, int]]:
    """Normalize a cached size/mtime-nanoseconds signature when valid."""
    if not isinstance(value, Mapping):
        return None
    size = value.get("size")
    mtime_ns = value.get("mtime_ns")
    if (
        isinstance(size, bool) or
        not isinstance(size, Integral) or
        size < 0 or
        isinstance(mtime_ns, bool) or
        not isinstance(mtime_ns, Integral)
    ):
        return None
    return {"size": int(size), "mtime_ns": int(mtime_ns)}


def _read_current_metadata(annotation_paths: list[Path]) -> dict:
    """Read the first nonempty metadata mapping from current sources."""
    for path in annotation_paths:
        try:
            stat_before = path.stat()
            _, metadata = _load_annotation_document(path)
            stat_after = path.stat()
            if _signature_from_stat(stat_before) != _signature_from_stat(stat_after):
                raise ValueError("annotation PKL changed while reading metadata")
        except Exception as error:
            raise RuntimeError(
                f"Failed to read annotation metadata: {path}"
            ) from error
        if metadata:
            return metadata
    return {}


def _atomic_pickle_dump(value, output_path: Path) -> None:
    """Atomically serialize a trusted pickle next to its destination."""
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    try:
        with open(temporary_path, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def build_lazy_index(
    annotation_source: os.PathLike | str,
    *,
    force: bool = False,
    num_workers: Optional[int] = None,
    write_camera_counts: bool = True,
    camera_counts_path: Optional[os.PathLike | str] = None,
) -> dict:
    """Build or incrementally refresh a Sparse4D lazy annotation index."""
    source = Path(annotation_source).expanduser()
    annotation_paths = resolve_annotation_paths(source)
    unique_paths = list(dict.fromkeys(annotation_paths))
    cache_path = get_lazy_index_path(source)
    sidecar_path = (
        Path(camera_counts_path).expanduser()
        if camera_counts_path is not None
        else get_camera_counts_path(source)
    )
    source_paths = {source.resolve(), *unique_paths}
    resolved_cache_path = cache_path.resolve()
    if resolved_cache_path in source_paths:
        raise ValueError(
            "Lazy-index output path conflicts with an annotation source: "
            f"{resolved_cache_path}"
        )
    if write_camera_counts:
        resolved_sidecar_path = sidecar_path.resolve()
        if (
            resolved_sidecar_path == resolved_cache_path or
            resolved_sidecar_path in source_paths
        ):
            raise ValueError(
                "Camera-count output path conflicts with an annotation source "
                f"or lazy index: {resolved_sidecar_path}"
            )

    existing_entries = {}
    previous_counts = {}
    previous_frame_counts = {}
    previous_signatures = {}
    if not force and cache_path.is_file():
        cached = _safe_load_pickle_mapping(cache_path)
        existing_entries = _group_cached_entries(cached.get("frame_index"))
        cached_counts = cached.get("pkl_cam_counts", {})
        if isinstance(cached_counts, Mapping):
            previous_counts = cached_counts
        cached_frame_counts = cached.get("pkl_frame_counts", {})
        if isinstance(cached_frame_counts, Mapping):
            previous_frame_counts = cached_frame_counts
        cached_signatures = cached.get("signatures", {})
        if isinstance(cached_signatures, Mapping):
            previous_signatures = cached_signatures
        if write_camera_counts and sidecar_path.is_file():
            sidecar_counts = _safe_load_pickle_mapping(sidecar_path)
            previous_counts = {**sidecar_counts, **previous_counts}

    reusable = set()
    paths_to_index = []
    current_mtimes = {}
    current_signatures = {}
    for path in unique_paths:
        key = str(path)
        stat_result = path.stat()
        current_mtimes[key] = stat_result.st_mtime
        current_signatures[key] = _signature_from_stat(stat_result)
        camera_count = _cached_nonnegative_integer(previous_counts, key)
        frame_count = _cached_nonnegative_integer(previous_frame_counts, key)
        entries = existing_entries.get(key, [])
        if (
            _normalize_signature(previous_signatures.get(key)) ==
            current_signatures[key] and
            camera_count is not None and
            frame_count is not None and
            len(entries) == frame_count and
            all(
                int(entry["local_idx"]) == local_idx
                for local_idx, entry in enumerate(entries)
            )
        ):
            reusable.add(key)
        else:
            paths_to_index.append(path)

    if num_workers is None:
        num_workers = min(32, os.cpu_count() or 1)
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_workers == 1:
        indexed_values = [_index_one_pkl(path) for path in paths_to_index]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            indexed_values = list(executor.map(_index_one_pkl, paths_to_index))
    indexed = {value[0]: value for value in indexed_values}

    frame_index = []
    mtimes = {}
    signatures = {}
    camera_counts = {}
    frame_counts = {}
    for path in annotation_paths:
        key = str(path)
        if key in indexed:
            (
                _, entries, _, mtime, signature, camera_count, frame_count
            ) = indexed[key]
        else:
            entries = existing_entries.get(key, [])
            mtime = current_mtimes[key]
            signature = current_signatures[key]
            camera_count = _cached_nonnegative_integer(previous_counts, key)
            frame_count = _cached_nonnegative_integer(previous_frame_counts, key)
        frame_index.extend(entries)
        mtimes[key] = mtime
        signatures[key] = signature
        camera_counts[key] = int(camera_count)
        frame_counts[key] = int(frame_count)

    metadata = _read_current_metadata(unique_paths)

    index_data = {
        "frame_index": frame_index,
        "metadata": metadata,
        "mtimes": mtimes,
        "pkl_cam_counts": camera_counts,
        "signatures": signatures,
        "pkl_frame_counts": frame_counts,
    }
    _atomic_pickle_dump(index_data, cache_path)
    if write_camera_counts:
        _atomic_pickle_dump(camera_counts, sidecar_path)

    return {
        "cache_path": str(cache_path.resolve()),
        "camera_counts_path": (
            str(sidecar_path.resolve()) if write_camera_counts else None
        ),
        "num_frames": len(frame_index),
        "num_pkls": len(unique_paths),
        "num_reused_pkls": len(reusable),
        "num_indexed_pkls": len(paths_to_index),
    }
