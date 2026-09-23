# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare versioned data artifacts consumed by TAO Sparse4D."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from nvidia_tao_ds.annotations.sparse4d.contracts import normalize_npz_path, validate_class_names
from nvidia_tao_ds.annotations.sparse4d.lazy_index import (
    build_lazy_index, get_lazy_index_path, get_camera_counts_path,
)
from nvidia_tao_ds.annotations.sparse4d.ltt_geometry import (
    build_name_to_id,
    extract_ltt_data,
    resolve_scene_paths,
)
from nvidia_tao_ds.annotations.sparse4d.sidecars import (
    build_ltt_2dgt_scene,
    build_rtdetr_archive_sidecar,
)
from nvidia_tao_ds.annotations.sparse4d.sv2d import (
    build_sv2d_artifacts,
    load_dataset_manifest,
    write_split_artifacts,
    validate_suffix,
)
from nvidia_tao_ds.config.annotations.sparse4d_prepare_config import (
    Sparse4DPrepareConfig,
)
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


def _optional(value):
    """Convert an empty optional path or string to None."""
    return value if value not in (None, "") else None


def _required(value, field_name: str) -> str:
    """Return a configured non-empty value or raise an actionable error."""
    if value is None or not str(value).strip():
        raise ValueError(
            f"{field_name} is required for the selected Sparse4D operation"
        )
    return str(value)


def _scene_paths(selection) -> list[Path]:
    """Resolve a structured scene-selection block."""
    return resolve_scene_paths(
        data_root=_optional(selection.data_root),
        scenes=list(selection.scenes),
        scenes_file=_optional(selection.scenes_file),
        scene_dirs=list(selection.scene_dirs),
        train_split=_optional(selection.train_split),
        dedup_regex=str(selection.dedup_regex),
    )


def _taxonomy(cfg: Sparse4DPrepareConfig):
    """Return ordered class names, alias IDs, and source-to-target aliases."""
    class_names = list(validate_class_names(list(cfg.class_names)))
    subclass_map = {
        str(parent): list(subclasses)
        for parent, subclasses in dict(cfg.subclass_map).items()
    }
    name_to_id = build_name_to_id(class_names, subclass_map)
    alias_map = {
        source_name: class_names[class_id]
        for source_name, class_id in name_to_id.items()
        if source_name not in class_names
    }
    return class_names, name_to_id, alias_map


def _run_lazy_index(cfg: Sparse4DPrepareConfig) -> dict:
    """Build or incrementally refresh the trusted annotation index."""
    values = cfg.lazy_index
    workers = int(values.workers)
    source = _required(values.annotation_source, "lazy_index.annotation_source")
    outputs = [get_lazy_index_path(source)]
    if values.write_camera_counts:
        outputs.append(_optional(values.camera_counts_path) or get_camera_counts_path(source))
    _guard_outputs(cfg, outputs)
    return build_lazy_index(
        _required(
            values.annotation_source,
            "lazy_index.annotation_source",
        ),
        force=bool(values.force),
        num_workers=workers or None,
        write_camera_counts=bool(values.write_camera_counts),
        camera_counts_path=_optional(values.camera_counts_path),
    )


def _run_ltt_2dgt(cfg: Sparse4DPrepareConfig) -> dict:
    """Build one visible-2D sidecar per selected raw scene."""
    values = cfg.ltt_2dgt
    if values.annotation_version not in {"v0.0", "v0.1"}:
        raise ValueError("ltt_2dgt.annotation_version must be v0.0 or v0.1")
    output_dir = Path(
        _required(values.output_dir, "ltt_2dgt.output_dir")
    ).expanduser().resolve()
    class_names, _, alias_map = _taxonomy(cfg)
    class_map = {
        **alias_map,
        **{str(name): None for name in values.drop_names},
    }
    scene_paths = _scene_paths(values.selection)
    expected_paths = [
        output_dir / f"{scene_path.name}__ltt2dgt.npz"
        for scene_path in scene_paths
    ]
    output_owners = {}
    for scene_path, expected_path in zip(scene_paths, expected_paths):
        prior_scene = output_owners.get(expected_path)
        if prior_scene is not None and prior_scene != scene_path:
            raise ValueError(
                "Distinct scenes would write the same LTT 2D ground-truth "
                f"sidecar {expected_path}: {prior_scene} and {scene_path}"
            )
        output_owners[expected_path] = scene_path
    _guard_outputs(cfg, expected_paths)
    outputs = [
        build_ltt_2dgt_scene(
            scene_path,
            output_dir,
            class_names=class_names,
            class_name_map=class_map,
            anno_version=str(values.annotation_version),
            frame_stride=int(values.frame_stride),
            max_frames=int(values.max_frames_per_scene),
        )
        for scene_path in scene_paths
    ]
    return {
        "operation": "ltt_2dgt",
        "num_scenes": len(outputs),
        "output_paths": outputs,
    }


def _run_ltt_data(cfg: Sparse4DPrepareConfig) -> dict:
    """Extract model-independent raw geometry for LTT fitting."""
    values = cfg.ltt_data
    class_names, name_to_id, _ = _taxonomy(cfg)
    width, height = int(values.image_width), int(values.image_height)
    if (width == 0) != (height == 0):
        raise ValueError(
            "ltt_data.image_width and image_height must both be zero or positive"
        )
    image_size = (width, height) if width else None
    _guard_outputs(cfg, [normalize_npz_path(_required(values.output_path, "ltt_data.output_path"))])
    return extract_ltt_data(
        _scene_paths(values.selection),
        _required(values.output_path, "ltt_data.output_path"),
        class_names,
        name_to_id,
        calibration_file=_optional(values.calibration_file),
        calibration_mode=str(values.calibration_mode),
        annotation_version=str(values.annotation_version),
        frame_stride=int(values.frame_stride),
        max_frames_per_scene=int(values.max_frames_per_scene),
        min_visibility=float(values.min_visibility),
        max_per_class=int(values.max_per_class),
        image_size_override=image_size,
        seed=int(values.seed),
        metadata={
            "train_split": _optional(values.selection.train_split),
            "dedup_regex": str(values.selection.dedup_regex),
        },
    )


def _run_rtdetr_2d(cfg: Sparse4DPrepareConfig) -> dict:
    """Normalize archived RT-DETR KITTI labels into a safe cache."""
    values = cfg.rtdetr_2d
    input_dir = _required(values.input_dir, "rtdetr_2d.input_dir")
    _guard_outputs(cfg, [normalize_npz_path(_required(values.output_path, "rtdetr_2d.output_path"))])
    class_names, _, alias_map = _taxonomy(cfg)
    class_map = {**alias_map, **dict(values.class_map)}
    return build_rtdetr_archive_sidecar(
        input_dir,
        _required(values.output_path, "rtdetr_2d.output_path"),
        class_names=class_names,
        class_name_map=class_map,
        camera_map=dict(values.camera_map),
        confidence_threshold=float(values.confidence_threshold),
        frame_stride=int(values.frame_stride),
        max_frames_per_camera=int(values.max_frames_per_camera),
        scene_name=_optional(values.scene_name),
    )


def _run_sv2d(cfg: Sparse4DPrepareConfig) -> dict:
    """Build GT-less Sparse4D PKLs and paired safe 2D caches."""
    values = cfg.sv2d
    class_names, _, alias_map = _taxonomy(cfg)
    category_map = {**alias_map, **dict(values.class_map)}
    datasets = load_dataset_manifest(
        _required(values.manifest_path, "sv2d.manifest_path")
    )
    by_name = {dataset["name"]: dataset for dataset in datasets}
    if values.dataset == "all":
        selected = datasets
    elif values.dataset in by_name:
        selected = [by_name[values.dataset]]
    else:
        raise ValueError(
            f"Unknown SV2D dataset {values.dataset!r}; choose one of "
            f"{sorted(by_name)} or 'all'"
        )
    cache_dir = _required(values.cache_dir, "sv2d.cache_dir")
    pkl_dir = _required(values.pkl_dir, "sv2d.pkl_dir")
    suffix = validate_suffix(str(values.suffix))
    split_output = _optional(values.split_output)
    if split_output is None and values.dataset == "all":
        split_output = str(Path(pkl_dir).expanduser() / f"sv2d_train_split{suffix}.txt")
    planned = []
    for dataset in selected:
        scene = dataset["scene_name"] + suffix
        planned.extend([
            Path(cache_dir).expanduser() / f"{scene}__rtdetr2d.npz",
            Path(pkl_dir).expanduser() / f"{scene}_infos_train.pkl",
        ])
    if split_output:
        split_path = Path(split_output).expanduser()
        planned.extend([split_path, split_path.with_suffix(".sv2d_weights.json")])
    _guard_outputs(cfg, planned)
    results = [
        build_sv2d_artifacts(
            dataset,
            cache_dir,
            pkl_dir,
            class_names=class_names,
            category_map=category_map,
            drop_names=list(values.drop_names),
            canonical_width=int(values.canonical_width),
            canonical_height=int(values.canonical_height),
            max_images=int(values.max_images),
            keep_empty=bool(values.keep_empty),
            suffix=str(values.suffix),
        )
        for dataset in selected
    ]
    split_artifacts = (
        write_split_artifacts(results, selected, split_output)
        if split_output
        else None
    )
    return {
        "operation": "sv2d",
        "datasets": results,
        "split_artifacts": split_artifacts,
    }


_OPERATIONS = {
    "lazy_index": _run_lazy_index,
    "ltt_2dgt": _run_ltt_2dgt,
    "ltt_data": _run_ltt_data,
    "rtdetr_2d": _run_rtdetr_2d,
    "sv2d": _run_sv2d,
}


def _guard_outputs(cfg: Sparse4DPrepareConfig, paths) -> None:
    """Preflight all artifacts before the selected operation writes any output."""
    outputs = [Path(path).expanduser().resolve() for path in paths]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Output paths collide within the selected operation")
    existing = [path for path in outputs if path.exists()]
    if existing and not cfg.overwrite:
        raise FileExistsError(
            f"Refusing to replace {len(existing)} output(s); first: {existing[0]}. "
            "Set overwrite=true to replace existing artifacts."
        )


def run_operation(cfg: Sparse4DPrepareConfig) -> dict:
    """Run the selected operation and return its JSON-safe summary."""
    operation = str(cfg.operation)
    try:
        runner = _OPERATIONS[operation]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Sparse4D operation {operation!r}; "
            f"choose one of {sorted(_OPERATIONS)}"
        ) from error
    return runner(cfg)


@monitor_status(name="Sparse4D", mode="data preparation")
def run_sparse4d_prepare(cfg: Sparse4DPrepareConfig) -> None:
    """Run a Sparse4D data preparation operation with status reporting."""
    summary_path = Path(cfg.results_dir) / "sparse4d_prepare_summary.json"
    try:
        _guard_outputs(cfg, [summary_path])
        summary = run_operation(cfg)
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".sparse4d-summary-", dir=summary_path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temporary, summary_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print(text, end="")
    except OSError as error:
        # monitor_status records ValueError failures for API callers.
        raise ValueError(f"Sparse4D preparation I/O failed: {error}") from error


@hydra_runner(
    config_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../experiment_specs",
    ),
    config_name="sparse4d_prepare",
    schema=Sparse4DPrepareConfig,
)
def main(cfg: Sparse4DPrepareConfig) -> None:
    """Launch Sparse4D data preparation from a Hydra experiment spec."""
    run_sparse4d_prepare(cfg)


if __name__ == "__main__":
    main()
