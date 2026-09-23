# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified Sparse4D data-preparation entrypoint."""

import json
from pathlib import Path
import pickle

from omegaconf import OmegaConf
import pytest

from nvidia_tao_ds.annotations.scripts.sparse4d_prepare import run_operation
from nvidia_tao_ds.config.annotations.sparse4d_prepare_config import (
    Sparse4DPrepareConfig,
)


def _config(tmp_path: Path):
    """Return a mutable structured config with a local results directory."""
    cfg = OmegaConf.structured(Sparse4DPrepareConfig)
    cfg.results_dir = str(tmp_path / "results")
    return cfg


def _write_annotation_pkl(path: Path):
    """Write one minimal trusted Sparse4D annotation PKL."""
    with path.open("wb") as stream:
        pickle.dump(
            {
                "metadata": {"version": "fixture"},
                "infos": [
                    {
                        "scene_name": "scene",
                        "frame_idx": 0,
                        "timestamp": 0.0,
                        "cams": {"cam0": {}},
                    }
                ],
            },
            stream,
        )


def _write_raw_scene(path: Path):
    """Write one empty raw frame with a discoverable camera."""
    path.mkdir()
    (path / "ground_truth.json").write_text(
        json.dumps({"0": []}),
        encoding="utf-8",
    )
    (path / "calibration.json").write_text(
        json.dumps(
            {
                "sensors": [
                    {
                        "id": "cam0",
                        "type": "camera",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_lazy_index_operation_builds_runtime_paths(tmp_path):
    """Dispatch lazy indexing and preserve its conventional sibling outputs."""
    source = tmp_path / "annotations"
    source.mkdir()
    _write_annotation_pkl(source / "scene.pkl")
    cfg = _config(tmp_path)
    cfg.operation = "lazy_index"
    cfg.lazy_index.annotation_source = str(source)
    cfg.lazy_index.workers = 1

    summary = run_operation(cfg)

    assert summary["num_frames"] == 1
    assert Path(summary["cache_path"]) == source / "_lazy_index.pkl"
    assert Path(summary["camera_counts_path"]) == (
        source / "_pkl_cam_counts.pkl"
    )


def test_ltt_2dgt_operation_writes_empty_scene_and_guards_overwrite(tmp_path):
    """Keep background-only scenes representable and avoid accidental writes."""
    scene = tmp_path / "scene_a"
    _write_raw_scene(scene)
    cfg = _config(tmp_path)
    cfg.operation = "ltt_2dgt"
    cfg.ltt_2dgt.selection.scene_dirs = [str(scene)]
    cfg.ltt_2dgt.output_dir = str(tmp_path / "sidecars")

    summary = run_operation(cfg)

    assert summary["num_scenes"] == 1
    assert Path(summary["output_paths"][0]).name == "scene_a__ltt2dgt.npz"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_operation(cfg)

    cfg.overwrite = True
    assert run_operation(cfg)["num_scenes"] == 1


def test_ltt_2dgt_rejects_duplicate_output_paths_before_writing(tmp_path):
    """Reject distinct same-name scenes before either sidecar is written."""
    scenes = []
    for source_name in ("source_a", "source_b"):
        source_dir = tmp_path / source_name
        source_dir.mkdir()
        scene = source_dir / "shared_scene"
        _write_raw_scene(scene)
        scenes.append(scene)

    output_dir = tmp_path / "sidecars"
    cfg = _config(tmp_path)
    cfg.operation = "ltt_2dgt"
    cfg.ltt_2dgt.selection.scene_dirs = [str(scene) for scene in scenes]
    cfg.ltt_2dgt.output_dir = str(output_dir)
    cfg.overwrite = True

    with pytest.raises(
        ValueError,
        match="Distinct scenes would write the same LTT 2D ground-truth",
    ):
        run_operation(cfg)

    assert not output_dir.exists()


def test_sv2d_operation_builds_default_split_bundle(tmp_path):
    """Dispatch a portable COCO source through PKL, cache, and split outputs."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    coco = tmp_path / "coco.json"
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "frame.jpg",
                        "width": 100,
                        "height": 50,
                    }
                ],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 5, 20, 10],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "fixture",
                        "scene_name": "SV2D__fixture",
                        "kind": "file",
                        "coco": str(coco),
                        "image_root": str(image_root),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = _config(tmp_path)
    cfg.operation = "sv2d"
    cfg.sv2d.manifest_path = str(manifest)
    cfg.sv2d.cache_dir = str(tmp_path / "cache")
    cfg.sv2d.pkl_dir = str(tmp_path / "pkls")

    summary = run_operation(cfg)

    assert summary["datasets"][0]["num_frames"] == 1
    assert Path(summary["datasets"][0]["npz_path"]).is_file()
    assert Path(summary["datasets"][0]["pkl_path"]).is_file()
    assert Path(summary["split_artifacts"]["split_path"]).name == (
        "sv2d_train_split.txt"
    )
    assert Path(summary["split_artifacts"]["weights_path"]).is_file()


def test_selected_operation_validation_is_local(tmp_path):
    """Reject incomplete selected blocks without requiring unused inputs."""
    cfg = _config(tmp_path)
    cfg.operation = "rtdetr_2d"
    with pytest.raises(ValueError, match="rtdetr_2d.input_dir"):
        run_operation(cfg)

    cfg.operation = "ltt_data"
    cfg.ltt_data.image_width = 1920
    with pytest.raises(ValueError, match="image_width and image_height"):
        run_operation(cfg)

    cfg.operation = "unknown"
    with pytest.raises(ValueError, match="Unsupported Sparse4D operation"):
        run_operation(cfg)


def test_example_spec_matches_the_structured_schema():
    """Prevent example-spec drift from its Hydra dataclass."""
    spec_path = (
        Path(__file__).parents[1] /
        "nvidia_tao_ds" /
        "annotations" /
        "experiment_specs" /
        "sparse4d_prepare.yaml"
    )
    merged = OmegaConf.merge(
        OmegaConf.structured(Sparse4DPrepareConfig),
        OmegaConf.load(spec_path),
    )

    assert merged.operation == "lazy_index"
    assert list(merged.class_names) == [
        "person",
        "gr1_t2",
        "agility_digit",
        "nova_carter",
    ]
    assert list(merged.subclass_map.person) == ["Person", "Human"]
    assert merged.ltt_data.frame_stride == 10
    assert merged.sv2d.canonical_width == 1920
