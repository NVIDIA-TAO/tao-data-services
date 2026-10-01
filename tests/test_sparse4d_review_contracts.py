# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for reviewed producer safety and service contracts."""

import io
import json
import pickle
import tarfile
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from nvidia_tao_ds.annotations.scripts import sparse4d_prepare as prepare
from nvidia_tao_ds.annotations.sparse4d import ltt_geometry, sidecars
from nvidia_tao_ds.annotations.sparse4d.contracts import (
    RTDETR_2D_SCHEMA_VERSION, load_contract, validate_scene_name,
)
from nvidia_tao_ds.config.annotations.sparse4d_prepare_config import Sparse4DPrepareConfig


def _config(tmp_path, operation):
    """Use only generated inputs and a private result directory."""
    cfg = OmegaConf.structured(Sparse4DPrepareConfig)
    cfg.results_dir = str(tmp_path / "results")
    cfg.operation = operation
    return cfg


@pytest.mark.parametrize("operation", ["lazy_index", "ltt_data", "rtdetr_2d", "sv2d"])
def test_preflight_checks_outputs_before_any_writer(tmp_path, monkeypatch, operation):
    """Every dispatch path refuses existing artifacts without partial output."""
    cfg = _config(tmp_path, operation)
    if operation == "lazy_index":
        cfg.lazy_index.annotation_source = str(tmp_path)
        output = tmp_path / "_pkl_cam_counts.pkl"
        writer = "build_lazy_index"
    elif operation == "ltt_data":
        cfg.ltt_data.output_path = str(tmp_path / "geometry")
        output = tmp_path / "geometry.npz"
        writer = "extract_ltt_data"
    elif operation == "rtdetr_2d":
        cfg.rtdetr_2d.input_dir = str(tmp_path / "labels")
        cfg.rtdetr_2d.output_path = str(tmp_path / "detector")
        output = tmp_path / "detector.npz"
        writer = "build_rtdetr_archive_sidecar"
    else:
        cfg.sv2d.manifest_path = str(tmp_path / "manifest.json")
        cfg.sv2d.cache_dir = str(tmp_path / "cache")
        cfg.sv2d.pkl_dir = str(tmp_path / "pkls")
        cfg.sv2d.split_output = str(tmp_path / "train.txt")
        output = tmp_path / "train.sv2d_weights.json"
        writer = "build_sv2d_artifacts"
        monkeypatch.setattr(prepare, "load_dataset_manifest", lambda _: [
            {"name": "fixture", "scene_name": "SV2D__fixture"},
        ])

    def unexpected_writer(*_args, **_kwargs):
        pytest.fail("Writer ran before output preflight completed")

    monkeypatch.setattr(prepare, writer, unexpected_writer)
    output.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        prepare.run_operation(cfg)
    assert output.read_bytes() == b"existing user artifact"
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "pkls").exists()


def test_monitored_operation_persists_summary_and_failure_status(tmp_path):
    """A successful summary is machine readable; malformed PKLs report FAILURE."""
    source = tmp_path / "annotations"
    source.mkdir()
    annotation = source / "scene.pkl"
    with annotation.open("wb") as stream:
        pickle.dump({"infos": [], "metadata": {"version": "fixture"}}, stream)
    cfg = _config(tmp_path, "lazy_index")
    cfg.lazy_index.annotation_source = str(source)
    cfg.lazy_index.workers = 1
    prepare.run_sparse4d_prepare(cfg)
    summary_path = Path(cfg.results_dir) / "sparse4d_prepare_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_pkls"] == 1
    assert Path(summary["cache_path"]).is_file()
    summary_before = summary_path.read_bytes()

    with pytest.raises(ValueError, match="overwrite=true"):
        prepare.run_sparse4d_prepare(cfg)
    assert summary_path.read_bytes() == summary_before

    cfg.overwrite = True
    cfg.lazy_index.force = True
    annotation.write_bytes(b"malformed pickle")
    with pytest.raises(ValueError, match="scene.pkl"):
        prepare.run_sparse4d_prepare(cfg)
    status = (Path(cfg.results_dir) / "status.json").read_text(encoding="utf-8")
    assert '"status": "FAILURE"' in status
    assert summary_path.read_bytes() == summary_before


def test_detector_requires_explicit_alias_or_drop(tmp_path):
    """Unknown foreground must not silently turn into a valid background frame."""
    camera = tmp_path / "scene" / "rt-detr" / "cam0"
    camera.mkdir(parents=True)
    payload = b"Human 0 0 0 1 2 20 30 0 0 0 0 0 0 0 0.9\n"
    with tarfile.open(camera / "labels.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("labels/frame_000001.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "detector.npz"
    kwargs = {"class_names": ["person"], "scene_name": "scene"}
    with pytest.raises(ValueError, match="Unmapped detector class"):
        sidecars.build_rtdetr_archive_sidecar(camera.parent, output, **kwargs)
    assert not output.exists()

    sidecars.build_rtdetr_archive_sidecar(
        camera.parent, output, class_name_map={"Human": "person"}, **kwargs,
    )
    assert load_contract(output, RTDETR_2D_SCHEMA_VERSION)["class_id"].tolist() == [0]
    sidecars.build_rtdetr_archive_sidecar(
        camera.parent, output, class_name_map={"Human": None}, **kwargs,
    )
    dropped = load_contract(output, RTDETR_2D_SCHEMA_VERSION)
    assert dropped["class_id"].size == 0
    assert dropped["valid_frame_id"].tolist() == [1]


@pytest.mark.parametrize("scene", ["scene+group0", "SV2D__data+variant"])
def test_scene_name_rejects_reserved_bev_group_separator(scene):
    """Runtime scene normalization must not truncate producer cache keys."""
    with pytest.raises(ValueError, match="BEV"):
        validate_scene_name(scene)


@pytest.mark.parametrize("streamed", [False, True])
def test_ltt_producers_share_sampling_and_metadata_policy(tmp_path, monkeypatch, streamed):
    """Both producers select exactly the same frames with either JSON parser."""
    if not streamed:
        monkeypatch.setattr(ltt_geometry, "_ijson", None)
    else:
        pytest.importorskip("ijson")
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "ground_truth.json").write_text(
        '{"metadata": {"version": "fixture"}, "4": [], "0": [], "2": []}',
        encoding="utf-8",
    )
    expected = list(ltt_geometry.iter_gt_frames(scene, frame_stride=2, max_frames=2))
    actual = list(sidecars.iter_aicity_annotation_frames(scene, frame_stride=2, max_frames=2))
    assert [(frame["frame_id"], frame["annotations"]) for frame in actual] == expected
    assert [frame[0] for frame in expected] == [4, 0]
