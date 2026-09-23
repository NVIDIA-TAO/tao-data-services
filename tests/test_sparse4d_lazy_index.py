# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tests for Sparse4D lazy annotation indexing."""

import os
from pathlib import Path
import pickle

import pytest

from nvidia_tao_ds.annotations.sparse4d.lazy_index import (
    build_lazy_index,
    resolve_annotation_paths,
)


@pytest.fixture(autouse=True)
def _runtime_cwd(tmp_path, monkeypatch):
    """Use the same working directory for producer and runtime relative paths."""
    monkeypatch.chdir(tmp_path)


def _write_document(path: Path, infos: list[dict], metadata=None):
    """Write one trusted annotation document for a focused fixture."""
    document = {"infos": infos}
    if metadata is not None:
        document["metadata"] = metadata
    with path.open("wb") as stream:
        pickle.dump(document, stream)


def _write_pkl(
    path: Path,
    scene: str,
    camera_count: int,
    frame_count: int = 2,
    metadata=None,
):
    """Write a minimal trusted Sparse4D annotation pickle."""
    cameras = {f"Camera{index}": {} for index in range(camera_count)}
    _write_document(
        path,
        [
            {
                "scene_name": scene,
                "timestamp": frame / 30.0,
                "frame_idx": frame,
                "cams": cameras,
            }
            for frame in range(frame_count)
        ],
        metadata={"version": "fixture"} if metadata is None else metadata,
    )


def test_split_paths_are_cwd_relative_and_incremental(tmp_path):
    """Match the runtime CWD contract while refreshing only changed PKLs."""
    data_dir = tmp_path / "pkls"
    data_dir.mkdir()
    first = data_dir / "a.pkl"
    second = data_dir / "b.pkl"
    _write_pkl(first, "scene_a", camera_count=2)
    _write_pkl(second, "scene_b", camera_count=4, frame_count=1)
    split = tmp_path / "train.txt"
    split.write_text("# fixture\npkls/a.pkl\npkls/b.pkl weight=2\n", encoding="utf-8")

    first_result = build_lazy_index(split, num_workers=1)

    assert first_result["num_frames"] == 3
    assert first_result["num_pkls"] == 2
    assert first_result["num_indexed_pkls"] == 2
    assert Path(first_result["cache_path"]) == tmp_path / "train_lazy_index.pkl"
    assert Path(first_result["camera_counts_path"]) == (
        tmp_path / "_pkl_cam_counts.pkl"
    )
    with open(first_result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert [entry["scene_name"] for entry in index["frame_index"]] == [
        "scene_a",
        "scene_a",
        "scene_b",
    ]
    assert all(Path(entry["pkl_path"]).is_absolute() for entry in index["frame_index"])
    assert index["pkl_cam_counts"] == {str(first): 2, str(second): 4}

    reused = build_lazy_index(split, num_workers=1)
    assert reused["num_reused_pkls"] == 2
    assert reused["num_indexed_pkls"] == 0

    stat = second.stat()
    os.utime(second, (stat.st_atime, stat.st_mtime + 2))
    refreshed = build_lazy_index(split, num_workers=1)
    assert refreshed["num_reused_pkls"] == 1
    assert refreshed["num_indexed_pkls"] == 1


def test_directory_input_excludes_generated_pickle_caches(tmp_path):
    """Never recursively treat generated indexes as source annotations."""
    source = tmp_path / "annotations"
    source.mkdir()
    _write_pkl(source / "scene.pkl", "scene", camera_count=1)

    first = build_lazy_index(source, num_workers=1)
    (source / "train_lazy_index.pkl").write_bytes(b"generated")
    (source / "looks_like.pkl").mkdir()
    second = build_lazy_index(source, num_workers=1)

    assert first["num_pkls"] == 1
    assert second["num_pkls"] == 1
    assert second["num_reused_pkls"] == 1
    assert [path.name for path in resolve_annotation_paths(source)] == ["scene.pkl"]


def test_split_rejects_generated_missing_and_empty_inputs(tmp_path):
    """Fail early rather than producing a misleading empty or recursive index."""
    generated = tmp_path / "_lazy_index.pkl"
    generated.write_bytes(b"not an annotation")
    split = tmp_path / "train.txt"
    split.write_text("_lazy_index.pkl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated cache"):
        resolve_annotation_paths(split)

    split.write_text("train_lazy_index.pkl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated cache"):
        resolve_annotation_paths(split)

    split.write_text("missing.pkl\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        resolve_annotation_paths(split)

    split.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No annotation PKLs"):
        resolve_annotation_paths(split)


@pytest.mark.parametrize("collision", ["split", "annotation", "index"])
def test_camera_count_output_cannot_overwrite_inputs_or_index(
    tmp_path, collision
):
    """Reject colliding outputs before any source or index is modified."""
    annotation = tmp_path / "scene.pkl"
    _write_pkl(annotation, "scene", camera_count=1)
    split = tmp_path / "train.txt"
    split.write_text("scene.pkl\n", encoding="utf-8")
    cache_path = tmp_path / "train_lazy_index.pkl"
    targets = {
        "split": split,
        "annotation": annotation,
        "index": cache_path,
    }
    split_before = split.read_bytes()
    annotation_before = annotation.read_bytes()

    with pytest.raises(ValueError, match="Camera-count output path conflicts"):
        build_lazy_index(
            split,
            num_workers=1,
            camera_counts_path=targets[collision],
        )

    assert split.read_bytes() == split_before
    assert annotation.read_bytes() == annotation_before
    assert not cache_path.exists()


def test_lazy_index_output_cannot_follow_a_source_symlink(tmp_path):
    """Do not replace an annotation through a colliding cache symlink."""
    annotation = tmp_path / "scene.pkl"
    _write_pkl(annotation, "scene", camera_count=1)
    annotation_before = annotation.read_bytes()
    cache_path = tmp_path / "_lazy_index.pkl"
    cache_path.symlink_to(annotation)

    with pytest.raises(ValueError, match="Lazy-index output path conflicts"):
        build_lazy_index(tmp_path, num_workers=1)

    assert cache_path.is_symlink()
    assert annotation.read_bytes() == annotation_before


def test_malformed_annotation_is_reported_with_its_path(tmp_path):
    """Wrap untrusted structural errors with actionable path context."""
    bad = tmp_path / "bad.pkl"
    with bad.open("wb") as stream:
        pickle.dump({"metadata": {}, "infos": [{"frame_idx": 0}]}, stream)

    with pytest.raises(ValueError, match=str(bad)):
        build_lazy_index(tmp_path, num_workers=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scene_name", "", "scene_name"),
        ("scene_name", "   ", "scene_name"),
        ("scene_name", 7, "scene_name"),
        ("timestamp", True, "timestamp"),
        ("timestamp", float("inf"), "timestamp"),
        ("timestamp", float("nan"), "timestamp"),
        ("timestamp", "0.0", "timestamp"),
        ("frame_idx", True, "frame_idx"),
        ("frame_idx", 1.5, "frame_idx"),
        ("frame_idx", "1", "frame_idx"),
        ("cams", [], "cams"),
    ],
)
def test_invalid_frame_sort_and_camera_fields_are_rejected(
    tmp_path, field, value, message
):
    """Reject values that would make lazy runtime sorting or loading unsafe."""
    info = {
        "scene_name": "scene",
        "timestamp": 0.0,
        "frame_idx": 0,
        "cams": {"Camera0": {}},
    }
    info[field] = value
    _write_document(tmp_path / "bad.pkl", [info], metadata={})

    with pytest.raises(ValueError, match=str(tmp_path / "bad.pkl")) as error:
        build_lazy_index(tmp_path, num_workers=1)
    assert message in str(error.value.__cause__)


def test_camera_names_are_consistent_and_empty_cameras_are_valid(tmp_path):
    """Require one camera set per PKL while accepting zero-camera frames."""
    inconsistent = tmp_path / "inconsistent.pkl"
    _write_document(
        inconsistent,
        [
            {
                "scene_name": "scene",
                "timestamp": 0.0,
                "cams": {"Camera0": {}},
            },
            {
                "scene_name": "scene",
                "timestamp": 1.0,
                "cams": {"Camera1": {}},
            },
        ],
        metadata={},
    )
    with pytest.raises(ValueError, match=str(inconsistent)) as error:
        build_lazy_index(tmp_path, num_workers=1)
    assert "camera names" in str(error.value.__cause__)

    inconsistent.unlink()
    empty_cameras = tmp_path / "empty_cameras.pkl"
    empty_infos = tmp_path / "empty_infos.pkl"
    _write_pkl(empty_cameras, "scene", camera_count=0)
    _write_pkl(empty_infos, "unused", camera_count=0, frame_count=0)

    result = build_lazy_index(tmp_path, num_workers=1)
    with open(result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["pkl_cam_counts"] == {
        str(empty_cameras): 0,
        str(empty_infos): 0,
    }
    assert index["pkl_frame_counts"][str(empty_infos)] == 0

    reused = build_lazy_index(tmp_path, num_workers=1)
    assert reused["num_reused_pkls"] == 2


def test_same_mtime_size_change_invalidates_cached_entries(tmp_path):
    """Use size plus nanosecond mtime instead of a float mtime alone."""
    source = tmp_path / "scene.pkl"
    _write_pkl(source, "scene", camera_count=1, frame_count=1)
    first = build_lazy_index(tmp_path, num_workers=1)
    original_stat = source.stat()

    _write_pkl(source, "scene", camera_count=1, frame_count=5)
    assert source.stat().st_size != original_stat.st_size
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    refreshed = build_lazy_index(tmp_path, num_workers=1)

    assert refreshed["num_indexed_pkls"] == 1
    assert refreshed["num_reused_pkls"] == 0
    assert refreshed["num_frames"] == 5
    with open(first["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["signatures"][str(source)] == {
        "size": source.stat().st_size,
        "mtime_ns": original_stat.st_mtime_ns,
    }


def test_older_mtimes_only_cache_is_safely_rebuilt(tmp_path):
    """Never reuse an old cache that has no strong source signatures."""
    source = tmp_path / "scene.pkl"
    _write_pkl(source, "scene", camera_count=1)
    first = build_lazy_index(tmp_path, num_workers=1)
    with open(first["cache_path"], "rb") as stream:
        cache = pickle.load(stream)
    cache.pop("signatures")
    cache.pop("pkl_frame_counts")
    with open(first["cache_path"], "wb") as stream:
        pickle.dump(cache, stream)

    rebuilt = build_lazy_index(tmp_path, num_workers=1)

    assert rebuilt["num_reused_pkls"] == 0
    assert rebuilt["num_indexed_pkls"] == 1


def test_metadata_is_refreshed_and_removed_deterministically(tmp_path):
    """Select current metadata by source order instead of retaining cache state."""
    first_source = tmp_path / "a.pkl"
    second_source = tmp_path / "b.pkl"
    _write_pkl(
        first_source,
        "scene_a",
        camera_count=1,
        metadata={"version": "old"},
    )
    _write_pkl(
        second_source,
        "scene_b",
        camera_count=1,
        metadata={"version": "fallback"},
    )
    first_result = build_lazy_index(tmp_path, num_workers=1)
    original_stat = first_source.stat()

    _write_pkl(
        first_source,
        "scene_a",
        camera_count=1,
        metadata={"version": "new"},
    )
    assert first_source.stat().st_size == original_stat.st_size
    os.utime(
        first_source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    reused = build_lazy_index(tmp_path, num_workers=1)
    assert reused["num_reused_pkls"] == 2
    with open(first_result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["metadata"] == {"version": "new"}

    _write_pkl(first_source, "scene_a", camera_count=1, metadata={})
    build_lazy_index(tmp_path, num_workers=1)
    with open(first_result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["metadata"] == {"version": "fallback"}

    second_source.unlink()
    build_lazy_index(tmp_path, num_workers=1)
    with open(first_result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    assert index["metadata"] == {}


@pytest.mark.parametrize("cache_value", [b"not a pickle", pickle.dumps([])])
def test_malformed_cache_is_rebuilt(cache_value, tmp_path):
    """Treat unreadable and wrong-shaped cache files as cache misses."""
    source = tmp_path / "scene.pkl"
    _write_pkl(source, "scene", camera_count=1)
    first = build_lazy_index(tmp_path, num_workers=1)
    Path(first["cache_path"]).write_bytes(cache_value)

    rebuilt = build_lazy_index(tmp_path, num_workers=1)

    assert rebuilt["num_reused_pkls"] == 0
    assert rebuilt["num_indexed_pkls"] == 1


def test_malformed_cached_bookkeeping_is_rebuilt(tmp_path):
    """Rebuild instead of crashing or reusing incomplete cached structures."""
    source = tmp_path / "scene.pkl"
    _write_pkl(source, "scene", camera_count=1)
    first = build_lazy_index(tmp_path, num_workers=1)
    with open(first["cache_path"], "rb") as stream:
        cache = pickle.load(stream)
    cache["frame_index"] = [{"pkl_path": str(source)}]
    cache["signatures"] = ["not", "a", "mapping"]
    cache["pkl_cam_counts"] = "not a mapping"
    cache["pkl_frame_counts"] = {str(source): True}
    with open(first["cache_path"], "wb") as stream:
        pickle.dump(cache, stream)

    rebuilt = build_lazy_index(tmp_path, num_workers=1)

    assert rebuilt["num_reused_pkls"] == 0
    assert rebuilt["num_indexed_pkls"] == 1


def test_symlink_mount_paths_are_preserved_in_all_index_keys(tmp_path):
    """Do not replace paths visible in the container with resolved host paths."""
    physical = tmp_path / "physical"
    physical.mkdir()
    _write_pkl(physical / "scene.pkl", "scene", camera_count=2)
    mount = tmp_path / "mounted"
    mount.symlink_to(physical, target_is_directory=True)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split = split_dir / "train.txt"
    split.write_text("mounted/scene.pkl\n", encoding="utf-8")

    result = build_lazy_index(split, num_workers=1)
    with open(result["cache_path"], "rb") as stream:
        index = pickle.load(stream)
    expected = str(mount / "scene.pkl")
    assert {row["pkl_path"] for row in index["frame_index"]} == {expected}
    assert set(index["signatures"]) == {expected}
    assert index["pkl_cam_counts"] == {expected: 2}


def test_duplicate_split_rows_fail_before_index_creation(tmp_path):
    """Sampling weights must not accidentally multiply indexed training frames."""
    source = tmp_path / "scene.pkl"
    _write_pkl(source, "scene", camera_count=1)
    split = tmp_path / "train.txt"
    split.write_text(f"{source}\n{source} weight=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        build_lazy_index(split, num_workers=1)
    assert not (tmp_path / "train_lazy_index.pkl").exists()
