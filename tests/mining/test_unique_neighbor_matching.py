# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the unique_neighbor_matching command; require a RAPIDS/GPU runtime."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

# Skip the whole module if the RAPIDS stack can't be imported. A broken install can
# fail with something other than ImportError, so catch broadly rather than using
# importorskip. Note this gates on import only — a usable import with an unusable
# GPU (e.g. no device) still surfaces at runtime, not here.
try:
    import cudf  # noqa: F401
    import cuml  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment gate
    pytest.skip(f"RAPIDS (cudf/cuml) unavailable: {_exc}", allow_module_level=True)

from nvidia_tao_ds.mining.tmm.selection import iterative_tmm_retrieval, perform_tmm_retrieval
from nvidia_tao_ds.mining.tmm.scripts.unique_neighbor_matching import UniqueNeighborMatchingMining, _allocate_quotas


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_parquet(path, filepaths, embeddings, filepath_col="filepath"):
    """Persist a parquet with ``filepath`` + ``embedding`` (embedding as a per-row list)."""
    pd.DataFrame({
        filepath_col: list(filepaths),
        "embedding": [list(map(float, e)) for e in embeddings],
    }).to_parquet(str(path), index=False)
    return str(path)


def _write_coco(path, images, categories, annotations):
    """Write a minimal COCO JSON.

    images: list of (id, file_name); categories: list of (id, name);
    annotations: list of (image_id, category_id, [x, y, w, h]).
    """
    coco = {
        "images": [{"id": i, "file_name": fn} for i, fn in images],
        "categories": [{"id": i, "name": n} for i, n in categories],
        "annotations": [
            {"id": k, "image_id": im, "category_id": c, "bbox": b}
            for k, (im, c, b) in enumerate(annotations)
        ],
    }
    Path(path).write_text(json.dumps(coco))
    return str(path)


def _write_detection(fmt, base, name, image_classes, start_id=0):
    """Write detection annotations (COCO ``.json`` or a KITTI label dir) from a
    ``{image_path: [class, ...]}`` map, letting one test drive both formats.

    An image mapped to an empty list carries no annotation, so it reads as non-rare.
    Returns the detection file/dir to pass as a ``*_detection_file``.
    """
    class_names = sorted({c for classes in image_classes.values() for c in classes})
    cat_id = {c: i + 1 for i, c in enumerate(class_names)}
    if fmt == "coco":
        imgs = [(start_id + i, p) for i, p in enumerate(image_classes)]
        anns = [
            (start_id + i, cat_id[c], [0, 0, 10, 10])
            for i, classes in enumerate(image_classes.values()) for c in classes
        ]
        return _write_coco(base / f"{name}.json", imgs, [(cat_id[c], c) for c in class_names], anns)
    # KITTI: a directory of <stem>.txt label files, one line per (image, class).
    label_dir = base / f"{name}_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for p, classes in image_classes.items():
        if classes:
            # each line: class trunc occ alpha x1 y1 x2 y2 h w l tx ty tz ry
            lines = "\n".join(f"{c} 0 0 0 0 0 10 10 0 0 0 0 0 0 0" for c in classes)
            (label_dir / f"{Path(p).stem}.txt").write_text(lines + "\n", encoding="utf-8")
    return str(label_dir)


def _retrieved_sources(output_dir, subset):
    """Union of source filepaths a subset's iteration parquets matched (incl. fallback)."""
    srcs = set()
    out = Path(output_dir)
    files = sorted(out.glob(f"{subset}_iteration_*.parquet")) + sorted(out.glob(f"{subset}_fallback_iteration_*.parquet"))
    for f in files:
        df = pd.read_parquet(f)
        for col in [c for c in df.columns if c.startswith("top") and c.endswith("_source_filepath")]:
            srcs |= set(df[col].dropna())
    return srcs


def _build_cfg(
    source_path,
    target_path,
    output_dir,
    desired_unique_count,
    mode="global",
    distance_metric="euclidean",
    candidate_expansion_factor=5,
    exclude_path=None,
    source_detection_file=None,
    target_detection_file=None,
    detection_format=None,
    rare_class_list="",
    save_embeddings=False,
    visualize=False,
    source_embedding_column="embedding",
    target_embedding_column="embedding",
    source_filepath_column="filepath",
    target_filepath_column="filepath",
):
    """Construct the OmegaConf cfg ``run()`` expects, using canonical field names."""
    return OmegaConf.create({
        "source_path": source_path,
        "target_path": target_path,
        "output_dir": output_dir,
        "desired_unique_count": desired_unique_count,
        "allocation_policy": mode,
        "distance_metric": distance_metric,
        "candidate_expansion_factor": candidate_expansion_factor,
        "exclude_path": exclude_path,
        "source_detection_file": source_detection_file,
        "target_detection_file": target_detection_file,
        "detection_format": detection_format,
        "rare_class_list": rare_class_list,
        "save_embeddings": save_embeddings,
        "visualize": visualize,
        "source_embedding_column": source_embedding_column,
        "target_embedding_column": target_embedding_column,
        "source_filepath_column": source_filepath_column,
        "target_filepath_column": target_filepath_column,
    })


@pytest.fixture
def basis_data(tmp_path):
    """8 orthogonal source vectors; 3 targets aligned to axes 0, 2, 4."""
    axes = np.eye(8, dtype=np.float32)
    source_paths = [f"src_{i}.png" for i in range(8)]
    target_emb = np.stack([axes[i] for i in (0, 2, 4)])
    target_paths = [f"tgt_{i}.png" for i in range(3)]
    return {
        "source": _write_parquet(tmp_path / "source.parquet", source_paths, axes),
        "target": _write_parquet(tmp_path / "target.parquet", target_paths, target_emb),
        "output": str(tmp_path / "out"),
    }


# ---------------------------------------------------------------------------
# Simple mode
# ---------------------------------------------------------------------------


def test_global_mode_file_source(basis_data):
    """Simple mode reaches the desired count and writes the standard outputs."""
    cfg = _build_cfg(
        basis_data["source"], basis_data["target"], basis_data["output"],
        desired_unique_count=3, mode="global", distance_metric="euclidean",
    )
    UniqueNeighborMatchingMining().run(cfg)

    out = Path(basis_data["output"])
    final = out / "final_unique_files.parquet"
    assert final.exists()
    df = pd.read_parquet(final)
    assert list(df.columns) == ["filepath"]
    assert len(df) >= 1

    summary = json.loads((out / "summary.json").read_text())
    assert summary["allocation_policy"] == "global"
    assert summary["desired_unique_count"] == 3
    assert "retrieved_unique_count" in summary

    # iterative retrieval writes per-iteration tables for the "global" subset
    assert sorted(out.glob("global_iteration_*.parquet"))


def test_global_mode_directory_source(tmp_path):
    """A directory of source parquet shards is concatenated and searched."""
    axes = np.eye(6, dtype=np.float32)
    src_dir = tmp_path / "source_dir"
    src_dir.mkdir()
    _write_parquet(src_dir / "a.parquet", [f"a_{i}.png" for i in range(3)], axes[:3])
    _write_parquet(src_dir / "b.parquet", [f"b_{i}.png" for i in range(3)], axes[3:])
    tgt = _write_parquet(
        tmp_path / "target.parquet",
        ["t0.png", "t1.png"],
        np.stack([axes[0], axes[4]]),
    )
    out = str(tmp_path / "out")
    cfg = _build_cfg(str(src_dir), tgt, out, desired_unique_count=2, mode="global")
    UniqueNeighborMatchingMining().run(cfg)

    df = pd.read_parquet(Path(out) / "final_unique_files.parquet")
    # retrieved files come from either shard
    assert len(df) >= 1
    assert all(fp.startswith(("a_", "b_")) for fp in df["filepath"])


def test_desired_unique_count_zero_raises(basis_data):
    """An explicit desired_unique_count of 0 is rejected."""
    cfg = _build_cfg(
        basis_data["source"], basis_data["target"], basis_data["output"],
        desired_unique_count=0, mode="global",
    )
    with pytest.raises(ValueError):
        UniqueNeighborMatchingMining().run(cfg)


# ---------------------------------------------------------------------------
# Class-stratified: prerequisites + detection-format resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing", ["rare_class_list", "source_detection_file", "target_detection_file", "detection_format"]
)
def test_class_stratified_missing_prereq_raises(basis_data, tmp_path, missing):
    """class_stratified requires rare_class_list, both detection files, and detection_format."""
    coco = _write_coco(
        tmp_path / "ann.json",
        images=[(1, "src_0.png")], categories=[(1, "person")],
        annotations=[(1, 1, [0, 0, 10, 10])],
    )
    kwargs = dict(
        mode="class_stratified",
        rare_class_list="person",
        source_detection_file=coco,
        target_detection_file=coco,
        detection_format="coco",
    )
    kwargs[missing] = "" if missing == "rare_class_list" else None
    cfg = _build_cfg(
        basis_data["source"], basis_data["target"], basis_data["output"],
        desired_unique_count=3, **kwargs,
    )
    with pytest.raises(ValueError):
        UniqueNeighborMatchingMining().run(cfg)


def test_resolve_detection_format(tmp_path):
    """detection_format is required only when a detection file is given, and maps to the enum."""
    from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat
    from nvidia_tao_ds.mining.tmm.scripts.unique_neighbor_matching import _resolve_detection_format

    out = str(tmp_path)
    # No detection file -> format is irrelevant (None), even if unset.
    assert _resolve_detection_format(_build_cfg("s", "t", out, 1)) is None
    # Detection file provided but no declared format -> error (never inferred).
    with pytest.raises(ValueError):
        _resolve_detection_format(_build_cfg("s", "t", out, 1, source_detection_file="ann.json"))
    # Declared format -> the matching AnnotationFormat enum.
    declared = _build_cfg("s", "t", out, 1, source_detection_file="ann.json", detection_format="coco")
    assert _resolve_detection_format(declared) is AnnotationFormat.COCO


# ---------------------------------------------------------------------------
# Class-stratified: main end-to-end functional test (multi-class, COCO + KITTI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detection_format", ["coco", "kitti"])
def test_class_stratified_end_to_end(tmp_path, detection_format):
    """The main class_stratified functional test: two rare classes, over COCO and KITTI.

    Exercises the full split -> per-class retrieval -> summary pipeline and primary-class
    ownership together. sp_0/sp_1 carry BOTH "person" and "car", and the person- and
    car-targets are both nearest sp_0, so the class processed first (car) claims it and
    the other (person) must fall through to sp_1 — the two class pools never share a
    source. nr_0/nr_1 are non-rare.
    """
    axes = np.eye(4, dtype=np.float32)
    source_paths = ["sp_0.png", "sp_1.png", "nr_0.png", "nr_1.png"]
    src = _write_parquet(tmp_path / "source.parquet", source_paths, axes)
    # person- and car-targets both point at sp_0 (contention); the non-rare target at nr_0.
    target_paths = ["tp_0.png", "tc_0.png", "tnr_0.png"]
    tgt_emb = np.array([[0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0], [0, 0, 0.9, 0.1]], dtype=np.float32)
    tgt = _write_parquet(tmp_path / "target.parquet", target_paths, tgt_emb)

    src_classes = {"sp_0.png": ["person", "car"], "sp_1.png": ["person", "car"], "nr_0.png": [], "nr_1.png": []}
    tgt_classes = {"tp_0.png": ["person"], "tc_0.png": ["car"], "tnr_0.png": []}
    src_det = _write_detection(detection_format, tmp_path, "src", src_classes, start_id=0)
    tgt_det = _write_detection(detection_format, tmp_path, "tgt", tgt_classes, start_id=100)

    out = str(tmp_path / "out")
    cfg = _build_cfg(
        src, tgt, out, desired_unique_count=3, mode="class_stratified",
        distance_metric="euclidean", rare_class_list="person,car",
        source_detection_file=src_det, target_detection_file=tgt_det,
        detection_format=detection_format,
    )
    UniqueNeighborMatchingMining().run(cfg)

    out_p = Path(out)
    summary = json.loads((out_p / "summary.json").read_text())
    assert summary["allocation_policy"] == "class_stratified"
    # both rare classes are allocated, alongside the non_rare pool
    assert {"person", "car"} <= set(summary["allocation"]["per_class"])
    assert "distance_stats" in summary
    for subset in ("person", "car", "non_rare"):
        assert sorted(out_p.glob(f"{subset}_iteration_*.parquet")), f"no iteration tables for {subset}"

    # Output is the unique matched set, consistent with the summary count.
    df = pd.read_parquet(out_p / "final_unique_files.parquet")
    assert list(df.columns) == ["filepath"]
    assert df["filepath"].is_unique
    assert len(df) == summary["retrieved_unique_count"]

    # Primary-class ownership: a source is never reselected across classes, so the
    # person and car pools stay disjoint (car claims sp_0, person falls to sp_1).
    person_srcs = _retrieved_sources(out_p, "person")
    car_srcs = _retrieved_sources(out_p, "car")
    assert person_srcs and car_srcs
    assert person_srcs.isdisjoint(car_srcs)


# ---------------------------------------------------------------------------
# Global allocation: result cap, partial fill, determinism, exclusions, rollback
# ---------------------------------------------------------------------------


def test_global_does_not_exceed_desired_count(basis_data):
    """The retrieved set is hard-capped at desired_unique_count, with unique files.

    3 targets would each yield 1 nearest source (3 candidates), but desired=2 caps
    the result to 2.
    """
    cfg = _build_cfg(
        basis_data["source"], basis_data["target"], basis_data["output"],
        desired_unique_count=2, mode="global",
    )
    UniqueNeighborMatchingMining().run(cfg)

    out = Path(basis_data["output"])
    df = pd.read_parquet(out / "final_unique_files.parquet")
    assert len(df) == 2
    assert df["filepath"].is_unique
    summary = json.loads((out / "summary.json").read_text())
    assert summary["retrieved_unique_count"] == 2


def test_global_partial_fill_when_sources_scarce(tmp_path):
    """Requesting more than the source pool holds fills partially, without error or overshoot."""
    axes = np.eye(3, dtype=np.float32)
    src = _write_parquet(tmp_path / "src.parquet", ["a.png", "b.png", "c.png"], axes)
    tgt = _write_parquet(tmp_path / "tgt.parquet", ["t0.png", "t1.png"], np.stack([axes[0], axes[1]]))
    out = str(tmp_path / "out")

    # desired=4 -> topn=ceil(4/2)=2 (<= the 3 available sources), so a target can fill
    # but the pool cannot satisfy all of desired: a graceful partial fill, no overshoot.
    cfg = _build_cfg(src, tgt, out, desired_unique_count=4, mode="global")
    UniqueNeighborMatchingMining().run(cfg)

    df = pd.read_parquet(Path(out) / "final_unique_files.parquet")
    assert 1 <= len(df) <= 3  # bounded by the 3 available sources, below desired=4
    assert df["filepath"].is_unique


def test_global_deterministic_across_runs(basis_data, tmp_path):
    """Two runs on the same inputs produce identical output (deterministic assignment)."""
    cfg_a = _build_cfg(basis_data["source"], basis_data["target"], str(tmp_path / "a"),
                       desired_unique_count=3, mode="global")
    cfg_b = _build_cfg(basis_data["source"], basis_data["target"], str(tmp_path / "b"),
                       desired_unique_count=3, mode="global")
    UniqueNeighborMatchingMining().run(cfg_a)
    UniqueNeighborMatchingMining().run(cfg_b)

    a = pd.read_parquet(Path(tmp_path / "a") / "final_unique_files.parquet")
    b = pd.read_parquet(Path(tmp_path / "b") / "final_unique_files.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_exclude_path_removes_sources(basis_data, tmp_path):
    """Sources listed in exclude_path are dropped from the pool and never retrieved."""
    exclude = _write_parquet(tmp_path / "exclude.parquet", ["src_0.png"], np.zeros((1, 8), dtype=np.float32))
    cfg = _build_cfg(
        basis_data["source"], basis_data["target"], basis_data["output"],
        desired_unique_count=8, mode="global", exclude_path=exclude,
    )
    UniqueNeighborMatchingMining().run(cfg)

    df = pd.read_parquet(Path(basis_data["output"]) / "final_unique_files.parquet")
    assert "src_0.png" not in set(df["filepath"])


def test_iterative_empty_targets_returns_empty():
    """An empty target set returns an empty result instead of dividing by zero."""
    result = iterative_tmm_retrieval(
        np.zeros((0, 4), dtype=np.float32), np.eye(4, dtype=np.float32),
        None, None, desired_count=3, knn_metric="euclidean",
    )
    assert result == set()


def test_perform_rollback_frees_partial_sources():
    """A target that cannot reach topn releases its tentative picks for later targets.

    topn=2 with a narrow candidate window (candidate_expansion_factor=1 -> k=2):
    T0 claims S1,S2. T1's free candidate (S3) alone is < topn, so T1 is deferred —
    but it must NOT lock S3, so T2 can still claim S3+S4 and be satisfied.
    """
    import cupy as cp  # noqa: PLC0415

    sources = cp.asarray(np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=np.float32))
    targets = cp.asarray(np.array([
        [0.8, 0.6, 0.0, 0.0],   # T0 nearest: S1, S2
        [0.8, 0.0, 0.6, 0.0],   # T1 nearest: S1, S3
        [0.0, 0.0, 0.8, 0.6],   # T2 nearest: S3, S4
    ], dtype=np.float32))
    source_df = cudf.DataFrame({"filepath": ["S1", "S2", "S3", "S4"]})
    target_df = cudf.DataFrame({"filepath": ["T0", "T1", "T2"]})

    result_df, targets_with_duplicates = perform_tmm_retrieval(
        targets, sources, source_df, target_df, topn=2, knn_metric="euclidean",
        candidate_expansion_factor=1,
    )

    file_cols = [c for c in result_df.columns if c.startswith("top") and c.endswith("_source_filepath")]
    retrieved = set(result_df[file_cols].melt()["value"].dropna())
    assert retrieved == {"S1", "S2", "S3", "S4"}
    assert targets_with_duplicates == [1]  # only T1 deferred; S3 was not locked away from T2
    assert len(result_df) == 2  # T0 and T2 satisfied


def test_candidate_expansion_factor_widens_candidate_pool():
    """A larger candidate_expansion_factor lets a single pass find more unique sources."""
    import cupy as cp  # noqa: PLC0415

    sources = cp.asarray(np.eye(4, dtype=np.float32))  # S0..S3
    targets = cp.asarray(np.array([
        [0.9, 0.3, 0.2, 0.1],    # both nearest S0, then S1, S2, S3
        [0.85, 0.35, 0.25, 0.15],
    ], dtype=np.float32))
    source_df = cudf.DataFrame({"filepath": ["S0", "S1", "S2", "S3"]})
    target_df = cudf.DataFrame({"filepath": ["T0", "T1"]})

    def _retrieved(expansion):
        result_df, _ = perform_tmm_retrieval(
            targets, sources, source_df, target_df, topn=1, knn_metric="euclidean",
            candidate_expansion_factor=expansion,
        )
        cols = [c for c in result_df.columns if c.startswith("top") and c.endswith("_source_filepath")]
        return set(result_df[cols].melt()["value"].dropna())

    narrow = _retrieved(1)  # k=1: T1's only candidate S0 is taken -> T1 unfilled
    wide = _retrieved(4)    # k=4: T1 falls through to S1
    assert len(wide) > len(narrow)


# ---------------------------------------------------------------------------
# Class-stratified: quota allocation (_allocate_quotas)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("desired", [1, 3, 5, 7, 100])
def test_allocate_quotas_sums_to_desired(desired):
    """Largest-remainder quotas are non-negative and sum to exactly desired."""
    quotas = _allocate_quotas({"a": 0.5, "b": 0.3, "non_rare": 0.2}, desired)
    assert sum(quotas.values()) == desired
    assert all(v >= 0 for v in quotas.values())


def test_allocate_quotas_no_overallocation_or_negative_common():
    """Many small classes never over-allocate or drive the common quota negative."""
    ratios = {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "e": 0.1, "non_rare": 0.5}
    quotas = _allocate_quotas(ratios, 3)  # naive max(1,...) per class would demand >= 6
    assert sum(quotas.values()) == 3
    assert quotas["non_rare"] >= 0
    assert all(v >= 0 for v in quotas.values())


def test_allocate_quotas_permutation_invariant():
    """Quota allocation does not depend on the ordering of the ratio dict."""
    r1 = {"a": 0.5, "b": 0.3, "non_rare": 0.2}
    r2 = {"non_rare": 0.2, "b": 0.3, "a": 0.5}
    assert _allocate_quotas(r1, 10) == _allocate_quotas(r2, 10)


def test_allocate_quotas_zero_capacity():
    """Zero desired or all-zero ratios yield all-zero quotas, not a crash."""
    assert _allocate_quotas({"a": 0.5, "non_rare": 0.5}, 0) == {"a": 0, "non_rare": 0}
    assert _allocate_quotas({"a": 0.0, "non_rare": 0.0}, 10) == {"a": 0, "non_rare": 0}
