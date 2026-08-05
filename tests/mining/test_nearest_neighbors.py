# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mining/tmm/scripts/nearest_neighbors.py script.

Each test generates dummy source / target parquets whose top-N matches
are known by construction, runs ``main(cfg)`` end-to-end through cudf +
cuML on the GPU, then compares the mined filepaths (and the
``mining_summary.txt``) against the expected result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from nvidia_tao_ds.mining.tmm.scripts.nearest_neighbors import main


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_parquet(path, filepaths, embeddings, labels=None):
    """Persist a parquet with ``filepath`` + ``embedding`` (+ optional ``label``).

    The ``embedding`` column is stored as a list per row, which is what
    ``cudf.read_parquet`` materializes as a list-typed column.
    """
    df = pd.DataFrame({
        "filepath": list(filepaths),
        "embedding": [list(map(float, e)) for e in embeddings],
    })
    if labels is not None:
        df["label"] = list(labels)
    df.to_parquet(str(path), index=False)
    return str(path)


def _build_cfg(
    source_parquet, target_parquet, output_parquet,
    topn=2, knn_metric="euclidean",
    source_embed_column_name="embedding",
    target_embed_column_name="embedding",
    filter_by_label="false",
    distance_threshold=-1.0,
):
    """Construct the OmegaConf cfg object ``main()`` expects."""
    return OmegaConf.create({
        "source_parquet": source_parquet,
        "target_parquet": target_parquet,
        "output_parquet": output_parquet,
        "topn": topn,
        "knn_metric": knn_metric,
        "source_embed_column_name": source_embed_column_name,
        "target_embed_column_name": target_embed_column_name,
        "filter_by_label": filter_by_label,
        "distance_threshold": distance_threshold,
    })


def _read_summary(output_parquet):
    return (Path(output_parquet).parent / "mining_summary.txt").read_text()


@pytest.fixture
def dummy_parquets(tmp_path):
    """Source / target parquets whose top-1 matches are known.

    Source embeddings are the canonical 4-D basis vectors, so each
    target's nearest source is unambiguous under every supported metric.
    """
    source_emb = np.array([
        [1.0, 0.0, 0.0, 0.0],   # src_a
        [0.0, 1.0, 0.0, 0.0],   # src_b
        [0.0, 0.0, 1.0, 0.0],   # src_c
        [0.0, 0.0, 0.0, 1.0],   # src_d
    ], dtype=np.float32)
    source_paths = ["src_a.png", "src_b.png", "src_c.png", "src_d.png"]
    source_labels = ["PASS", "NO_PASS", "PASS", "NO_PASS"]

    target_emb = np.array([
        [1.0, 0.0, 0.0, 0.0],   # -> src_a
        [0.0, 1.0, 0.0, 0.0],   # -> src_b
        [0.05, 0.0, 0.95, 0.0], # -> src_c
    ], dtype=np.float32)
    target_paths = ["tgt_0.png", "tgt_1.png", "tgt_2.png"]
    target_labels = ["PASS", "NO_PASS", "PASS"]

    return {
        "source": _write_parquet(
            tmp_path / "source.parquet",
            source_paths, source_emb, source_labels,
        ),
        "target": _write_parquet(
            tmp_path / "target.parquet",
            target_paths, target_emb, target_labels,
        ),
        "output": str(tmp_path / "out" / "mined.parquet"),
        "source_paths": source_paths,
        "target_paths": target_paths,
        "source_labels": source_labels,
        "target_labels": target_labels,
    }


# ---------------------------------------------------------------------------
# Simple per-behavior tests
# ---------------------------------------------------------------------------


def test_main_top1_retrieves_nearest_source(dummy_parquets):
    """Each target's top-1 neighbor should be its matching source image."""
    cfg = _build_cfg(
        dummy_parquets["source"], dummy_parquets["target"],
        dummy_parquets["output"], topn=1, knn_metric="euclidean",
    )
    main(cfg)

    assert Path(dummy_parquets["output"]).exists()
    out_df = pd.read_parquet(dummy_parquets["output"])
    assert list(out_df.columns) == ["filepath"]
    assert sorted(out_df["filepath"].tolist()) == [
        "src_a.png", "src_b.png", "src_c.png",
    ]


def test_main_creates_output_directory_tree(tmp_path, dummy_parquets):
    """A nested output path should be created if missing."""
    nested = str(tmp_path / "a" / "b" / "c" / "mined.parquet")
    cfg = _build_cfg(
        dummy_parquets["source"], dummy_parquets["target"], nested, topn=1,
    )
    main(cfg)
    assert Path(nested).exists()


def test_main_writes_summary_file(dummy_parquets):
    cfg = _build_cfg(
        dummy_parquets["source"], dummy_parquets["target"],
        dummy_parquets["output"], topn=2,
    )
    main(cfg)

    text = _read_summary(dummy_parquets["output"])
    assert "Target queries: 3" in text
    assert "Similar items per query: 2" in text
    assert "Unique items saved:" in text


def test_main_deduplicates_mined_filepaths(tmp_path):
    """Multiple targets that resolve to the same source produce one row."""
    source_emb = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32,
    )
    target_emb = np.tile([[1.0, 0.0]], (4, 1)).astype(np.float32)

    src = _write_parquet(
        tmp_path / "s.parquet", ["src_a.png", "src_b.png"], source_emb,
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet",
        [f"tgt_{i}.png" for i in range(4)], target_emb,
    )
    out = str(tmp_path / "mined.parquet")

    cfg = _build_cfg(src, tgt, out, topn=1, knn_metric="euclidean")
    main(cfg)

    out_df = pd.read_parquet(out)
    # Four targets × top-1 = 4 candidates; dedup collapses to one source.
    assert out_df["filepath"].tolist() == ["src_a.png"]


def test_main_label_filter_drops_cross_label_pairs(tmp_path):
    """With the filter on, source/target pairs with mismatched labels are dropped."""
    # Two sources per identical vector: one PASS, one NO_PASS.
    source_emb = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    source_paths = ["src_a.png", "src_b.png", "src_c.png", "src_d.png"]
    source_labels = ["PASS", "NO_PASS", "PASS", "NO_PASS"]

    target_emb = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    target_paths = ["tgt_0.png", "tgt_1.png"]
    target_labels = ["PASS", "NO_PASS"]

    src = _write_parquet(
        tmp_path / "s.parquet", source_paths, source_emb, source_labels,
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet", target_paths, target_emb, target_labels,
    )
    out = str(tmp_path / "mined.parquet")

    cfg = _build_cfg(
        src, tgt, out, topn=2, knn_metric="euclidean", filter_by_label="true",
    )
    main(cfg)

    mined = set(pd.read_parquet(out)["filepath"].tolist())
    # PASS target keeps src_a (PASS) only; NO_PASS target keeps src_d
    # (NO_PASS) only — mismatched labels are filtered out.
    assert mined == {"src_a.png", "src_d.png"}
    assert "After label mismatch filtering" in _read_summary(out)


def test_main_label_filter_false_keeps_cross_label_pairs(tmp_path):
    """Without the filter, cross-label neighbors are retained."""
    source_emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    src = _write_parquet(
        tmp_path / "s.parquet",
        ["src_pass.png", "src_no_pass.png"],
        source_emb,
        labels=["PASS", "NO_PASS"],
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet",
        ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
        labels=["PASS"],
    )
    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(
        src, tgt, out, topn=2, filter_by_label="false",
    )
    main(cfg)

    assert set(pd.read_parquet(out)["filepath"].tolist()) == {
        "src_pass.png", "src_no_pass.png",
    }
    assert "Label filtering" not in _read_summary(out)


def test_main_label_filter_noop_when_label_column_missing(tmp_path):
    """filter_by_label='true' must no-op (not raise) without label columns."""
    source_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    src = _write_parquet(
        tmp_path / "s.parquet", ["src_a.png", "src_b.png"], source_emb,
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet",
        ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(src, tgt, out, topn=1, filter_by_label="true")
    main(cfg)

    assert pd.read_parquet(out)["filepath"].tolist() == ["src_a.png"]


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
def test_main_supports_all_knn_metrics(dummy_parquets, tmp_path, metric):
    """Top-1 on orthogonal unit vectors is unambiguous for every metric."""
    out_path = str(tmp_path / f"mined-{metric}.parquet")
    cfg = _build_cfg(
        dummy_parquets["source"], dummy_parquets["target"],
        out_path, topn=1, knn_metric=metric,
    )
    main(cfg)
    assert sorted(pd.read_parquet(out_path)["filepath"].tolist()) == [
        "src_a.png", "src_b.png", "src_c.png",
    ]


def test_main_honors_custom_embed_column_names(tmp_path):
    """Custom embedding column names must be read instead of 'embedding'."""
    src_path = str(tmp_path / "s.parquet")
    tgt_path = str(tmp_path / "t.parquet")
    pd.DataFrame({
        "filepath": ["src_a.png", "src_b.png"],
        "src_vec": [[1.0, 0.0], [0.0, 1.0]],
    }).to_parquet(src_path, index=False)
    pd.DataFrame({
        "filepath": ["tgt.png"],
        "tgt_vec": [[1.0, 0.0]],
    }).to_parquet(tgt_path, index=False)

    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(
        src_path, tgt_path, out, topn=1, knn_metric="euclidean",
        source_embed_column_name="src_vec",
        target_embed_column_name="tgt_vec",
    )
    main(cfg)
    assert pd.read_parquet(out)["filepath"].tolist() == ["src_a.png"]


# ---------------------------------------------------------------------------
# End-to-end: known-answer test on a larger, deterministic input
# ---------------------------------------------------------------------------

def test_main_end_to_end_known_answer(tmp_path):
    """Full pipeline: 8 source vectors, 4 target vectors, top-2, cosine.

    Each target is aligned with a distinct source axis, so top-2 should
    return the two strongest-aligned sources per target. Dedup collapses
    overlapping candidates into the expected unique set.
    """
    axes = np.eye(8, dtype=np.float32)  # src_0..src_7 on 8 basis axes
    source_paths = [f"src_{i}.png" for i in range(8)]

    # Targets 0..3 each align with source axes 0, 2, 4, 6 respectively.
    target_emb = np.stack([axes[i] for i in (0, 2, 4, 6)])
    target_paths = [f"tgt_{i}.png" for i in range(4)]

    src = _write_parquet(tmp_path / "s.parquet", source_paths, axes)
    tgt = _write_parquet(tmp_path / "t.parquet", target_paths, target_emb)
    out = str(tmp_path / "mined.parquet")

    cfg = _build_cfg(src, tgt, out, topn=2, knn_metric="cosine")
    main(cfg)

    # Each target's top-1 is its own axis; top-2 is any other orthogonal
    # source (cosine distance is identical across them and argsort is
    # stable). The guaranteed invariant: the primary match for every
    # target is included, so {src_0, src_2, src_4, src_6} must be a
    # subset of the mined set.
    mined = set(pd.read_parquet(out)["filepath"].tolist())
    assert {"src_0.png", "src_2.png", "src_4.png", "src_6.png"}.issubset(mined)

    # 4 targets × top-2 = 8 candidate pairs before dedup.
    summary = _read_summary(out)
    assert "Target queries: 4" in summary
    assert "Similar items per query: 2" in summary
    assert "Total candidates: 8" in summary
    assert f"Unique items saved: {len(mined)}" in summary
    

def test_distance_filter_disabled_when_negative(tmp_path):
    """distance_threshold=-1.0 keeps all pairs regardless of distance; no summary line emitted."""
    src = _write_parquet(
        tmp_path / "s.parquet",
        ["src_close.png", "src_far.png"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet", ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(src, tgt, out, topn=2)  # distance_threshold defaults to -1.0
    main(cfg)

    assert set(pd.read_parquet(out)["filepath"].tolist()) == {"src_close.png", "src_far.png"}
    assert "Distance filtering" not in _read_summary(out)


def test_distance_filter_keeps_pair_at_exact_threshold(tmp_path):
    """A pair whose distance equals the threshold exactly is kept (filter is strict >)."""
    src = _write_parquet(
        tmp_path / "s.parquet",
        ["src_boundary.png", "src_far.png"],
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet", ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(src, tgt, out, topn=2, knn_metric="euclidean", distance_threshold=1.0)
    main(cfg)

    mined = pd.read_parquet(out)["filepath"].tolist()
    assert "src_boundary.png" in mined   
    assert "src_far.png" not in mined   


def test_distance_and_label_filters_cascade_correctly(tmp_path):
    """With both filters active, distance counts are from label-passing pairs only.

    Setup: 1 target (PASS), topn=3, threshold=1.5 (euclidean).
      src_a [1,0]  PASS   dist=0.0  → passes label, passes distance  → kept
      src_c [0,1]  NO_PASS dist=1.41 → fails label (skipped by distance check)
      src_b [-1,0] PASS   dist=2.0  → passes label, fails distance   → dropped

    Expected cascade:
      Total pairs:        3
      Label filtering:    kept 2/3,  dropped 1 mismatch
      Distance filtering: kept 1/2,  dropped 1 exceeding threshold  (denominator=2, not 3)
      Unique items saved: 1
    """
    src = _write_parquet(
        tmp_path / "s.parquet",
        ["src_a.png", "src_b.png", "src_c.png"],
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=["PASS", "PASS", "NO_PASS"],
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet", ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
        labels=["PASS"],
    )
    out = str(tmp_path / "mined.parquet")
    cfg = _build_cfg(
        src, tgt, out, topn=3, knn_metric="euclidean",
        filter_by_label="true", distance_threshold=1.5,
    )
    main(cfg)

    assert pd.read_parquet(out)["filepath"].tolist() == ["src_a.png"]

    summary = _read_summary(out)
    assert "After label mismatch filtering: kept 2/3" in summary
    assert "After distance filtering: kept 1/2" in summary
    assert "Unique items saved: 1" in summary


@pytest.mark.parametrize("metric,threshold", [
    ("cosine", 0.5),
    ("manhattan", 1.5),
    ("euclidean", 1.0),
])
def test_distance_filter_works_across_metrics(tmp_path, metric, threshold):
    """Distance filtering drops the far source under cosine, manhattan, and euclidean metrics."""
    src = _write_parquet(
        tmp_path / "s.parquet",
        ["src_close.png", "src_far.png"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    tgt = _write_parquet(
        tmp_path / "t.parquet", ["tgt.png"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    out = str(tmp_path / f"mined-{metric}.parquet")
    cfg = _build_cfg(src, tgt, out, topn=2, knn_metric=metric, distance_threshold=threshold)
    main(cfg)

    mined = pd.read_parquet(out)["filepath"].tolist()
    assert "src_close.png" in mined
    assert "src_far.png" not in mined