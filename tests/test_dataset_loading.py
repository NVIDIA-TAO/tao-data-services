# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nvidia_tao_ds/core/utils/dataset_loading.py.

CPU-only: exercises the pure-Python KITTI/COCO parsers, no RAPIDS/GPU required.
"""

import json
from pathlib import Path

import pytest

from nvidia_tao_ds.core.utils.dataset_loading import (
    AnnotationFormat,
    _load_coco,
    load_annotations,
    load_scored_annotations,
)


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
    Path(path).write_text(json.dumps(coco), encoding="utf-8")
    return str(path)


def test_load_coco_drops_ambiguous_basename(tmp_path):
    """Two images sharing a basename don't merge — the basename fallback drops the collision."""
    coco = _write_coco(
        tmp_path / "ann.json",
        images=[(0, "dirA/frame.png"), (1, "dirB/frame.png")],
        categories=[(1, "person"), (2, "car")],
        annotations=[(0, 1, [0, 0, 1, 1]), (1, 2, [0, 0, 1, 1])],
    )
    fp_to_classes, bn_to_classes, _, _ = _load_coco(coco)

    # Full-path lookups stay precise...
    assert fp_to_classes["dirA/frame.png"] == {"person"}
    assert fp_to_classes["dirB/frame.png"] == {"car"}
    # ...but the ambiguous basename is dropped, so it can't return {person, car}.
    assert "frame.png" not in bn_to_classes


def test_load_coco_keeps_unambiguous_basename(tmp_path):
    """A basename owned by a single image is retained for fallback lookup."""
    coco = _write_coco(
        tmp_path / "ann.json",
        images=[(0, "only/cat.png")], categories=[(1, "dog")],
        annotations=[(0, 1, [0, 0, 1, 1])],
    )
    _, bn_to_classes, _, _ = _load_coco(coco)
    assert bn_to_classes["cat.png"] == {"dog"}


# ---------------------------------------------------------------------------
# load_scored_annotations — KITTI
# ---------------------------------------------------------------------------

_KITTI_LINE = "Car 0.00 0 -1.57 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 8.20 -1.56"


@pytest.fixture()
def kitti_scored(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    (d / "img_a.txt").write_text(_KITTI_LINE + " 0.92\n")
    (d / "img_b.txt").write_text(_KITTI_LINE + "\n")  # no score
    return str(d)


def test_load_scored_kitti_returns_three_tuples(kitti_scored):
    _, _, _, bn = load_scored_annotations(kitti_scored, AnnotationFormat.KITTI)
    assert all(len(b) == 3 for b in bn["img_a"])


def test_load_scored_kitti_score_parsed(kitti_scored):
    _, _, _, bn = load_scored_annotations(kitti_scored, AnnotationFormat.KITTI)
    assert bn["img_a"][0][2] == pytest.approx(0.92)


def test_load_scored_kitti_missing_score_is_none(kitti_scored):
    _, _, _, bn = load_scored_annotations(kitti_scored, AnnotationFormat.KITTI)
    assert bn["img_b"][0][2] is None


def test_load_annotations_ignores_kitti_score(kitti_scored):
    """load_annotations must still return 2-tuples even when the source has scores."""
    _, _, _, bn = load_annotations(kitti_scored, AnnotationFormat.KITTI)
    assert all(len(b) == 2 for b in bn["img_a"])


# ---------------------------------------------------------------------------
# load_scored_annotations — COCO
# ---------------------------------------------------------------------------

def test_load_scored_coco_score_populated(tmp_path):
    preds_path = tmp_path / "preds.json"
    _write_coco(
        preds_path,
        images=[(1, "img.jpg")], categories=[(1, "car")],
        annotations=[(1, 1, [0, 0, 10, 10])],
    )
    # Manually inject score into annotation
    data = json.loads(preds_path.read_text())
    data["annotations"][0]["score"] = 0.85
    preds_path.write_text(json.dumps(data))

    _, _, fp, _ = load_scored_annotations(str(preds_path), AnnotationFormat.COCO)
    assert fp["img.jpg"][0][2] == pytest.approx(0.85)


def test_load_scored_coco_missing_score_is_none(tmp_path):
    coco = _write_coco(
        tmp_path / "gt.json",
        images=[(1, "img.jpg")], categories=[(1, "car")],
        annotations=[(1, 1, [0, 0, 10, 10])],
    )
    _, _, fp, _ = load_scored_annotations(coco, AnnotationFormat.COCO)
    assert fp["img.jpg"][0][2] is None
