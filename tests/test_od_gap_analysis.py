# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RCCA OD gap-analysis helpers and integration pipeline."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat
from nvidia_tao_ds.rcca.gap_analysis.scripts.object_detection import (
    box_iou,
    calculate_image_metrics,
    extract_fp_fn_boxes,
    main,
    match_boxes,
    select_weak_images,
    write_artifacts,
    _load_image_records,
    _process_image,
)


# ---------------------------------------------------------------------------
# Fixtures — annotation files
# ---------------------------------------------------------------------------

KITTI_GT_LINE = "Car 0.00 0 0.00 10.0 20.0 110.0 70.0 0 0 0 0 0 0 0"
KITTI_PRED_LINE = "Car 0.00 0 0.00 12.0 22.0 108.0 68.0 0 0 0 0 0 0 0 0.90"
KITTI_FP_LINE = "Car 0.00 0 0.00 300.0 300.0 400.0 400.0 0 0 0 0 0 0 0 0.80"
KITTI_FN_GT_LINE = "Pedestrian 0.00 0 0.00 50.0 50.0 80.0 100.0 0 0 0 0 0 0 0"


@pytest.fixture()
def kitti_gt_dir(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    (d / "frame_001.txt").write_text(KITTI_GT_LINE + "\n" + KITTI_FN_GT_LINE + "\n")
    (d / "frame_002.txt").write_text(KITTI_GT_LINE + "\n")
    return str(d)


@pytest.fixture()
def kitti_pred_dir(tmp_path):
    d = tmp_path / "preds"
    d.mkdir()
    (d / "frame_001.txt").write_text(KITTI_PRED_LINE + "\n" + KITTI_FP_LINE + "\n")
    (d / "frame_002.txt").write_text(KITTI_PRED_LINE + "\n")
    return str(d)


@pytest.fixture()
def images_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "frame_001.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "frame_002.jpg").write_bytes(b"\xff\xd8\xff")
    return str(d)


# ---------------------------------------------------------------------------
# box_iou
# ---------------------------------------------------------------------------

class TestBoxIou:
    def test_identical_boxes(self):
        assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert box_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0,0,4,4] and [2,2,6,6] → intersection 2×2=4, union = 16+16-4 = 28
        iou = box_iou([0, 0, 4, 4], [2, 2, 6, 6])
        assert iou == pytest.approx(4 / 28)

    def test_zero_area_box(self):
        assert box_iou([0, 0, 0, 0], [0, 0, 5, 5]) == pytest.approx(0.0)

    def test_contained_box(self):
        # inner box entirely inside outer
        iou = box_iou([1, 1, 3, 3], [0, 0, 4, 4])
        inner_area = 4.0
        outer_area = 16.0
        assert iou == pytest.approx(inner_area / outer_area)


# ---------------------------------------------------------------------------
# match_boxes
# ---------------------------------------------------------------------------

class TestMatchBoxes:
    def test_perfect_match(self):
        gt = [[0, 0, 10, 10]]
        preds = [[0, 0, 10, 10]]
        matches, unmatched_p, unmatched_g = match_boxes(gt, preds, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0][2] == pytest.approx(1.0)
        assert not unmatched_p
        assert not unmatched_g

    def test_below_threshold_is_fp(self):
        gt = [[0, 0, 10, 10]]
        preds = [[50, 50, 60, 60]]  # no overlap
        matches, unmatched_p, unmatched_g = match_boxes(gt, preds, iou_threshold=0.5)
        assert not matches
        assert unmatched_p == [0]
        assert unmatched_g == [0]

    def test_one_pred_two_gt_picks_highest_iou(self):
        gt = [[0, 0, 10, 10], [0, 0, 9, 9]]
        preds = [[0, 0, 10, 10]]
        matches, _, unmatched_g = match_boxes(gt, preds, iou_threshold=0.5)
        assert len(matches) == 1
        matched_gt_idx = matches[0][1]
        # Should pick gt[0] (exact match, IoU=1.0) over gt[1] (slightly smaller)
        assert matched_gt_idx == 0
        assert len(unmatched_g) == 1

    def test_each_gt_matched_at_most_once(self):
        gt = [[0, 0, 10, 10]]
        preds = [[0, 0, 10, 10], [0, 0, 10, 10]]  # two identical preds
        matches, unmatched_p, _ = match_boxes(gt, preds, iou_threshold=0.5)
        assert len(matches) == 1
        assert len(unmatched_p) == 1

    def test_empty_gt(self):
        matches, unmatched_p, unmatched_g = match_boxes([], [[0, 0, 5, 5]], 0.5)
        assert not matches
        assert unmatched_p == [0]
        assert not unmatched_g

    def test_empty_preds(self):
        matches, unmatched_p, unmatched_g = match_boxes([[0, 0, 5, 5]], [], 0.5)
        assert not matches
        assert not unmatched_p
        assert unmatched_g == [0]


# ---------------------------------------------------------------------------
# extract_fp_fn_boxes
# ---------------------------------------------------------------------------

class TestExtractFpFnBoxes:
    def _run(self, gt_boxes, pred_records, iou_threshold=0.5):
        sorted_bboxes = [r["bbox"] for r in pred_records]
        _, unmatched_p, unmatched_g = match_boxes(gt_boxes, sorted_bboxes, iou_threshold)
        return extract_fp_fn_boxes(
            "img001", "/images/img001.jpg", "kpi1", "car",
            unmatched_p, unmatched_g,
            pred_records, gt_boxes,
            "/gt", "/preds",
        )

    def test_fp_row_fields(self):
        pred = [{"bbox": [50, 50, 100, 100], "score": 0.9}]
        rows = self._run(gt_boxes=[], pred_records=pred)
        assert len(rows) == 1
        r = rows[0]
        assert r["gap_type"] == "FP"
        assert r["class"] == "car"
        assert r["confidence"] == pytest.approx(0.9)
        assert r["bbox"] == [50, 50, 100, 100]

    def test_fn_row_fields(self):
        gt = [[10, 10, 50, 50]]
        rows = self._run(gt_boxes=gt, pred_records=[])
        assert len(rows) == 1
        r = rows[0]
        assert r["gap_type"] == "FN"
        assert math.isnan(r["confidence"])
        assert r["bbox"] == [10, 10, 50, 50]

    def test_best_iou_populated(self):
        gt = [[0, 0, 10, 10]]
        pred = [{"bbox": [2, 2, 8, 8], "score": 0.5}]
        # IoU is below 0.9 threshold so it becomes FP, but best_iou is recorded
        rows = self._run(gt_boxes=gt, pred_records=pred, iou_threshold=0.9)
        fp_rows = [r for r in rows if r["gap_type"] == "FP"]
        assert fp_rows[0]["best_iou"] > 0


# ---------------------------------------------------------------------------
# calculate_image_metrics
# ---------------------------------------------------------------------------

class TestCalculateImageMetrics:
    """ap50 is always NaN from calculate_image_metrics — it is filled in by
    _process_class via torchmetrics after the fact."""

    def test_perfect_detection(self):
        row = calculate_image_metrics("img", "/f.jpg", "kpi", "car", tp=5, fp_count=0, fn=0)
        assert row["precision"] == pytest.approx(1.0)
        assert row["recall"] == pytest.approx(1.0)
        assert math.isnan(row["ap50"])

    def test_zero_preds(self):
        row = calculate_image_metrics("img", "/f.jpg", "kpi", "car", tp=0, fp_count=0, fn=3)
        assert math.isnan(row["precision"])
        assert row["recall"] == pytest.approx(0.0)
        assert math.isnan(row["ap50"])

    def test_zero_gt(self):
        row = calculate_image_metrics("img", "/f.jpg", "kpi", "car", tp=0, fp_count=2, fn=0)
        assert row["precision"] == pytest.approx(0.0)
        assert math.isnan(row["recall"])
        assert math.isnan(row["ap50"])

    def test_mixed(self):
        row = calculate_image_metrics("img", "/f.jpg", "kpi", "car", tp=2, fp_count=1, fn=1)
        assert row["precision"] == pytest.approx(2 / 3)
        assert row["recall"] == pytest.approx(2 / 3)
        assert math.isnan(row["ap50"])

    def test_all_zero(self):
        row = calculate_image_metrics("img", "/f.jpg", "kpi", "car", tp=0, fp_count=0, fn=0)
        assert math.isnan(row["precision"])
        assert math.isnan(row["recall"])
        assert math.isnan(row["ap50"])


# ---------------------------------------------------------------------------
# select_weak_images
# ---------------------------------------------------------------------------

class TestSelectWeakImages:
    def _metric(self, image_id, cls, tp, fp_count, fn):
        return calculate_image_metrics(image_id, f"/{image_id}.jpg", "kpi", cls, tp, fp_count, fn)

    def test_weak_image_selected(self):
        rows = [self._metric("img1", "car", tp=0, fp_count=0, fn=3)]
        weak = select_weak_images(rows, {}, 0.5, 0.0, 0.5)
        assert len(weak) == 1
        assert weak[0]["image_id"] == "img1"

    def test_strong_image_not_selected(self):
        rows = [self._metric("img1", "car", tp=5, fp_count=1, fn=0)]
        weak = select_weak_images(rows, {}, 0.5, 0.0, 0.5)
        assert not weak

    def test_per_class_threshold(self):
        rows = [
            self._metric("img1", "car", tp=3, fp_count=0, fn=3),      # recall=0.5
            self._metric("img1", "pedestrian", tp=5, fp_count=0, fn=0),  # recall=1.0
        ]
        # car threshold = 0.8, pedestrian threshold = 0.5
        weak = select_weak_images(rows, {"car": {"recall": 0.8}, "pedestrian": {"recall": 0.5}}, 0.5, 0.0, 0.5)
        assert len(weak) == 1
        assert "car" in weak[0]["weak_classes"]
        assert "pedestrian" not in weak[0]["weak_classes"]

    def test_fp_only_not_selected(self):
        # tp=0, fp=3, fn=0 → recall is NaN (no GT) → skip
        rows = [self._metric("img1", "car", tp=0, fp_count=3, fn=0)]
        weak = select_weak_images(rows, {}, 0.5, 0.0, 0.5)
        assert not weak

    def test_recall_weak_flag_set(self):
        rows = [self._metric("img1", "car", tp=0, fp_count=0, fn=1)]
        weak = select_weak_images(rows, {"car": {"recall": 0.9}}, 0.5, 0.0, 0.5)
        assert weak[0]["weak_recall"] == [True]
        assert weak[0]["weak_precision"] == [False]
        assert weak[0]["weak_ap50"] == [False]

    def test_default_recall_threshold_applies(self):
        rows = [self._metric("img1", "truck", tp=0, fp_count=0, fn=1)]
        weak = select_weak_images(rows, {}, 0.5, 0.0, 0.5)
        assert weak[0]["weak_recall"] == [True]

    def test_precision_threshold_selects_fp_heavy_image(self):
        # tp=0, fp=5, fn=0 → precision=0.0, recall=NaN → only precision gate fires
        rows = [self._metric("img1", "car", tp=0, fp_count=5, fn=0)]
        weak = select_weak_images(rows, {"car": {"precision": 0.5}}, 0.5, 0.5, 0.5)
        assert len(weak) == 1
        assert weak[0]["weak_precision"] == [True]
        assert weak[0]["weak_recall"] == [False]

    def test_precision_gate_disabled_when_zero(self):
        rows = [self._metric("img1", "car", tp=0, fp_count=5, fn=0)]
        weak = select_weak_images(rows, {}, 0.5, 0.0, 0.5)
        assert not weak  # recall is NaN and precision gate disabled

    def test_ap50_threshold_selects_weak_image(self):
        row = self._metric("img1", "car", tp=1, fp_count=1, fn=1)
        row["ap50"] = 0.33
        weak = select_weak_images([row], {"car": {"ap50": 0.5}}, 0.5, 0.0, 0.5)
        assert len(weak) == 1
        assert weak[0]["weak_ap50"] == [True]

    def test_per_class_precision_threshold(self):
        rows = [
            self._metric("img1", "car", tp=0, fp_count=3, fn=0),
            self._metric("img1", "truck", tp=0, fp_count=3, fn=0),
        ]
        # car has explicit precision threshold, truck uses default (disabled=0.0)
        weak = select_weak_images(rows, {"car": {"precision": 0.5}}, 0.5, 0.0, 0.5)
        assert len(weak) == 1
        assert "car" in weak[0]["weak_classes"]
        assert "truck" not in weak[0]["weak_classes"]

    def test_multiple_metrics_fire_for_same_class(self):
        row = self._metric("img1", "car", tp=0, fp_count=3, fn=3)
        row["ap50"] = 0.0
        weak = select_weak_images([row], {}, 0.5, 0.5, 0.5)
        assert len(weak) == 1
        assert weak[0]["weak_recall"] == [True]
        assert weak[0]["weak_precision"] == [True]
        assert weak[0]["weak_ap50"] == [True]


# ---------------------------------------------------------------------------
# write_artifacts
# ---------------------------------------------------------------------------

class TestWriteArtifacts:
    def test_empty_artifacts_have_correct_schemas(self, tmp_path):
        write_artifacts(str(tmp_path), [], [], [], {
            "kpi": "k", "counts_by_type": {}, "counts_by_class": {},
            "settings": {"iou_threshold": 0.5, "conf_threshold": 0.0, "min_area": 0},
        })
        gaps = pd.read_parquet(tmp_path / "box_gaps.parquet")
        metrics = pd.read_parquet(tmp_path / "image_metrics.parquet")
        weak = pd.read_parquet(tmp_path / "weak_images.parquet")
        assert set(gaps.columns) >= {"kpi", "gap_type", "bbox", "confidence", "best_iou"}
        assert set(metrics.columns) >= {"tp", "fp", "fn", "precision", "recall"}
        assert set(weak.columns) >= {"weak_classes", "weak_recall", "weak_precision", "weak_ap50"}
        assert (tmp_path / "gap_report.json").is_file()

    def test_gap_report_contents(self, tmp_path):
        report = {
            "kpi": "test_kpi", "counts_by_type": {"FP": 2},
            "counts_by_class": {"car": {"FP": 2}},
            "settings": {"iou_threshold": 0.5, "conf_threshold": 0.0, "min_area": 0},
        }
        write_artifacts(str(tmp_path), [], [], [], report)
        loaded = json.loads((tmp_path / "gap_report.json").read_text())
        assert loaded["kpi"] == "test_kpi"
        assert loaded["counts_by_type"]["FP"] == 2


# ---------------------------------------------------------------------------
# Integration: full pipeline with KITTI fixtures
# ---------------------------------------------------------------------------

class TestKittiIntegration:
    def test_artifacts_created(self, kitti_gt_dir, kitti_pred_dir, images_dir, tmp_path):
        results_dir = str(tmp_path / "results")
        cfg = OmegaConf.create({
            "ground_truth_ann_path": kitti_gt_dir,
            "inference_ann_path": kitti_pred_dir,
            "images_dir": images_dir,
            "results_dir": results_dir,
            "kpi": "test_kpi",
            "input_format": "kitti",
            "iou_threshold": 0.5,
            "conf_threshold": 0.0,
            "min_area": 0,
            "class_mapping": {},
            "weak_thresholds": {},
            "default_recall_threshold": 0.5,
            "default_precision_threshold": 0.0,
            "default_ap50_threshold": 0.0,
        })

        image_records = _load_image_records(
            cfg.ground_truth_ann_path, cfg.inference_ann_path,
            cfg.images_dir, AnnotationFormat.KITTI, {},
        )
        assert len(image_records) == 2

        all_gap_rows, all_metric_rows = [], []
        for img_path, recs in image_records.items():
            g_rows, m_rows = _process_image(
                img_path, recs["gt"], recs["preds"],
                cfg.conf_threshold, cfg.min_area, cfg.iou_threshold,
                cfg.kpi, cfg.ground_truth_ann_path, cfg.inference_ann_path,
            )
            all_gap_rows.extend(g_rows)
            all_metric_rows.extend(m_rows)

        weak_rows = select_weak_images(all_metric_rows, {}, cfg.default_recall_threshold, 0.0, 0.5)
        report = {
            "kpi": cfg.kpi, "counts_by_type": {}, "counts_by_class": {},
            "settings": {"iou_threshold": 0.5, "conf_threshold": 0.0, "min_area": 0},
        }
        write_artifacts(results_dir, all_gap_rows, all_metric_rows, weak_rows, report)

        out = Path(results_dir)
        assert (out / "box_gaps.parquet").is_file()
        assert (out / "image_metrics.parquet").is_file()
        assert (out / "weak_images.parquet").is_file()
        assert (out / "gap_report.json").is_file()

    def test_ap50_populated_by_process_image(self, kitti_gt_dir, kitti_pred_dir, images_dir):
        """ap50 comes from torchmetrics (not calculate_image_metrics) and must be non-NaN."""
        image_records = _load_image_records(
            kitti_gt_dir, kitti_pred_dir, images_dir, AnnotationFormat.KITTI, {}
        )
        img_path = next(p for p in image_records if "frame_002" in p)
        _, m_rows = _process_image(
            img_path, image_records[img_path]["gt"], image_records[img_path]["preds"],
            0.0, 0, 0.5, "kpi", kitti_gt_dir, kitti_pred_dir,
        )
        car_row = next(r for r in m_rows if r["class"] == "Car")
        assert not math.isnan(car_row["ap50"])
        assert 0.0 <= car_row["ap50"] <= 1.0

    def test_gap_recorded_for_unmatched_class(self, kitti_gt_dir, kitti_pred_dir, images_dir):
        image_records = _load_image_records(
            kitti_gt_dir, kitti_pred_dir, images_dir, AnnotationFormat.KITTI, {}
        )
        # frame_001 has a second-class GT (Pedestrian) with no matching prediction → FN
        img_path = next(p for p in image_records if "frame_001" in p)
        g_rows, _ = _process_image(
            img_path, image_records[img_path]["gt"], image_records[img_path]["preds"],
            0.0, 0, 0.5, "kpi", kitti_gt_dir, kitti_pred_dir,
        )
        fn_rows = [r for r in g_rows if r["gap_type"] == "FN" and r["class"] == "Pedestrian"]
        assert len(fn_rows) == 1

    def test_fp_gap_recorded_for_far_prediction(self, kitti_gt_dir, kitti_pred_dir, images_dir):
        image_records = _load_image_records(
            kitti_gt_dir, kitti_pred_dir, images_dir, AnnotationFormat.KITTI, {}
        )
        img_path = next(p for p in image_records if "frame_001" in p)
        g_rows, _ = _process_image(
            img_path, image_records[img_path]["gt"], image_records[img_path]["preds"],
            0.0, 0, 0.5, "kpi", kitti_gt_dir, kitti_pred_dir,
        )
        fp_rows = [r for r in g_rows if r["gap_type"] == "FP"]
        assert len(fp_rows) >= 1

    def test_conf_threshold_filters_predictions(self, kitti_gt_dir, kitti_pred_dir, images_dir):
        image_records = _load_image_records(
            kitti_gt_dir, kitti_pred_dir, images_dir, AnnotationFormat.KITTI, {}
        )
        img_path = next(p for p in image_records if "frame_001" in p)
        # Both preds have score <0.95 → filtered out → all GT become FN
        g_rows, _ = _process_image(
            img_path, image_records[img_path]["gt"], image_records[img_path]["preds"],
            0.95, 0, 0.5, "kpi", kitti_gt_dir, kitti_pred_dir,
        )
        assert all(r["gap_type"] == "FN" for r in g_rows)

    def test_weak_images_selected_on_low_recall(self, kitti_gt_dir, kitti_pred_dir, images_dir):
        image_records = _load_image_records(
            kitti_gt_dir, kitti_pred_dir, images_dir, AnnotationFormat.KITTI, {}
        )
        all_metric_rows = []
        for img_path, recs in image_records.items():
            _, m_rows = _process_image(
                img_path, recs["gt"], recs["preds"],
                0.0, 0, 0.5, "kpi", kitti_gt_dir, kitti_pred_dir,
            )
            all_metric_rows.extend(m_rows)
        # frame_001 has a Pedestrian FN so its recall = 0 → weak
        weak = select_weak_images(all_metric_rows, {}, 0.5, 0.0, 0.5)
        weak_ids = {w["image_id"] for w in weak}
        assert "frame_001" in weak_ids

    def test_prediction_only_class_produces_fp_gap(self, tmp_path, images_dir):
        """A class present only in predictions (no GT) must produce an FP gap row
        and a metric row — it must not be silently dropped."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        pred_dir = tmp_path / "preds"
        pred_dir.mkdir()
        # GT has Car only; preds have Car + Cyclist (no Cyclist GT)
        (gt_dir / "frame_001.txt").write_text(
            "Car 0.00 0 0.00 10.0 20.0 110.0 70.0 0 0 0 0 0 0 0\n"
        )
        (pred_dir / "frame_001.txt").write_text(
            "Car 0.00 0 0.00 12.0 22.0 108.0 68.0 0 0 0 0 0 0 0 0.90\n"
            "Cyclist 0.00 0 0.00 200.0 200.0 300.0 300.0 0 0 0 0 0 0 0 0.80\n"
        )
        image_records = _load_image_records(
            str(gt_dir), str(pred_dir), images_dir, AnnotationFormat.KITTI, {}
        )
        img_path = next(p for p in image_records if "frame_001" in p)
        g_rows, m_rows = _process_image(
            img_path, image_records[img_path]["gt"], image_records[img_path]["preds"],
            0.0, 0, 0.5, "kpi", str(gt_dir), str(pred_dir),
        )
        cyclist_fp = [r for r in g_rows if r["class"] == "Cyclist" and r["gap_type"] == "FP"]
        assert len(cyclist_fp) == 1, "Cyclist FP gap must be recorded even with no GT"
        cyclist_metrics = [r for r in m_rows if r["class"] == "Cyclist"]
        assert len(cyclist_metrics) == 1
        assert cyclist_metrics[0]["fp"] == 1
        assert cyclist_metrics[0]["precision"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end: the command actually runs and writes artifacts
# ---------------------------------------------------------------------------

_ARTIFACTS = ("box_gaps.parquet", "image_metrics.parquet", "weak_images.parquet", "gap_report.json")


def _od_cfg(gt, preds, images, out):
    """Minimal config for a full object_detection run over the KITTI fixtures."""
    return OmegaConf.create({
        "ground_truth_ann_path": gt, "inference_ann_path": preds,
        "images_dir": images, "results_dir": str(out), "kpi": "od_kpi",
        "input_format": "kitti", "iou_threshold": 0.5, "conf_threshold": 0.0,
        "min_area": 0, "class_mapping": {}, "weak_thresholds": {},
        "default_recall_threshold": 0.5, "default_precision_threshold": 0.0,
        "default_ap50_threshold": 0.5,
    })


def test_object_detection_main_writes_artifacts(kitti_gt_dir, kitti_pred_dir, images_dir, tmp_path):
    """main() end-to-end writes all four artifacts (a no-op main would leave results_dir empty)."""
    out = tmp_path / "out"
    main(_od_cfg(kitti_gt_dir, kitti_pred_dir, images_dir, out))

    for name in _ARTIFACTS:
        assert (out / name).exists(), f"missing artifact {name}"
    report = json.loads((out / "gap_report.json").read_text())
    assert report["kpi"] == "od_kpi"
    # frame_001 has an FP (spurious Car) and an FN (unmatched Pedestrian) → gaps are non-empty.
    assert len(pd.read_parquet(out / "box_gaps.parquet")) >= 1


def test_object_detection_runs_as_script(kitti_gt_dir, kitti_pred_dir, images_dir, tmp_path):
    """Running the module as a script must execute main() and write artifacts.

    Guards the ``if __name__ == "__main__": main()`` entrypoint: without it the module
    imports, defines main(), never calls it, and exits 0 — a silent PASS with no output.
    """
    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "nvidia_tao_ds.rcca.gap_analysis.scripts.object_detection",
         f"ground_truth_ann_path={kitti_gt_dir}", f"inference_ann_path={kitti_pred_dir}",
         f"images_dir={images_dir}", f"results_dir={out}", "kpi=od_kpi", "input_format=kitti"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "weak_images.parquet").exists(), \
        "running the module as a script produced no artifacts — main() never executed"
