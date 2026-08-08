# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RCCA object-detection gap-analysis command."""

import json
import logging
import math
from collections import defaultdict
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torchmetrics.detection import MeanAveragePrecision

from nvidia_tao_ds.config.rcca.gap_analysis.object_detection import ObjectDetectionGapConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.utils.dataset_loading import (
    AnnotationFormat,
    coco_lookup,
    load_annotations,
    load_scored_annotations,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Typed empty-DataFrame schemas so artifacts are always well-formed even when
# no gaps are found.
_BOX_GAPS_COLS = {
    "kpi": "object", "image_id": "object", "filepath": "object",
    "class": "object", "gap_type": "object", "bbox": "object",
    "confidence": "float64", "best_iou": "float64",
    "gt_label_path": "object", "pred_label_path": "object",
}
_IMAGE_METRICS_COLS = {
    "kpi": "object", "image_id": "object", "filepath": "object",
    "class": "object", "tp": "int64", "fp": "int64", "fn": "int64",
    "precision": "float64", "recall": "float64", "ap50": "float64",
}
_WEAK_IMAGES_COLS = {
    "kpi": "object", "image_id": "object", "filepath": "object",
    "weak_classes": "object",
    "weak_recall": "object", "weak_precision": "object", "weak_ap50": "object",
}


# ── Low-level geometry ────────────────────────────────────────────────────────

def _xywh_to_xyxy(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _box_area(xyxy: List[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _is_valid_box(xyxy: List[float]) -> bool:
    return all(math.isfinite(v) for v in xyxy) and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]


# ── Public helpers ────────────────────────────────────────────────────────────

def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-union for two ``[xmin, ymin, xmax, ymax]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) +
        max(0.0, bx2 - bx1) * max(0.0, by2 - by1) -
        inter
    )
    return inter / union if union > 0 else 0.0


def match_boxes(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy IoU matching: each prediction paired with the highest-IoU unmatched GT.

    ``pred_boxes`` must already be sorted in descending priority order (highest
    confidence first with a deterministic tie-break).

    Args:
        gt_boxes: list of ``[xmin, ymin, xmax, ymax]``.
        pred_boxes: list of ``[xmin, ymin, xmax, ymax]`` in priority order.
        iou_threshold: Minimum IoU to accept a match as a true positive.

    Returns:
        ``(matches, unmatched_pred_indices, unmatched_gt_indices)`` where
        ``matches`` is a list of ``(pred_idx, gt_idx, iou)`` tuples.
    """
    matched_gt = set()
    matches = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou, best_gt_idx = -1.0, -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)

    matched_pred = {m[0] for m in matches}
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return matches, unmatched_preds, unmatched_gts


def extract_fp_fn_boxes(
    image_id: str,
    filepath: str,
    kpi: str,
    cls: str,
    unmatched_pred_idx: List[int],
    unmatched_gt_idx: List[int],
    sorted_pred_records: List[Dict[str, Any]],
    cls_gt_boxes: List[List[float]],
    gt_label_path: str,
    pred_label_path: str,
) -> List[Dict[str, Any]]:
    """Build FP and FN gap rows from ``match_boxes`` results.

    Args:
        image_id: Image stem used as row identifier.
        filepath: Absolute image path.
        kpi: KPI tag for this run.
        cls: Normalized class name.
        unmatched_pred_idx: Indices into ``sorted_pred_records`` with no GT match.
        unmatched_gt_idx: Indices into ``cls_gt_boxes`` not covered by a prediction.
        sorted_pred_records: Prediction dicts in the order passed to ``match_boxes``.
        cls_gt_boxes: GT ``[xmin,ymin,xmax,ymax]`` boxes for this class.
        gt_label_path: Source annotation path for ground truth.
        pred_label_path: Source annotation path for predictions.

    Returns:
        List of gap row dicts (one per unmatched box).
    """
    rows = []
    for pred_idx in unmatched_pred_idx:
        pred = sorted_pred_records[pred_idx]
        best = max((box_iou(pred["bbox"], gt_box) for gt_box in cls_gt_boxes), default=0.0)
        rows.append({
            "kpi": kpi, "image_id": image_id, "filepath": filepath,
            "class": cls, "gap_type": "FP",
            "bbox": pred["bbox"], "confidence": pred["score"],
            "best_iou": best,
            "gt_label_path": gt_label_path, "pred_label_path": pred_label_path,
        })
    for gt_idx in unmatched_gt_idx:
        gt_box = cls_gt_boxes[gt_idx]
        best = max((box_iou(gt_box, pred["bbox"]) for pred in sorted_pred_records), default=0.0)
        rows.append({
            "kpi": kpi, "image_id": image_id, "filepath": filepath,
            "class": cls, "gap_type": "FN",
            "bbox": gt_box, "confidence": float("nan"),
            "best_iou": best,
            "gt_label_path": gt_label_path, "pred_label_path": pred_label_path,
        })
    return rows


def calculate_image_metrics(
    image_id: str,
    filepath: str,
    kpi: str,
    cls: str,
    tp: int,
    fp_count: int,
    fn: int,
) -> Dict[str, Any]:
    """Return a single image/class metric row dict.

    Precision and recall are ``NaN`` when the denominator is zero (e.g. no GT
    boxes for recall, no predictions for precision). ``ap50`` is populated
    separately by ``_calculate_ap50`` using torchmetrics; this function
    initialises the key to ``NaN`` as a placeholder.
    """
    precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "kpi": kpi, "image_id": image_id, "filepath": filepath,
        "class": cls, "tp": tp, "fp": fp_count, "fn": fn,
        "precision": precision, "recall": recall, "ap50": float("nan"),
    }


_AP50_METRIC: Optional[MeanAveragePrecision] = None


def _get_ap50_metric() -> MeanAveragePrecision:
    """Return the module-level AP50 metric, initialising it on first use.

    Lazy init avoids paying the MeanAveragePrecision construction cost when
    other gap_analysis subtasks (vcn_aoi, vlm_bcq) are invoked instead.
    """
    global _AP50_METRIC  # pylint: disable=global-statement
    if _AP50_METRIC is None:
        _AP50_METRIC = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[0.5],
            rec_thresholds=torch.linspace(0, 1, 101).tolist(),
            class_metrics=False,
        )
    return _AP50_METRIC


def _calculate_ap50(
    gt_boxes: List[List[float]],
    pred_records: List[Dict[str, Any]],
) -> float:
    """Compute per-image AP50 via 101-point interpolated PR curve at IoU=0.5.

    Uses torchmetrics ``MeanAveragePrecision`` with all area-filtered predictions
    (confidence threshold is NOT applied — torchmetrics sweeps thresholds
    internally to build the full curve). Returns ``NaN`` when there are no GT
    boxes (undefined), ``0.0`` when there are GT boxes but no predictions.

    Args:
        gt_boxes: List of ``[xmin, ymin, xmax, ymax]`` ground-truth boxes.
        pred_records: Prediction dicts with ``bbox`` and ``score`` keys.

    Returns:
        AP50 as a float.
    """
    if not gt_boxes:
        return float("nan")
    if not pred_records:
        return 0.0

    scores = [r["score"] if r["score"] is not None else 1.0 for r in pred_records]
    preds = [{
        "boxes": torch.tensor([r["bbox"] for r in pred_records], dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.zeros(len(pred_records), dtype=torch.int64),
    }]
    target = [{
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.zeros(len(gt_boxes), dtype=torch.int64),
    }]

    metric = _get_ap50_metric()
    metric.reset()
    metric.update(preds, target)
    return metric.compute()["map_50"].item()


def select_weak_images(
    image_metric_rows: List[Dict[str, Any]],
    weak_thresholds: Dict[str, Dict[str, float]],
    default_recall_threshold: float,
    default_precision_threshold: float,
    default_ap50_threshold: float,
) -> List[Dict[str, Any]]:
    """Select images where any class metric falls below its per-class threshold.

    ``weak_thresholds`` is a nested dict ``{class_name: {recall, precision, ap50}}``.
    Any metric key absent from a class entry falls back to the corresponding
    ``default_*_threshold``. Set a default to 0.0 to disable that gate.
    NaN metrics (undefined denominator) are treated as passing.

    Args:
        image_metric_rows: List of metric row dicts from ``calculate_image_metrics``.
        weak_thresholds: Nested per-class thresholds (class → metric → float).
        default_recall_threshold: Fallback recall threshold for unlisted classes.
        default_precision_threshold: Fallback precision threshold; 0.0 disables it.
        default_ap50_threshold: Fallback AP50 threshold; 0.0 disables it.

    Returns:
        List of weak-image row dicts with ``weak_recall``, ``weak_precision``,
        and ``weak_ap50`` boolean lists parallel to ``weak_classes``.
    """
    by_image = defaultdict(list)
    for row in image_metric_rows:
        by_image[(row["image_id"], row["filepath"], row["kpi"])].append(row)

    weak_rows = []
    for (image_id, filepath, kpi), rows in by_image.items():
        weak_classes: List[str] = []
        wr_flags: List[bool] = []
        wp_flags: List[bool] = []
        wa_flags: List[bool] = []
        for row in rows:
            cls = row["class"]
            cls_thresh = weak_thresholds.get(cls, {})

            r_thresh = cls_thresh.get("recall", default_recall_threshold)
            recall = row["recall"]
            is_wr = not math.isnan(recall) and recall < r_thresh

            p_thresh = cls_thresh.get("precision", default_precision_threshold)
            precision = row["precision"]
            is_wp = p_thresh > 0 and not math.isnan(precision) and precision < p_thresh

            a_thresh = cls_thresh.get("ap50", default_ap50_threshold)
            ap50 = row["ap50"]
            is_wa = a_thresh > 0 and not math.isnan(ap50) and ap50 < a_thresh

            if is_wr or is_wp or is_wa:
                weak_classes.append(cls)
                wr_flags.append(is_wr)
                wp_flags.append(is_wp)
                wa_flags.append(is_wa)

        if weak_classes:
            weak_rows.append({
                "kpi": kpi, "image_id": image_id, "filepath": filepath,
                "weak_classes": weak_classes,
                "weak_recall": wr_flags,
                "weak_precision": wp_flags,
                "weak_ap50": wa_flags,
            })
    return weak_rows


def write_artifacts(
    results_dir: str,
    gap_rows: List[Dict[str, Any]],
    metric_rows: List[Dict[str, Any]],
    weak_rows: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> None:
    """Write all four gap-analysis artifacts to ``results_dir``.

    All four files are always written — even when the input lists are empty —
    so downstream consumers can rely on the files existing with correct schemas.

    Args:
        results_dir: Output directory (created if absent).
        gap_rows: List of box gap dicts (FP/FN).
        metric_rows: List of image/class metric dicts.
        weak_rows: List of weak-image dicts.
        report: Gap report dict written as ``gap_report.json``.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _to_df(rows, schema):
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame({c: pd.Series(dtype=d) for c, d in schema.items()})

    _to_df(gap_rows, _BOX_GAPS_COLS).to_parquet(out / "box_gaps.parquet", index=False)
    _to_df(metric_rows, _IMAGE_METRICS_COLS).to_parquet(out / "image_metrics.parquet", index=False)
    _to_df(weak_rows, _WEAK_IMAGES_COLS).to_parquet(out / "weak_images.parquet", index=False)
    with open(out / "gap_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# ── Adapter and per-image orchestration ──────────────────────────────────────

def _load_image_records(
    gt_path: str,
    pred_path: str,
    images_dir: str,
    fmt: AnnotationFormat,
    class_mapping: Dict[str, str],
) -> Dict[str, Dict[str, List]]:
    """Return per-image GT and prediction records from both annotation sources.

    Args:
        gt_path: Ground-truth annotation source path.
        pred_path: Inference annotation source path.
        images_dir: Root image directory; all image files are enumerated here.
        fmt: ``AnnotationFormat`` value, or ``None`` for auto-detection.
        class_mapping: dict mapping raw label strings to canonical class names.

    Returns:
        Dict of ``img_path -> {"gt": [...], "preds": [...]}``.
        GT entries are ``{"class": str, "bbox": [xmin,ymin,xmax,ymax]}``.
        Pred entries add ``"score": float|None``.
    """
    # GT uses the unscored loader — GT files are 15-field KITTI / no-score COCO.
    # Using the scored loader on GT would silently mis-parse a 16-field GT file.
    _, _, gt_fp_boxes, gt_bn_boxes = load_annotations(gt_path, fmt)
    _, _, pred_fp_boxes, pred_bn_boxes = load_scored_annotations(pred_path, fmt)

    images_root = Path(images_dir)
    if not images_root.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    img_paths = sorted(
        str(p) for p in images_root.rglob("*")
        if p.suffix.lower() in _IMAGE_EXTS
    )
    if not img_paths:
        raise ValueError(f"No supported images found in images_dir: {images_dir}")

    records = {}
    for img_path in img_paths:
        raw_gt = coco_lookup(img_path, gt_fp_boxes, gt_bn_boxes, [])
        if not raw_gt:
            # COCO file_name values are often relative (e.g. dirA/frame.png).
            # Try the path relative to images_dir as a fallback.
            try:
                rel = str(Path(img_path).relative_to(images_root))
                raw_gt = gt_fp_boxes.get(rel, [])
            except ValueError:
                pass
        raw_preds = coco_lookup(img_path, pred_fp_boxes, pred_bn_boxes, [])
        if not raw_preds:
            try:
                rel = str(Path(img_path).relative_to(images_root))
                raw_preds = pred_fp_boxes.get(rel, [])
            except ValueError:
                pass

        gt_recs = []
        for cls, bbox_xywh in raw_gt:
            mapped = class_mapping.get(cls, cls)
            xyxy = _xywh_to_xyxy(bbox_xywh)
            if not _is_valid_box(xyxy):
                logger.warning("Discarding invalid GT box %s in %s", xyxy, img_path)
                continue
            gt_recs.append({"class": mapped, "bbox": xyxy})

        pred_recs = []
        for cls, bbox_xywh, score in raw_preds:
            mapped = class_mapping.get(cls, cls)
            xyxy = _xywh_to_xyxy(bbox_xywh)
            if not _is_valid_box(xyxy):
                logger.warning("Discarding invalid pred box %s in %s", xyxy, img_path)
                continue
            pred_recs.append({"class": mapped, "bbox": xyxy, "score": score})

        records[img_path] = {"gt": gt_recs, "preds": pred_recs}
    return records


def _process_class(
    image_id: str,
    img_path: str,
    kpi: str,
    cls: str,
    cls_gt_boxes: List[List[float]],
    cls_pred_recs: List[Dict[str, Any]],
    conf_threshold: float,
    min_area: int,
    iou_threshold: float,
    gt_label_path: str,
    pred_label_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process one image/class pair and return ``(gap_rows, metric_row)``."""
    # Filter and sort predictions
    valid_preds = [
        r for r in cls_pred_recs
        if (r["score"] is None or r["score"] >= conf_threshold) and
        _box_area(r["bbox"]) >= min_area
    ]
    valid_gt = [b for b in cls_gt_boxes if _box_area(b) >= min_area]

    indexed = list(enumerate(valid_preds))
    indexed.sort(key=lambda t: (-(t[1]["score"] or 0.0), t[0]))
    sorted_records = [valid_preds[t[0]] for t in indexed]
    sorted_bboxes = [r["bbox"] for r in sorted_records]

    matches, unmatched_pred_idx, unmatched_gt_idx = match_boxes(
        valid_gt, sorted_bboxes, iou_threshold
    )

    tp = len(matches)
    fp_count = len(unmatched_pred_idx)
    fn = len(unmatched_gt_idx)

    gap_rows = extract_fp_fn_boxes(
        image_id, img_path, kpi, cls,
        unmatched_pred_idx, unmatched_gt_idx,
        sorted_records, valid_gt,
        gt_label_path, pred_label_path,
    )
    metric_row = calculate_image_metrics(image_id, img_path, kpi, cls, tp, fp_count, fn)
    # AP50 uses all area-filtered predictions without conf_threshold so that
    # torchmetrics can sweep confidence thresholds internally.
    all_area_preds = [r for r in cls_pred_recs if _box_area(r["bbox"]) >= min_area]
    metric_row["ap50"] = _calculate_ap50(valid_gt, all_area_preds)
    return gap_rows, metric_row


def _process_image(
    img_path: str,
    gt_recs: List[Dict[str, Any]],
    pred_recs: List[Dict[str, Any]],
    conf_threshold: float,
    min_area: int,
    iou_threshold: float,
    kpi: str,
    gt_path: str,
    pred_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(gap_rows, metric_rows)`` for one image across all classes."""
    image_id = Path(img_path).stem

    gt_by_class = defaultdict(list)
    for r in gt_recs:
        gt_by_class[r["class"]].append(r["bbox"])

    pred_by_class = defaultdict(list)
    for r in pred_recs:
        pred_by_class[r["class"]].append(r)

    all_classes = sorted(set(gt_by_class) | set(pred_by_class))
    gap_rows, metric_rows = [], []
    for cls in all_classes:
        g_rows, m_row = _process_class(
            image_id, img_path, kpi, cls,
            gt_by_class.get(cls, []),
            pred_by_class.get(cls, []),
            conf_threshold, min_area, iou_threshold,
            str(gt_path), str(pred_path),
        )
        gap_rows.extend(g_rows)
        metric_rows.append(m_row)
    return gap_rows, metric_rows


# ── CLI entrypoint ────────────────────────────────────────────────────────────

spec_root = Path(__file__).resolve().parent


@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="object_detection",
    schema=ObjectDetectionGapConfig
)
def main(cfg: ObjectDetectionGapConfig):
    """CLI entrypoint for OD KPI gap analysis."""
    _log_level = getattr(logging, getenv("TAO_LOGGING_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    class_mapping = dict(cfg.class_mapping) if cfg.class_mapping else {}
    weak_thresholds = dict(cfg.weak_thresholds) if cfg.weak_thresholds else {}
    fmt = AnnotationFormat(cfg.input_format)

    image_records = _load_image_records(
        cfg.ground_truth_ann_path, cfg.inference_ann_path,
        cfg.images_dir, fmt, class_mapping,
    )
    logger.info("Processing %d images", len(image_records))

    all_gap_rows, all_metric_rows = [], []
    for img_path, recs in image_records.items():
        g_rows, m_rows = _process_image(
            img_path, recs["gt"], recs["preds"],
            cfg.conf_threshold, cfg.min_area, cfg.iou_threshold,
            cfg.kpi, cfg.ground_truth_ann_path, cfg.inference_ann_path,
        )
        all_gap_rows.extend(g_rows)
        all_metric_rows.extend(m_rows)

    weak_rows = select_weak_images(
        all_metric_rows, weak_thresholds,
        cfg.default_recall_threshold,
        cfg.default_precision_threshold,
        cfg.default_ap50_threshold,
    )

    counts_by_type: dict = defaultdict(int)
    counts_by_class: dict = defaultdict(lambda: defaultdict(int))
    for row in all_gap_rows:
        counts_by_type[row["gap_type"]] += 1
        counts_by_class[row["class"]][row["gap_type"]] += 1

    report = {
        "kpi": cfg.kpi,
        "counts_by_type": dict(counts_by_type),
        "counts_by_class": {cls: dict(c) for cls, c in counts_by_class.items()},
        "settings": {
            "iou_threshold": cfg.iou_threshold,
            "conf_threshold": cfg.conf_threshold,
            "min_area": cfg.min_area,
        },
    }

    write_artifacts(cfg.results_dir, all_gap_rows, all_metric_rows, weak_rows, report)
    logger.info(
        "Done. FP=%d FN=%d weak_images=%d → %s",
        counts_by_type.get("FP", 0), counts_by_type.get("FN", 0),
        len(weak_rows), cfg.results_dir,
    )


if __name__ == "__main__":
    main()
