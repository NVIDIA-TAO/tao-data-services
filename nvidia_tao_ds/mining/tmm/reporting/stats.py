# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribution, per-class, and distance summary statistics for unique_neighbor_matching."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Set

import numpy as np
import pandas as pd

from nvidia_tao_ds.core.utils.dataset_loading import coco_lookup, load_annotations

if TYPE_CHECKING:
    from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat


def calculate_distribution_stats(
    filepaths: set, detection_file: Optional[str],
    rare_class_list: Set[str], filepath_column_name: str = "filepath",
    *, fmt: AnnotationFormat,
) -> dict:
    """Return rare_count, common_count, rare_instance_count, total_files for a set of filepaths."""
    if not detection_file or not rare_class_list:
        return {
            "rare_count": 0, "common_count": len(filepaths),
            "rare_instance_count": 0, "total_files": len(filepaths),
        }

    fp_to_cls, bn_to_cls, _, _ = load_annotations(detection_file, fmt)
    rare_lower = {cls.lower() for cls in rare_class_list}

    rare_count = common_count = rare_instance_count = 0
    for fp in filepaths:
        rare_hits = {c.lower() for c in coco_lookup(fp, fp_to_cls, bn_to_cls, set())} & rare_lower
        rare_instance_count += len(rare_hits)
        if rare_hits:
            rare_count += 1
        else:
            common_count += 1

    return {"rare_count": rare_count, "common_count": common_count,
            "rare_instance_count": rare_instance_count, "total_files": len(filepaths)}


def calculate_per_class_counts(
    filepaths: set, detection_file: Optional[str], *, fmt: AnnotationFormat,
) -> Dict[str, Dict[str, int]]:
    """
    For each class, count image_count and instance_count within filepaths.
    Supports COCO (JSON) and KITTI (directory). Returns {} if no detection file.
    """
    if not detection_file:
        return {}

    _, _, fp_to_boxes, bn_to_boxes = load_annotations(detection_file, fmt)

    per_class: Dict[str, Dict[str, int]] = {}
    class_to_image_keys: Dict[str, Set[str]] = {}

    for fp in filepaths:
        boxes = coco_lookup(fp, fp_to_boxes, bn_to_boxes, [])
        for cat_name, _ in boxes:
            per_class.setdefault(cat_name, {"image_count": 0, "instance_count": 0})["instance_count"] += 1
            class_to_image_keys.setdefault(cat_name, set()).add(fp)

    for cat_name, fps in class_to_image_keys.items():
        per_class[cat_name]["image_count"] = len(fps)

    return per_class


def calculate_distance_stats_per_class(
    output_dir: str,
    subset_name_for_class: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute top-1 distance summary stats per class from iteration parquets.

    Reads {subset}_iteration_*.parquet (and *_fallback_* for rare classes). Each
    class maps to its own subset name, so its iteration parquets already contain
    only that class's targets — no per-class bbox re-filtering is needed. Also
    emits an "overall" entry aggregating across all iteration parquets.

    Returns {class_name: {count, min, max, mean, median, std, p25, p75}}.
    """
    if not subset_name_for_class:
        return {}

    output_path = Path(output_dir)

    def _arr_stats(arr: np.ndarray) -> dict:
        return {
            "count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std()),
            "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)),
        }

    stats: Dict[str, Dict[str, float]] = {}

    for class_name in sorted(subset_name_for_class):
        subset = subset_name_for_class.get(class_name)
        if not subset:
            continue
        is_non_rare = class_name.lower() == "non_rare"

        # Class-pool iterations first; fallback files fill gaps for rare-class retrievals.
        # "non_rare" has no fallback path.
        iter_files = sorted(output_path.glob(f"{subset}_iteration_*.parquet"))
        if not is_non_rare:
            iter_files += sorted(output_path.glob(f"{subset}_fallback_iteration_*.parquet"))
        if not iter_files:
            continue

        distances: list = []
        seen_targets: set = set()
        for itf in iter_files:
            df = pd.read_parquet(itf)
            if "top1_distance" not in df.columns or "target_filepath" not in df.columns:
                continue
            for _, row in df.iterrows():
                tgt = row.get("target_filepath")
                if pd.isna(tgt) or tgt in seen_targets:
                    continue
                d = row.get("top1_distance")
                if pd.notna(d):
                    distances.append(float(d))
                    seen_targets.add(tgt)

        if distances:
            stats[class_name] = _arr_stats(np.asarray(distances))

    overall_distances: list = []
    overall_seen: set = set()
    for itf in sorted(output_path.glob("*_iteration_*.parquet")):
        df = pd.read_parquet(itf)
        if "top1_distance" not in df.columns or "target_filepath" not in df.columns:
            continue
        for _, row in df.iterrows():
            tgt = row.get("target_filepath")
            if pd.isna(tgt) or tgt in overall_seen:
                continue
            d = row.get("top1_distance")
            if pd.notna(d):
                overall_distances.append(float(d))
                overall_seen.add(tgt)

    if overall_distances:
        stats["overall"] = _arr_stats(np.asarray(overall_distances))

    return stats
