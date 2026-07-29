# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for RCCA object-detection gap-analysis."""

from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_ds.config.utils.types import DICT_FIELD, FLOAT_FIELD, INT_FIELD, STR_FIELD


@dataclass
class ObjectDetectionGapConfig:
    """Configuration for object-detection KPI gap analysis.

    Required fields:
        ground_truth_ann_path: KITTI label directory or COCO JSON containing
            ground-truth boxes.
        inference_ann_path: KITTI label directory or COCO JSON containing
            model predictions (with optional per-box confidence scores).
        images_dir: Root directory of images. Used to establish the complete
            image universe, including images that have no GT or prediction
            annotations.
        results_dir: Output directory for gap-analysis artifacts.
        kpi: Identifier tag attached to every output row.
        input_format: Annotation format — ``kitti`` or ``coco``. Must be
            declared explicitly; format is never inferred from the path.

    Optional fields:
        iou_threshold: IoU at or above which a prediction is accepted as a TP.
        conf_threshold: Predictions below this confidence are dropped before
            matching. Has no effect when the source does not carry scores.
        min_area: Boxes whose pixel area (w × h) is strictly below this value
            are discarded before matching.
        class_mapping: Maps raw annotation label strings to canonical class
            names. Labels absent from the mapping are kept as-is.
        weak_thresholds: Per-class metric thresholds as a nested dict:
            ``{class_name: {recall: float, precision: float, ap50: float}}``.
            Any metric key absent from a class entry falls back to the
            corresponding ``default_*_threshold``.
        default_recall_threshold: Fallback recall threshold for classes not
            listed in ``weak_thresholds``.
        default_precision_threshold: Fallback precision threshold. Set to 0.0
            to disable precision-based weak selection entirely.
        default_ap50_threshold: Fallback AP50 threshold. Set to 0.0 to disable
            AP50-based weak selection entirely.
    """

    ground_truth_ann_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to KITTI label directory or COCO JSON>",
        description="Ground-truth annotation source: a KITTI label directory or a COCO .json file."
    )
    inference_ann_path: str = STR_FIELD(
        value=MISSING,
        default_value="<path to KITTI label directory or COCO JSON>",
        description="Inference annotation source: a KITTI label directory or a COCO .json file."
    )
    images_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<path to image root directory>",
        description="Root directory of images. Establishes the full image universe including unannotated images."
    )
    results_dir: str = STR_FIELD(
        value=MISSING,
        default_value="<path to output directory>",
        description="Output directory for box_gaps.parquet, image_metrics.parquet, weak_images.parquet, and gap_report.json."
    )
    kpi: str = STR_FIELD(
        value=MISSING,
        default_value="<kpi_name>",
        description="KPI identifier tag written to every output row."
    )
    input_format: str = STR_FIELD(
        value=MISSING,
        default_value="kitti",
        valid_options=["kitti", "coco"],
        description="Annotation format: 'kitti' or 'coco'. Must be declared explicitly."
    )
    iou_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="IoU threshold for accepting a prediction as a true positive."
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Minimum prediction confidence. Predictions below this value are dropped before matching."
    )
    min_area: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        description="Minimum box area in pixels (w × h). Boxes strictly below this are discarded."
    )
    class_mapping: dict = DICT_FIELD(
        hashMap={},
        default_value={},
        description="Maps raw annotation label strings to canonical class names. Absent labels are kept as-is."
    )
    weak_thresholds: dict = DICT_FIELD(
        hashMap={},
        default_value={},
        description=(
            "Per-class metric thresholds as a nested dict: "
            "{class_name: {recall: float, precision: float, ap50: float}}. "
            "Any omitted metric key falls back to the corresponding default_*_threshold."
        )
    )
    default_recall_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Fallback recall threshold for classes not listed in weak_thresholds."
    )
    default_precision_threshold: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Fallback precision threshold. Set to 0.0 to disable precision-based weak selection."
    )
    default_ap50_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Fallback AP50 threshold. Set to 0.0 to disable AP50-based weak selection."
    )
