# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for two kpi.py bugs fixed in TAO-2319.

Bug 1 — FN sentinel (line 176):
    Unmatched GT boxes were appended to P[] with 0.0. At conf_threshold=0.0
    the classifier branch p >= conf_threshold evaluates as 0.0 >= 0.0 = True,
    counting every unmatched GT as TP instead of FN. Fixed by using -1.0.

Bug 2 — voc_ap argument order (line 339):
    voc_ap(rec, prec, ...) was called as voc_ap(prec, rec, ...), computing
    the area under the inverse curve. Fixed by swapping the arguments.
"""

from functools import partial

import numpy as np
import pytest

from nvidia_tao_ds.data_analytics.utils.kpi import (
    _per_img_match,
    evaluate_class,
    voc_ap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_img_match_fn(matching_iou_threshold=0.5, min_area=0):
    return partial(
        _per_img_match,
        n_classes=1,
        sorting_algorithm="quicksort",
        matching_iou_threshold=matching_iou_threshold,
        min_area=min_area,
    )


def _gt(box):
    """Single GT box: [class=0, x1, y1, x2, y2]."""
    return np.array([[0, *box]], dtype=float)


def _pred(box, conf):
    """Single pred box: [class=0, conf, x1, y1, x2, y2]."""
    return np.array([[0, conf, *box]], dtype=float)


# ---------------------------------------------------------------------------
# Bug 1 — FN sentinel: unmatched GT must land in FN at any conf_threshold
# ---------------------------------------------------------------------------

class TestFNSentinel:
    """_per_img_match must use -1.0 (not 0.0) for unmatched GT confidence.

    At conf_threshold=0.0, a 0.0 sentinel passes p >= 0.0 and is counted
    as TP.  -1.0 is below every valid confidence so FN is always chosen.
    """

    def test_unmatched_gt_sentinel_is_negative(self):
        # 1 GT, 0 predictions — the unmatched GT entry must be < 0
        gt = _gt([0, 0, 10, 10])
        pred = np.empty((0, 6), dtype=float)
        T, P, _ = _per_img_match_fn()((gt, pred))
        assert len(P[0]) == 1
        assert P[0][0] < 0, "unmatched GT sentinel must be negative to avoid TP collision"

    def test_unmatched_gt_is_fn_at_conf_threshold_zero(self):
        # The default analytics threshold used to be 0.5 but users can set 0.0.
        # At 0.0 a 0.0 sentinel would be TP — ensure FN instead.
        gt = _gt([0, 0, 10, 10])
        pred = np.empty((0, 6), dtype=float)
        fn = _per_img_match_fn()
        TP, FP, FN, *_ = evaluate_class(fn, [gt], [pred], conf_threshold=0.0)
        assert TP == 0
        assert FN == 1

    def test_unmatched_gt_is_fn_at_conf_threshold_half(self):
        # conf_threshold=0.5 is the evaluate() default — must also give FN=1
        gt = _gt([0, 0, 10, 10])
        pred = np.empty((0, 6), dtype=float)
        fn = _per_img_match_fn()
        TP, FP, FN, *_ = evaluate_class(fn, [gt], [pred], conf_threshold=0.5)
        assert TP == 0
        assert FN == 1

    def test_matched_gt_is_tp_not_affected(self):
        # Confirm the fix doesn't touch matched GT boxes
        gt = _gt([0, 0, 10, 10])
        pred = _pred([0, 0, 10, 10], conf=0.9)
        fn = _per_img_match_fn()
        TP, FP, FN, *_ = evaluate_class(fn, [gt], [pred], conf_threshold=0.5)
        assert TP == 1
        assert FN == 0


# ---------------------------------------------------------------------------
# Bug 2 — voc_ap argument order: AP must be precision area, not recall area
# ---------------------------------------------------------------------------

class TestVocApArgumentOrder:
    """voc_ap(rec, prec, ...) was called as voc_ap(prec, rec, ...).

    When the args are swapped the function computes max-recall at each
    precision threshold — the inverse of average precision.  The two values
    are numerically different whenever the PR curve is not symmetric.
    """

    def test_perfect_detector_ap_is_one(self):
        # 1 GT, 1 matching prediction: rec=[1.0], prec=[1.0] → AP=1.0
        rec = np.array([1.0])
        prec = np.array([1.0])
        assert voc_ap(rec, prec, "sample", 11) == pytest.approx(1.0)

    def test_partial_recall_ap_matches_hand_computation(self):
        # rec=[0.5], prec=[1.0]: 6 of 11 recall thresholds (0.0–0.5) have
        # max precision = 1.0; the rest have p=0. Hand-computed AP = 6/11.
        rec = np.array([0.5])
        prec = np.array([1.0])
        expected = 6 / 11
        assert voc_ap(rec, prec, "sample", 11) == pytest.approx(expected)

    def test_swapped_args_gives_different_result(self):
        # Demonstrates the bug: voc_ap(prec, rec) computes a different value.
        # rec=[0.5], prec=[1.0]: correct=6/11≈0.5455, swapped=0.5.
        rec = np.array([0.5])
        prec = np.array([1.0])
        correct = voc_ap(rec, prec, "sample", 11)
        swapped = voc_ap(prec, rec, "sample", 11)
        assert correct != pytest.approx(swapped), "swapped and correct AP must differ"
        assert correct == pytest.approx(6 / 11)
        assert swapped == pytest.approx(0.5)

    def test_zero_predictions_ap_is_zero(self):
        rec = np.array([])
        prec = np.array([])
        assert voc_ap(rec, prec, "sample", 11) == pytest.approx(0.0)
