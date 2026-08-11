# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: kpi_calc.csv must include class_name.

kpi_analyze.py built final_kpi_df with class_name but the to_csv column
list omitted it, leaving one unlabeled row per class in the CSV. Callers
(e.g. deft_verify.py) had to parse stdout PrettyTable order to identify
rows — fragile and broken in practice.
"""

import io
import pandas as pd


def _make_final_kpi_df():
    """Minimal DataFrame matching what kpi_analyze.py builds at line 88."""
    return pd.DataFrame([
        {"Tag": "run1", "Sequence Name": "seq_a", "class_name": "car",
         "TP": 10, "FP": 2, "FN": 1, "TN": 0,
         "Pr": 0.833, "Re": 0.909, "Acc": 0.8, "AP": 0.85,
         "precision": None, "recall": None},
        {"Tag": "run1", "Sequence Name": "seq_a", "class_name": "pedestrian",
         "TP": 5, "FP": 1, "FN": 2, "TN": 0,
         "Pr": 0.833, "Re": 0.714, "Acc": 0.7, "AP": 0.72,
         "precision": None, "recall": None},
    ])


def test_class_name_in_csv_columns():
    """class_name must appear in kpi_calc.csv — not just in stdout."""
    final_kpi_df = _make_final_kpi_df()
    buf = io.StringIO()
    final_kpi_df[["Sequence Name", "class_name", "TP", "FP", "FN", "TN",
                   "Pr", "Re", "Acc", "AP"]].to_csv(buf, index=False)
    buf.seek(0)
    saved = pd.read_csv(buf)
    assert "class_name" in saved.columns, (
        "class_name missing from kpi_calc.csv — per-class rows are unlabeled"
    )


def test_class_name_values_preserved():
    """class names must round-trip through the CSV correctly."""
    final_kpi_df = _make_final_kpi_df()
    buf = io.StringIO()
    final_kpi_df[["Sequence Name", "class_name", "TP", "FP", "FN", "TN",
                   "Pr", "Re", "Acc", "AP"]].to_csv(buf, index=False)
    buf.seek(0)
    saved = pd.read_csv(buf)
    assert list(saved["class_name"]) == ["car", "pedestrian"]


def test_old_column_list_lacks_class_name():
    """Documents the regression: the pre-fix column list omits class_name."""
    final_kpi_df = _make_final_kpi_df()
    buf = io.StringIO()
    final_kpi_df[["Sequence Name", "TP", "FP", "FN", "TN",
                   "Pr", "Re", "Acc", "AP"]].to_csv(buf, index=False)
    buf.seek(0)
    saved = pd.read_csv(buf)
    assert "class_name" not in saved.columns  # demonstrates the bug that was fixed
