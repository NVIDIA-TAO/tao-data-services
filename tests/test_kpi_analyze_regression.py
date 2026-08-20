# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: kpi_calc.csv must include class_name.

kpi_analyze.py built final_kpi_df with class_name but the to_csv column
list omitted it, leaving the CSV without any identifiers for each metric row.
Callers (e.g. deft_verify.py) had to parse stdout PrettyTable order to identify
rows — fragile and broken in practice.

The test runs analyze() end to end against a small KITTI fixture and reads the
kpi_calc.csv it writes, so it fails if the column list regresses. Asserting
against a column list restated in the test would pass either way.

Nothing in the KPI path opens an image -- image_dir is used only to build path
strings and to derive the sequence name -- so the fixture is label files, a
class mapping, and two directory levels of image_dir that never need to exist.
"""

import os

import pandas as pd
import pytest
from omegaconf import OmegaConf

from nvidia_tao_ds.data_analytics.scripts.kpi_analyze import analyze

# type truncated occluded alpha xmin ymin xmax ymax dim(3) loc(3) rot [score]
_GT_KITTI = (
    "car 0.0 0 0.0 10.0 10.0 210.0 210.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    "pedestrian 0.0 0 0.0 300.0 300.0 500.0 500.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
)
# Predictions overlap the GT boxes closely enough to match at IoU 0.5.
_PRED_KITTI = (
    "car 0.0 0 0.0 12.0 12.0 208.0 208.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.9\n"
    "pedestrian 0.0 0 0.0 305.0 305.0 498.0 498.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.8\n"
)

_MAPPING = """\
- car:
  - car
- pedestrian:
  - pedestrian
"""


@pytest.fixture
def kpi_run(tmp_path):
    """A single-sequence KITTI KPI run, ready to hand to analyze()."""
    sequence = tmp_path / "seq_a"
    gt_dir = sequence / "labels"
    pred_dir = sequence / "inference"
    for directory in (gt_dir, pred_dir):
        directory.mkdir(parents=True)
    (gt_dir / "frame_000.txt").write_text(_GT_KITTI, encoding="utf-8")
    (pred_dir / "frame_000.txt").write_text(_PRED_KITTI, encoding="utf-8")

    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(_MAPPING, encoding="utf-8")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Sequence Name is derived as image_dir.split('/')[-2], so the parent of
    # the image dir names the sequence. The images themselves are never read.
    cfg = OmegaConf.create({
        "results_dir": str(results_dir),
        "data": {
            "input_format": "KITTI",
            "mapping": str(mapping),
            "kpi_sources": [{
                "image_dir": str(sequence / "images"),
                "ground_truth_ann_path": str(gt_dir),
                "inference_ann_path": str(pred_dir),
            }],
        },
        "kpi": {
            "iou_threshold": 0.5,
            "conf_threshold": 0.0,
            "ignore_sqwidth": 40,
            "num_recall_points": 11,
            "is_internal": False,
        },
        "visualize": {"platform": "local", "tag": "regression"},
    })
    return cfg, results_dir


def _read_kpi_csv(results_dir):
    output_csv_path = os.path.join(results_dir, "kpi_calc.csv")
    assert os.path.exists(output_csv_path), "analyze() did not write kpi_calc.csv"
    return pd.read_csv(output_csv_path)


def test_kpi_calc_csv_has_a_class_name_column(kpi_run):
    """class_name must reach the CSV, not just the stdout PrettyTable."""
    cfg, results_dir = kpi_run
    analyze(cfg)

    saved = _read_kpi_csv(results_dir)
    assert "class_name" in saved.columns, (
        "class_name missing from kpi_calc.csv -- every per-class row is left "
        f"with no class identifier. Columns written: {list(saved.columns)}"
    )


def test_kpi_calc_csv_identifies_each_per_class_row(kpi_run):
    """Each row must name the class it scores, so callers need no row order."""
    cfg, results_dir = kpi_run
    analyze(cfg)

    saved = _read_kpi_csv(results_dir)
    assert sorted(saved["class_name"]) == ["car", "pedestrian"]
    # One row per class, and AP is attributable without consulting stdout.
    assert len(saved) == 2
    assert saved.set_index("class_name").loc["car", "AP"] >= 0.0
