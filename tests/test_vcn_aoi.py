# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RCCA gap analysis vcn_aoi script."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from nvidia_tao_ds.rcca.gap_analysis.scripts.vcn_aoi import (
    analyze_vcn_inference_gaps,
    compute_vcn_optimal_threshold,
)
from nvidia_tao_ds.config.rcca.gap_analysis.vcn_aoi import GapAnalysisConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_COLS = ["label", "siamese_score", "input_path", "object_name"]


def _write_inference_csv(tmp_path, rows, subdir="inference"):
    """Write rows to a CSV inside ``tmp_path/subdir`` and return the dir."""
    results_dir = tmp_path / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLS).to_csv(results_dir / "inference.csv", index=False)
    return str(results_dir)


def _write_train_config(tmp_path, lightings=("IR", "RGB"), ext=".png"):
    """Write a minimal train config YAML and return its path."""
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump({
            "dataset": {
                "classify": {
                    "input_map": list(lightings),
                    "image_ext": ext,
                }
            }
        }),
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# compute_vcn_optimal_threshold
# ---------------------------------------------------------------------------

class TestComputeOptimalThreshold:
    """Unit tests for compute_vcn_optimal_threshold."""

    def test_perfect_separation(self, tmp_path):
        """Clean split: threshold should fall between the two clusters."""
        rows = [
            {"label": "PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "PASS", "siamese_score": 0.2, "input_path": "a", "object_name": "o2"},
            {"label": "NO_PASS", "siamese_score": 0.8, "input_path": "a", "object_name": "o3"},
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "o4"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        output_path = str(tmp_path / "out" / "threshold.txt")
        threshold = compute_vcn_optimal_threshold(
            results_dir=results_dir, output_path=output_path, min_recall=1.0,
        )
        # Threshold must be at or above max PASS score and below min NO_PASS.
        # Decision rule is strict (score > threshold), so threshold == 0.2
        # still predicts both NO_PASS correctly and both PASS correctly.
        assert 0.2 <= threshold < 0.8
        assert Path(output_path).is_file()
        assert float(Path(output_path).read_text()) == threshold

    def test_min_recall_filters_thresholds(self, tmp_path):
        """With min_recall=1.0, the optimum must catch every NO_PASS sample."""
        rows = [
            {"label": "PASS",    "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "PASS",    "siamese_score": 0.4, "input_path": "a", "object_name": "o2"},
            {"label": "NO_PASS", "siamese_score": 0.5, "input_path": "a", "object_name": "o3"},
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "o4"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        output_path = str(tmp_path / "out" / "threshold.txt")
        threshold = compute_vcn_optimal_threshold(
            results_dir=results_dir, output_path=output_path, min_recall=1.0,
        )
        # To keep 100% NO_PASS recall the threshold must be < 0.5.
        assert threshold < 0.5

    def test_relaxed_min_recall_allows_higher_threshold(self, tmp_path):
        """A low-score outlier NO_PASS forces strict min_recall to sweep very
        low; relaxing min_recall lets the optimum move above it."""
        rows = [
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "outlier"},
            {"label": "PASS",    "siamese_score": 0.3, "input_path": "a", "object_name": "p1"},
            {"label": "PASS",    "siamese_score": 0.3, "input_path": "a", "object_name": "p2"},
            {"label": "PASS",    "siamese_score": 0.3, "input_path": "a", "object_name": "p3"},
            {"label": "NO_PASS", "siamese_score": 0.8, "input_path": "a", "object_name": "n1"},
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "n2"},
            {"label": "NO_PASS", "siamese_score": 1.0, "input_path": "a", "object_name": "n3"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        strict = compute_vcn_optimal_threshold(
            results_dir=results_dir,
            output_path=str(tmp_path / "out" / "strict.txt"),
            min_recall=1.0,
        )
        relaxed = compute_vcn_optimal_threshold(
            results_dir=results_dir,
            output_path=str(tmp_path / "out" / "relaxed.txt"),
            min_recall=0.5,
        )
        # Strict must catch the outlier (score 0.1) → threshold below it.
        assert strict < 0.1
        # Relaxed can skip the outlier, yielding a better F1 above 0.1.
        assert relaxed > strict
        assert relaxed >= 0.3

    def test_no_csv_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Expected inference CSV"):
            compute_vcn_optimal_threshold(
                results_dir=str(empty_dir),
                output_path=str(tmp_path / "threshold.txt"),
                min_recall=1.0,
            )

    def test_no_feasible_threshold_raises(self, tmp_path):
        """If the dataset has no NO_PASS samples, min_recall cannot be met."""
        rows = [
            {"label": "PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "PASS", "siamese_score": 0.2, "input_path": "a", "object_name": "o2"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        with pytest.raises(ValueError, match="No threshold achieves"):
            compute_vcn_optimal_threshold(
                results_dir=results_dir,
                output_path=str(tmp_path / "out" / "threshold.txt"),
                min_recall=1.0,
            )

    def test_empty_inference_csv_raises(self, tmp_path):
        """A header-only inference CSV must surface a clear error,
        not an IndexError from the threshold sweep."""
        results_dir = _write_inference_csv(tmp_path, rows=[])
        with pytest.raises(ValueError, match="contains no rows"):
            compute_vcn_optimal_threshold(
                results_dir=results_dir,
                output_path=str(tmp_path / "out" / "threshold.txt"),
                min_recall=1.0,
            )

    def test_only_top_level_inference_csv_is_used(self, tmp_path):
        """The lookup is non-recursive: a nested CSV must not be found."""
        nested = tmp_path / "inference" / "deeply" / "nested"
        nested.mkdir(parents=True)
        rows = [
            {"label": "PASS", "siamese_score": 0.2, "input_path": "a", "object_name": "o1"},
            {"label": "NO_PASS", "siamese_score": 0.8, "input_path": "a", "object_name": "o2"},
        ]
        pd.DataFrame(rows, columns=CSV_COLS).to_csv(nested / "inference.csv", index=False)
        output_path = str(tmp_path / "out" / "threshold.txt")
        with pytest.raises(FileNotFoundError, match="Expected inference CSV"):
            compute_vcn_optimal_threshold(
                results_dir=str(tmp_path / "inference"),
                output_path=output_path,
                min_recall=1.0,
            )


# ---------------------------------------------------------------------------
# analyze_vcn_inference_gaps
# ---------------------------------------------------------------------------

class TestAnalyzeInferenceGaps:
    """Unit tests for analyze_vcn_inference_gaps."""

    def test_top_k_selects_weakest_per_label(self, tmp_path):
        """With top_k_per_label=1, only the single weakest sample per
        label is kept — the misclassified one, not the correct one."""
        rows = [
            # FN: weakness = 0.5 - 0.1 = 0.4 (positive — misclassified)
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "sess/1", "object_name": "obj1"},
            # FP: weakness = 0.9 - 0.5 = 0.4 (positive — misclassified)
            {"label": "PASS",    "siamese_score": 0.9, "input_path": "sess/2", "object_name": "obj2"},
            # TP: weakness = 0.5 - 0.8 = -0.3 (negative — correct)
            {"label": "NO_PASS", "siamese_score": 0.8, "input_path": "sess/3", "object_name": "obj3"},
            # TN: weakness = 0.2 - 0.5 = -0.3 (negative — correct)
            {"label": "PASS",    "siamese_score": 0.2, "input_path": "sess/4", "object_name": "obj4"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR", "RGB"), ext=".png")
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        media = str(tmp_path / "media")

        returned = analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=media,
            train_config=train_config,
            threshold=0.5,
            top_k_per_label=1,
        )
        assert returned == gaps_parquet
        assert Path(gaps_parquet).is_file()

        df = pd.read_parquet(gaps_parquet)
        # 1 weakest per label × 2 labels × 2 lightings = 4 rows
        assert len(df) == 4
        media_root = Path(media)
        expected_paths = {
            str(media_root / "sess/1" / "obj1_IR.png"),
            str(media_root / "sess/1" / "obj1_RGB.png"),
            str(media_root / "sess/2" / "obj2_IR.png"),
            str(media_root / "sess/2" / "obj2_RGB.png"),
        }
        assert set(df["filepath"]) == expected_paths
        assert set(df.columns) == {"filepath", "label", "siamese_score", "weakness"}

    def test_large_top_k_keeps_all_including_correct(self, tmp_path):
        """With a large budget, correctly-classified samples are included
        with negative weakness scores."""
        rows = [
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "o1"},
            {"label": "PASS",    "siamese_score": 0.1, "input_path": "a", "object_name": "o2"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR",))
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
            top_k_per_label=50,
        )
        df = pd.read_parquet(gaps_parquet)
        # Both samples kept (1 per label × 1 lighting each = 2 rows)
        assert len(df) == 2
        assert set(df.columns) == {"filepath", "label", "siamese_score", "weakness"}
        # Both are correctly classified → negative weakness
        assert (df["weakness"] < 0).all()

    def test_top_k_keeps_the_k_weakest_per_label(self, tmp_path):
        """With more candidates than the budget, top_k keeps the K rows
        with the highest weakness per label and drops the rest."""
        rows = [
            # PASS rows — weakness = score - threshold (threshold=0.5):
            {"label": "PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "p_high"},   # +0.4
            {"label": "PASS", "siamese_score": 0.7, "input_path": "a", "object_name": "p_mid"},    # +0.2
            {"label": "PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "p_low"},    # -0.4 (drop)
            # NO_PASS rows — weakness = threshold - score:
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "n_high"}, # +0.4
            {"label": "NO_PASS", "siamese_score": 0.3, "input_path": "a", "object_name": "n_mid"},  # +0.2
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "a", "object_name": "n_low"},  # -0.4 (drop)
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR",))
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
            top_k_per_label=2,
        )
        df = pd.read_parquet(gaps_parquet)
        # 2 weakest per label × 2 labels × 1 lighting = 4 rows.
        assert len(df) == 4
        # The two "low" rows (most-correct, most-negative weakness) must be
        # dropped; the four kept rows are the per-label top-2 by weakness.
        kept_objects = {Path(p).stem.removesuffix("_IR") for p in df["filepath"]}
        assert kept_objects == {"p_high", "p_mid", "n_high", "n_mid"}
        # All kept rows have weakness >= the dropped rows' weakness (-0.4).
        assert (df["weakness"] >= 0.0).all()

    def test_weakness_values(self, tmp_path):
        """Weakness is computed correctly for PASS and NO_PASS rows."""
        rows = [
            # PASS: weakness = score - threshold = 0.9 - 0.5 = 0.4
            {"label": "PASS",    "siamese_score": 0.9, "input_path": "a", "object_name": "o1"},
            # NO_PASS: weakness = threshold - score = 0.5 - 0.1 = 0.4
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "o2"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR",))
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
        )
        df = pd.read_parquet(gaps_parquet)
        weakness_by_label = dict(zip(df["label"], df["weakness"]))
        assert weakness_by_label["PASS"] == pytest.approx(0.4)
        assert weakness_by_label["NO_PASS"] == pytest.approx(0.4)

    def test_threshold_boundary_is_strict(self, tmp_path):
        """Decision rule is ``score > threshold``; equality should predict PASS.
        A NO_PASS sample at the boundary has weakness = 0 (borderline)."""
        rows = [
            # score == threshold => predicted PASS. Actual NO_PASS.
            # weakness = threshold - score = 0.
            {"label": "NO_PASS", "siamese_score": 0.5, "input_path": "a", "object_name": "o1"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR",), ext=".png")
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
        )
        df = pd.read_parquet(gaps_parquet)
        assert len(df) == 1
        assert df["weakness"].iloc[0] == pytest.approx(0.0)

    def test_breakdown_report_written(self, tmp_path):
        """A weak_samples_breakdown.txt is written next to the parquet."""
        rows = [
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "PASS",    "siamese_score": 0.9, "input_path": "a", "object_name": "o2"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path)
        out_dir = tmp_path / "out"
        gaps_parquet = str(out_dir / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
        )
        breakdown = out_dir / "weak_samples_breakdown.txt"
        assert breakdown.is_file()
        text = breakdown.read_text(encoding="utf-8")
        assert "Total KPI samples: 2" in text
        assert "Total weak samples kept: 2" in text
        assert "threshold=0.5" in text
        assert "top_k_per_label=50" in text
        assert "misclassified" in text

    def test_label_whitespace_stripped(self, tmp_path):
        """Surrounding whitespace on the CSV label is stripped on output."""
        rows = [
            {"label": " NO_PASS ", "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR",))
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=str(tmp_path / "media"),
            train_config=train_config,
            threshold=0.5,
        )
        df = pd.read_parquet(gaps_parquet)
        assert df["label"].tolist() == ["NO_PASS"]

    def test_no_csv_raises_file_not_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Expected inference CSV"):
            analyze_vcn_inference_gaps(
                results_dir=str(empty_dir),
                gaps_parquet=str(tmp_path / "out" / "gaps.parquet"),
                kpi_media_path=str(tmp_path / "media"),
                train_config=_write_train_config(tmp_path),
                threshold=0.5,
            )

    def test_custom_lightings_and_extension(self, tmp_path):
        """The filepath respects arbitrary input_map entries and image_ext."""
        rows = [
            {"label": "PASS", "siamese_score": 0.9, "input_path": "sess/5", "object_name": "widget"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(
            tmp_path, lightings=("bright", "dim", "backlit"), ext=".jpg",
        )
        gaps_parquet = str(tmp_path / "out" / "gaps.parquet")
        media = str(tmp_path / "media")
        analyze_vcn_inference_gaps(
            results_dir=results_dir,
            gaps_parquet=gaps_parquet,
            kpi_media_path=media,
            train_config=train_config,
            threshold=0.5,
        )
        df = pd.read_parquet(gaps_parquet)
        assert len(df) == 3
        media_root = Path(media)
        assert set(df["filepath"]) == {
            str(media_root / "sess/5" / "widget_bright.jpg"),
            str(media_root / "sess/5" / "widget_dim.jpg"),
            str(media_root / "sess/5" / "widget_backlit.jpg"),
        }


# ---------------------------------------------------------------------------
# End-to-end via main() (hydra_runner.__wrapped__)
# ---------------------------------------------------------------------------

class TestMain:
    """Integration tests for the main() entrypoint via hydra_runner.__wrapped__."""

    def _make_cfg(
        self, inference_results_dir, train_config, results_dir,
        kpi_media_path="", threshold=-1.0, min_recall=1.0,
        top_k_per_label=50,
    ):
        cfg = OmegaConf.structured(GapAnalysisConfig)
        cfg.inference_results_dir = inference_results_dir
        cfg.train_config = train_config
        cfg.kpi_media_path = kpi_media_path
        cfg.threshold = threshold
        cfg.min_recall = min_recall
        cfg.top_k_per_label = top_k_per_label
        cfg.results_dir = results_dir
        return cfg

    def _basic_rows(self):
        return [
            {"label": "PASS",    "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "PASS",    "siamese_score": 0.2, "input_path": "b", "object_name": "o2"},
            {"label": "NO_PASS", "siamese_score": 0.8, "input_path": "c", "object_name": "o3"},
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "d", "object_name": "o4"},
            {"label": "NO_PASS", "siamese_score": 0.1, "input_path": "e", "object_name": "o5"},  # FN
            {"label": "PASS",    "siamese_score": 0.9, "input_path": "f", "object_name": "o6"},  # FP
        ]

    def test_outputs_created_with_auto_threshold(self, tmp_path):
        results_dir = _write_inference_csv(tmp_path, self._basic_rows())
        train_config = _write_train_config(tmp_path, lightings=("IR", "RGB"))
        output_dir = tmp_path / "output"
        cfg = self._make_cfg(
            inference_results_dir=results_dir,
            train_config=train_config,
            results_dir=str(output_dir),
            kpi_media_path=str(tmp_path / "media"),
            threshold=-1.0,
        )

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vcn_aoi import main
        main.__wrapped__(cfg)

        assert (output_dir / "threshold.txt").is_file()
        assert (output_dir / "kpi_gaps.parquet").is_file()
        assert (output_dir / "weak_samples_breakdown.txt").is_file()

    def test_parquet_contents_match_expected_gaps(self, tmp_path):
        """With top_k_per_label=1, only the FN (o5) and FP (o6) survive
        because they have the highest weakness in their label groups."""
        rows = self._basic_rows()
        results_dir = _write_inference_csv(tmp_path, rows)
        train_config = _write_train_config(tmp_path, lightings=("IR", "RGB"))
        output_dir = tmp_path / "output"
        media = tmp_path / "media"
        cfg = self._make_cfg(
            inference_results_dir=results_dir,
            train_config=train_config,
            results_dir=str(output_dir),
            kpi_media_path=str(media),
            threshold=0.5,
            top_k_per_label=1,
        )

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vcn_aoi import main
        main.__wrapped__(cfg)

        # threshold.txt is only written when threshold is auto-computed.
        assert not (output_dir / "threshold.txt").exists()

        df = pd.read_parquet(output_dir / "kpi_gaps.parquet")
        # 1 weakest per label × 2 labels × 2 lightings = 4 rows.
        assert len(df) == 4
        assert set(df["filepath"]) == {
            str(media / "e" / "o5_IR.png"),
            str(media / "e" / "o5_RGB.png"),
            str(media / "f" / "o6_IR.png"),
            str(media / "f" / "o6_RGB.png"),
        }
        # Both kept samples are misclassified (positive weakness).
        assert (df["weakness"] > 0).all()


# ---------------------------------------------------------------------------
# Config validation and error surface
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for config validation and main()'s error handling."""

    def _make_cfg(
        self, inference_results_dir, train_config, results_dir,
        kpi_media_path="", threshold=-1.0, min_recall=1.0,
        top_k_per_label=50,
    ):
        cfg = OmegaConf.structured(GapAnalysisConfig)
        cfg.inference_results_dir = inference_results_dir
        cfg.train_config = train_config
        cfg.kpi_media_path = kpi_media_path
        cfg.threshold = threshold
        cfg.min_recall = min_recall
        cfg.top_k_per_label = top_k_per_label
        cfg.results_dir = results_dir
        return cfg

    def test_inference_dir_missing_raises(self, tmp_path):
        train_config = _write_train_config(tmp_path)
        cfg = self._make_cfg(
            inference_results_dir=str(tmp_path / "nonexistent"),
            train_config=train_config,
            results_dir=str(tmp_path / "output"),
            threshold=0.5,
        )
        from nvidia_tao_ds.rcca.gap_analysis.scripts.vcn_aoi import main
        with pytest.raises(FileNotFoundError, match="Expected inference CSV"):
            main.__wrapped__(cfg)

    def test_train_config_missing_raises(self, tmp_path):
        """A nonexistent train_config path bubbles up from yaml.safe_load."""
        rows = [
            {"label": "PASS",    "siamese_score": 0.1, "input_path": "a", "object_name": "o1"},
            {"label": "NO_PASS", "siamese_score": 0.9, "input_path": "b", "object_name": "o2"},
        ]
        results_dir = _write_inference_csv(tmp_path, rows)
        cfg = self._make_cfg(
            inference_results_dir=results_dir,
            train_config=str(tmp_path / "nope.yaml"),
            results_dir=str(tmp_path / "output"),
            threshold=0.5,
        )
        from nvidia_tao_ds.rcca.gap_analysis.scripts.vcn_aoi import main
        with pytest.raises(FileNotFoundError):
            main.__wrapped__(cfg)


