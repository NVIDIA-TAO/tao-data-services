# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RCCA gap analysis vlm_bcq script."""

import json
import os
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import extract_yes_no, extract_fp_fn
from nvidia_tao_ds.config.rcca.gap_analysis.vlm_bcq import GapAnalysisConfig


class TestExtractYesNo:
    """Tests for the extract_yes_no helper."""

    @pytest.mark.parametrize("text, expected", [
        ("yes", "yes"),
        ("Yes", "yes"),
        ("YES", "yes"),
        ("yes, there is a collision", "yes"),
        ("no", "no"),
        ("No", "no"),
        ("NO", "no"),
        ("no collision detected", "no"),
    ])
    def test_clear_responses(self, text, expected):
        assert extract_yes_no(text) == expected

    @pytest.mark.parametrize("text", [
        "yes and no",
        "The answer could be yes or no",
    ])
    def test_ambiguous_returns_none(self, text):
        assert extract_yes_no(text) is None

    @pytest.mark.parametrize("text", [
        "",
        "maybe",
        "uncertain",
        "I cannot determine",
    ])
    def test_no_match_returns_none(self, text):
        assert extract_yes_no(text) is None

    def test_word_boundary(self):
        """'nothing' should not match 'no'."""
        assert extract_yes_no("nothing happened") is None


class TestExtractFpFn:
    """Tests for the extract_fp_fn function."""

    def test_false_positive(self):
        data = [{"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "collision?"}]
        cases = extract_fp_fn(data)
        assert len(cases) == 1
        assert cases[0]["error_type"] == "FP"
        assert cases[0]["video_id"] == "v1.mp4"

    def test_false_negative(self):
        data = [{"video_id": "v2.mp4", "response": "no", "gt": "yes", "question": "collision?"}]
        cases = extract_fp_fn(data)
        assert len(cases) == 1
        assert cases[0]["error_type"] == "FN"

    def test_correct_predictions_excluded(self):
        data = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "yes", "question": "q"},
            {"video_id": "v2.mp4", "response": "no", "gt": "no", "question": "q"},
        ]
        cases = extract_fp_fn(data)
        assert len(cases) == 0

    def test_ambiguous_response_skipped(self):
        data = [{"video_id": "v1.mp4", "response": "yes and no", "gt": "yes", "question": "q"}]
        cases = extract_fp_fn(data)
        assert len(cases) == 0

    def test_ambiguous_gt_skipped(self):
        data = [{"video_id": "v1.mp4", "response": "yes", "gt": "yes or no", "question": "q"}]
        cases = extract_fp_fn(data)
        assert len(cases) == 0

    def test_mixed_batch(self):
        data = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "q"},
            {"video_id": "v2.mp4", "response": "no", "gt": "yes", "question": "q"},
            {"video_id": "v3.mp4", "response": "yes", "gt": "yes", "question": "q"},
            {"video_id": "v4.mp4", "response": "maybe", "gt": "yes", "question": "q"},
        ]
        cases = extract_fp_fn(data)
        assert len(cases) == 2
        error_types = {c["video_id"]: c["error_type"] for c in cases}
        assert error_types["v1.mp4"] == "FP"
        assert error_types["v2.mp4"] == "FN"

    def test_empty_input(self):
        assert extract_fp_fn([]) == []


class TestMain:
    """Integration tests for the main() entrypoint via hydra_runner.__wrapped__."""

    def _write_predictions(self, tmp_path, predictions):
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")
        return str(pred_path)

    def _make_cfg(self, predictions_json, results_dir, videos_dir=""):
        cfg = OmegaConf.structured(GapAnalysisConfig)
        cfg.predictions_json = predictions_json
        cfg.videos_dir = videos_dir
        cfg.results_dir = results_dir
        return cfg

    def test_outputs_created(self, tmp_path):
        predictions = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "collision?"},
            {"video_id": "v2.mp4", "response": "no", "gt": "yes", "question": "collision?"},
        ]
        pred_path = self._write_predictions(tmp_path, predictions)
        output_dir = str(tmp_path / "output")
        cfg = self._make_cfg(pred_path, output_dir)

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        main.__wrapped__(cfg)

        assert os.path.exists(os.path.join(output_dir, "kpi_gaps.jsonl"))
        assert os.path.exists(os.path.join(output_dir, "kpi_gaps_report.txt"))

    def test_jsonl_content(self, tmp_path):
        predictions = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "collision?"},
            {"video_id": "v2.mp4", "response": "maybe", "gt": "yes", "question": "collision?"},
            {"video_id": "v3.mp4", "response": "yes and no", "gt": "no", "question": "collision?"},
            {"video_id": "v4.mp4", "response": "no", "gt": "yes", "question": "collision?"},
            {"video_id": "v5.mp4", "response": "yes", "gt": "yes", "question": "collision?"},
        ]
        pred_path = self._write_predictions(tmp_path, predictions)
        output_dir = str(tmp_path / "output")
        cfg = self._make_cfg(pred_path, output_dir)

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        main.__wrapped__(cfg)

        jsonl_path = os.path.join(output_dir, "kpi_gaps.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        # v2 (no match) and v3 (ambiguous) skipped, v5 (correct) excluded
        assert len(lines) == 2
        assert lines[0]["error_type"] == "FP"
        assert lines[1]["error_type"] == "FN"

    def test_no_gaps_found(self, tmp_path):
        predictions = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "yes", "question": "q"},
        ]
        pred_path = self._write_predictions(tmp_path, predictions)
        output_dir = str(tmp_path / "output")
        cfg = self._make_cfg(pred_path, output_dir)

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        main.__wrapped__(cfg)

        # No output files when there are no gaps
        assert not os.path.exists(os.path.join(output_dir, "kpi_gaps.jsonl"))

    def test_videos_dir_resolves_paths(self, tmp_path):
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        predictions = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "collision?"},
        ]
        pred_path = self._write_predictions(tmp_path, predictions)
        output_dir = str(tmp_path / "output")
        cfg = self._make_cfg(pred_path, output_dir, videos_dir=str(videos_dir))

        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        main.__wrapped__(cfg)

        jsonl_path = os.path.join(output_dir, "kpi_gaps.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["video_id"] == str(videos_dir / "v1.mp4")


class TestConfig:
    """Tests for config validation and error handling."""

    def _make_cfg(self, predictions_json, results_dir, videos_dir=""):
        cfg = OmegaConf.structured(GapAnalysisConfig)
        cfg.predictions_json = predictions_json
        cfg.videos_dir = videos_dir
        cfg.results_dir = results_dir
        return cfg

    def test_missing_required_fields(self):
        """OmegaConf raises if MISSING fields are accessed without being set."""
        cfg = OmegaConf.structured(GapAnalysisConfig)
        with pytest.raises(Exception):
            _ = cfg.predictions_json
        with pytest.raises(Exception):
            _ = cfg.results_dir

    def test_predictions_json_not_found(self, tmp_path):
        cfg = self._make_cfg(
            predictions_json=str(tmp_path / "nonexistent.json"),
            results_dir=str(tmp_path / "output"),
        )
        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        with pytest.raises(Exception):
            main.__wrapped__(cfg)

    def test_predictions_json_not_a_list(self, tmp_path):
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps({"video_id": "v1.mp4"}), encoding="utf-8")
        cfg = self._make_cfg(
            predictions_json=str(pred_path),
            results_dir=str(tmp_path / "output"),
        )
        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        with pytest.raises(ValueError, match="must be a JSON array"):
            main.__wrapped__(cfg)

    @pytest.mark.parametrize("missing_key,item", [
        ("gt",        {"video_id": "v1.mp4", "response": "yes"}),
        ("response",  {"video_id": "v1.mp4", "gt": "yes"}),
        ("video_id",  {"response": "yes", "gt": "yes"}),
    ])
    def test_predictions_json_item_missing_required_key(self, tmp_path, missing_key, item):
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps([item]), encoding="utf-8")
        cfg = self._make_cfg(
            predictions_json=str(pred_path),
            results_dir=str(tmp_path / "output"),
        )
        from nvidia_tao_ds.rcca.gap_analysis.scripts.vlm_bcq import main
        with pytest.raises(ValueError, match=f"missing '{missing_key}'"):
            main.__wrapped__(cfg)


class TestHydraConfigLoading:
    """Tests that the script can be invoked via CLI with a YAML config file."""

    def test_load_yaml_config(self, tmp_path):
        predictions = [
            {"video_id": "v1.mp4", "response": "yes", "gt": "no", "question": "collision?"},
        ]
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")
        output_dir = tmp_path / "output"

        config_dir = tmp_path / "specs"
        config_dir.mkdir()
        (config_dir / "test.yaml").write_text(
            f"predictions_json: {pred_path}\n"
            f"results_dir: {output_dir}\n"
            "videos_dir: \"\"\n",
            encoding="utf-8",
        )

        script = os.path.join(
            os.path.dirname(__file__),
            "../nvidia_tao_ds/rcca/gap_analysis/scripts/vlm_bcq.py",
        )
        result = subprocess.run(
            [sys.executable, script, f"--config-path={config_dir}", "--config-name=test"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))},
        )
        assert result.returncode == 0, result.stderr

        assert os.path.exists(os.path.join(str(output_dir), "kpi_gaps.jsonl"))
        assert os.path.exists(os.path.join(str(output_dir), "kpi_gaps_report.txt"))
