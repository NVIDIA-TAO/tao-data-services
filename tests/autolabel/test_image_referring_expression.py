# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image_referring_expression (referring-data-engine) annotation pipeline."""

import os
from unittest.mock import MagicMock

import pytest

from nvidia_tao_core.config.auto_label.default_config import (
    ImageREConfig,
    ImageREDataConfig,
    ImageREWorkflowConfig,
    LLMBackendConfig,
)
from nvidia_tao_ds.auto_label.common.annotation import (
    load_records,
    save_records,
)
from nvidia_tao_ds.auto_label.image_referring_expression.io_utils import (
    clean_response,
    format_bboxes,
    format_grounding_text,
    list_images,
    parse_grounding_response,
    parse_kitti_label,
    parse_regions_response,
    scale_bbox,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import (
    PROMPT_TEMPLATES,
    get_prompt,
)
from nvidia_tao_ds.auto_label.image_referring_expression.steps import (
    step0_region_expr,
    step1_image_caption,
    step2_grounding_expr,
    step3_double_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_icfg(steps=None, max_workers=1, force=False, output_format="jsonl"):
    """Create a mock ``image_referring_expression`` sub-config."""
    icfg = MagicMock()
    icfg.workflow.steps = steps or ["0", "1", "2", "3"]
    icfg.workflow.max_workers = max_workers
    icfg.workflow.force_reprocess = force
    icfg.workflow.output_format = output_format
    icfg.workflow.long_video_threshold_sec = 60
    icfg.workflow.long_video_sample_fps = 0.5
    icfg.workflow.long_video_max_frames = 60
    icfg.data.image_dir = ""
    icfg.data.kitti_label_dir = ""
    icfg.data.input_annotations_jsonl = ""
    return icfg


def _write_kitti(path, rows):
    """Write a KITTI-format label file.

    Each row is (obj_type, x1, y1, x2, y2). All other columns are filled
    with zeros / placeholders to match the KITTI text format.
    """
    lines = []
    for obj_type, x1, y1, x2, y2 in rows:
        # type, truncated, occluded, alpha, x1, y1, x2, y2, 3xdim, 3xloc, ry
        lines.append(
            f"{obj_type} 0.00 0 0.00 "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ===========================================================================
# Config dataclass tests
# ===========================================================================

class TestImageREConfig:
    def test_workflow_defaults(self):
        cfg = ImageREWorkflowConfig()
        assert cfg.steps == ["0", "1", "2", "3"], f"Unexpected steps: {cfg.steps}"
        assert cfg.max_workers == 4, f"Expected 4, got {cfg.max_workers}"
        assert cfg.output_format == "jsonl", f"Expected 'jsonl', got {cfg.output_format}"

    def test_data_defaults(self):
        cfg = ImageREDataConfig()
        assert cfg.image_dir == "", f"Expected empty image_dir"
        assert cfg.kitti_label_dir == "", f"Expected empty kitti_label_dir"
        assert cfg.input_annotations_jsonl == "", f"Expected empty input_annotations_jsonl"

    def test_image_re_config_defaults(self):
        cfg = ImageREConfig()
        assert isinstance(cfg.vlm, LLMBackendConfig), "vlm should be LLMBackendConfig"
        assert isinstance(cfg.workflow, ImageREWorkflowConfig), "workflow type check"
        assert isinstance(cfg.data, ImageREDataConfig), "data type check"


# ===========================================================================
# io_utils tests
# ===========================================================================

class TestListImages:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert list_images(str(tmp_path)) == [], "Empty dir → []"

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_images(str(tmp_path / "missing")) == [], "Missing dir → []"

    def test_lists_jpg_and_png(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.PNG").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("x")
        result = list_images(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert "a.jpg" in names, "Should include .jpg"
        assert "b.PNG" in names, "Should include .PNG (case-insensitive)"
        assert "c.txt" not in names, "Should exclude non-image files"

    def test_empty_path(self):
        assert list_images("") == [], "Empty path → []"


class TestParseKittiLabel:
    def test_parses_rows(self, tmp_path):
        path = tmp_path / "001.txt"
        _write_kitti(str(path), [
            ("Car", 100, 200, 300, 400),
            ("Pedestrian", 50, 60, 70, 90),
        ])
        rows = parse_kitti_label(str(path))
        assert len(rows) == 2, f"Expected 2 rows"
        assert rows[0] == [100, 200, 300, 400, "Car"], f"Row 0 mismatch: {rows[0]}"
        assert rows[1] == [50, 60, 70, 90, "Pedestrian"], f"Row 1 mismatch: {rows[1]}"

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_kitti_label(str(tmp_path / "missing.txt")) == [], "Missing file → []"

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "001.txt"
        path.write_text("Car 0.0 0\n")
        rows = parse_kitti_label(str(path))
        assert rows == [], "Too-short line should be skipped"


class TestFormatBboxes:
    def test_empty(self):
        assert format_bboxes([], 1000, 1000) == "[]", "Empty → '[]'"

    def test_normalizes_to_0_1000(self):
        # width=1000, height=1000 → coordinates map 1:1 to the 0-1000 space
        result = format_bboxes([[100, 200, 300, 400, "Car"]], 1000, 1000)
        assert "Car" in result, "Type should be present"
        assert "[100, 200, 300, 400]" in result, f"Expected raw coords in normalized space: {result}"

    def test_scaling(self):
        # width=2000 → x coords should be halved
        result = format_bboxes([[500, 200, 1000, 400, "Car"]], 1000, 2000)
        assert "[250, 200, 500, 400]" in result, f"Unexpected scaled output: {result}"


class TestScaleBbox:
    def test_round_trip(self):
        # 0-1000 normalized → pixels
        assert scale_bbox([250, 250, 750, 750], 1000, 1000) == (250, 250, 750, 750), \
            "Round-trip at scale=1000"

    def test_clamps_negative(self):
        assert scale_bbox([-100, -100, 2000, 2000], 1000, 1000) == (0, 0, 1000, 1000), \
            "OOB should clamp"


class TestCleanResponse:
    def test_strips_json_fence(self):
        raw = "```json\n{\"a\": 1}\n```"
        cleaned = clean_response(raw).strip()
        assert cleaned == '{"a": 1}', f"Expected stripped content: {cleaned!r}"

    def test_strips_generic_fence(self):
        raw = "```\n[1, 2, 3]\n```"
        cleaned = clean_response(raw).strip()
        assert cleaned == "[1, 2, 3]", f"Expected stripped content: {cleaned!r}"

    def test_passthrough_when_no_fence(self):
        raw = '{"a": 1}'
        assert clean_response(raw) == raw, "Unfenced content should pass through"


class TestParseRegionsResponse:
    def test_json_array(self):
        text = (
            '[{"bbox_2d": [1, 2, 3, 4], "type": "Car", "color": "red", "description": "d1"},'
            ' {"bbox_2d": [5, 6, 7, 8], "type": "Van", "color": "blue", "description": "d2"}]'
        )
        regions = parse_regions_response(text)
        assert len(regions) == 2, f"Expected 2 regions, got {len(regions)}"
        assert regions[0]["bbox"] == [1, 2, 3, 4], "bbox normalized from bbox_2d"
        assert regions[1]["description"] == "d2", "description preserved"

    def test_ndjson(self):
        text = (
            '{"bbox_2d": [1, 2, 3, 4], "type": "A", "color": "red", "description": "d1"}\n'
            '{"bbox_2d": [5, 6, 7, 8], "type": "B", "color": "blue", "description": "d2"}\n'
        )
        regions = parse_regions_response(text)
        assert len(regions) == 2, f"Expected 2 NDJSON regions, got {len(regions)}"

    def test_truncated_recovery(self):
        text = (
            '[{"bbox_2d": [1, 2, 3, 4], "type": "A", "color": "red", "description": "d1"},'
            '{"bbox_2d": [5, 6, 7, 8], "type": "B", "color": "blue", "description": "d2"'
        )
        regions = parse_regions_response(text)
        assert len(regions) >= 1, "Should recover at least one complete region"

    def test_fenced(self):
        text = (
            "```json\n"
            '[{"bbox_2d": [1, 2, 3, 4], "type": "A", "color": "red", "description": "d1"}]\n'
            "```"
        )
        regions = parse_regions_response(text)
        assert len(regions) == 1, "Should parse fenced response"

    def test_empty(self):
        assert parse_regions_response("") == [], "Empty response → []"


class TestParseGroundingResponse:
    def test_parses_phrase_bbox_lines(self):
        text = (
            "The red car: [[1, 2, 3, 4], [5, 6, 7, 8]]\n"
            "The blue van: [[10, 20, 30, 40]]\n"
        )
        expressions = parse_grounding_response(text)
        assert len(expressions) == 2, f"Expected 2 expressions"
        assert expressions[0]["text"] == "The red car", "Text preserved"
        assert len(expressions[0]["instances"]) == 2, "Two bboxes for red car"
        assert expressions[0]["instances"][0]["bbox"] == [1, 2, 3, 4], "Bbox values"

    def test_skips_malformed_lines(self):
        text = (
            "good: [[1, 2, 3, 4]]\n"
            "no colon here\n"
            "bad: not-a-list\n"
        )
        expressions = parse_grounding_response(text)
        assert len(expressions) == 1, f"Only the good line should parse"
        assert expressions[0]["text"] == "good", "Good line preserved"

    def test_empty_instances_dropped(self):
        text = "nothing: []\n"
        expressions = parse_grounding_response(text)
        assert expressions == [], "Empty instances should be dropped"


class TestFormatGroundingText:
    def test_roundtrip_with_parse(self):
        exprs = [
            {"text": "red car",
             "instances": [{"bbox": [1, 2, 3, 4]}, {"bbox": [5, 6, 7, 8]}]},
            {"text": "blue van", "instances": [{"bbox": [10, 20, 30, 40]}]},
        ]
        formatted = format_grounding_text(exprs)
        parsed = parse_grounding_response(formatted)
        assert len(parsed) == 2, "Round-trip should preserve expression count"
        assert parsed[0]["instances"][0]["bbox"] == [1, 2, 3, 4], "Bbox values preserved"

    def test_empty_instances_skipped(self):
        exprs = [
            {"text": "kept", "instances": [{"bbox": [1, 2, 3, 4]}]},
            {"text": "dropped", "instances": []},
        ]
        text = format_grounding_text(exprs)
        assert "kept" in text, "Kept expression"
        assert "dropped" not in text, "Dropped expression"


# ===========================================================================
# Prompt templates
# ===========================================================================

class TestPromptTemplates:
    def test_all_keys_present(self):
        for key in ("region_expr", "image_caption", "grounding_expr", "double_check"):
            assert key in PROMPT_TEMPLATES, f"Missing prompt: {key}"

    def test_region_expr_renders(self):
        prompt = get_prompt("region_expr", bboxes="[Car: [100,200,300,400]]")
        assert "Car" in prompt, "Bboxes substituted"
        assert "bbox_2d" in prompt, "Output schema described"

    def test_grounding_expr_renders_with_caption(self):
        prompt = get_prompt(
            "grounding_expr",
            bboxes=["car:[1,2,3,4]"],
            caption_section="caption: A scene.",
        )
        assert "car:[1,2,3,4]" in prompt, "Bboxes substituted"
        assert "caption: A scene." in prompt, "Caption substituted"

    def test_grounding_expr_renders_without_caption(self):
        prompt = get_prompt("grounding_expr", bboxes=["car:[1,2,3,4]"], caption_section="")
        assert "car:[1,2,3,4]" in prompt, "Bboxes substituted"

    def test_double_check_renders(self):
        prompt = get_prompt("double_check", expr="car: [[1,2,3,4]]")
        assert "car: [[1,2,3,4]]" in prompt, "Expressions substituted"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="No prompt template"):
            get_prompt("nonexistent_key")


# ===========================================================================
# Step 0: region_expr
# ===========================================================================

class TestStep0RegionExpr:
    def _mock_seed(self, width=1000, height=1000, kitti_bboxes=None):
        return [{
            "image_id": "000001",
            "image_path": "/fake/000001.jpg",
            "width": width,
            "height": height,
            "kitti_bboxes": kitti_bboxes if kitti_bboxes is not None else [[100, 200, 300, 400, "Car"]],
            "source": "image_referring_expression",
            "pipeline_steps": [],
        }]

    def test_run_with_mock_vlm(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["0"])
        seed = self._mock_seed()

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = (
            '[{"bbox_2d": [100, 200, 300, 400], "type": "Car", "color": "white", '
            '"description": "white sedan in center lane"}]'
        )

        output = step0_region_expr.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed)
        records = load_records(output)
        assert len(records) == 1, f"Expected 1 record"
        assert len(records[0]["regions"]) == 1, "One region parsed"
        assert records[0]["regions"][0]["bbox"] == [100, 200, 300, 400], "bbox set"
        assert records[0]["regions"][0]["description"] == "white sedan in center lane", \
            "description set"
        assert "step0_region_expr" in records[0]["pipeline_steps"], "Step name recorded"

    def test_run_writes_legacy_when_configured(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["0"], output_format="both")
        seed = self._mock_seed()

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = (
            '[{"bbox_2d": [1, 2, 3, 4], "type": "Car", "color": "red", "description": "red car"}]'
        )

        step0_region_expr.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed)

        legacy_path = os.path.join(
            results_dir, "step_0_region_expr", "labels", "000001.txt.step0",
        )
        assert os.path.exists(legacy_path), f"Expected legacy file at {legacy_path}"

    def test_run_empty_kitti_emits_empty_regions(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["0"])
        seed = self._mock_seed(kitti_bboxes=[])

        mock_vlm = MagicMock()
        output = step0_region_expr.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed)
        records = load_records(output)
        assert records[0]["regions"] == [], "No kitti → no regions"
        mock_vlm.generate_with_image.assert_not_called()

    def test_run_skips_if_output_exists(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["0"], force=False)

        step_dir = os.path.join(results_dir, "step_0_region_expr")
        os.makedirs(step_dir, exist_ok=True)
        output_file = os.path.join(step_dir, "annotations.jsonl")
        save_records([{"image_id": "pre", "regions": []}], output_file)

        mock_vlm = MagicMock()
        step0_region_expr.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, self._mock_seed(),
        )
        mock_vlm.generate_with_image.assert_not_called()


# ===========================================================================
# Step 1: image_caption
# ===========================================================================

class TestStep1ImageCaption:
    def _mock_seed(self):
        return [{
            "image_id": "000001",
            "image_path": "/fake/000001.jpg",
            "pipeline_steps": [],
        }]

    def test_run_writes_caption(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["1"])
        seed = self._mock_seed()

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = "A busy city intersection at dusk."

        output = step1_image_caption.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed,
        )
        records = load_records(output)
        assert records[0]["caption"] == "A busy city intersection at dusk.", "Caption set"
        assert "step1_image_caption" in records[0]["pipeline_steps"], "Step recorded"

    def test_run_legacy_output(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["1"], output_format="legacy")
        seed = self._mock_seed()

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = "  Caption text.  "

        step1_image_caption.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed,
        )
        legacy_path = os.path.join(
            results_dir, "step_1_image_caption", "labels", "000001.txt.step1",
        )
        assert os.path.exists(legacy_path), "Legacy file should exist"
        assert open(legacy_path).read().strip() == "Caption text.", "Caption stripped"

    def test_handles_vlm_failure_gracefully(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["1"])
        seed = self._mock_seed()

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.side_effect = RuntimeError("API error")

        output = step1_image_caption.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, seed,
        )
        records = load_records(output)
        assert records[0]["caption"] == "", "Failed call → empty caption"


# ===========================================================================
# Step 2: grounding_expr
# ===========================================================================

class TestStep2GroundingExpr:
    def _write_step0(self, results_dir, records):
        d = os.path.join(results_dir, "step_0_region_expr")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "annotations.jsonl")
        save_records(records, path)
        return path

    def _write_step1(self, results_dir, records):
        d = os.path.join(results_dir, "step_1_image_caption")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "annotations.jsonl")
        save_records(records, path)
        return path

    def test_run_with_regions_and_caption(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["2"])

        step0 = self._write_step0(results_dir, [{
            "image_id": "000001",
            "image_path": "/fake/000001.jpg",
            "regions": [
                {"bbox": [100, 200, 300, 400], "type": "Car", "color": "white",
                 "description": "white sedan"},
                {"bbox": [500, 200, 700, 400], "type": "Car", "color": "red",
                 "description": "red sedan"},
            ],
            "pipeline_steps": ["step0_region_expr"],
        }])
        step1 = self._write_step1(results_dir, [{
            "image_id": "000001",
            "caption": "A busy street.",
        }])

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = (
            "white sedan in the left lane: [[100, 200, 300, 400]]\n"
            "red sedan in the right lane: [[500, 200, 700, 400]]\n"
        )

        output = step2_grounding_expr.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir,
            step0_file=step0, step1_file=step1,
        )
        records = load_records(output)
        assert len(records) == 1, f"Expected 1 record"
        rec = records[0]
        assert rec["caption"] == "A busy street.", "Caption merged from step 1"
        assert len(rec["expressions"]) == 2, f"Expected 2 expressions"
        assert rec["expressions"][0]["text"] == "white sedan in the left lane", \
            "First expression parsed"
        assert rec["expressions"][0]["instances"][0]["bbox"] == [100, 200, 300, 400], \
            "Bbox propagated"
        assert "expression_id" in rec["expressions"][0], "Expression ID assigned"
        assert "bbox_id" in rec["expressions"][0]["instances"][0], "Bbox ID assigned"
        assert "step2_grounding_expr" in rec["pipeline_steps"], "Step recorded"

    def test_run_no_regions_gives_empty_expressions(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["2"])

        step0 = self._write_step0(results_dir, [{
            "image_id": "1", "image_path": "/fake.jpg", "regions": [],
        }])

        mock_vlm = MagicMock()
        output = step2_grounding_expr.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir,
            step0_file=step0, step1_file=None,
        )
        records = load_records(output)
        assert records[0]["expressions"] == [], "No regions → no expressions"
        mock_vlm.generate_with_image.assert_not_called()

    def test_run_without_step1(self, tmp_path):
        """Step 2 should run even if step 1 (caption) wasn't produced."""
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["2"])

        step0 = self._write_step0(results_dir, [{
            "image_id": "1",
            "image_path": "/fake.jpg",
            "regions": [{"bbox": [1, 2, 3, 4], "type": "Car",
                         "color": "red", "description": "red car"}],
        }])

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = "red car: [[1, 2, 3, 4]]\n"

        output = step2_grounding_expr.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir,
            step0_file=step0, step1_file=None,
        )
        records = load_records(output)
        assert len(records[0]["expressions"]) == 1, "Should work without caption"


# ===========================================================================
# Step 3: double_check
# ===========================================================================

class TestStep3DoubleCheck:
    def _write_step2(self, results_dir, records):
        d = os.path.join(results_dir, "step_2_grounding_expr")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "annotations.jsonl")
        save_records(records, path)
        return path

    def test_run_drops_rejected_and_marks_verified(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["3"])

        step2 = self._write_step2(results_dir, [{
            "image_id": "1",
            "image_path": "/fake.jpg",
            "expressions": [
                {"expression_id": "expr_0", "text": "kept",
                 "instances": [{"bbox_id": "box_0", "bbox": [1, 2, 3, 4], "bbox_score": 0.9}]},
                {"expression_id": "expr_1", "text": "rejected",
                 "instances": [{"bbox_id": "box_1", "bbox": [5, 6, 7, 8], "bbox_score": 0.9}]},
            ],
            "pipeline_steps": ["step2_grounding_expr"],
        }])

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = "kept: [[1, 2, 3, 4]]\n"

        output = step3_double_check.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, step2_file=step2,
        )
        records = load_records(output)
        rec = records[0]
        assert len(rec["expressions"]) == 1, f"Rejected expr dropped"
        assert rec["expressions"][0]["text"] == "kept", "Kept expression preserved"
        assert rec["expressions"][0]["verified"] is True, "Should be flagged verified"
        assert rec["expressions"][0]["instances"][0]["bbox_id"] == "box_0", \
            "Original ID preserved when bbox unchanged"
        assert "step3_double_check" in rec["pipeline_steps"], "Step recorded"

    def test_run_preserves_updated_bboxes(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["3"])

        step2 = self._write_step2(results_dir, [{
            "image_id": "1",
            "image_path": "/fake.jpg",
            "expressions": [
                {"expression_id": "expr_0", "text": "the car",
                 "instances": [{"bbox_id": "box_0", "bbox": [10, 20, 30, 40], "bbox_score": 0.9}]},
            ],
        }])

        mock_vlm = MagicMock()
        mock_vlm.generate_with_image.return_value = "the car: [[12, 22, 32, 42]]\n"

        output = step3_double_check.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, step2_file=step2,
        )
        records = load_records(output)
        inst = records[0]["expressions"][0]["instances"][0]
        assert inst["bbox"] == [12, 22, 32, 42], "Updated bbox applied"
        assert inst["bbox_id"] == "box_0", "ID preserved across coord update"

    def test_empty_expressions_passthrough(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(steps=["3"])

        step2 = self._write_step2(results_dir, [{
            "image_id": "1", "image_path": "/fake.jpg", "expressions": [],
        }])

        mock_vlm = MagicMock()
        output = step3_double_check.run(
            icfg, mock_vlm, PROMPT_TEMPLATES, results_dir, step2_file=step2,
        )
        records = load_records(output)
        assert records[0]["expressions"] == [], "Empty passthrough"
        mock_vlm.generate_with_image.assert_not_called()


# ===========================================================================
# Orchestrator seeding
# ===========================================================================

class TestSeedAnnotations:
    def test_seed_from_input_jsonl_when_provided(self, tmp_path):
        """When ``input_annotations_jsonl`` is set, those records are used as-is."""
        from nvidia_tao_ds.auto_label.image_referring_expression.inference import _seed_annotations

        input_path = tmp_path / "input.jsonl"
        save_records(
            [{"image_id": "x", "image_path": "/x.jpg", "pipeline_steps": []}],
            str(input_path),
        )

        data_cfg = MagicMock()
        data_cfg.input_annotations_jsonl = str(input_path)
        data_cfg.image_dir = ""
        data_cfg.kitti_label_dir = ""

        records = _seed_annotations(data_cfg, str(tmp_path))
        assert len(records) == 1, "Should use provided input"
        assert records[0]["image_id"] == "x", "Record content preserved"
