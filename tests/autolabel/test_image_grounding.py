# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for image_grounding (grounding-data-engine) annotation pipeline."""

import json
import os
from unittest.mock import MagicMock

import pytest

from nvidia_tao_core.config.auto_label.default_config import (
    ImageGDConfig,
    ImageGDDataConfig,
    ImageGDWorkflowConfig,
    LLMBackendConfig,
)
from nvidia_tao_ds.auto_label.common.annotation import (
    IdCounter,
    clamp_bbox,
    image_id_from_path,
    is_valid_output,
    load_records,
    make_expression,
    make_instance,
    merge_records,
    parse_legacy_grounding_file,
    parse_legacy_region_file,
    save_records,
    write_legacy_caption_file,
    write_legacy_grounding_file,
    write_legacy_region_file,
)
from nvidia_tao_ds.auto_label.image_grounding.prompts import (
    PROMPT_TEMPLATES,
    get_prompt,
)
from nvidia_tao_ds.auto_label.image_grounding.steps import (
    step0_expression_extraction,
    step1_grounding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_icfg(steps=None, max_workers=1, force=False):
    """Create a mock ``image_grounding`` sub-config."""
    icfg = MagicMock()
    icfg.workflow.steps = steps or ["0", "1"]
    icfg.workflow.max_workers = max_workers
    icfg.workflow.force_reprocess = force
    icfg.workflow.long_video_threshold_sec = 60
    icfg.workflow.long_video_sample_fps = 0.5
    icfg.workflow.long_video_max_frames = 60
    icfg.data.input_jsonl = ""
    icfg.data.image_root = ""
    return icfg


# ===========================================================================
# Config dataclass tests
# ===========================================================================

class TestImageGDConfig:
    def test_workflow_defaults(self):
        cfg = ImageGDWorkflowConfig()
        assert cfg.steps == ["0", "1"], f"Expected ['0', '1'], got {cfg.steps}"
        assert cfg.max_workers == 4, f"Expected 4, got {cfg.max_workers}"
        assert cfg.force_reprocess is False, f"Expected False, got {cfg.force_reprocess}"

    def test_data_defaults(self):
        cfg = ImageGDDataConfig()
        assert cfg.input_jsonl == "", f"Expected '', got {cfg.input_jsonl}"
        assert cfg.image_root == "", f"Expected '', got {cfg.image_root}"

    def test_image_gd_config_defaults(self):
        cfg = ImageGDConfig()
        assert isinstance(cfg.vlm, LLMBackendConfig), f"Expected LLMBackendConfig, got {type(cfg.vlm).__name__}"
        assert isinstance(cfg.workflow, ImageGDWorkflowConfig), f"Expected ImageGDWorkflowConfig, got {type(cfg.workflow).__name__}"
        assert isinstance(cfg.data, ImageGDDataConfig), f"Expected ImageGDDataConfig, got {type(cfg.data).__name__}"

    def test_llm_backend_alias(self):
        """LLMBackendConfig should alias VideoCotLLMConfig (same fields)."""
        cfg = LLMBackendConfig()
        assert cfg.backend == "gemini", f"Expected 'gemini', got {cfg.backend}"


# ===========================================================================
# Unified annotation schema tests
# ===========================================================================

class TestIdCounter:
    def test_monotonic(self):
        c = IdCounter("expr")
        assert c.next() == "expr_00000", f"Unexpected id: {c.next}"
        assert c.next() == "expr_00001", f"Unexpected id"
        assert c.next() == "expr_00002", f"Unexpected id"

    def test_custom_width(self):
        c = IdCounter("box", width=3)
        assert c.next() == "box_000", f"Unexpected id"
        assert c.next() == "box_001", f"Unexpected id"


class TestAnnotationBuilders:
    def test_make_instance(self):
        inst = make_instance([1.2, 2.8, 3, 4], score=0.75, bbox_id="box_0")
        assert inst["bbox"] == [1, 2, 3, 4], f"Unexpected bbox: {inst['bbox']}"
        assert inst["bbox_score"] == 0.75, f"Unexpected score: {inst['bbox_score']}"
        assert inst["bbox_id"] == "box_0", f"Unexpected bbox_id: {inst['bbox_id']}"

    def test_make_instance_default_score(self):
        inst = make_instance([1, 2, 3, 4])
        assert inst["bbox_score"] == 0.9, f"Unexpected default score: {inst['bbox_score']}"
        assert "bbox_id" not in inst, f"Expected no bbox_id when not provided: {inst}"

    def test_make_expression_minimal(self):
        expr = make_expression("the car", instances=[])
        assert expr["text"] == "the car", f"Unexpected text: {expr['text']}"
        assert expr["instances"] == [], f"Unexpected instances: {expr['instances']}"

    def test_make_expression_full(self):
        expr = make_expression(
            "the red car",
            instances=[{"bbox": [1, 2, 3, 4]}],
            expression_id="expr_0",
            char_span=[0, 11],
            noun_chunk="car",
            verified=True,
        )
        assert expr["expression_id"] == "expr_0", f"Unexpected id"
        assert expr["char_span"] == [0, 11], f"Unexpected span"
        assert expr["noun_chunk"] == "car", f"Unexpected noun_chunk"
        assert expr["verified"] is True, f"Unexpected verified flag"

    def test_image_id_from_path(self):
        assert image_id_from_path("/a/b/000042.jpg") == "000042", f"Unexpected id"
        assert image_id_from_path("img.PNG") == "img", f"Unexpected id"
        assert image_id_from_path("") == "", f"Empty path should give empty id"

    def test_clamp_bbox_within_bounds(self):
        assert clamp_bbox([10, 20, 30, 40], 100, 100) == [10, 20, 30, 40], "Should preserve in-bounds bbox"

    def test_clamp_bbox_out_of_bounds(self):
        # x1,y1 < 0 and x2,y2 > bounds -> clamp to bounds
        assert clamp_bbox([-5, -10, 150, 200], 100, 100) == [0, 0, 100, 100], "Should clamp OOB bbox"

    def test_merge_records(self):
        a = [{"image_id": "1", "caption": "cap1"}, {"image_id": "2", "caption": "cap2"}]
        b = [{"image_id": "1", "regions": ["r1"]}, {"image_id": "3", "regions": ["r3"]}]
        merged = merge_records(a, b)
        by_id = {r["image_id"]: r for r in merged}
        assert "caption" in by_id["1"] and "regions" in by_id["1"], "Image 1 should have both fields"
        assert by_id["2"].get("caption") == "cap2", "Image 2 preserved"
        assert by_id["3"].get("regions") == ["r3"], "Image 3 added"


class TestJsonlIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "a.jsonl")
        records = [{"image_id": "1", "caption": "hi"}, {"image_id": "2", "caption": "bye"}]
        save_records(records, path)
        loaded = load_records(path)
        assert loaded == records, f"Round-trip mismatch: {loaded}"

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_records(str(tmp_path / "missing.jsonl")) == [], "Missing file should yield []"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "a.jsonl")
        save_records([{"a": 1}], path)
        assert os.path.exists(path), "Parent dirs should be created"


class TestIsValidOutput:
    def test_missing_file(self, tmp_path):
        assert is_valid_output(str(tmp_path / "missing.txt")) is False, "Missing file is invalid"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        assert is_valid_output(str(p)) is False, "Empty file is invalid"

    def test_failure_marker(self, tmp_path):
        p = tmp_path / "fail.txt"
        p.write_text("Failed to obtain answer via API.")
        assert is_valid_output(str(p)) is False, "Failure marker is invalid"

    def test_valid_content(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("real content")
        assert is_valid_output(str(p)) is True, "Valid content should be accepted"


class TestLegacyFormat:
    def test_region_file_roundtrip(self, tmp_path):
        path = str(tmp_path / "img.txt.step0")
        regions = [
            {"bbox": [10, 20, 30, 40], "type": "sedan", "color": "white",
             "description": "white sedan"},
            {"bbox": [50, 60, 70, 80], "type": "suv", "color": "red",
             "description": "red SUV"},
        ]
        write_legacy_region_file(regions, path)
        assert os.path.exists(path), "Legacy region file should exist"
        parsed = parse_legacy_region_file(path)
        assert len(parsed) == 2, f"Expected 2 regions, got {len(parsed)}"
        assert parsed[0]["bbox"] == [10, 20, 30, 40], f"Unexpected bbox: {parsed[0]['bbox']}"
        assert parsed[0]["description"] == "white sedan", f"Unexpected description"

    def test_region_file_ndjson_parsing(self, tmp_path):
        path = str(tmp_path / "img.txt.step0")
        path_obj = tmp_path / "img.txt.step0"
        path_obj.write_text(
            '{"bbox_2d": [1, 2, 3, 4], "type": "sedan", "color": "red", "description": "d1"}\n'
            '{"bbox_2d": [5, 6, 7, 8], "type": "van", "color": "blue", "description": "d2"}\n'
        )
        parsed = parse_legacy_region_file(path)
        assert len(parsed) == 2, f"Expected 2 NDJSON entries, got {len(parsed)}"
        assert parsed[1]["bbox"] == [5, 6, 7, 8], f"Unexpected bbox after normalize"

    def test_caption_file(self, tmp_path):
        path = str(tmp_path / "img.txt.step1")
        write_legacy_caption_file("A test caption.", path)
        assert open(path).read() == "A test caption.", "Caption content mismatch"

    def test_grounding_file_roundtrip(self, tmp_path):
        path = str(tmp_path / "img.txt.step2")
        expressions = [
            {"text": "red car",
             "instances": [{"bbox": [1, 2, 3, 4]}, {"bbox": [5, 6, 7, 8]}]},
            {"text": "blue van",
             "instances": [{"bbox": [10, 20, 30, 40]}]},
        ]
        write_legacy_grounding_file(expressions, path)
        parsed = parse_legacy_grounding_file(path)
        assert len(parsed) == 2, f"Expected 2 expressions, got {len(parsed)}"
        assert parsed[0]["text"] == "red car", f"Unexpected text: {parsed[0]['text']}"
        assert len(parsed[0]["instances"]) == 2, f"Expected 2 instances"
        assert parsed[0]["instances"][0]["bbox"] == [1, 2, 3, 4], "Bbox mismatch"

    def test_grounding_file_omits_empty(self, tmp_path):
        path = str(tmp_path / "img.txt.step3")
        expressions = [
            {"text": "kept", "instances": [{"bbox": [1, 2, 3, 4]}]},
            {"text": "dropped", "instances": []},
        ]
        write_legacy_grounding_file(expressions, path)
        parsed = parse_legacy_grounding_file(path)
        assert len(parsed) == 1, f"Empty expressions should be omitted: {parsed}"
        assert parsed[0]["text"] == "kept", "Kept expression preserved"


# ===========================================================================
# Prompt template tests
# ===========================================================================

class TestPromptTemplates:
    def test_all_keys_present(self):
        for key in ("expression_extraction", "phrase_grounding"):
            assert key in PROMPT_TEMPLATES, f"Missing prompt key: {key}"

    def test_expression_extraction_renders(self):
        prompt = get_prompt("expression_extraction", caption="A cat sits.")
        assert "A cat sits." in prompt, "Caption should be substituted"
        assert "cleaned_caption" in prompt, "Response schema should be described"

    def test_phrase_grounding_renders(self):
        prompt = get_prompt(
            "phrase_grounding",
            expressions_block='  - "the red car"\n  - "the blue van"',
        )
        assert "red car" in prompt, "Expression should be substituted"
        assert "bboxes" in prompt, "Output format should mention bboxes"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="No prompt template"):
            get_prompt("nonexistent_key")


# ===========================================================================
# Step 0: expression extraction
# ===========================================================================

class TestStep0ExpressionExtraction:
    def _write_input(self, tmp_path, records):
        path = str(tmp_path / "input.jsonl")
        save_records(records, path)
        return path

    def test_run_with_caption_only(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg()
        icfg.data.input_jsonl = self._write_input(tmp_path, [
            {"image_id": "000001",
             "image_path": "/fake/does_not_exist.jpg",
             "caption": "A red car drives down a busy street."},
        ])

        mock_vlm = MagicMock()
        mock_vlm.generate_text.return_value = json.dumps({
            "cleaned_caption": "A red car drives down a busy street.",
            "expressions": [
                {"text": "red car", "char_span": [2, 9], "noun_chunk": "car"},
                {"text": "busy street", "char_span": [23, 34], "noun_chunk": "street"},
            ],
        })

        output = step0_expression_extraction.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        assert os.path.exists(output), f"Output jsonl should exist: {output}"

        records = load_records(output)
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        rec = records[0]
        assert rec["cleaned_caption"] == "A red car drives down a busy street.", "Caption should be set"
        assert len(rec["expressions"]) == 2, f"Expected 2 expressions, got {len(rec['expressions'])}"
        assert rec["expressions"][0]["text"] == "red car", f"Unexpected text"
        assert rec["expressions"][0]["instances"] == [], "Instances empty after step 0"
        assert rec["source"] == "image_grounding", "Source tagged"
        assert "step0_expression_extraction" in rec["pipeline_steps"], "Step name recorded"

    def test_run_parses_markdown_fenced_json(self, tmp_path):
        """Step 0 should recover when the VLM wraps JSON in a markdown fence."""
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg()
        icfg.data.input_jsonl = self._write_input(tmp_path, [
            {"image_id": "1", "image_path": "/fake.jpg", "caption": "A scene."},
        ])

        mock_vlm = MagicMock()
        mock_vlm.generate_text.return_value = (
            "```json\n"
            '{"cleaned_caption": "A scene.", "expressions": '
            '[{"text": "scene", "char_span": [2, 7], "noun_chunk": "scene"}]}\n'
            "```"
        )

        output = step0_expression_extraction.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        records = load_records(output)
        assert len(records[0]["expressions"]) == 1, "Should parse fenced JSON"

    def test_run_skips_if_output_exists_without_force(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(force=False)
        icfg.data.input_jsonl = self._write_input(tmp_path, [
            {"image_id": "1", "image_path": "/fake.jpg", "caption": "A scene."},
        ])

        step_dir = os.path.join(results_dir, "step_0_expression_extraction")
        os.makedirs(step_dir, exist_ok=True)
        existing = os.path.join(step_dir, "annotations.jsonl")
        save_records([{"image_id": "1", "expressions": [{"text": "pre-existing"}]}], existing)

        mock_vlm = MagicMock()
        step0_expression_extraction.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        mock_vlm.generate_with_image.assert_not_called()
        mock_vlm.generate_text.assert_not_called()

        records = load_records(existing)
        assert records[0]["expressions"][0]["text"] == "pre-existing", "Existing content preserved"


# ===========================================================================
# Step 1: phrase grounding
# ===========================================================================

class TestStep1Grounding:
    def _seed_step0(self, results_dir, records):
        step_dir = os.path.join(results_dir, "step_0_expression_extraction")
        os.makedirs(step_dir, exist_ok=True)
        save_records(records, os.path.join(step_dir, "annotations.jsonl"))

    def test_run_fills_instances(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg()
        self._seed_step0(results_dir, [
            {
                "image_id": "1",
                "image_path": "/fake.jpg",
                "width": 1000,
                "height": 1000,
                "cleaned_caption": "A red car.",
                "expressions": [
                    {"expression_id": "expr_0", "text": "red car", "instances": []},
                    {"expression_id": "expr_1", "text": "street", "instances": []},
                ],
                "pipeline_steps": ["step0_expression_extraction"],
                "source": "image_grounding",
            },
        ])

        mock_vlm = MagicMock()
        mock_vlm.generate_text.return_value = json.dumps({
            "red car": {"bboxes": [[100, 200, 300, 400]], "scores": [0.9]},
            "street": {"bboxes": [], "scores": []},
        })

        output = step1_grounding.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        records = load_records(output)
        assert len(records) == 1, f"Expected 1 record"
        exprs = records[0]["expressions"]
        assert len(exprs[0]["instances"]) == 1, "Red car should have 1 instance"
        assert exprs[0]["instances"][0]["bbox"] == [100, 200, 300, 400], "Bbox propagated"
        assert exprs[0]["instances"][0]["bbox_score"] == 0.9, "Score propagated"
        assert "bbox_id" in exprs[0]["instances"][0], "Bbox ID should be assigned"
        assert exprs[1]["instances"] == [], "Empty response → empty instances"
        assert "step1_grounding" in records[0]["pipeline_steps"], "Step name recorded"

    def test_run_clamps_oob_bboxes(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg()
        self._seed_step0(results_dir, [
            {
                "image_id": "1",
                "image_path": "/fake.jpg",
                "width": 100,
                "height": 100,
                "expressions": [{"text": "thing", "instances": []}],
                "pipeline_steps": [],
            },
        ])

        mock_vlm = MagicMock()
        mock_vlm.generate_text.return_value = json.dumps({
            "thing": {"bboxes": [[-10, -20, 150, 200]], "scores": [0.5]},
        })

        output = step1_grounding.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        records = load_records(output)
        assert records[0]["expressions"][0]["instances"][0]["bbox"] == [0, 0, 100, 100], \
            f"Bbox should be clamped to image bounds"

    def test_run_with_empty_expressions_passthrough(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg()
        self._seed_step0(results_dir, [
            {"image_id": "1", "image_path": "/fake.jpg", "expressions": []},
        ])

        mock_vlm = MagicMock()
        output = step1_grounding.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        records = load_records(output)
        assert records[0]["expressions"] == [], "Empty expressions preserved"
        mock_vlm.generate_with_image.assert_not_called()

    def test_run_skips_if_output_exists(self, tmp_path):
        results_dir = str(tmp_path / "results")
        icfg = _make_icfg(force=False)
        self._seed_step0(results_dir, [
            {"image_id": "1", "image_path": "/fake.jpg", "expressions": []},
        ])

        step_dir = os.path.join(results_dir, "step_1_grounding")
        os.makedirs(step_dir, exist_ok=True)
        existing = os.path.join(step_dir, "annotations.jsonl")
        save_records([{"image_id": "1", "expressions": [{"text": "keep"}]}], existing)

        mock_vlm = MagicMock()
        step1_grounding.run(icfg, mock_vlm, PROMPT_TEMPLATES, results_dir)
        mock_vlm.generate_with_image.assert_not_called()
        mock_vlm.generate_text.assert_not_called()
