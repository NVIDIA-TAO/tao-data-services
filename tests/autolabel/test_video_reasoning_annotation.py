# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for video reasoning annotation pipeline."""

import json
import os
import re
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from nvidia_tao_core.config.auto_label.default_config import (
    VideoReasoningAnnotationDataConfig as DataConfig,
    VideoReasoningAnnotationGeminiConfig as GeminiConfig,
    VideoReasoningAnnotationLLMConfig as LLMConfig,
    VideoReasoningAnnotationOpenAIConfig as OpenAIConfig,
    VideoReasoningAnnotationConfig,
    VideoReasoningAnnotationWorkflowConfig as WorkflowConfig,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    build_question_str,
    format_chunk_captions,
    get_entries_from_jsonl,
    get_processed_video_prompt_keys,
    get_processed_videos,
    load_mode_map,
    parse_qa_output,
    permute_mcq,
    resolve_mode,
    save_result_to_jsonl,
)
from nvidia_tao_ds.core.llm_clients import (
    GeminiClient,
    OpenAICompatibleClient,
    create_client,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import (
    PROMPT_TEMPLATES,
    get_prompt,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.steps import (
    step1c_highlight,
    step2_description,
    step3_qa,
    step4_parse_qa,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vcfg(mode="normal", steps=None, qa_types=None):
    """Create a mock VideoReasoningAnnotationConfig."""
    vcfg = MagicMock()
    vcfg.workflow.mode = mode
    vcfg.workflow.steps = steps or ["1a", "1b", "2", "3", "4"]
    vcfg.workflow.max_workers = 1
    vcfg.workflow.max_video_length_sec = 300
    vcfg.workflow.chunk_duration_options = [5, 10]
    vcfg.workflow.max_chunks = 10
    vcfg.workflow.highlight_before_sec = 3.0
    vcfg.workflow.highlight_after_sec = 3.0
    vcfg.workflow.long_video_threshold_sec = 60
    vcfg.workflow.long_video_sample_fps = 0.5
    vcfg.workflow.long_video_max_frames = 60
    vcfg.workflow.qa_types = qa_types or [
        "mcq", "bcq", "open_qa",
        "causal_linkage", "temporal_localization", "temporal_event_desc",
        "scene_description", "event_summary",
    ]
    vcfg.data.video_root = ""
    vcfg.data.input_jsonl_files = []
    vcfg.data.filter_field = None
    vcfg.license = ""
    vcfg.description_extra = ""
    vcfg.prompts_module = ""
    return vcfg


@dataclass
class _MockGeminiCfg:
    api_key: str = "test-key"
    model: str = "gemini-3.1-flash-lite-preview"
    media_resolution: str = "media_resolution_low"
    temperature: float = 0.3
    max_output_tokens: int = 8192
    timeout: int = 120


@dataclass
class _MockOpenAICfg:
    api_key: str = "test-key"
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "test-model"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


# ===========================================================================
# Config dataclass tests
# ===========================================================================

class TestConfig:
    def test_gemini_config_defaults(self):
        cfg = GeminiConfig()
        assert cfg.model == "gemini-3.1-flash-lite-preview", f"Expected 'gemini-3.1-flash-lite-preview', got {cfg.model}"
        assert cfg.media_resolution == "MEDIA_RESOLUTION_LOW", f"Expected 'MEDIA_RESOLUTION_LOW', got {cfg.media_resolution}"
        assert cfg.temperature == 0.3, f"Expected 0.3, got {cfg.temperature}"
        assert cfg.max_output_tokens == 8192, f"Expected 8192, got {cfg.max_output_tokens}"
        assert cfg.timeout == 120, f"Expected 120, got {cfg.timeout}"

    def test_openai_config_defaults(self):
        cfg = OpenAIConfig()
        assert cfg.base_url == "", f"Expected '', got {cfg.base_url}"
        assert cfg.model_name == "", f"Expected '', got {cfg.model_name}"
        assert cfg.max_tokens == 4096, f"Expected 4096, got {cfg.max_tokens}"

    def test_llm_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.backend == "gemini", f"Expected 'gemini', got {cfg.backend}"
        assert isinstance(cfg.gemini, GeminiConfig), f"Expected GeminiConfig, got {type(cfg.gemini).__name__}"
        assert isinstance(cfg.openai, OpenAIConfig), f"Expected OpenAIConfig, got {type(cfg.openai).__name__}"

    def test_workflow_config_defaults(self):
        cfg = WorkflowConfig()
        assert cfg.mode == "auto", f"Expected 'auto', got {cfg.mode}"
        assert cfg.max_workers == 4, f"Expected 4, got {cfg.max_workers}"
        assert cfg.max_video_length_sec == 300, f"Expected 300, got {cfg.max_video_length_sec}"
        assert "1a" in cfg.steps, f"Expected '1a' to be in {cfg.steps}"
        assert "1b" in cfg.steps, f"Expected '1b' to be in {cfg.steps}"
        assert cfg.max_chunks == 10, f"Expected 10, got {cfg.max_chunks}"

    def test_data_config_defaults(self):
        cfg = DataConfig()
        assert cfg.video_root == "", f"Expected '', got {cfg.video_root}"
        assert cfg.input_jsonl_files == [], f"Expected [], got {cfg.input_jsonl_files}"
        assert cfg.filter_field is None, f"Expected None, got {cfg.filter_field}"

    def test_video_reasoning_annotation_config_defaults(self):
        cfg = VideoReasoningAnnotationConfig()
        assert isinstance(cfg.vlm, LLMConfig), f"Expected LLMConfig, got {type(cfg.vlm).__name__}"
        assert isinstance(cfg.llm, LLMConfig), f"Expected LLMConfig, got {type(cfg.llm).__name__}"
        assert isinstance(cfg.workflow, WorkflowConfig), f"Expected WorkflowConfig, got {type(cfg.workflow).__name__}"
        assert isinstance(cfg.data, DataConfig), f"Expected DataConfig, got {type(cfg.data).__name__}"
        assert cfg.license == "", f"Expected '', got {cfg.license}"
        assert cfg.description_extra == "", f"Expected '', got {cfg.description_extra}"

    def test_video_reasoning_annotation_config_custom(self):
        cfg = VideoReasoningAnnotationConfig(
            vlm=LLMConfig(backend="openai"),
            workflow=WorkflowConfig(mode="anomaly", steps=["1a", "1c", "2"]),
            license="CC-BY-4.0",
            description_extra="My dataset.",
        )
        assert cfg.vlm.backend == "openai", f"Expected 'openai', got {cfg.vlm.backend}"
        assert cfg.workflow.mode == "anomaly", f"Expected 'anomaly', got {cfg.workflow.mode}"
        assert "1c" in cfg.workflow.steps, f"Expected '1c' to be in {cfg.workflow.steps}"
        assert cfg.license == "CC-BY-4.0", f"Expected 'CC-BY-4.0', got {cfg.license}"
        assert cfg.description_extra == "My dataset.", f"Expected 'My dataset.', got {cfg.description_extra}"


# ===========================================================================
# I/O utilities tests
# ===========================================================================

class TestJsonlRoundTrip:
    def test_save_and_read(self, tmp_path):
        output_file = str(tmp_path / "test.jsonl")
        entry1 = {"video": "a.mp4", "caption": "hello"}
        entry2 = {"video": "b.mp4", "caption": "world"}
        save_result_to_jsonl(entry1, output_file)
        save_result_to_jsonl(entry2, output_file)
        entries = get_entries_from_jsonl(output_file)
        assert len(entries) == 2, f"Expected 2 items, got {len(entries)}"
        assert entries[0]["video"] == "a.mp4", f"Unexpected video: {entries[0]['video']}"
        assert entries[1]["caption"] == "world", f"Unexpected caption: {entries[1]['caption']}"

    def test_read_nonexistent(self, tmp_path):
        entries = get_entries_from_jsonl(str(tmp_path / "missing.jsonl"))
        assert entries == [], f"Expected empty list, got {entries}"

    def test_get_processed_videos(self, tmp_path):
        output_file = str(tmp_path / "test.jsonl")
        save_result_to_jsonl({"video": "a.mp4"}, output_file)
        save_result_to_jsonl({"video": "b.mp4"}, output_file)
        processed = get_processed_videos(output_file)
        assert processed == {"a.mp4", "b.mp4"}, f"Expected {{'a.mp4', 'b.mp4'}}, got {processed}"

    def test_get_processed_video_prompt_keys(self, tmp_path):
        output_file = str(tmp_path / "test.jsonl")
        save_result_to_jsonl({"video": "a.mp4", "prompt_key": "mcq"}, output_file)
        save_result_to_jsonl({"video": "a.mp4", "prompt_key": "bcq"}, output_file)
        processed = get_processed_video_prompt_keys(output_file)
        assert ("a.mp4", "mcq") in processed, f"Expected ('a.mp4', 'mcq') to be in {processed}"
        assert ("a.mp4", "bcq") in processed, f"Expected ('a.mp4', 'bcq') to be in {processed}"


class TestFormatChunkCaptions:
    def test_empty(self):
        assert format_chunk_captions([]) == "", f"Expected empty string, got {format_chunk_captions([])!r}"

    def test_single_chunk(self):
        chunks = [{"chunk_index": 0, "timestamp_start": 0, "timestamp_end": 5, "caption": "Hello"}]
        result = format_chunk_captions(chunks)
        assert "Chunk 0" in result, f"Expected 'Chunk 0' to be in result: {result}"
        assert "Hello" in result, f"Expected 'Hello' to be in result: {result}"

    def test_multiple_chunks(self):
        chunks = [
            {"chunk_index": 0, "timestamp_start": 0, "timestamp_end": 5, "caption": "First"},
            {"chunk_index": 1, "timestamp_start": 5, "timestamp_end": 10, "caption": "Second"},
        ]
        result = format_chunk_captions(chunks)
        assert "Chunk 0" in result, f"Expected 'Chunk 0' to be in result: {result}"
        assert "Chunk 1" in result, f"Expected 'Chunk 1' to be in result: {result}"
        assert "---" in result, f"Expected '---' to be in result: {result}"


class TestParseQaOutput:
    def test_mcq(self):
        text = (
            "=====\n"
            "Multiple-Choice Question: What happened?\n"
            "A. Event X\n"
            "B. Event Y\n"
            "C. Event Z\n"
            "D. Nothing\n"
            "======\n"
            "Answer: A\n"
            "=====\n"
            "Reasoning: Because of event X."
        )
        result = parse_qa_output(text)
        assert len(result) == 1, f"Expected 1 items, got {len(result)}"
        assert result[0]["type"] == "mcq", f"Expected 'mcq', got {result[0]['type']}"
        assert result[0]["answer"] == "A", f"Expected 'A', got {result[0]['answer']}"
        assert len(result[0]["choices"]) == 4, f"Expected 4 items, got {len(result[0]['choices'])}"

    def test_binary(self):
        text = (
            "=====\n"
            "1. Question: Is there a collision?\n"
            "======\n"
            "Answer: Yes. There is a clear collision.\n"
            "=====\n"
            "Reasoning: The video shows impact."
        )
        result = parse_qa_output(text)
        assert len(result) == 1, f"Expected 1 items, got {len(result)}"
        assert result[0]["type"] == "binary", f"Expected 'binary', got {result[0]['type']}"

    def test_openended(self):
        text = (
            "=====\n"
            "Open-ended Question: What caused the event?\n"
            "======\n"
            "Answer: The event was caused by failure.\n"
            "=====\n"
            "Reasoning: Analysis shows failure."
        )
        result = parse_qa_output(text)
        assert len(result) == 1, f"Expected 1 items, got {len(result)}"
        assert result[0]["type"] == "openended", f"Expected 'openended', got {result[0]['type']}"

    def test_empty(self):
        assert parse_qa_output("") == [], f"Expected [] for empty string, got {parse_qa_output('')}"
        assert parse_qa_output(None) == [], f"Expected [] for None, got {parse_qa_output(None)}"


class TestPermuteMcq:
    def test_shuffle_preserves_answer(self):
        qa = {
            "type": "mcq",
            "question": "Q?",
            "choices": ["A. First", "B. Second", "C. Third", "D. Fourth"],
            "answer": "B",
            "reasoning": "Because.",
        }
        result = permute_mcq(qa)
        assert result["answer"] in "ABCD", f"Expected answer in 'ABCD', got {result['answer']}"
        for c in result["choices"]:
            if c.startswith(f"{result['answer']}."):
                assert "Second" in c, f"Expected 'Second' in answer choice, got {c}"

    def test_single_choice_unchanged(self):
        qa = {
            "type": "mcq",
            "question": "Q?",
            "choices": ["A. Only"],
            "answer": "A",
            "reasoning": "",
        }
        result = permute_mcq(qa)
        assert result["choices"] == ["A. Only"], f"Expected ['A. Only'], got {result['choices']}"


class TestResolveMode:
    def test_entry_mode_takes_priority(self):
        entry = {"video": "a.mp4", "mode": "anomaly"}
        assert resolve_mode(entry, "normal") == "anomaly", f"Expected 'anomaly', got {resolve_mode(entry, 'normal')}"

    def test_mode_map_used_when_entry_has_no_mode(self):
        entry = {"video": "a.mp4"}
        mode_map = {"a.mp4": "anomaly"}
        assert resolve_mode(entry, "normal", mode_map) == "anomaly", f"Expected 'anomaly', got {resolve_mode(entry, 'normal', mode_map)}"

    def test_global_mode_fallback(self):
        entry = {"video": "a.mp4"}
        assert resolve_mode(entry, "anomaly") == "anomaly", f"Expected 'anomaly', got {resolve_mode(entry, 'anomaly')}"

    def test_auto_mode_defaults_to_normal(self):
        entry = {"video": "a.mp4"}
        assert resolve_mode(entry, "auto") == "normal", f"Expected 'normal', got {resolve_mode(entry, 'auto')}"

    def test_auto_mode_with_mode_map(self):
        entry = {"video": "a.mp4"}
        mode_map = {"a.mp4": "anomaly"}
        assert resolve_mode(entry, "auto", mode_map) == "anomaly", f"Expected 'anomaly', got {resolve_mode(entry, 'auto', mode_map)}"


class TestLoadModeMap:
    def test_loads_from_step0_output(self, tmp_path):
        results_dir = str(tmp_path)
        step0_dir = os.path.join(results_dir, "step_0_filter")
        os.makedirs(step0_dir)
        output_file = os.path.join(step0_dir, "filter_results.jsonl")
        save_result_to_jsonl(
            {"video": "/fake/a.mp4", "is_valid": True, "mode": "anomaly"},
            output_file,
        )
        save_result_to_jsonl(
            {"video": "/fake/b.mp4", "is_valid": True, "mode": "normal"},
            output_file,
        )
        mode_map = load_mode_map(results_dir)
        assert mode_map == {"/fake/a.mp4": "anomaly", "/fake/b.mp4": "normal"}, f"Unexpected mode_map: {mode_map}"

    def test_missing_step0_returns_empty(self, tmp_path):
        assert load_mode_map(str(tmp_path)) == {}, f"Expected empty dict, got {load_mode_map(str(tmp_path))}"

    def test_entries_without_mode_skipped(self, tmp_path):
        results_dir = str(tmp_path)
        step0_dir = os.path.join(results_dir, "step_0_filter")
        os.makedirs(step0_dir)
        output_file = os.path.join(step0_dir, "filter_results.jsonl")
        save_result_to_jsonl(
            {"video": "/fake/a.mp4", "is_valid": True},
            output_file,
        )
        mode_map = load_mode_map(results_dir)
        assert mode_map == {}, f"Expected empty dict, got {mode_map}"


class TestBuildQuestionStr:
    def test_mcq(self):
        qa = {
            "type": "mcq",
            "question": "What happened?",
            "choices": ["A. X", "B. Y"],
            "answer": "A",
            "reasoning": "",
        }
        result = build_question_str(qa)
        assert "What happened?" in result, f"Expected 'What happened?' to be in result: {result}"
        assert "A. X" in result, f"Expected 'A. X' to be in result: {result}"
        assert "Answer with the option's letter" in result, f"Expected 'Answer with the option\\'s letter' to be in result: {result}"

    def test_openended(self):
        qa = {"type": "openended", "question": "Explain.", "choices": [], "answer": "", "reasoning": ""}
        result = build_question_str(qa)
        assert result == "Explain.", f"Expected 'Explain.', got {result}"


# ===========================================================================
# LLM client tests
# ===========================================================================

class TestGeminiClient:
    @patch("nvidia_tao_ds.core.llm_clients.gemini_client.GeminiClient.__init__", return_value=None)
    def test_extract_text(self, mock_init):
        client = GeminiClient.__new__(GeminiClient)
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        result = GeminiClient._extract_text(mock_response)
        assert result == "This is a test response.", f"Expected 'This is a test response.', got {result}"

    @patch("nvidia_tao_ds.core.llm_clients.gemini_client.GeminiClient.__init__", return_value=None)
    def test_extract_text_strips_thinking(self, mock_init):
        client = GeminiClient.__new__(GeminiClient)
        mock_response = MagicMock()
        mock_response.text = "<think>internal reasoning</think>Final answer."
        result = GeminiClient._extract_text(mock_response)
        assert result == "Final answer.", f"Expected 'Final answer.', got {result}"

    @patch("nvidia_tao_ds.core.llm_clients.gemini_client.GeminiClient.__init__", return_value=None)
    def test_extract_text_empty_raises(self, mock_init):
        client = GeminiClient.__new__(GeminiClient)
        mock_response = MagicMock()
        mock_response.text = ""
        with pytest.raises(ValueError, match="Empty response"):
            GeminiClient._extract_text(mock_response)


class TestCreateClient:
    def test_create_gemini_client(self):
        mock_config = MagicMock()
        mock_config.backend = "gemini"
        mock_config.gemini = _MockGeminiCfg()

        with patch("nvidia_tao_ds.core.llm_clients.gemini_client.GeminiClient") as MockGemini:
            create_client(mock_config)
            MockGemini.assert_called_once_with(mock_config.gemini)

    def test_create_openai_client(self):
        mock_config = MagicMock()
        mock_config.backend = "openai"
        mock_config.openai = _MockOpenAICfg()

        with patch("nvidia_tao_ds.core.llm_clients.openai_compatible_client.OpenAICompatibleClient") as MockOAI:
            create_client(mock_config)
            MockOAI.assert_called_once_with(mock_config.openai)

    def test_create_unknown_backend_raises(self):
        mock_config = MagicMock()
        mock_config.backend = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_client(mock_config)


# ===========================================================================
# Prompt template tests
# ===========================================================================

class TestPromptTemplates:
    def test_all_keys_present(self):
        expected_keys = [
            "video_filtering",
            "video_anomaly_classification",
            "anomaly_global_caption", "anomaly_dense_caption", "anomaly_chunk_caption",
            "normal_global_caption", "normal_dense_caption", "normal_chunk_caption",
            "highlight_timestamp_extraction", "highlight_chunk_caption",
            "anomaly_description", "normal_description",
            "anomaly_mcq", "normal_mcq",
            "anomaly_bcq", "normal_bcq",
            "anomaly_open_qa", "normal_open_qa",
            "anomaly_temporal_event_desc", "normal_temporal_event_desc",
            "anomaly_causal_linkage", "normal_causal_linkage",
            "anomaly_temporal_localization", "normal_temporal_localization",
            "scene_description", "event_summary",
        ]
        for key in expected_keys:
            assert key in PROMPT_TEMPLATES, f"Missing prompt key: {key}"

    def test_no_domain_placeholders(self):
        """General config prompts should not contain [DOMAIN] style placeholders."""
        placeholder_re = re.compile(r"\[(?:DOMAIN|DATA_TYPE|KEY_ASPECT|DOMAIN_\w+|ANOMALY_TOPIC|NORMAL_TOPIC)\w*\]")
        for key, template in PROMPT_TEMPLATES.items():
            matches = placeholder_re.findall(template)
            assert not matches, f"Prompt '{key}' contains unfilled placeholders: {matches}"

    def test_video_filtering_renders(self):
        prompt = get_prompt("video_filtering")
        assert "Yes" in prompt, f"Expected 'Yes' to be in prompt"
        assert "No" in prompt, f"Expected 'No' to be in prompt"

    def test_anomaly_classification_renders(self):
        prompt = get_prompt("video_anomaly_classification")
        assert "Anomaly" in prompt, f"Expected 'Anomaly' to be in prompt"
        assert "Normal" in prompt, f"Expected 'Normal' to be in prompt"
        assert "collision" in prompt.lower() or "collision" in prompt, f"Expected 'collision' to be in prompt"

    def test_chunk_caption_renders_with_duration(self):
        prompt = get_prompt("anomaly_chunk_caption", chunk_duration=10)
        assert "10-second" in prompt or "10" in prompt, f"Expected '10' to be in prompt"

    def test_highlight_timestamp_renders(self):
        prompt = get_prompt(
            "highlight_timestamp_extraction",
            global_caption="A scene.",
            dense_caption="<00:00:00><00:00:05> Event.",
            chunk_captions_str="Chunk 0: detail.",
        )
        assert "A scene." in prompt, f"Expected 'A scene.' to be in prompt"
        assert "Event." in prompt, f"Expected 'Event.' to be in prompt"

    def test_description_renders(self):
        prompt = get_prompt(
            "anomaly_description",
            video_length=30,
            global_caption="Scene.",
            dense_section="[Dense Caption]\nEvents.",
            chunk_captions_str="Chunks.",
            highlight_section="Highlight.",
        )
        assert "Scene." in prompt, f"Expected 'Scene.' to be in prompt"
        assert "30" in prompt, f"Expected '30' to be in prompt"

    def test_mcq_renders(self):
        prompt = get_prompt(
            "anomaly_mcq",
            video_length=30,
            global_caption="Cap.",
            dense_caption="Dense.",
            chunk_captions_str="Chunks.",
            step_2_output="Description.",
        )
        assert "Multiple-Choice Question" in prompt, f"Expected 'Multiple-Choice Question' to be in prompt"
        assert "Description." in prompt, f"Expected 'Description.' to be in prompt"

    def test_get_prompt_missing_key(self):
        with pytest.raises(ValueError, match="No prompt template"):
            get_prompt("nonexistent_key")


# ===========================================================================
# Pipeline step integration tests (mocked LLM clients)
# ===========================================================================

class TestStep2Description:
    def test_run_with_mock_input(self, tmp_path):
        results_dir = str(tmp_path)
        vcfg = _make_vcfg()

        step1a_dir = os.path.join(results_dir, "step_1a_caption")
        os.makedirs(step1a_dir)
        save_result_to_jsonl(
            {
                "video": "/fake/video.mp4",
                "video_length": 30,
                "mode": "normal",
                "global_caption": "A person walks.",
                "dense_caption": "<00:00:00><00:00:10> Walking.",
            },
            os.path.join(step1a_dir, "captions.jsonl"),
        )

        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            "1. Scene: A person walking.\n"
            "2. <00:00><00:10> [center] Person walks.\n"
            "3. Activity: Walking observed."
        )

        step2_description.run(vcfg, mock_client, PROMPT_TEMPLATES, results_dir)

        output_file = os.path.join(results_dir, "step_2_description", "descriptions.jsonl")
        assert os.path.exists(output_file), f"Expected file to exist: {output_file}"
        entries = get_entries_from_jsonl(output_file)
        assert len(entries) == 1, f"Expected 1 items, got {len(entries)}"
        assert "detailed_description" in entries[0], f"Expected 'detailed_description' to be in {list(entries[0].keys())}"


class TestStep3QA:
    def test_run_with_mock_input(self, tmp_path):
        results_dir = str(tmp_path)
        vcfg = _make_vcfg(qa_types=["mcq"])

        step2_dir = os.path.join(results_dir, "step_2_description")
        os.makedirs(step2_dir)
        save_result_to_jsonl(
            {
                "video": "/fake/video.mp4",
                "video_length": 30,
                "detailed_description": "Description text.",
                "original_captions": {
                    "global_caption": "Caption.",
                    "dense_caption": "Dense.",
                },
            },
            os.path.join(step2_dir, "descriptions.jsonl"),
        )

        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            "=====\n"
            "Multiple-Choice Question: What is happening?\n"
            "A. Walking\nB. Running\nC. Standing\nD. Sitting\n"
            "======\n"
            "Answer: A\n"
            "=====\n"
            "Reasoning: The person is walking."
        )

        step3_qa.run(vcfg, mock_client, PROMPT_TEMPLATES, results_dir)

        output_file = os.path.join(results_dir, "step_3_qa", "qa_output.jsonl")
        assert os.path.exists(output_file), f"Expected file to exist: {output_file}"
        entries = get_entries_from_jsonl(output_file)
        assert len(entries) == 1, f"Expected 1 items, got {len(entries)}"
        assert entries[0]["prompt_key"] == "normal_mcq", f"Expected 'normal_mcq', got {entries[0]['prompt_key']}"


class TestStep4ParseQA:
    def test_parse_mcq_output(self, tmp_path):
        results_dir = str(tmp_path)
        vcfg = _make_vcfg()

        step3_dir = os.path.join(results_dir, "step_3_qa")
        os.makedirs(step3_dir)
        save_result_to_jsonl(
            {
                "video": "/fake/video.mp4",
                "video_length": 30,
                "qa_output": (
                    "=====\n"
                    "Multiple-Choice Question: What happened?\n"
                    "A. Event X\nB. Event Y\nC. Event Z\nD. Nothing\n"
                    "======\n"
                    "Answer: A\n"
                    "=====\n"
                    "Reasoning: Event X occurred."
                ),
                "prompt_key": "normal_mcq",
            },
            os.path.join(step3_dir, "qa_output.jsonl"),
        )

        step4_parse_qa.run(vcfg, results_dir)

        output_dir = os.path.join(results_dir, "step_4_output")
        assert os.path.exists(output_dir), f"Expected dir to exist: {output_dir}"

        json_files = [f for f in os.listdir(output_dir) if f.endswith(".json")]
        assert "mcq.json" in json_files, f"Expected mcq.json to be in {json_files}"
        assert "mcq_openended.json" in json_files, f"Expected mcq_openended.json to be in {json_files}"

        with open(os.path.join(output_dir, "mcq.json")) as f:
            data = json.load(f)

        # tao-vl-reason-v1.0 envelope structure
        assert data["format"] == "tao-vl-reason-v1.0", f"Expected envelope format 'tao-vl-reason-v1.0', got {data.get('format')}"
        assert data["metadata"]["type"] == "annotation", f"Expected metadata.type 'annotation', got {data['metadata'].get('type')}"
        assert data["metadata"]["task"] == "mcq", f"Expected metadata.task 'mcq', got {data['metadata'].get('task')}"
        assert "date" in data["metadata"], f"Expected 'date' to be in metadata: {data['metadata']}"
        assert data["media_root"] is None, f"Expected media_root None (video_root empty), got {data['media_root']}"

        # Items
        items = data["items"]
        assert len(items) >= 1, f"Expected at least 1 item, got {len(items)}"
        item = items[0]
        assert set(item.keys()) == {"video_id", "question", "answer", "reasoning"}, (
            f"Expected item keys {{'video_id','question','answer','reasoning'}}, got {set(item.keys())}"
        )
        # MCQ answer is a single letter; question carries the choices and the instruction
        assert len(item["answer"]) == 1 and item["answer"].isalpha(), (
            f"Expected MCQ answer to be a single letter, got {item['answer']!r}"
        )
        assert "Answer with a single letter." in item["question"], (
            f"Expected MCQ instruction in question, got {item['question']!r}"
        )


class TestStep3PerVideoMode:
    def test_anomaly_entry_gets_anomaly_prompts(self, tmp_path):
        """When entry has mode=anomaly, step 3 should use anomaly_* prompt keys."""
        results_dir = str(tmp_path)
        vcfg = _make_vcfg(mode="auto", qa_types=["mcq"])

        step2_dir = os.path.join(results_dir, "step_2_description")
        os.makedirs(step2_dir)
        save_result_to_jsonl(
            {
                "video": "/fake/video.mp4",
                "video_length": 30,
                "mode": "anomaly",
                "detailed_description": "Description.",
                "original_captions": {
                    "global_caption": "Caption.",
                    "dense_caption": "Dense.",
                },
            },
            os.path.join(step2_dir, "descriptions.jsonl"),
        )

        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            "=====\nMultiple-Choice Question: Q?\n"
            "A. X\nB. Y\nC. Z\nD. W\n======\nAnswer: A\n=====\nReasoning: R."
        )

        step3_qa.run(vcfg, mock_client, PROMPT_TEMPLATES, results_dir)

        entries = get_entries_from_jsonl(
            os.path.join(results_dir, "step_3_qa", "qa_output.jsonl")
        )
        assert len(entries) == 1, f"Expected 1 items, got {len(entries)}"
        assert entries[0]["prompt_key"] == "anomaly_mcq", f"Expected 'anomaly_mcq', got {entries[0]['prompt_key']}"


class TestStepSkip:
    def test_step1c_skipped_in_normal_mode(self, tmp_path):
        """Step 1c should be skipped when mode is normal."""
        results_dir = str(tmp_path)
        vcfg = _make_vcfg(mode="normal")

        mock_vlm = MagicMock()
        mock_llm = MagicMock()

        step1c_highlight.run(vcfg, mock_vlm, mock_llm, PROMPT_TEMPLATES, results_dir)

        output_file = os.path.join(results_dir, "step_1c_highlight", "highlight_captions.jsonl")
        assert not os.path.exists(output_file), f"File should not exist: {output_file}"
        mock_vlm.generate_with_video.assert_not_called()

    def test_step1c_skips_normal_entries_in_auto_mode(self, tmp_path):
        """In auto mode, step 1c should only process entries with mode=anomaly."""
        results_dir = str(tmp_path)
        vcfg = _make_vcfg(mode="auto")

        step1a_dir = os.path.join(results_dir, "step_1a_caption")
        os.makedirs(step1a_dir)
        save_result_to_jsonl(
            {
                "video": "/fake/normal_video.mp4",
                "video_length": 30,
                "mode": "normal",
                "global_caption": "Cap.",
                "dense_caption": "Dense.",
            },
            os.path.join(step1a_dir, "captions.jsonl"),
        )

        mock_vlm = MagicMock()
        mock_llm = MagicMock()

        step1c_highlight.run(vcfg, mock_vlm, mock_llm, PROMPT_TEMPLATES, results_dir)

        mock_vlm.generate_with_video.assert_not_called()
        mock_llm.generate_text.assert_not_called()


class TestResume:
    def test_already_processed_skipped(self, tmp_path):
        """Videos already in the output JSONL should be skipped."""
        results_dir = str(tmp_path)
        vcfg = _make_vcfg()

        step1a_dir = os.path.join(results_dir, "step_1a_caption")
        os.makedirs(step1a_dir)
        for vid in ["a.mp4", "b.mp4"]:
            save_result_to_jsonl(
                {
                    "video": f"/fake/{vid}",
                    "video_length": 30,
                    "global_caption": "Cap.",
                    "dense_caption": "Dense.",
                },
                os.path.join(step1a_dir, "captions.jsonl"),
            )

        step2_dir = os.path.join(results_dir, "step_2_description")
        os.makedirs(step2_dir)
        save_result_to_jsonl(
            {"video": "/fake/a.mp4", "detailed_description": "Already done."},
            os.path.join(step2_dir, "descriptions.jsonl"),
        )

        mock_client = MagicMock()
        mock_client.generate_text.return_value = "New description."

        step2_description.run(vcfg, mock_client, PROMPT_TEMPLATES, results_dir)

        assert mock_client.generate_text.call_count == 1, f"Expected 1 call, got {mock_client.generate_text.call_count}"
