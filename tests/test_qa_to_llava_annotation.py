# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for qa_to_llava_annotation conversion."""

import json
from pathlib import Path

import pytest

from nvidia_tao_ds.annotations.scripts.qa_to_llava_annotation import convert_qa_to_llava


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows) -> Path:
    """Write rows (list[dict]) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConvertQAToLLaVA:
    """Tests for convert_qa_to_llava."""

    def test_basic_conversion(self, tmp_path):
        input_jsonl = _write_jsonl(tmp_path / "qa.jsonl", [
            {"id": "clip_001", "video_path": "/v/a.mp4", "question": "What happens?", "answer": "A car stops."},
            {"id": "clip_002", "video_path": "/v/b.mp4", "question": "Describe.", "answer": "Rain falls."},
        ])
        output_path = tmp_path / "out.json"

        convert_qa_to_llava(str(input_jsonl), str(output_path))

        annotations = json.loads(output_path.read_text())
        assert len(annotations) == 2
        assert annotations[0] == {
            "id": "clip_001",
            "video": "/v/a.mp4",
            "conversations": [
                {"from": "human", "value": "What happens?"},
                {"from": "gpt", "value": "A car stops."},
            ],
        }
        assert annotations[1]["id"] == "clip_002"
        assert annotations[1]["video"] == "/v/b.mp4"

    def test_missing_field_raises(self, tmp_path):
        input_jsonl = _write_jsonl(tmp_path / "qa.jsonl", [
            {"id": "clip_001", "video_path": "/v/a.mp4", "question": "What?"},
        ])
        output_path = tmp_path / "out.json"

        with pytest.raises(ValueError, match="missing fields.*answer"):
            convert_qa_to_llava(str(input_jsonl), str(output_path))

    def test_invalid_json_raises(self, tmp_path):
        input_jsonl = tmp_path / "qa.jsonl"
        input_jsonl.write_text("not json\n", encoding="utf-8")
        output_path = tmp_path / "out.json"

        with pytest.raises(ValueError, match="Invalid JSON at line 1"):
            convert_qa_to_llava(str(input_jsonl), str(output_path))

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input JSONL not found"):
            convert_qa_to_llava(str(tmp_path / "missing.jsonl"), str(tmp_path / "out.json"))

    def test_skips_empty_lines(self, tmp_path):
        input_jsonl = tmp_path / "qa.jsonl"
        input_jsonl.write_text(
            '{"id": "a", "video_path": "/v/a.mp4", "question": "Q", "answer": "A"}\n'
            "\n"
            "   \n"
            '{"id": "b", "video_path": "/v/b.mp4", "question": "Q2", "answer": "A2"}\n',
            encoding="utf-8",
        )
        output_path = tmp_path / "out.json"

        convert_qa_to_llava(str(input_jsonl), str(output_path))

        annotations = json.loads(output_path.read_text())
        assert len(annotations) == 2

    def test_empty_input_produces_empty_array(self, tmp_path):
        input_jsonl = tmp_path / "qa.jsonl"
        input_jsonl.write_text("", encoding="utf-8")
        output_path = tmp_path / "out.json"

        convert_qa_to_llava(str(input_jsonl), str(output_path))

        annotations = json.loads(output_path.read_text())
        assert annotations == []

    def test_output_is_valid_json_array(self, tmp_path):
        input_jsonl = _write_jsonl(tmp_path / "qa.jsonl", [
            {"id": "1", "video_path": "/v/a.mp4", "question": "Q", "answer": "A"},
        ])
        output_path = tmp_path / "out.json"

        convert_qa_to_llava(str(input_jsonl), str(output_path))

        # Should be a JSON array, not JSONL
        content = output_path.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, list)
