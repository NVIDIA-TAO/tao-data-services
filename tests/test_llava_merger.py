# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLaVAMerger."""

import json
from pathlib import Path

import pytest

from nvidia_tao_ds.annotations.merger import LLaVAMerger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_llava_json(path: Path, annotations) -> Path:
    """Write a LLaVA annotation JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False)
    return path


def _make_entry(id, video="/v/test.mp4", question="Q", answer="A"):
    """Create a LLaVA annotation entry."""
    return {
        "id": id,
        "video": video,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLaVAMerger:
    """Tests for LLaVAMerger."""

    def test_merge_two_files(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1", "/v/a.mp4"),
            _make_entry("2", "/v/b.mp4"),
        ])
        file2 = _write_llava_json(tmp_path / "b.json", [
            _make_entry("3", "/v/c.mp4"),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1), str(file2)])
        merger.merge(str(output))

        result = json.loads(output.read_text())
        assert len(result) == 3
        assert [r["id"] for r in result] == ["1", "2", "3"]

    def test_merge_single_file(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1"),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1)])
        merger.merge(str(output))

        result = json.loads(output.read_text())
        assert len(result) == 1

    def test_merge_empty_files(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [])
        file2 = _write_llava_json(tmp_path / "b.json", [])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1), str(file2)])
        merger.merge(str(output))

        result = json.loads(output.read_text())
        assert result == []

    def test_duplicate_id_error_by_default(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1", "/v/a.mp4"),
        ])
        file2 = _write_llava_json(tmp_path / "b.json", [
            _make_entry("1", "/v/b.mp4"),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1), str(file2)])
        with pytest.raises(ValueError, match="Duplicate id"):
            merger.merge(str(output))

    def test_duplicate_id_skip(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1", "/v/a.mp4", "Q1", "A1"),
        ])
        file2 = _write_llava_json(tmp_path / "b.json", [
            _make_entry("1", "/v/b.mp4", "Q2", "A2"),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1), str(file2)], on_duplicate="skip")
        merger.merge(str(output))

        result = json.loads(output.read_text())
        assert len(result) == 1
        assert result[0]["video"] == "/v/a.mp4"  # keeps first

    def test_duplicate_id_keep(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1", "/v/a.mp4"),
        ])
        file2 = _write_llava_json(tmp_path / "b.json", [
            _make_entry("1", "/v/b.mp4"),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1), str(file2)], on_duplicate="keep")
        merger.merge(str(output))

        result = json.loads(output.read_text())
        assert len(result) == 2

    def test_invalid_on_duplicate_raises(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [])

        with pytest.raises(ValueError, match="Invalid on_duplicate"):
            LLaVAMerger([str(file1)], on_duplicate="invalid")

    def test_empty_annotation_list_raises(self):
        with pytest.raises(AssertionError):
            LLaVAMerger([])

    def test_preserves_conversation_content(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [
            _make_entry("1", "/v/a.mp4", "What happens?", "A car stops."),
        ])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1)])
        merger.merge(str(output))

        result = json.loads(output.read_text())
        convos = result[0]["conversations"]
        assert convos[0] == {"from": "human", "value": "What happens?"}
        assert convos[1] == {"from": "gpt", "value": "A car stops."}

    def test_output_is_valid_json_array(self, tmp_path):
        file1 = _write_llava_json(tmp_path / "a.json", [_make_entry("1")])
        output = tmp_path / "merged.json"

        merger = LLaVAMerger([str(file1)])
        merger.merge(str(output))

        parsed = json.loads(output.read_text())
        assert isinstance(parsed, list)
