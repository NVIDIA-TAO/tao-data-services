# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a QA JSONL file to LLaVA training annotation format.

Input JSONL: one JSON object per line with id, video_path, question, answer.
Output JSON: a LLaVA-style annotation array.

Example input line:
    {"id": "clip_001", "video_path": "/data/videos/clip_001.mp4", "question": "What happens?", "answer": "A car stops."}

Example output entry:
    {"id": "clip_001", "video": "/data/videos/clip_001.mp4", "conversations": [
        {"from": "human", "value": "What happens?"},
        {"from": "gpt", "value": "A car stops."}
    ]}
"""

import json
import logging
import os
from dataclasses import dataclass

from nvidia_tao_ds.config.utils.types import STR_FIELD
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner

logger = logging.getLogger(__name__)


@dataclass
class QAToLLaVAConfig:
    """Configuration for QA to LLaVA annotation conversion."""

    input_jsonl: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to input JSONL file. Each line must have id, video_path, question, and answer fields."
    )
    output_file: str = STR_FIELD(
        value="llava_annotations.json",
        default_value="llava_annotations.json",
        description="Output filename (written inside results_dir)."
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Output directory for results."
    )


def convert_qa_to_llava(input_jsonl: str, output_path: str) -> None:
    """Convert a QA JSONL file to LLaVA annotation format.

    Args:
        input_jsonl: Path to input JSONL file.
        output_path: Path to write the output JSON file.

    Raises:
        FileNotFoundError: If input_jsonl does not exist.
        ValueError: If a line is missing required fields.
    """
    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    required_fields = {"id", "video_path", "question", "answer"}
    annotations = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e

            missing = required_fields - row.keys()
            if missing:
                raise ValueError(f"Line {i}: missing fields {missing}")

            annotations.append({
                "id": row["id"],
                "video": row["video_path"],
                "conversations": [
                    {"from": "human", "value": row["question"]},
                    {"from": "gpt", "value": row["answer"]},
                ],
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d LLaVA annotations to %s", len(annotations), output_path)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="qa_to_llava",
    schema=QAToLLaVAConfig,
)
def main(cfg: QAToLLaVAConfig) -> None:
    """Convert QA JSONL to LLaVA annotation format."""
    os.makedirs(cfg.results_dir, exist_ok=True)
    output_path = os.path.join(cfg.results_dir, cfg.output_file)
    convert_qa_to_llava(cfg.input_jsonl, output_path)


if __name__ == "__main__":
    main()
