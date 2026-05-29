# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to extract FP/FN KPI gaps from model predictions.

The input predictions JSON must be a list of objects, each with the
following required fields:
  - video_id  (str): Path to the video file.
  - response  (str): The model's free-form response (must contain 'yes' or 'no').
  - gt        (str): Ground truth label (must contain 'yes' or 'no').

The optional field 'question' (str) is carried through to the output but
is not required.

Usage:
    Use the default experiment spec (experiment_specs/vlm_bcq.yaml), overriding
    required fields on the command line:

        python vlm_bcq.py predictions_json=<path> results_dir=<path>

    Or point to a custom YAML config file matching the GapAnalysisConfig schema:

        python vlm_bcq.py --config-path <directory> --config-name <filename_without_extension>
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

logger = logging.getLogger(__name__)

KPI_GAPS_JSONL: Final[str] = "kpi_gaps.jsonl"
KPI_GAPS_REPORT: Final[str] = "kpi_gaps_report.txt"

from nvidia_tao_ds.config.rcca.gap_analysis.vlm_bcq import GapAnalysisConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


def extract_yes_no(response: str) -> Optional[str]:
    """Extract a binary 'yes' or 'no' answer from free-form text.

    Uses word-boundary matching so that substrings like "nothing" do not
    match "no". Returns None if the text contains both 'yes' and 'no'
    (ambiguous) or neither.

    Args:
        response: The free-form text to parse.

    Returns:
        'yes', 'no', or None.
    """
    text = response.lower().strip()
    has_yes = re.search(r'\byes\b', text) is not None
    has_no = re.search(r'\bno\b', text) is not None
    if has_yes and has_no:
        return None
    if has_yes:
        return 'yes'
    if has_no:
        return 'no'
    return None


def extract_fp_fn(predictions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify false positive and false negative predictions.

    Compares each prediction's response against its ground truth by
    extracting a binary yes/no from both. Samples where either the
    response or ground truth is ambiguous or unrecognizable are skipped
    with a warning to stderr.

    Args:
        predictions_data: List of dicts. Required keys per item: 'video_id',
            'response', and 'gt' (ground truth). Optional: 'question'.

    Returns:
        List of dicts for FP/FN cases, each containing 'video_id',
        'error_type' ('FP' or 'FN'), 'question', 'ground_truth',
        and 'response'.
    """
    fp_fn_cases = []
    for item in predictions_data:
        video_id = item.get('video_id', '')
        response = item.get('response', '')
        gt_raw = item.get('gt', '')
        question = item.get('question', '')
        extracted = extract_yes_no(response)
        if extracted is None:
            snippet = (response[:80] + "...") if len(response) > 80 else response
            logger.warning(
                "Skipping sample (video_id=%r): response has no single clear 'yes' or 'no' "
                "(missing, or both present). Response snippet: %r", video_id, snippet
            )
            continue
        gt_extracted = extract_yes_no(gt_raw)
        if gt_extracted is None:
            snippet = (gt_raw[:80] + "...") if len(gt_raw) > 80 else gt_raw
            logger.warning(
                "Skipping sample (video_id=%r): ground truth has no single clear 'yes' or 'no' "
                "(missing, or both present). GT snippet: %r", video_id, snippet
            )
            continue
        error_type = None
        if extracted == 'yes' and gt_extracted == 'no':
            error_type = 'FP'
        elif extracted == 'no' and gt_extracted == 'yes':
            error_type = 'FN'
        if error_type:
            fp_fn_cases.append({
                'video_id': video_id,
                'error_type': error_type,
                'question': question,
                'ground_truth': gt_raw,
                'response': response,
            })
    return fp_fn_cases


def write_gap_report(cases: List[Dict[str, Any]], report_path: Path) -> None:
    """Write a formatted table report of FP/FN gap analysis results.

    Args:
        cases: List of FP/FN case dicts, each with an 'error_type' field ('FP' or 'FN').
        report_path: Path to write the report file to.
    """
    n_fp = sum(1 for c in cases if c["error_type"] == "FP")
    n_fn = sum(1 for c in cases if c["error_type"] == "FN")
    col1, col2, col3 = "Error Type", "Count", "Description"
    rows = [
        ("FP", str(n_fp), "model said yes, ground truth no"),
        ("FN", str(n_fn), "model said no, ground truth yes"),
        ("Total", str(len(cases)), ""),
    ]
    w1 = max(len(col1), max(len(r[0]) for r in rows))
    w2 = max(len(col2), max(len(r[1]) for r in rows))
    w3 = max(len(col3), max(len(r[2]) for r in rows))
    sep = f"+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+{'-' * (w3 + 2)}+"
    header = f"| {col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}} |"
    report_lines = [
        "KPI Gaps Report (binary ground truth: yes/no)",
        sep,
        header,
        sep,
    ] + [f"| {r[0]:<{w1}} | {r[1]:<{w2}} | {r[2]:<{w3}} |" for r in rows] + [sep]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


spec_root = os.path.dirname(os.path.abspath(__file__))

@hydra_runner(
    config_path=os.path.join(spec_root, "../experiment_specs"),
    config_name="vlm_bcq",
    schema=GapAnalysisConfig
)
def main(cfg: GapAnalysisConfig):
    """CLI entrypoint for KPI gap analysis.

    Reads a predictions JSON file, extracts false positives and false
    negatives by comparing model responses to ground truth, and writes
    the results to cfg.results_dir:

    - A JSONL file with one object per line, each representing an FP or FN
      case with fields: 'video_id' (absolute path), 'error_type' ('FP' or
      'FN'), 'question', 'ground_truth', and 'response'.
    - A human-readable report with total FP/FN counts.

    Video IDs are resolved to absolute paths using cfg.videos_dir if
    provided, otherwise video_id values in the predictions are used as-is.

    If no FP/FN cases are found, logs a message and returns without writing
    any output files.

    Raises:
        FileNotFoundError: If cfg.predictions_json does not exist.
        ValueError: If predictions_json is not a JSON array, any item is not
            an object, or any item is missing a required field ('gt',
            'response', or 'video_id').
    """
    _log_level = getattr(logging, os.getenv("TAO_LOGGING_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    videos_dir = Path(cfg.videos_dir).resolve() if cfg.videos_dir.strip() else None
    if videos_dir is not None:
        logger.info("Video directory: %s", videos_dir)
    else:
        logger.info("videos_dir is empty; assuming video_id in predictions_json are absolute paths.")

    logger.info("Loading predictions from %s...", cfg.predictions_json)
    with open(cfg.predictions_json, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)
    if not isinstance(predictions_data, list):
        msg = "predictions_json must be a JSON array of prediction objects"
        logger.error(msg)
        raise ValueError(msg)
    for i, item in enumerate(predictions_data):
        if not isinstance(item, dict):
            msg = f"predictions_json item {i} is not an object"
            logger.error(msg)
            raise ValueError(msg)
        for key in ("gt", "response", "video_id"):
            if key not in item:
                msg = f"predictions_json item {i}: missing '{key}'"
                logger.error(msg)
                raise ValueError(msg)

    logger.info("Extracting KPI Gaps...")
    cases = extract_fp_fn(predictions_data)
    if not cases:
        logger.info("No KPI Gaps found!")
        return

    if videos_dir is not None:
        logger.info("Converting video_id paths to absolute paths using base: %s", videos_dir)
        for c in cases:
            c["video_id"] = str((videos_dir / c["video_id"]).resolve())
    else:
        logger.info("Using video_id from predictions as absolute paths.")
        for c in cases:
            c["video_id"] = str(Path(c["video_id"]).resolve())

    gaps_output = Path(cfg.results_dir)
    gaps_output.mkdir(parents=True, exist_ok=True)
    out_path = gaps_output / KPI_GAPS_JSONL
    logger.info("Saving %d cases to %s...", len(cases), out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_fp = sum(1 for c in cases if c["error_type"] == "FP")
    n_fn = sum(1 for c in cases if c["error_type"] == "FN")
    report_path = gaps_output / KPI_GAPS_REPORT
    write_gap_report(cases, report_path)

    logger.info("Summary: total=%d, FP=%d, FN=%d", len(cases), n_fp, n_fn)
    logger.info("Results saved to %s", out_path)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
