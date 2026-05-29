# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 0: Video filtering and anomaly classification.

Sub-steps:
  0a — Domain filter: is the video suitable for analysis?
  0b — Anomaly classification: does the video contain an anomaly or normal activity?
       Only runs when workflow.mode is "auto".
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    get_processed_videos,
    get_videos_from_dir,
    get_videos_from_jsonl,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _filter_single_video(video_path, vlm_client, prompts):
    """Classify a single video for domain suitability. Returns a result dict."""
    try:
        prompt = prompts.get("video_filtering") or get_prompt("video_filtering")
        response = vlm_client.generate_with_video(video_path, prompt)
        is_valid = response.strip().lower().startswith("yes")
        return {
            "video_path": video_path,
            "video": video_path,
            "is_valid": is_valid,
            "raw_response": response.strip(),
        }
    except Exception as e:
        logger.warning("Error filtering %s: %s", video_path, e)
        return None


def _classify_anomaly(video_path, vlm_client, prompts):
    """Classify a video as anomaly or normal. Returns 'anomaly' or 'normal'."""
    try:
        prompt = prompts.get("video_anomaly_classification") or get_prompt(
            "video_anomaly_classification"
        )
        response = vlm_client.generate_with_video(video_path, prompt)
        last_line = response.strip().splitlines()[-1].strip().lower()
        logger.info(
            "Step 0 classification reasoning for %s:\n%s",
            os.path.basename(video_path), response.strip(),
        )
        if last_line.startswith("anomaly"):
            return "anomaly"
        return "normal"
    except Exception as e:
        logger.warning("Error classifying %s: %s", video_path, e)
        return "normal"


def _process_single_video(video_path, vlm_client, prompts, auto_mode):
    """Filter + optionally classify a single video."""
    result = _filter_single_video(video_path, vlm_client, prompts)
    if result is None:
        return None
    if auto_mode and result["is_valid"]:
        result["mode"] = _classify_anomaly(video_path, vlm_client, prompts)
        logger.info(
            "Step 0: %s -> %s", os.path.basename(video_path), result["mode"],
        )
    return result


def run(vcfg, vlm_client, prompts, results_dir):
    """Run step 0: video filtering and optional anomaly classification.

    Filters videos for domain suitability and, when the workflow mode is
    ``"auto"``, classifies each valid video as anomaly or normal.
    Results are written to ``step_0_filter/filter_results.jsonl``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``data`` and ``workflow``
            sections.
        vlm_client (LLMClient): VLM client used for video-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    step_dir = os.path.join(results_dir, "step_0_filter")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "filter_results.jsonl")

    auto_mode = vcfg.workflow.mode == "auto"

    video_paths = []
    if vcfg.data.video_root:
        video_paths.extend(get_videos_from_dir(vcfg.data.video_root))
    for jsonl_path in vcfg.data.input_jsonl_files:
        video_paths.extend(get_videos_from_jsonl(jsonl_path, vcfg.data.filter_field))

    processed = get_processed_videos(output_file)
    to_process = [v for v in video_paths if v not in processed]

    if not to_process:
        logger.info("Step 0: No new videos to filter.")
        return

    logger.info(
        "Step 0: Filtering %d videos%s...",
        len(to_process),
        " (with anomaly classification)" if auto_mode else "",
    )
    max_workers = vcfg.workflow.max_workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_video, vp, vlm_client, prompts, auto_mode,
            ): vp
            for vp in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 0: Done. Results at %s", output_file)
