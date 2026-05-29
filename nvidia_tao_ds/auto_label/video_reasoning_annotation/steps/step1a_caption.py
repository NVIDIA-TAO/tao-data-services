# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1a: Global and dense caption generation using VLM."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    get_processed_videos,
    get_videos_from_dir,
    get_videos_from_jsonl,
    load_mode_map,
    resolve_mode,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.video_utils import get_video_length_sec


def _process_single_video(video_path, vlm_client, prompts, mode, max_video_length_sec):
    """Generate global and dense captions for a single video."""
    try:
        video_length = get_video_length_sec(video_path)
        if max_video_length_sec and video_length and video_length > max_video_length_sec:
            logger.info("Skipping %s: %.0fs > %ds", video_path, video_length, max_video_length_sec)
            return None

        global_key = f"{mode}_global_caption"
        dense_key = f"{mode}_dense_caption"

        global_prompt = prompts.get(global_key) or get_prompt(global_key)
        dense_prompt = prompts.get(dense_key) or get_prompt(dense_key)

        global_caption = vlm_client.generate_with_video(video_path, global_prompt)
        dense_caption = vlm_client.generate_with_video(video_path, dense_prompt)

        if not global_caption or not dense_caption:
            return None

        return {
            "video": video_path,
            "video_length": video_length,
            "mode": mode,
            "global_caption": global_caption,
            "dense_caption": dense_caption,
        }
    except Exception as e:
        logger.warning("Error processing %s: %s", video_path, e)
        return None


def run(vcfg, vlm_client, prompts, results_dir):
    """Run step 1a: global and dense caption generation.

    Produces a global narrative caption and a timestamped dense caption
    for each video. Results are written to ``step_1a_caption/captions.jsonl``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``data`` and ``workflow``
            sections.
        vlm_client (LLMClient): VLM client used for video-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    step_dir = os.path.join(results_dir, "step_1a_caption")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "captions.jsonl")

    global_mode = vcfg.workflow.mode
    mode_map = load_mode_map(results_dir)
    max_video_length_sec = vcfg.workflow.max_video_length_sec

    # Gather video paths from step 0 output or raw inputs
    video_paths = []
    step0_output = os.path.join(results_dir, "step_0_filter", "filter_results.jsonl")
    if os.path.exists(step0_output):
        video_paths.extend(get_videos_from_jsonl(step0_output, filter_field="is_valid"))
    else:
        if vcfg.data.video_root:
            video_paths.extend(get_videos_from_dir(vcfg.data.video_root))
        for jsonl_path in vcfg.data.input_jsonl_files:
            video_paths.extend(get_videos_from_jsonl(jsonl_path, vcfg.data.filter_field))

    processed = get_processed_videos(output_file)
    to_process = [v for v in video_paths if v not in processed]

    if not to_process:
        logger.info("Step 1a: No new videos to caption.")
        return

    logger.info("Step 1a: Captioning %d videos (mode=%s)...", len(to_process), global_mode)

    with ThreadPoolExecutor(max_workers=vcfg.workflow.max_workers) as executor:
        futures = {}
        for vp in to_process:
            mode = resolve_mode({"video": vp}, global_mode, mode_map)
            futures[executor.submit(
                _process_single_video, vp, vlm_client, prompts,
                mode, max_video_length_sec,
            )] = vp
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 1a: Done. Results at %s", output_file)
