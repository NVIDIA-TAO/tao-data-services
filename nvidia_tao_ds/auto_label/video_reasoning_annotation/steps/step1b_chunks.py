# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1b: Chunk caption generation — split video into chunks and caption each."""

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    get_entries_from_jsonl,
    get_processed_videos,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.video_utils import (
    get_video_length_sec,
    select_chunk_duration,
    split_video_into_chunks,
)


def _caption_single_chunk(chunk_path, vlm_client, chunk_query):
    """Caption a single video chunk."""
    try:
        return vlm_client.generate_with_video(chunk_path, chunk_query)
    except Exception as e:
        logger.warning("Error captioning chunk %s: %s", chunk_path, e)
        return ""


def _process_video_entry(entry, vlm_client, prompts, vcfg, temp_dir):
    """Process a single video: split into chunks and caption each."""
    video_path = entry.get("video")
    if not video_path or not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return None

    mode = entry.get("mode", vcfg.workflow.mode)
    if mode == "auto":
        mode = "normal"
    video_length = get_video_length_sec(video_path)
    chunk_duration = select_chunk_duration(
        video_length,
        options=vcfg.workflow.chunk_duration_options,
        max_chunks=vcfg.workflow.max_chunks,
    )

    prompt_key = f"{mode}_chunk_caption" if f"{mode}_chunk_caption" in prompts else "anomaly_chunk_caption"
    chunk_query = prompts.get(prompt_key) or get_prompt(prompt_key, chunk_duration=chunk_duration)
    if "{chunk_duration}" in chunk_query:
        chunk_query = chunk_query.format(chunk_duration=chunk_duration)

    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_temp_dir = os.path.join(temp_dir, f"{video_name}_{path_hash}")

    chunks = split_video_into_chunks(video_path, video_temp_dir, chunk_duration)

    if video_length is None:
        video_length = len(chunks) * chunk_duration

    # Caption chunks in parallel
    chunk_captions = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_idx = {
            executor.submit(_caption_single_chunk, cp, vlm_client, chunk_query): i
            for i, cp in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            caption_text = future.result()
            ts_start = idx * chunk_duration
            ts_end = min((idx + 1) * chunk_duration, video_length) if video_length else (idx + 1) * chunk_duration
            chunk_captions[idx] = {
                "chunk_index": idx,
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "caption": caption_text,
            }

    # Validate all chunks have captions
    if any(not c or not c.get("caption") for c in chunk_captions):
        logger.warning("Empty chunk captions for %s, skipping.", video_path)
        if os.path.exists(video_temp_dir):
            shutil.rmtree(video_temp_dir)
        return None

    entry["chunk_captions"] = chunk_captions

    if os.path.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir)

    return entry


def run(vcfg, vlm_client, prompts, results_dir):
    """Run step 1b: split videos into temporal chunks and caption each.

    Reads step 1a output, splits each video into fixed-duration chunks,
    and generates a per-chunk caption via the VLM. Results are written
    to ``step_1b_chunks/chunk_captions.jsonl``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``data`` and ``workflow``
            sections (including ``chunk_duration_options`` and ``max_chunks``).
        vlm_client (LLMClient): VLM client used for video-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    step_dir = os.path.join(results_dir, "step_1b_chunks")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "chunk_captions.jsonl")
    temp_dir = os.path.join(step_dir, "temp_video_chunks")

    # Read input from step 1a
    input_file = os.path.join(results_dir, "step_1a_caption", "captions.jsonl")
    if not os.path.exists(input_file):
        logger.warning("Step 1b: No input from step 1a at %s", input_file)
        return

    entries = get_entries_from_jsonl(input_file)
    processed = get_processed_videos(output_file)
    to_process = [e for e in entries if e.get("video") and e["video"] not in processed]

    if not to_process:
        logger.info("Step 1b: No new videos to process.")
        return

    logger.info("Step 1b: Processing %d videos for chunk captions...", len(to_process))

    with ThreadPoolExecutor(max_workers=vcfg.workflow.max_workers) as executor:
        futures = {
            executor.submit(_process_video_entry, e, vlm_client, prompts, vcfg, temp_dir): e
            for e in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 1b: Done. Results at %s", output_file)
