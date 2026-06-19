# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1c: Anomaly highlight chunk — extract and caption the anomaly moment."""

import hashlib
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    format_chunk_captions,
    get_entries_from_jsonl,
    get_processed_videos,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.video_utils import (
    extract_highlight_clip,
    get_video_length_sec,
)


def _extract_anomaly_timestamp(entry, llm_client, prompts):
    """Use LLM (text-to-text) to identify anomaly timestamp from captions."""
    global_caption = entry.get("global_caption", "")
    dense_caption = entry.get("dense_caption", "")
    chunk_captions = entry.get("chunk_captions", [])
    chunk_captions_str = format_chunk_captions(chunk_captions) if chunk_captions else "N/A"

    prompt = prompts.get("highlight_timestamp_extraction") or get_prompt(
        "highlight_timestamp_extraction",
        global_caption=global_caption,
        dense_caption=dense_caption,
        chunk_captions_str=chunk_captions_str,
    )
    if "{global_caption}" in prompt:
        prompt = prompt.format(
            global_caption=global_caption,
            dense_caption=dense_caption,
            chunk_captions_str=chunk_captions_str,
        )

    response = llm_client.generate_text(prompt)
    if response:
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            return float(match.group(1))
        logger.warning("Could not parse timestamp from: %s", response)
    return None


def _process_video_entry(entry, vlm_client, llm_client, prompts, vcfg, temp_dir):
    """Full pipeline for one video: extract timestamp -> cut clip -> caption."""
    video_path = entry.get("video")
    if not video_path or not os.path.exists(video_path):
        logger.warning("Video not found: %s", video_path)
        return None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    video_length = get_video_length_sec(video_path) or entry.get("video_length") or 0

    anomaly_time = _extract_anomaly_timestamp(entry, llm_client, prompts)
    if anomaly_time is None:
        logger.warning("Could not identify anomaly timestamp for %s.", video_name)
        entry["highlight_chunk"] = None
        return entry

    anomaly_time = max(0, min(anomaly_time, video_length))
    logger.info("Anomaly at %.1fs for %s (video: %.1fs)", anomaly_time, video_name, video_length)

    highlight_dir = os.path.join(temp_dir, f"{video_name}_{path_hash}_highlight")
    highlight_path = os.path.join(highlight_dir, "highlight.webm")

    before_sec = vcfg.workflow.highlight_before_sec
    after_sec = vcfg.workflow.highlight_after_sec

    clip_path, start_time, end_time = extract_highlight_clip(
        video_path, anomaly_time, highlight_path,
        before_sec=before_sec, after_sec=after_sec,
    )
    if clip_path is None:
        logger.warning("Failed to extract highlight clip for %s.", video_name)
        entry["highlight_chunk"] = None
        if os.path.exists(highlight_dir):
            shutil.rmtree(highlight_dir)
        return entry

    duration = end_time - start_time
    highlight_prompt = prompts.get("highlight_chunk_caption") or get_prompt(
        "highlight_chunk_caption",
        anomaly_time=f"{anomaly_time:.1f}",
        start_time=f"{start_time:.1f}",
        end_time=f"{end_time:.1f}",
        duration=f"{duration:.1f}",
    )
    if "{anomaly_time}" in highlight_prompt:
        highlight_prompt = highlight_prompt.format(
            anomaly_time=f"{anomaly_time:.1f}",
            start_time=f"{start_time:.1f}",
            end_time=f"{end_time:.1f}",
            duration=f"{duration:.1f}",
        )

    try:
        caption_text = vlm_client.generate_with_video(clip_path, highlight_prompt)
    except Exception as e:
        logger.warning("Failed to caption highlight clip for %s: %s", video_name, e)
        caption_text = ""

    if os.path.exists(highlight_dir):
        shutil.rmtree(highlight_dir)

    if not caption_text:
        return None

    entry["highlight_chunk"] = {
        "anomaly_timestamp_sec": anomaly_time,
        "timestamp_start": start_time,
        "timestamp_end": end_time,
        "caption": caption_text,
    }
    return entry


def run(vcfg, vlm_client, llm_client, prompts, results_dir):
    """Run step 1c: extract and caption anomaly highlight clips.

    For each anomaly-mode video, uses the LLM to identify the anomaly
    timestamp from existing captions, extracts a short highlight clip
    around that moment, and captions it with the VLM. Skips entirely
    when the global mode is ``"normal"``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``data`` and ``workflow``
            sections (including ``highlight_before_sec`` and
            ``highlight_after_sec``).
        vlm_client (LLMClient): VLM client for captioning the highlight clip.
        llm_client (LLMClient): Text-only LLM client for anomaly timestamp
            extraction.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    global_mode = vcfg.workflow.mode
    if global_mode == "normal":
        logger.info("Step 1c: Skipping (mode=normal, highlight is anomaly-only).")
        return

    step_dir = os.path.join(results_dir, "step_1c_highlight")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "highlight_captions.jsonl")
    temp_dir = os.path.join(step_dir, "temp_highlights")

    # Prefer step 1b output (has chunk captions), fallback to 1a
    input_file = os.path.join(results_dir, "step_1b_chunks", "chunk_captions.jsonl")
    if not os.path.exists(input_file):
        input_file = os.path.join(results_dir, "step_1a_caption", "captions.jsonl")
    if not os.path.exists(input_file):
        logger.warning("Step 1c: No input found.")
        return

    entries = get_entries_from_jsonl(input_file)
    processed = get_processed_videos(output_file)
    to_process = [
        e for e in entries
        if e.get("video") and
        e["video"] not in processed and
        e.get("mode", global_mode) == "anomaly"
    ]

    if not to_process:
        logger.info("Step 1c: No anomaly videos to process.")
        return

    logger.info("Step 1c: Processing %d anomaly videos for highlight captions...", len(to_process))

    with ThreadPoolExecutor(max_workers=vcfg.workflow.max_workers) as executor:
        futures = {
            executor.submit(
                _process_video_entry, e, vlm_client, llm_client, prompts, vcfg, temp_dir
            ): e
            for e in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 1c: Done. Results at %s", output_file)
