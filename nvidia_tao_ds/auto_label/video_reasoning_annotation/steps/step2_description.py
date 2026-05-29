# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 2: Description synthesis — combine captions into structured descriptions."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    format_chunk_captions,
    get_entries_from_jsonl,
    get_processed_videos,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _generate_description(entry, llm_client, prompts, global_mode):
    """Generate a cohesive description from captions (text-to-text)."""
    mode = entry.get("mode", global_mode)
    if mode == "auto":
        mode = "normal"
    video_name = entry.get("video", "Unknown")
    global_caption = entry.get("global_caption", "")
    dense_caption = entry.get("dense_caption", "")
    chunk_captions = entry.get("chunk_captions", [])
    video_length = entry.get("video_length", 0)
    highlight_chunk = entry.get("highlight_chunk")

    chunk_captions_str = format_chunk_captions(chunk_captions) if chunk_captions else "N/A"
    dense_section = f"[Dense Caption]\n{dense_caption}"

    prompt_key = f"{mode}_description"

    prompt_kwargs = {
        "video_length": video_length,
        "global_caption": global_caption,
        "dense_section": dense_section,
        "chunk_captions_str": chunk_captions_str,
    }

    if highlight_chunk and highlight_chunk.get("caption"):
        start = highlight_chunk.get("timestamp_start", 0)
        end = highlight_chunk.get("timestamp_end", 0)
        anomaly = highlight_chunk.get("anomaly_timestamp_sec", 0)
        prompt_kwargs["highlight_section"] = (
            f"[Highlight Chunk Caption ({start:.1f}s - {end:.1f}s, anomaly at {anomaly:.1f}s)]\n"
            f"{highlight_chunk['caption']}"
        )
    else:
        prompt_kwargs["highlight_section"] = ""

    try:
        prompt = prompts.get(prompt_key) or get_prompt(prompt_key, **prompt_kwargs)
        if "{video_length}" in prompt:
            prompt = prompt.format(**prompt_kwargs)
    except (KeyError, IndexError):
        prompt = get_prompt(prompt_key, **prompt_kwargs)

    try:
        description = llm_client.generate_text(prompt)
    except Exception as e:
        logger.warning("Error generating description for %s: %s", video_name, e)
        return None

    if not description:
        return None

    result = {
        "video": video_name,
        "video_length": video_length,
        "mode": mode,
        "prompt_key": prompt_key,
        "detailed_description": description,
        "original_captions": {
            "global_caption": global_caption,
            "dense_caption": dense_caption,
        },
    }
    if chunk_captions:
        result["chunk_captions"] = chunk_captions
    return result


def run(vcfg, llm_client, prompts, results_dir):
    """Run step 2: synthesize structured descriptions from captions.

    Merges outputs from steps 1a/1b/1c and generates a cohesive,
    multi-part video description using the text-only LLM. Results are
    written to ``step_2_description/descriptions.jsonl``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``workflow`` section.
        llm_client (LLMClient): Text-only LLM client for description
            generation.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    step_dir = os.path.join(results_dir, "step_2_description")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "descriptions.jsonl")

    global_mode = vcfg.workflow.mode

    # Merge entries from all caption steps.
    # Later steps override earlier ones (1c > 1b > 1a) per video.
    entries_by_video = {}
    for candidate in [
        os.path.join(results_dir, "step_1a_caption", "captions.jsonl"),
        os.path.join(results_dir, "step_1b_chunks", "chunk_captions.jsonl"),
        os.path.join(results_dir, "step_1c_highlight", "highlight_captions.jsonl"),
    ]:
        if os.path.exists(candidate):
            for e in get_entries_from_jsonl(candidate):
                video = e.get("video")
                if video:
                    entries_by_video.setdefault(video, {}).update(e)

    if not entries_by_video:
        logger.warning("Step 2: No input from prior caption steps.")
        return

    entries = list(entries_by_video.values())
    processed = get_processed_videos(output_file)
    to_process = [e for e in entries if e.get("video") and e["video"] not in processed]

    if not to_process:
        logger.info("Step 2: No new videos to process.")
        return

    logger.info("Step 2: Generating descriptions for %d videos...", len(to_process))

    with ThreadPoolExecutor(max_workers=vcfg.workflow.max_workers) as executor:
        futures = {
            executor.submit(
                _generate_description, e, llm_client, prompts, global_mode,
            ): e
            for e in to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 2: Done. Results at %s", output_file)
