# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 3: QA generation for 10 tasks with reasoning traces."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    format_chunk_captions,
    get_entries_from_jsonl,
    get_processed_video_prompt_keys,
    save_result_to_jsonl,
)
from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


_MODE_INDEPENDENT_TYPES = frozenset({"scene_description", "event_summary"})


def _build_prompt_keys(mode, qa_types):
    """Build the list of prompt keys from mode and QA type configuration.

    Most QA types are mode-dependent (e.g. ``anomaly_mcq``, ``normal_bcq``).
    Scene description and event summary are mode-independent — they use a
    single prompt regardless of anomaly/normal classification.
    """
    keys = []
    for qt in qa_types:
        if qt in _MODE_INDEPENDENT_TYPES:
            keys.append(qt)
        else:
            keys.append(f"{mode}_{qt}")
    return keys


def _process_single_video(entry, prompt_key, llm_client, prompts):
    """Generate QA for a single video entry."""
    video_name = entry.get("video", "Unknown")
    global_caption = (
        entry.get("global_caption", "") or
        entry.get("original_captions", {}).get("global_caption", "")
    )
    dense_caption = (
        entry.get("dense_caption", "") or
        entry.get("original_captions", {}).get("dense_caption", "")
    )
    chunk_captions = entry.get("chunk_captions", [])
    video_length = entry.get("video_length", 0)
    description = entry.get("detailed_description", "")

    chunk_captions_str = format_chunk_captions(chunk_captions) if chunk_captions else "N/A"

    fmt_kwargs = {
        "video_length": video_length,
        "global_caption": global_caption,
        "dense_caption": dense_caption,
        "chunk_captions_str": chunk_captions_str,
    }
    if description:
        fmt_kwargs["step_2_output"] = description

    try:
        prompt = prompts.get(prompt_key) or get_prompt(prompt_key, **fmt_kwargs)
        if "{video_length}" in prompt:
            prompt = prompt.format(**fmt_kwargs)
    except KeyError:
        prompt = get_prompt(
            prompt_key,
            video_length=video_length,
            global_caption=global_caption,
            dense_caption=dense_caption,
        )

    try:
        output = llm_client.generate_text(prompt)
    except Exception as e:
        logger.warning("Error generating QA for %s [%s]: %s", video_name, prompt_key, e)
        return None

    if not output:
        return None

    result = {
        "video": video_name,
        "video_length": video_length,
        "qa_output": output,
        "prompt_key": prompt_key,
        "original_captions": {
            "global_caption": global_caption,
            "dense_caption": dense_caption,
        },
    }
    if description:
        result["detailed_description"] = description
    return result


def run(vcfg, llm_client, prompts, results_dir):
    """Run step 3: generate QA pairs (MCQ, binary, open-ended) with reasoning.

    Merges outputs from all prior steps and produces QA output for each
    configured QA type. Results are written to ``step_3_qa/qa_output.jsonl``.

    Args:
        vcfg (object): Video reasoning annotation sub-config with ``workflow`` section
            (including ``qa_types`` list).
        llm_client (LLMClient): Text-only LLM client for QA generation.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    step_dir = os.path.join(results_dir, "step_3_qa")
    os.makedirs(step_dir, exist_ok=True)
    output_file = os.path.join(step_dir, "qa_output.jsonl")

    global_mode = vcfg.workflow.mode
    qa_types = list(vcfg.workflow.qa_types)

    # Merge entries from all prior steps.
    # Later steps override earlier ones per video.
    entries_by_video = {}
    for candidate in [
        os.path.join(results_dir, "step_1a_caption", "captions.jsonl"),
        os.path.join(results_dir, "step_1b_chunks", "chunk_captions.jsonl"),
        os.path.join(results_dir, "step_1c_highlight", "highlight_captions.jsonl"),
        os.path.join(results_dir, "step_2_description", "descriptions.jsonl"),
    ]:
        if os.path.exists(candidate):
            for e in get_entries_from_jsonl(candidate):
                video = e.get("video")
                if video:
                    entries_by_video.setdefault(video, {}).update(e)

    if not entries_by_video:
        logger.warning("Step 3: No input from prior steps.")
        return

    entries = list(entries_by_video.values())
    processed_keys = get_processed_video_prompt_keys(output_file)

    tasks = []
    for entry in entries:
        if not entry.get("video"):
            continue
        mode = entry.get("mode", global_mode)
        if mode == "auto":
            mode = "normal"
        for pk in _build_prompt_keys(mode, qa_types):
            if (entry["video"], pk) not in processed_keys:
                tasks.append((entry, pk))

    if not tasks:
        logger.info("Step 3: No new tasks to process.")
        return

    logger.info("Step 3: Processing %d tasks across %d videos...", len(tasks), len(entries))

    with ThreadPoolExecutor(max_workers=vcfg.workflow.max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_video, entry, pk, llm_client, prompts,
            ): (entry.get("video", "Unknown"), pk)
            for entry, pk in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                save_result_to_jsonl(result, output_file)

    logger.info("Step 3: Done. Results at %s", output_file)
