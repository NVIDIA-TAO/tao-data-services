# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1: image caption generation."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    is_valid_output,
    save_records,
    write_legacy_caption_file,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def run(icfg, vlm_client, prompts, results_dir, seed_records):
    """Run step 1: image captioning.

    Calls the VLM with each image and the (image-only) caption prompt,
    storing the result as the ``caption`` field on each record. If
    ``output_format in {legacy, both}``, a plain-text ``.txt.step1`` is
    also written per image.

    Args:
        icfg: ``image_referring_expression`` sub-config.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.
        seed_records (list[dict]): Seed records from the orchestrator.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step_dir = os.path.join(results_dir, "step_1_image_caption")
    labels_dir = os.path.join(step_dir, "labels")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(labels_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess
    output_format = icfg.workflow.output_format

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 1 (image_caption): %s already exists — skipping.", output_file)
        return output_file

    if not seed_records:
        logger.warning("Step 1 (image_caption): no seed records.")
        save_records([], output_file)
        return output_file

    logger.info("Step 1 (image_caption): processing %d images...", len(seed_records))

    template = prompts.get("image_caption") or get_prompt("image_caption")
    query = template.strip()
    results: List[Optional[Dict]] = [None] * len(seed_records)
    lock = threading.Lock()

    def _process(idx_record):
        idx, record = idx_record
        image_path = record.get("image_path", "")
        stem = os.path.splitext(os.path.basename(image_path))[0]
        legacy_path = os.path.join(labels_dir, f"{stem}.txt.step1")

        out = dict(record)
        out.setdefault("pipeline_steps", list(record.get("pipeline_steps", [])))
        if "step1_image_caption" not in out["pipeline_steps"]:
            out["pipeline_steps"].append("step1_image_caption")

        if not force and is_valid_output(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    out["caption"] = f.read().strip()
                results[idx] = out
                return
            except Exception:
                pass

        try:
            response = vlm_client.generate_with_image(image_path, query)
        except Exception as e:
            logger.warning("Step 1 (image_caption) VLM call failed for %s: %s", stem, e)
            out["caption"] = ""
            results[idx] = out
            return

        caption = (response or "").strip()
        out["caption"] = caption
        results[idx] = out

        with lock:
            if output_format in ("legacy", "both"):
                write_legacy_caption_file(caption, legacy_path)

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, (i, r)): i for i, r in enumerate(seed_records)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 1 (image_caption) worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(seed_records):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            r = dict(record)
            r.setdefault("caption", "")
            output_records.append(r)

    save_records(output_records, output_file)
    logger.info("Step 1 (image_caption): wrote %d records → %s",
                len(output_records), output_file)
    return output_file
