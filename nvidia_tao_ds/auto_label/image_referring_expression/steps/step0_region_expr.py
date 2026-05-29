# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 0: region expression generation from KITTI labels."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    is_valid_output,
    parse_legacy_region_file,
    save_records,
    write_legacy_region_file,
)
from nvidia_tao_ds.auto_label.image_referring_expression.io_utils import (
    clean_response,
    format_bboxes,
    parse_regions_response,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _write_output_format(output_format: str, legacy_path: str,
                         response: str, regions: List[Dict]) -> None:
    """Optionally dump legacy-format ``.txt.step0`` alongside the JSONL."""
    if output_format in ("legacy", "both"):
        if regions:
            write_legacy_region_file(regions, legacy_path)
        else:
            os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
            with open(legacy_path, "w", encoding="utf-8") as f:
                f.write(response)


def run(icfg, vlm_client, prompts, results_dir, seed_records):
    """Run step 0: region expression generation.

    For each seed record with ``kitti_bboxes``, build a prompt that lists
    the normalized bboxes, call the VLM with the image, parse the
    resulting regions (JSON array or NDJSON), and write unified
    ``annotations.jsonl`` (and, if ``output_format in {legacy, both}``,
    the byte-compatible ``step_0_region_expr/labels/<stem>.txt.step0``).

    Args:
        icfg: ``image_referring_expression`` sub-config.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.
        seed_records (list[dict]): Seed records from the orchestrator.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step_dir = os.path.join(results_dir, "step_0_region_expr")
    labels_dir = os.path.join(step_dir, "labels")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(labels_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess
    output_format = icfg.workflow.output_format

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 0 (region_expr): %s already exists — skipping.", output_file)
        return output_file

    if not seed_records:
        logger.warning("Step 0 (region_expr): no seed records.")
        save_records([], output_file)
        return output_file

    logger.info("Step 0 (region_expr): processing %d images...", len(seed_records))

    results: List[Optional[Dict]] = [None] * len(seed_records)
    template = prompts.get("region_expr") or get_prompt("region_expr")
    lock = threading.Lock()

    def _process(idx_record):
        idx, record = idx_record
        image_path = record.get("image_path", "")
        stem = os.path.splitext(os.path.basename(image_path))[0]
        legacy_path = os.path.join(labels_dir, f"{stem}.txt.step0")

        out = dict(record)
        out.setdefault("pipeline_steps", list(record.get("pipeline_steps", [])))
        if "step0_region_expr" not in out["pipeline_steps"]:
            out["pipeline_steps"].append("step0_region_expr")

        if not force and is_valid_output(legacy_path):
            try:
                out["regions"] = parse_legacy_region_file(legacy_path)
                results[idx] = out
                return
            except Exception:
                pass

        kitti_bboxes = record.get("kitti_bboxes", [])
        if not kitti_bboxes:
            out["regions"] = []
            results[idx] = out
            _write_output_format(output_format, legacy_path, "[]", [])
            return

        width = int(record.get("width") or 1920)
        height = int(record.get("height") or 1080)
        bboxes_str = format_bboxes(kitti_bboxes, height, width)
        query = template.format(bboxes=bboxes_str)

        try:
            response = vlm_client.generate_with_image(image_path, query)
        except Exception as e:
            logger.warning("Step 0 (region_expr) VLM call failed for %s: %s", stem, e)
            out["regions"] = []
            results[idx] = out
            return

        response = clean_response(response)
        regions = parse_regions_response(response)
        out["regions"] = regions
        results[idx] = out

        with lock:
            _write_output_format(output_format, legacy_path, response, regions)

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, (i, r)): i for i, r in enumerate(seed_records)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 0 (region_expr) worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(seed_records):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            r = dict(record)
            r["regions"] = []
            output_records.append(r)

    save_records(output_records, output_file)
    logger.info("Step 0 (region_expr): wrote %d records → %s",
                len(output_records), output_file)
    return output_file
