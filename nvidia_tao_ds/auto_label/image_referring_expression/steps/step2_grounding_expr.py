# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 2: grounding expression generation (merges step 0 regions + step 1 caption)."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    IdCounter,
    is_valid_output,
    load_records,
    make_expression,
    make_instance,
    merge_records,
    parse_legacy_grounding_file,
    save_records,
    write_legacy_grounding_file,
)
from nvidia_tao_ds.auto_label.image_referring_expression.io_utils import (
    clean_response,
    parse_grounding_response,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _format_region_chunks(regions: List[Dict]) -> List[str]:
    """Convert step-0 regions into ``<desc>: [x1,y1,x2,y2]`` chunks for the prompt."""
    chunks: List[str] = []
    for item in regions:
        bbox = item.get("bbox") or item.get("bbox_2d") or []
        if item.get("description"):
            chunks.append(f"{item['description']}:{list(bbox)}")
        elif item.get("color") and item.get("type"):
            chunks.append(f"{item['color']} {item['type']}:{list(bbox)}")
        else:
            chunks.append(str(list(bbox)))
    return chunks


def run(icfg, vlm_client, prompts, results_dir, step0_file=None, step1_file=None):
    """Run step 2: grounding expression generation.

    Merges step-0 and step-1 outputs (inner-joined on ``image_id``),
    builds a prompt that supplies both the region list and the caption
    (when available), calls the VLM with the image, and parses the
    resulting ``<phrase>: [[bbox], ...]`` lines into the unified
    ``expressions[]`` schema.

    Args:
        icfg: ``image_referring_expression`` sub-config.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.
        step0_file (str | None): Path to step-0 annotations.
        step1_file (str | None): Path to step-1 annotations.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step_dir = os.path.join(results_dir, "step_2_grounding_expr")
    labels_dir = os.path.join(step_dir, "labels")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(labels_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess
    output_format = icfg.workflow.output_format

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 2 (grounding_expr): %s already exists — skipping.", output_file)
        return output_file

    step0_records = load_records(step0_file) if step0_file else []
    step1_records = load_records(step1_file) if step1_file else []

    if not step0_records:
        logger.warning("Step 2 (grounding_expr): no step-0 records to process.")
        save_records([], output_file)
        return output_file

    merged = merge_records(step0_records, step1_records)
    logger.info("Step 2 (grounding_expr): processing %d merged records...", len(merged))

    template = prompts.get("grounding_expr") or get_prompt("grounding_expr")
    results: List[Optional[Dict]] = [None] * len(merged)
    lock = threading.Lock()
    expr_counter = IdCounter("expr")
    box_counter = IdCounter("box")

    def _process(idx_record):
        idx, record = idx_record
        image_path = record.get("image_path", "")
        stem = os.path.splitext(os.path.basename(image_path))[0]
        legacy_path = os.path.join(labels_dir, f"{stem}.txt.step2")

        out = dict(record)
        out.setdefault("pipeline_steps", list(record.get("pipeline_steps", [])))
        if "step2_grounding_expr" not in out["pipeline_steps"]:
            out["pipeline_steps"].append("step2_grounding_expr")

        if not force and is_valid_output(legacy_path):
            try:
                parsed = parse_legacy_grounding_file(legacy_path)
                out["expressions"] = _rehydrate_expressions(
                    parsed, expr_counter, box_counter,
                )
                results[idx] = out
                return
            except Exception:
                pass

        regions = record.get("regions", [])
        if not regions:
            out["expressions"] = []
            results[idx] = out
            if output_format in ("legacy", "both"):
                with lock:
                    write_legacy_grounding_file([], legacy_path)
            return

        region_chunks = _format_region_chunks(regions)
        caption = record.get("caption", "")
        caption_section = f"caption: {caption}" if caption else ""

        query = template.format(
            bboxes=region_chunks,
            caption_section=caption_section,
        )

        try:
            response = vlm_client.generate_with_image(image_path, query)
        except Exception as e:
            logger.warning("Step 2 (grounding_expr) VLM call failed for %s: %s", stem, e)
            out["expressions"] = []
            results[idx] = out
            return

        response = clean_response(response)
        parsed_exprs = parse_grounding_response(response)
        expressions = _rehydrate_expressions(parsed_exprs, expr_counter, box_counter)
        out["expressions"] = expressions
        results[idx] = out

        with lock:
            if output_format in ("legacy", "both"):
                write_legacy_grounding_file(expressions, legacy_path)

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, (i, r)): i for i, r in enumerate(merged)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 2 (grounding_expr) worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(merged):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            r = dict(record)
            r.setdefault("expressions", [])
            output_records.append(r)

    save_records(output_records, output_file)
    logger.info("Step 2 (grounding_expr): wrote %d records → %s",
                len(output_records), output_file)
    return output_file


def _rehydrate_expressions(parsed_exprs: List[Dict],
                           expr_counter: IdCounter,
                           box_counter: IdCounter) -> List[Dict]:
    """Turn ``[{text, instances:[{bbox}]}, ...]`` into fully-ID'd expressions."""
    out: List[Dict] = []
    for expr in parsed_exprs:
        text = expr.get("text", "").strip()
        if not text:
            continue
        instances = []
        for inst in expr.get("instances", []):
            bbox = inst.get("bbox")
            if not bbox:
                continue
            instances.append(make_instance(
                bbox=bbox,
                score=inst.get("bbox_score", 0.9),
                bbox_id=box_counter.next(),
            ))
        out.append(make_expression(
            text=text,
            expression_id=expr_counter.next(),
            instances=instances,
        ))
    return out
