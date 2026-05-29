# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 3: double-check verification of step-2 grounding expressions."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    is_valid_output,
    load_records,
    parse_legacy_grounding_file,
    save_records,
    write_legacy_grounding_file,
)
from nvidia_tao_ds.auto_label.image_referring_expression.io_utils import (
    clean_response,
    format_grounding_text,
    parse_grounding_response,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _bbox_key(bbox):
    return tuple(int(v) for v in bbox) if bbox else None


def _reconcile_expressions(original: List[Dict],
                           verified: List[Dict]) -> List[Dict]:
    """Update the original step-2 expressions in place using verified output.

    - An expression whose phrase no longer appears in the verified output
      is dropped (all its bboxes were rejected).
    - A bbox present in the verified output keeps its original id/score;
      a bbox whose coords were updated retains the original id.
    - Each surviving expression is tagged ``verified=True``.
    """
    verified_by_text = {
        expr.get("text", "").strip(): expr for expr in verified
    }

    out: List[Dict] = []
    for expr in original:
        text = expr.get("text", "").strip()
        ver = verified_by_text.get(text)
        if ver is None:
            continue

        ver_bboxes = [inst.get("bbox") for inst in ver.get("instances", [])]
        if not ver_bboxes:
            continue

        original_by_key = {
            _bbox_key(inst.get("bbox")): inst
            for inst in expr.get("instances", [])
        }

        new_instances: List[Dict] = []
        original_instances_list = list(expr.get("instances", []))
        for i, bbox in enumerate(ver_bboxes):
            key = _bbox_key(bbox)
            match = original_by_key.get(key)
            if match is not None:
                new_inst = dict(match)
            elif i < len(original_instances_list):
                new_inst = dict(original_instances_list[i])
                new_inst["bbox"] = [int(v) for v in bbox]
            else:
                new_inst = {"bbox": [int(v) for v in bbox], "bbox_score": 0.9}
            new_instances.append(new_inst)

        new_expr = dict(expr)
        new_expr["instances"] = new_instances
        new_expr["verified"] = True
        out.append(new_expr)
    return out


def run(icfg, vlm_client, prompts, results_dir, step2_file):
    """Run step 3: double-check verification.

    Reads the step-2 ``annotations.jsonl``, renders each record's
    expressions back into the legacy ``<phrase>: [[bbox], ...]`` format
    for the prompt, calls the VLM with the image, parses the verified
    output, and updates ``expressions[]`` in place (dropping rejected
    expressions, tagging survivors as ``verified=True``).

    Args:
        icfg: ``image_referring_expression`` sub-config.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.
        step2_file (str): Path to step-2 annotations.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step_dir = os.path.join(results_dir, "step_3_double_check")
    labels_dir = os.path.join(step_dir, "labels")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(labels_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess
    output_format = icfg.workflow.output_format

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 3 (double_check): %s already exists — skipping.", output_file)
        return output_file

    records = load_records(step2_file) if step2_file else []
    if not records:
        logger.warning("Step 3 (double_check): no step-2 records to process.")
        save_records([], output_file)
        return output_file

    logger.info("Step 3 (double_check): verifying %d records...", len(records))

    template = prompts.get("double_check") or get_prompt("double_check")
    results: List[Optional[Dict]] = [None] * len(records)
    lock = threading.Lock()

    def _process(idx_record):
        idx, record = idx_record
        image_path = record.get("image_path", "")
        stem = os.path.splitext(os.path.basename(image_path))[0]
        legacy_path = os.path.join(labels_dir, f"{stem}.txt.step3")

        out = dict(record)
        out.setdefault("pipeline_steps", list(record.get("pipeline_steps", [])))
        if "step3_double_check" not in out["pipeline_steps"]:
            out["pipeline_steps"].append("step3_double_check")

        original_expressions = record.get("expressions", [])
        if not original_expressions:
            out["expressions"] = []
            results[idx] = out
            if output_format in ("legacy", "both"):
                with lock:
                    write_legacy_grounding_file([], legacy_path)
            return

        if not force and is_valid_output(legacy_path):
            try:
                parsed = parse_legacy_grounding_file(legacy_path)
                out["expressions"] = _reconcile_expressions(
                    original_expressions, parsed,
                )
                results[idx] = out
                return
            except Exception:
                pass

        expr_text = format_grounding_text(original_expressions)
        if not expr_text.strip():
            out["expressions"] = []
            results[idx] = out
            return

        query = template.format(expr=expr_text)

        try:
            response = vlm_client.generate_with_image(image_path, query)
        except Exception as e:
            logger.warning("Step 3 (double_check) VLM call failed for %s: %s", stem, e)
            out["expressions"] = original_expressions
            results[idx] = out
            return

        response = clean_response(response)
        verified_parsed = parse_grounding_response(response)
        updated = _reconcile_expressions(original_expressions, verified_parsed)
        out["expressions"] = updated
        results[idx] = out

        with lock:
            if output_format in ("legacy", "both"):
                write_legacy_grounding_file(updated, legacy_path)

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, (i, r)): i for i, r in enumerate(records)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 3 (double_check) worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(records):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            output_records.append(dict(record))

    save_records(output_records, output_file)
    logger.info("Step 3 (double_check): wrote %d records → %s",
                len(output_records), output_file)
    return output_file
