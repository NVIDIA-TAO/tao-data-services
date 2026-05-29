# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 1: phrase grounding — produce bboxes for each expression."""

import ast
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    IdCounter,
    clamp_bbox,
    load_records,
    make_instance,
    save_records,
)
from nvidia_tao_ds.auto_label.image_grounding.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _build_query(template: str, expressions: List[Dict]) -> str:
    lines = "\n".join(f'  - "{e["text"]}"' for e in expressions)
    return template.format(expressions_block=lines)


def _parse_response(raw: str) -> Optional[Dict]:
    """Parse VLM JSON response, with a regex fallback for truncated output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()

    brace = text.find("{")
    if brace > 0:
        text = text[brace:]

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass

    end = text.rfind("}") + 1
    if end > 1:
        try:
            return json.loads(text[:end])
        except Exception:
            pass

    pattern = re.compile(
        r'"([^"]+)"\s*:\s*\{\s*"bboxes"\s*:\s*(\[[^\]]*(?:\[[^\]]*\][^\]]*)*\])'
        r'\s*,\s*"scores"\s*:\s*(\[[^\]]*\])\s*\}',
        re.DOTALL,
    )
    result: Dict = {}
    for m in pattern.finditer(text):
        try:
            key = m.group(1)
            bboxes = json.loads(m.group(2))
            scores = json.loads(m.group(3))
            result[key] = {"bboxes": bboxes, "scores": scores}
        except Exception:
            continue
    return result if result else None


def run(icfg, vlm_client, prompts, results_dir):
    """Run step 1: phrase grounding.

    Reads step 0's ``annotations.jsonl``, groups records by ``image_path``
    (one VLM call per unique image), asks the VLM to return pixel-space
    bounding boxes for all expressions at once, and fills the
    ``instances`` field of each expression with clamped integer bboxes.

    Args:
        icfg (object): ``image_grounding`` sub-config.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step0_file = os.path.join(
        results_dir, "step_0_expression_extraction", "annotations.jsonl",
    )
    step_dir = os.path.join(results_dir, "step_1_grounding")
    ckpt_dir = os.path.join(step_dir, ".ckpt")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(ckpt_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 1: %s already exists — skipping.", output_file)
        return output_file

    records = load_records(step0_file)
    if not records:
        logger.warning("Step 1: no step-0 output at %s", step0_file)
        save_records([], output_file)
        return output_file

    logger.info(
        "Step 1: phrase grounding for %d records across %d unique images...",
        len(records), len({r.get("image_path", "") for r in records}),
    )

    results: List[Optional[Dict]] = [None] * len(records)
    template = prompts.get("phrase_grounding") or get_prompt("phrase_grounding")
    lock = threading.Lock()
    bbox_counter = IdCounter("box")

    image_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        image_to_indices[record.get("image_path", "")].append(idx)

    def _process(image_item):
        image_path, indices = image_item

        for idx in indices:
            record = records[idx]
            sample_id = str(record.get("image_id") or idx)
            ckpt_file = os.path.join(ckpt_dir, f"{sample_id}.json")

            if not force and os.path.exists(ckpt_file):
                try:
                    with open(ckpt_file, "r", encoding="utf-8") as f:
                        results[idx] = json.load(f)
                    continue
                except Exception:
                    pass

            expressions = record.get("expressions", [])
            out = dict(record)
            out.setdefault("pipeline_steps", list(record.get("pipeline_steps", [])))
            if "step1_grounding" not in out["pipeline_steps"]:
                out["pipeline_steps"].append("step1_grounding")

            if not expressions:
                out["expressions"] = []
                results[idx] = out
                with lock:
                    with open(ckpt_file, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
                continue

            query = _build_query(template, expressions)
            try:
                if image_path and os.path.exists(image_path):
                    response = vlm_client.generate_with_image(image_path, query)
                else:
                    response = vlm_client.generate_text(query)
            except Exception as e:
                logger.warning("Step 1: VLM call failed for %s: %s", sample_id, e)
                continue

            parsed = _parse_response(response)
            width = int(record.get("width") or 1920)
            height = int(record.get("height") or 1080)

            out_expressions: List[Dict] = []
            for expr in expressions:
                expr_out = dict(expr)
                key = expr.get("text", "")
                grounded = (parsed or {}).get(key, {})
                if isinstance(grounded, dict):
                    bboxes_raw = grounded.get("bboxes", [])
                    scores_raw = grounded.get("scores", [])
                else:
                    bboxes_raw, scores_raw = [], []

                if bboxes_raw and isinstance(bboxes_raw[0], (int, float)):
                    bboxes_raw = [bboxes_raw]
                    scores_raw = (
                        scores_raw if isinstance(scores_raw, list) else [scores_raw]
                    )

                instances: List[Dict] = []
                for b_idx, bbox in enumerate(bboxes_raw):
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    pixel_bbox = clamp_bbox(bbox, width, height)
                    score = (
                        float(scores_raw[b_idx])
                        if b_idx < len(scores_raw)
                        else 0.9
                    )
                    instances.append(make_instance(
                        bbox=pixel_bbox,
                        score=score,
                        bbox_id=bbox_counter.next(),
                    ))

                expr_out["instances"] = instances
                out_expressions.append(expr_out)

            out["expressions"] = out_expressions
            results[idx] = out
            with lock:
                with open(ckpt_file, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False)

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, item): item
            for item in image_to_indices.items()
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 1 worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(records):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            r = dict(record)
            for expr in r.get("expressions", []):
                expr.setdefault("instances", [])
            output_records.append(r)

    save_records(output_records, output_file)
    logger.info("Step 1: wrote %d records → %s", len(output_records), output_file)
    return output_file
