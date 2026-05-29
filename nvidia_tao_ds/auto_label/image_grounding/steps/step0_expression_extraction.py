# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 0: expression extraction from (image, caption) pairs via VLM."""

import ast
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from nvidia_tao_ds.auto_label.common.annotation import (
    IdCounter,
    image_id_from_path,
    load_records,
    make_expression,
    save_records,
)
from nvidia_tao_ds.auto_label.image_grounding.prompts import get_prompt
from nvidia_tao_ds.core.logging.logging import logging as logger


def _parse_response(raw: str, sample_id: str) -> Optional[Dict]:
    """Parse VLM JSON response robustly, handling markdown fences and partials."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if 0 <= start < end:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    logger.warning("Could not parse response for %s", sample_id)
    logger.warning("It's possible that the response is truncated. Try increasing max_tokens in the VLM config.")
    return None


def _resolve_image_path(image_path: str, image_root: str) -> str:
    """Resolve relative image paths against an optional root directory."""
    if not image_path:
        return image_path
    if os.path.isabs(image_path) or not image_root:
        return image_path
    return os.path.join(image_root, image_path)


def _load_input_records(data_cfg) -> List[Dict]:
    """Load input JSONL records and normalize minimally-required fields."""
    records = load_records(data_cfg.input_jsonl)
    image_root = getattr(data_cfg, "image_root", "") or ""

    normalized: List[Dict] = []
    for idx, rec in enumerate(records):
        r = dict(rec)
        image_path = r.get("image_path", "")
        r["image_path"] = _resolve_image_path(image_path, image_root)
        r.setdefault(
            "image_id",
            r.get("image_id") or image_id_from_path(r["image_path"]) or str(idx),
        )
        normalized.append(r)
    return normalized


def run(icfg, vlm_client, prompts, results_dir):
    """Run step 0: expression extraction.

    Reads ``icfg.data.input_jsonl``, calls the VLM once per record with the
    image + caption, parses the JSON response, and writes enriched records
    to ``step_0_expression_extraction/annotations.jsonl``. Each record in
    the input is augmented with ``cleaned_caption`` and ``expressions[]``.

    Args:
        icfg (object): ``image_grounding`` sub-config with ``data`` and ``workflow``.
        vlm_client (LLMClient): VLM client for image-based inference.
        prompts (dict[str, str]): Prompt template mapping.
        results_dir (str): Root results directory.

    Returns:
        str: Path to the step's ``annotations.jsonl`` output.
    """
    step_dir = os.path.join(results_dir, "step_0_expression_extraction")
    ckpt_dir = os.path.join(step_dir, ".ckpt")
    output_file = os.path.join(step_dir, "annotations.jsonl")
    os.makedirs(ckpt_dir, exist_ok=True)

    force = icfg.workflow.force_reprocess

    if not force and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        logger.info("Step 0: %s already exists — skipping.", output_file)
        return output_file

    records = _load_input_records(icfg.data)
    if not records:
        logger.warning("Step 0: no input records at %s", icfg.data.input_jsonl)
        save_records([], output_file)
        return output_file

    logger.info("Step 0: extracting expressions from %d samples...", len(records))

    results: List[Optional[Dict]] = [None] * len(records)
    template = prompts.get("expression_extraction") or get_prompt("expression_extraction")
    lock = threading.Lock()
    expr_counter = IdCounter("expr")

    def _process(idx_record):
        idx, record = idx_record
        sample_id = str(record.get("image_id") or record.get("dataset_sample_id") or idx)
        ckpt_file = os.path.join(ckpt_dir, f"{sample_id}.json")

        if not force and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, "r", encoding="utf-8") as f:
                    results[idx] = json.load(f)
                return
            except Exception:
                pass

        caption = record.get("caption", "")
        image_path = record.get("image_path", "")
        query = template.format(caption=caption)

        try:
            if image_path and os.path.exists(image_path):
                response = vlm_client.generate_with_image(image_path, query)
            else:
                response = vlm_client.generate_text(query)
        except Exception as e:
            logger.warning("Step 0: VLM call failed for %s: %s", sample_id, e)
            return

        parsed = _parse_response(response, sample_id)
        if parsed is None:
            return

        out = dict(record)
        out["cleaned_caption"] = parsed.get("cleaned_caption", caption)

        expressions: List[Dict] = []
        for expr in parsed.get("expressions", []):
            text = expr.get("text", "").strip()
            if not text:
                continue
            char_span = expr.get("char_span")
            if isinstance(char_span, list) and len(char_span) == 2:
                char_span = [int(char_span[0]), int(char_span[1])]
            else:
                char_span = [0, len(text)]
            noun_chunk = expr.get("noun_chunk") or text.split()[-1]
            expressions.append(make_expression(
                text=text,
                expression_id=expr_counter.next(),
                char_span=char_span,
                noun_chunk=noun_chunk,
                instances=[],
            ))
        out["expressions"] = expressions
        out["source"] = "image_grounding"
        out.setdefault("pipeline_steps", []).append("step0_expression_extraction")

        with lock:
            with open(ckpt_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        results[idx] = out

    max_workers = max(1, icfg.workflow.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process, (i, r)): i for i, r in enumerate(records)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Step 0 worker error: %s", e)

    output_records: List[Dict] = []
    for idx, record in enumerate(records):
        if results[idx] is not None:
            output_records.append(results[idx])
        else:
            r = dict(record)
            r.setdefault("cleaned_caption", record.get("caption", ""))
            r["expressions"] = []
            r["source"] = "image_grounding"
            r.setdefault("pipeline_steps", []).append("step0_expression_extraction")
            output_records.append(r)

    save_records(output_records, output_file)
    logger.info("Step 0: wrote %d records → %s", len(output_records), output_file)
    return output_file
