# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""image_referring_expression pipeline orchestrator.

Mirrors the 2d-data-engine referring-data-engine workflow:

    0_region_expr  ──┐
                     ├──▶  2_grounding_expr  ──▶  [3_double_check]
    1_image_caption ─┘

Steps 0 and 1 run in parallel (in their own thread pool), since both only
depend on the seed annotations. Step 2 merges their outputs and calls the
VLM to produce grounded expressions. Step 3 is an optional verification
pass that updates expressions in place.
"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from nvidia_tao_ds.auto_label.common.annotation import (
    image_id_from_path,
    load_records,
    merge_records,
    save_records,
)
from nvidia_tao_ds.auto_label.image_referring_expression.io_utils import (
    get_image_dimensions,
    list_images,
    parse_kitti_label,
)
from nvidia_tao_ds.auto_label.image_referring_expression.prompts import PROMPT_TEMPLATES
from nvidia_tao_ds.auto_label.image_referring_expression.steps import (
    step0_region_expr,
    step1_image_caption,
    step2_grounding_expr,
    step3_double_check,
)
from nvidia_tao_ds.core.llm_clients import create_client
from nvidia_tao_ds.core.logging.logging import logging as logger


def _seed_annotations(data_cfg, results_dir: str) -> List[Dict]:
    """Build the initial annotations.jsonl from image_dir + KITTI labels.

    If ``data_cfg.input_annotations_jsonl`` is set, that file is used as-is
    (for resuming a run). Otherwise one record per image is emitted,
    containing ``image_id``, ``image_path``, ``width``, ``height``, and
    ``kitti_bboxes`` (pre-parsed KITTI rows for step 0 consumption).
    """
    if data_cfg.input_annotations_jsonl and os.path.exists(data_cfg.input_annotations_jsonl):
        records = load_records(data_cfg.input_annotations_jsonl)
        logger.info("Seeded %d records from %s", len(records), data_cfg.input_annotations_jsonl)
        return records

    images = list_images(data_cfg.image_dir)
    records: List[Dict] = []
    for image_path in images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(data_cfg.kitti_label_dir or "", stem + ".txt")
        kitti_bboxes = parse_kitti_label(label_path) if data_cfg.kitti_label_dir else []
        width, height = get_image_dimensions(image_path)
        records.append({
            "image_id": image_id_from_path(image_path),
            "image_path": image_path,
            "width": int(width) if width else 0,
            "height": int(height) if height else 0,
            "kitti_bboxes": kitti_bboxes,
            "source": "image_referring_expression",
            "pipeline_steps": [],
        })

    seed_path = os.path.join(results_dir, "seed_annotations.jsonl")
    save_records(records, seed_path)
    logger.info("Seeded %d records from %s → %s",
                len(records), data_cfg.image_dir, seed_path)
    return records


def run_image_referring_expression_inference(cfg, results_dir):
    """Run the image referring-data-engine pipeline.

    Args:
        cfg (object): Top-level experiment config with an ``image_referring_expression``
            sub-config.
        results_dir (str): Directory where per-step outputs are written.
    """
    icfg = cfg.image_referring_expression
    prompts = PROMPT_TEMPLATES

    os.makedirs(results_dir, exist_ok=True)
    vlm_client = create_client(icfg.vlm, icfg.workflow)

    seed_records = _seed_annotations(icfg.data, results_dir)
    if not seed_records:
        logger.warning("image_referring_expression: no input records — nothing to do.")
        return

    steps = list(icfg.workflow.steps)
    logger.info("image_referring_expression pipeline starting. Steps: %s", steps)

    step0_file = None
    step1_file = None
    step2_file = None
    step3_file = None

    # Steps 0 and 1 depend only on seed_records → run in parallel.
    if "0" in steps or "1" in steps:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut0 = None
            fut1 = None
            if "0" in steps:
                fut0 = pool.submit(
                    step0_region_expr.run,
                    icfg, vlm_client, prompts, results_dir, seed_records,
                )
            if "1" in steps:
                fut1 = pool.submit(
                    step1_image_caption.run,
                    icfg, vlm_client, prompts, results_dir, seed_records,
                )
            if fut0 is not None:
                step0_file = fut0.result()
            if fut1 is not None:
                step1_file = fut1.result()

    if "2" in steps:
        step2_file = step2_grounding_expr.run(
            icfg, vlm_client, prompts, results_dir,
            step0_file=step0_file, step1_file=step1_file,
        )

    if "3" in steps:
        if step2_file is None:
            step2_file = os.path.join(
                results_dir, "step_2_grounding_expr", "annotations.jsonl",
            )
        step3_file = step3_double_check.run(
            icfg, vlm_client, prompts, results_dir, step2_file=step2_file,
        )

    final_source = step3_file or step2_file or step1_file or step0_file
    if final_source and os.path.exists(final_source):
        final_path = os.path.join(results_dir, "annotations.jsonl")
        if "2" in steps or "3" in steps:
            shutil.copyfile(final_source, final_path)
        else:
            merged = merge_records(
                load_records(step0_file) if step0_file else [],
                load_records(step1_file) if step1_file else [],
            )
            save_records(merged, final_path)
        logger.info("image_referring_expression pipeline complete. Final output: %s", final_path)
    else:
        logger.info("image_referring_expression pipeline complete. No output produced.")
