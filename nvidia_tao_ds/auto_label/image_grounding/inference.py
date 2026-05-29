# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""image_grounding pipeline orchestrator — runs steps 0 and 1 in sequence."""

import os
import shutil

from nvidia_tao_ds.auto_label.image_grounding.prompts import PROMPT_TEMPLATES
from nvidia_tao_ds.auto_label.image_grounding.steps import (
    step0_expression_extraction,
    step1_grounding,
)
from nvidia_tao_ds.core.llm_clients import create_client
from nvidia_tao_ds.core.logging.logging import logging as logger


def run_image_grounding_inference(cfg, results_dir):
    """Run the image grounding-data-engine pipeline.

    Runs step 0 (expression extraction) and/or step 1 (phrase grounding)
    based on ``cfg.image_grounding.workflow.steps``. Writes one progressively
    enriched ``annotations.jsonl`` per step and a final
    ``annotations.jsonl`` at *results_dir* root.

    Args:
        cfg (object): Top-level experiment config with an ``image_grounding``
            sub-config.
        results_dir (str): Directory where per-step outputs are written.
    """
    icfg = cfg.image_grounding
    prompts = PROMPT_TEMPLATES

    vlm_client = create_client(icfg.vlm, icfg.workflow)

    steps = list(icfg.workflow.steps)
    logger.info("image_grounding pipeline starting. Steps: %s", steps)

    last_step_jsonl = None

    if "0" in steps:
        last_step_jsonl = step0_expression_extraction.run(
            icfg, vlm_client, prompts, results_dir,
        )

    if "1" in steps:
        last_step_jsonl = step1_grounding.run(
            icfg, vlm_client, prompts, results_dir,
        )

    if last_step_jsonl and os.path.exists(last_step_jsonl):
        final_path = os.path.join(results_dir, "annotations.jsonl")
        shutil.copyfile(last_step_jsonl, final_path)
        logger.info("image_grounding pipeline complete. Final output: %s", final_path)
    else:
        logger.info("image_grounding pipeline complete. No output produced.")
