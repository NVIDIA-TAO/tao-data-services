# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video reasoning annotation pipeline orchestrator — runs all steps in sequence."""

import importlib

from nvidia_tao_ds.auto_label.video_reasoning_annotation.prompts import PROMPT_TEMPLATES
from nvidia_tao_ds.auto_label.video_reasoning_annotation.steps import (
    step0_filter,
    step1a_caption,
    step1b_chunks,
    step1c_highlight,
    step2_description,
    step3_qa,
    step4_parse_qa,
)
from nvidia_tao_ds.core.llm_clients import create_client
from nvidia_tao_ds.core.logging.logging import logging as logger


def _load_prompts(vcfg):
    """Load prompt templates — default or from a custom module."""
    prompts_module = getattr(vcfg, "prompts_module", "")
    if prompts_module:
        try:
            mod = importlib.import_module(prompts_module)
            return getattr(mod, "PROMPT_TEMPLATES", PROMPT_TEMPLATES)
        except ImportError:
            logger.warning(
                "Could not import custom prompts module '%s', using defaults.",
                prompts_module,
            )
    return PROMPT_TEMPLATES


def run_video_reasoning_annotation_inference(cfg, results_dir):
    """Run the full video reasoning annotation pipeline.

    Orchestrates steps 0 through 4 (filtering, captioning, description
    synthesis, QA generation, and output parsing) based on the workflow
    configuration in ``cfg.video_reasoning_annotation``.

    Args:
        cfg (object): Top-level experiment config. Must contain a
            ``video_reasoning_annotation`` sub-config with ``vlm``, ``llm``,
            ``workflow``, and ``data`` sections.
        results_dir (str): Directory where per-step outputs are written.
    """
    vcfg = cfg.video_reasoning_annotation
    prompts = _load_prompts(vcfg)

    vlm_client = create_client(vcfg.vlm, vcfg.workflow)
    llm_client = create_client(vcfg.llm, vcfg.workflow)

    steps = list(vcfg.workflow.steps)
    mode = vcfg.workflow.mode

    if mode == "auto":
        if "0" not in steps:
            steps.insert(0, "0")
        if "1c" not in steps and "1b" in steps:
            idx = steps.index("1b") + 1
            steps.insert(idx, "1c")

    logger.info("Video reasoning annotation pipeline starting. Steps: %s, Mode: %s", steps, mode)

    if "0" in steps:
        step0_filter.run(vcfg, vlm_client, prompts, results_dir)

    if "1a" in steps:
        step1a_caption.run(vcfg, vlm_client, prompts, results_dir)

    if "1b" in steps:
        step1b_chunks.run(vcfg, vlm_client, prompts, results_dir)

    if "1c" in steps:
        step1c_highlight.run(vcfg, vlm_client, llm_client, prompts, results_dir)

    if "2" in steps:
        step2_description.run(vcfg, llm_client, prompts, results_dir)

    if "3" in steps:
        step3_qa.run(vcfg, llm_client, prompts, results_dir)

    if "4" in steps:
        step4_parse_qa.run(vcfg, results_dir)

    logger.info("Video reasoning annotation pipeline complete. Results at %s", results_dir)
