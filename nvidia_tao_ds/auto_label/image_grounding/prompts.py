# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for the image_grounding (grounding) pipeline."""

PROMPT_TEMPLATES = {

    # =========================================================================
    # STEP 0 — EXPRESSION EXTRACTION
    # =========================================================================
    "expression_extraction": (
        "You are analyzing a scene image and its caption to extract referring "
        "expressions.\n\n"
        "Caption: \"{caption}\"\n\n"
        "Tasks:\n"
        "1. Clean the caption — remove speech artifacts such as \"we can see\", "
        "\"there is/are\", \"you can see\", \"in the image\", \"in this photo\", "
        "\"I can see\". Rephrase into natural written English.\n"
        "2. Extract referring expressions — noun phrases that describe specific "
        "visible objects or groups. Include spatial context when helpful "
        "(e.g. \"car near the tree\" not just \"car\").\n"
        "3. For each expression provide its exact character span in the cleaned "
        "caption and the core noun.\n\n"
        "Rules:\n"
        "- Only include expressions referring to something visible in the image.\n"
        "- Avoid abstract or uncountable nouns (\"traffic\", \"scenery\") unless "
        "they refer to something specific and countable.\n"
        "- Each expression must be a noun phrase, not a full sentence.\n\n"
        "Respond ONLY with a JSON object — no markdown, no explanation:\n"
        "{{\n"
        "  \"cleaned_caption\": \"...\",\n"
        "  \"expressions\": [\n"
        "    {{\"text\": \"...\", \"char_span\": [start, end], \"noun_chunk\": \"...\"}},\n"
        "    ...\n"
        "  ]\n"
        "}}"
    ),

    # =========================================================================
    # STEP 1 — PHRASE GROUNDING
    # =========================================================================
    "phrase_grounding": (
        "You are a visual grounding model. Locate visible instances of each "
        "referring expression listed below and return bounding boxes in the image.\n\n"
        "Referring expressions:\n"
        "{expressions_block}\n\n"
        "Rules:\n"
        "- Coordinates are pixel-space: [x1, y1, x2, y2] where (x1,y1) is "
        "top-left and (x2,y2) is bottom-right.\n"
        "- Return at most 10 instances per expression (the most salient/confident ones).\n"
        "- Estimate a confidence score 0.0–1.0 for each bbox.\n"
        "- If an expression matches nothing visible, use empty lists.\n\n"
        "Respond ONLY with a valid JSON object — no markdown fences, no explanation, "
        "nothing before or after the JSON:\n"
        "{{\n"
        "  \"<expression_text>\": {{\"bboxes\": [[x1,y1,x2,y2], ...], \"scores\": [0.9, ...]}},\n"
        "  ...\n"
        "}}"
    ),
}


def get_prompt(key, **kwargs):
    """Retrieve and optionally format a prompt template by key.

    Args:
        key (str): Template name (e.g. ``"expression_extraction"``).
        **kwargs: Substitution values passed to ``str.format()``.

    Returns:
        str: The prompt text, with placeholders filled if *kwargs* were provided.

    Raises:
        ValueError: If *key* is not found in ``PROMPT_TEMPLATES``.
    """
    template = PROMPT_TEMPLATES.get(key)
    if template is None:
        raise ValueError(f"No prompt template found for key: {key}")
    if kwargs:
        return template.format(**kwargs)
    return template
