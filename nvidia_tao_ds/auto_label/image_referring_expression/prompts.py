# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for the image_referring_expression (referring) pipeline.

Ports from 2d-data-engine/referring-data-engine step scripts, with minor
adaptations for VLM-friendly JSON-only output when useful.
"""

PROMPT_TEMPLATES = {

    # =========================================================================
    # STEP 0 — REGION EXPRESSION
    # =========================================================================
    "region_expr": (
        "You are an AI visual assistant. Describe each object in one short, "
        "discriminative phrase.\n\n"
        "Objects in the image at [x1, y1, x2, y2]: {bboxes}\n\n"
        "For each object, write a \"description\" that would let someone pick it "
        "out from the other objects. Follow these rules:\n"
        "1. Lead with color (if visible) and object type (sedan, SUV, van, bus, "
        "truck, pickup, motorcycle, scooter, bicycle, pedestrian, etc.).\n"
        "2. Add ONE distinguishing detail — pick whichever is most discriminative:\n"
        "   - Spatial position: lane, side of road, relative to a landmark "
        "(e.g., \"in the center lane\", \"near the crosswalk\").\n"
        "   - Relation to a neighbor: \"behind the red bus\", \"to the left of the white van\".\n"
        "   - Action or state: \"turning right\", \"stopped at the light\", \"walking across the street\".\n"
        "3. Keep each description under 15 words. Do not list multiple attributes if one is enough to distinguish the object.\n"
        "4. Base descriptions only on what is visible — no guessing.\n\n"
        "Good examples:\n"
        "- \"The white sedan driving in the center lane\"\n"
        "- \"A red SUV stopped behind the blue truck\"\n"
        "- \"Pedestrian in a dark jacket crossing near the traffic light\"\n"
        "- \"Silver pickup parked on the right shoulder\"\n\n"
        "Output ONLY a JSON array. Each element must have exactly these fields:\n"
        "  \"bbox_2d\": [x1, y1, x2, y2], \"type\": <object type>, "
        "\"color\": <primary color or \"unknown\">, \"description\": <short phrase>\n"
    ),

    # =========================================================================
    # STEP 1 — IMAGE CAPTION
    # =========================================================================
    "image_caption": (
        "You are an AI visual assistant that specializes in providing clear and "
        "accurate descriptions of images without any ambiguity or uncertainty. "
        "Your descriptions should focus solely on the content of the image "
        "itself and avoid mentioning any location-specific details such as "
        "regions or countries where the image might have been captured."
    ),

    # =========================================================================
    # STEP 2 — GROUNDING EXPRESSION
    # =========================================================================
    "grounding_expr": (
        "As an AI visual assistant, your role involves analyzing a single image.\n"
        "You are supplied with data about specific attributes of objects within the image, "
        "including categories, colors, and precise coordinates [x1, y1, x2, y2].\n"
        "{bboxes}\n"
        "{caption_section}\n"
        "Your task is to classify the provided objects into groups based on shared characteristics "
        "(e.g., direction of travel, type, color, lane position), while justifying each grouping "
        "with direct visual evidence from the image.\n\n"
        "IMPORTANT — output format rules (follow exactly):\n"
        "- Output ONLY the final classification lines. Do not include any reasoning, headers, "
        "bullet points, or markdown formatting.\n"
        "- Each line must follow this exact format:\n"
        "  <descriptive phrase>: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]\n"
        "- Each line represents ONE group. All bounding boxes for that group must appear on the "
        "same line inside double brackets.\n"
        "- Do not use single brackets. Do not use asterisks, dashes, or any other decoration.\n"
        "- Do not output anything before or after the classification lines.\n"
    ),

    # =========================================================================
    # STEP 3 — DOUBLE CHECK / VERIFICATION
    # =========================================================================
    "double_check": (
        "You are verifying grounding expressions against an image. Each line below is a "
        "grounding phrase with one or more bounding boxes [x1, y1, x2, y2]:\n"
        "{expr}\n\n"
        "For each bounding box, check these criteria by looking at the image:\n"
        "1. Does the box contain an object that matches the described type (e.g., sedan, SUV, pedestrian)?\n"
        "2. Does the object's color match the phrase?\n"
        "3. Is the spatial description accurate (e.g., \"in the left lane\", \"parked on the right\")?\n"
        "4. Does the box tightly frame the object, or is it on empty space / a different object?\n\n"
        "Actions:\n"
        "- KEEP a bounding box unchanged if it passes all checks.\n"
        "- REMOVE a bounding box from its group if the object inside does not match the phrase.\n"
        "- UPDATE a bounding box's coordinates only if the box is on the right object but poorly aligned.\n"
        "- Do NOT add new bounding boxes that were not in the input.\n"
        "- Do NOT rewrite phrases — keep them exactly as given.\n\n"
        "Output ONLY the corrected list in the same format:\n"
        "  <original phrase>: [[x1, y1, x2, y2], ...]\n"
        "Omit any line whose bounding boxes were all removed. Do not include reasoning or "
        "commentary in the output.\n"
    ),
}


def get_prompt(key, **kwargs):
    """Retrieve and optionally format a prompt template by key.

    Args:
        key (str): Template name.
        **kwargs: Substitution values passed to ``str.format()``.

    Returns:
        str: The formatted prompt string.

    Raises:
        ValueError: If *key* is not found in ``PROMPT_TEMPLATES``.
    """
    template = PROMPT_TEMPLATES.get(key)
    if template is None:
        raise ValueError(f"No prompt template found for key: {key}")
    if kwargs:
        return template.format(**kwargs)
    return template
