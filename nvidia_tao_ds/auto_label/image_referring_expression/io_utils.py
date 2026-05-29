# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I/O utilities for the image_referring_expression pipeline (image listing, KITTI, responses)."""

import ast
import json
import os
from typing import List, Optional, Tuple

from PIL import Image


def list_images(image_dir: str) -> List[str]:
    """Return a sorted list of jpg/png files under *image_dir*."""
    if not image_dir or not os.path.isdir(image_dir):
        return []
    return sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def parse_kitti_label(label_path: str) -> List[List]:
    """Parse a KITTI-format label file into ``[x1, y1, x2, y2, type]`` rows."""
    bboxes: List[List] = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            obj_type = str(parts[0])
            bbox_left = float(parts[4])
            bbox_top = float(parts[5])
            bbox_right = float(parts[6])
            bbox_bottom = float(parts[7])
            bboxes.append([
                int(bbox_left), int(bbox_top),
                int(bbox_right), int(bbox_bottom),
                obj_type,
            ])
    return bboxes


def get_image_dimensions(image_path: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (width, height) using PIL. Returns (None, None) on error."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None, None


def format_bboxes(bboxes: List[List], h: int, w: int) -> str:
    """Format KITTI boxes as normalized 0-1000 coordinate strings for the VLM."""
    if not bboxes:
        return "[]"
    bbox_strs = [
        f"{b[-1]}: [{int(b[0] / w * 1000)}, {int(b[1] / h * 1000)}, "
        f"{int(b[2] / w * 1000)}, {int(b[3] / h * 1000)}] "
        for b in bboxes
    ]
    return "[" + ", ".join(bbox_strs) + " ]"


def scale_bbox(bbox: List, height: int, width: int,
               scale_factor: int = 1000) -> Tuple[int, int, int, int]:
    """Scale a normalized (0-1000) bbox back to absolute pixel coords."""
    abs_x1 = max(int(bbox[0] / scale_factor * width), 0)
    abs_y1 = max(int(bbox[1] / scale_factor * height), 0)
    abs_x2 = min(int(bbox[2] / scale_factor * width), width)
    abs_y2 = min(int(bbox[3] / scale_factor * height), height)
    return abs_x1, abs_y1, abs_x2, abs_y2


def clean_response(response: str) -> str:
    """Strip markdown ```json fences from a VLM response."""
    if "```json" in response:
        lines = response.splitlines()
        for i, line in enumerate(lines):
            if "```json" in line:
                response = "\n".join(lines[i + 1:])
                response = response.split("```", maxsplit=1)[0]
                break
    elif response.strip().startswith("```"):
        lines = response.strip().splitlines()
        if len(lines) >= 2:
            response = "\n".join(lines[1:])
            if response.endswith("```"):
                response = response[: -3]
    return response


def parse_regions_response(response: str) -> List[dict]:
    """Parse a step-0 VLM response into a list of region dicts.

    Mirrors ``_parse_region_str`` from 2d-data-engine: accepts both
    JSON-array and NDJSON formats; also accepts truncated output by
    attempting to close the array at the last complete object.
    """
    text = clean_response(response).strip()
    if not text:
        return []

    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [_normalize(o) for o in result]
        if isinstance(result, dict):
            return [_normalize(result)]
    except Exception:
        pass

    try:
        end_idx = text.rfind('"}') + len('"}')
        if end_idx > len('"}'):
            truncated = text[:end_idx] + "]"
            result = ast.literal_eval(truncated)
            if isinstance(result, list):
                return [_normalize(o) for o in result]
    except Exception:
        pass

    decoder = json.JSONDecoder()
    objects: List[dict] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r,":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except Exception:
            break
        if isinstance(obj, dict):
            objects.append(_normalize(obj))
        idx = end
    return objects


def _normalize(obj: dict) -> dict:
    """Normalize a parsed region dict (copy ``bbox_2d`` → ``bbox`` when missing)."""
    out = dict(obj)
    if "bbox" not in out and "bbox_2d" in out:
        out["bbox"] = out["bbox_2d"]
    return out


def parse_grounding_response(response: str) -> List[dict]:
    """Parse a step-2 / step-3 VLM response into expression dicts.

    Each line is expected to have the form::

        <phrase>: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]

    Returns a list of ``{"text": str, "instances": [{"bbox": [..]}, ...]}``.
    """
    expressions: List[dict] = []
    text = clean_response(response).strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        phrase, _, rest = line.partition(":")
        rest = rest.strip()
        if not rest.startswith("["):
            continue
        try:
            bboxes = ast.literal_eval(rest)
        except Exception:
            continue
        if not isinstance(bboxes, list):
            continue
        instances = []
        for b in bboxes:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                instances.append({"bbox": [int(v) for v in b]})
        if not instances:
            continue
        expressions.append({
            "text": phrase.strip(),
            "instances": instances,
        })
    return expressions


def format_grounding_text(expressions: List[dict]) -> str:
    """Render expressions back into the ``<phrase>: [[...]]`` legacy format."""
    lines: List[str] = []
    for expr in expressions:
        instances = expr.get("instances", [])
        if not instances:
            continue
        bboxes = [list(inst["bbox"]) for inst in instances if "bbox" in inst]
        if not bboxes:
            continue
        text = expr.get("text", "").strip() or "object"
        lines.append(f"{text}: {json.dumps(bboxes)}")
    return "\n".join(lines)
