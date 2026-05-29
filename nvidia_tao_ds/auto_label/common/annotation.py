# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified annotation schema and I/O helpers for image_grounding / image_referring_expression pipelines.

A single JSONL record per image, progressively enriched across pipeline
steps so that both the grounding and referring pipelines converge on the
same final shape:

    {
      "image_id":        str,
      "image_path":      str,
      "width":           int,
      "height":          int,
      "caption":         str | None,
      "cleaned_caption": str | None,
      "regions":  [ {"bbox": [x1,y1,x2,y2], "type": str, "color": str,
                     "description": str}, ... ],
      "expressions": [
        {
          "expression_id": str,
          "text":          str,
          "char_span":     [start, end] | None,
          "noun_chunk":    str | None,
          "instances": [
            {"bbox_id": str, "bbox": [x1,y1,x2,y2], "bbox_score": float}, ...
          ],
          "verified":      bool | None
        }, ...
      ],
      "source":          "image_grounding" | "image_referring_expression",
      "pipeline_steps":  [str, ...]
    }

Legacy per-image text-file helpers (``write_legacy_*`` / ``parse_legacy_*``)
provide byte-compatibility with the 2d-data-engine output layout so existing
downstream consumers and partially-processed output directories can be
used interchangeably with the new pipelines.
"""

import ast
import json
import os
import re
import threading
from typing import Dict, Iterable, List, Optional

FAIL_MSG = "Failed to obtain answer via API."


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts. Missing file returns []."""
    if not os.path.exists(path):
        return []
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_records(records: Iterable[Dict], path: str) -> None:
    """Atomically write records to a JSONL file (creates parent dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


_append_lock = threading.Lock()


def append_record(record: Dict, path: str) -> None:
    """Thread-safe append of a single record to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _append_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Builders / ID generators
# ---------------------------------------------------------------------------

class IdCounter:
    """Thread-safe monotonic id generator (prefix + zero-padded counter)."""

    def __init__(self, prefix: str, width: int = 5):
        """Initialize the id counter with a prefix and width."""
        self._prefix = prefix
        self._width = width
        self._count = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        """Return the next id."""
        with self._lock:
            v = self._count
            self._count += 1
        return f"{self._prefix}_{v:0{self._width}d}"


def make_instance(bbox: List[float], score: float = 0.9,
                  bbox_id: Optional[str] = None) -> Dict:
    """Build a single instance dict: {bbox_id, bbox, bbox_score}."""
    inst = {
        "bbox": [int(v) for v in bbox],
        "bbox_score": round(float(score), 4),
    }
    if bbox_id is not None:
        inst["bbox_id"] = bbox_id
    return inst


def make_expression(text: str,
                    instances: Optional[List[Dict]] = None,
                    expression_id: Optional[str] = None,
                    char_span: Optional[List[int]] = None,
                    noun_chunk: Optional[str] = None,
                    verified: Optional[bool] = None) -> Dict:
    """Build an expression dict with optional image_grounding / image_referring_expression fields."""
    expr: Dict = {"text": text, "instances": instances or []}
    if expression_id is not None:
        expr["expression_id"] = expression_id
    if char_span is not None:
        expr["char_span"] = list(char_span)
    if noun_chunk is not None:
        expr["noun_chunk"] = noun_chunk
    if verified is not None:
        expr["verified"] = bool(verified)
    return expr


def image_id_from_path(image_path: str) -> str:
    """Derive a stable image_id from an image file path (basename stem)."""
    return os.path.splitext(os.path.basename(image_path))[0]


def clamp_bbox(bbox: List, width: int, height: int) -> List[int]:
    """Clamp bbox coordinates to image bounds and return as integers."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    return [x1, y1, x2, y2]


def merge_records(base: List[Dict], other: List[Dict],
                  key: str = "image_id") -> List[Dict]:
    """Merge two record lists by ``key`` (right record wins on conflict)."""
    index = {r.get(key): dict(r) for r in base}
    for r in other:
        k = r.get(key)
        if k in index:
            index[k].update(r)
        else:
            index[k] = dict(r)
    return list(index.values())


# ---------------------------------------------------------------------------
# Legacy text-file writers (2d-data-engine byte-compatible output)
# ---------------------------------------------------------------------------

def write_legacy_region_file(regions: List[Dict], path: str) -> None:
    """Write step-0 ``.txt.step0`` — a JSON array of region objects.

    Mirrors the 2d-data-engine referring step-0 output: each element has
    ``bbox_2d``, ``type``, ``color``, ``description`` keys.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out: List[Dict] = []
    for r in regions:
        bbox = r.get("bbox") or r.get("bbox_2d") or [0, 0, 0, 0]
        out.append({
            "bbox_2d": list(bbox),
            "type": r.get("type", ""),
            "color": r.get("color", ""),
            "description": r.get("description", ""),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def write_legacy_caption_file(caption: str, path: str) -> None:
    """Write step-1 ``.txt.step1`` — plain caption text."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(caption or "")


def write_legacy_grounding_file(expressions: List[Dict], path: str) -> None:
    """Write step-2/3 ``.txt.stepN`` — ``<phrase>: [[x,y,x,y], ...]`` lines.

    Expressions with no instances (e.g. all bboxes removed by verification)
    are omitted, matching step-3 behaviour in the legacy engine.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------------------------------------------------------------------------
# Legacy text-file parsers (read 2d-data-engine outputs as unified records)
# ---------------------------------------------------------------------------

def parse_legacy_region_file(path: str) -> List[Dict]:
    """Parse a ``.txt.step0`` file into a list of region dicts.

    Accepts both JSON-array (qwen3_30b style) and NDJSON (qwen3_235b style)
    output formats, mirroring ``_parse_region_str`` in the 2d-data-engine.
    Returned dicts use the unified ``bbox`` key (mirroring ``bbox_2d``).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return _parse_region_text(text)


def _parse_region_text(text: str) -> List[Dict]:
    """Parse region text (JSON array or NDJSON) into a list of dicts."""
    text = _strip_markdown_fence(text).strip()
    if not text:
        return []
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [_normalize_region(o) for o in result]
        return [_normalize_region(result)]
    except Exception:
        pass
    decoder = json.JSONDecoder()
    objects: List[Dict] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except Exception:
            break
        if isinstance(obj, dict):
            objects.append(_normalize_region(obj))
        idx = end
    return objects


def _normalize_region(obj: Dict) -> Dict:
    """Normalize a parsed region dict to use ``bbox`` (not ``bbox_2d``)."""
    out = dict(obj)
    if "bbox" not in out and "bbox_2d" in out:
        out["bbox"] = out.pop("bbox_2d")
    return out


_LEGACY_GROUNDING_LINE = re.compile(
    r"^(?P<phrase>.+?):\s*(?P<bboxes>\[\[.*?\]\])\s*$"
)


def parse_legacy_grounding_file(path: str) -> List[Dict]:
    """Parse a ``.txt.step2`` / ``.txt.step3`` file into expression dicts.

    Each output dict has ``text`` and ``instances: [{bbox}, ...]`` fields
    (no ids / scores — those are legacy-free additions).
    """
    expressions: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LEGACY_GROUNDING_LINE.match(line)
            if not m:
                continue
            try:
                bboxes = ast.literal_eval(m.group("bboxes"))
            except Exception:
                continue
            if not isinstance(bboxes, list):
                continue
            instances = [
                {"bbox": [int(v) for v in b]}
                for b in bboxes
                if isinstance(b, (list, tuple)) and len(b) == 4
            ]
            expressions.append({
                "text": m.group("phrase").strip(),
                "instances": instances,
            })
    return expressions


def _strip_markdown_fence(text: str) -> str:
    """Strip ```json fences or ``` fences if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2:
            t = "\n".join(lines[1:])
            if t.endswith("```"):
                t = t[: -len("```")]
    return t.strip()


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------

def is_valid_output(path: str) -> bool:
    """Return True only if path is a non-empty file that isn't a failure marker."""
    try:
        if os.path.getsize(path) == 0:
            return False
    except OSError:
        return False
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(200).strip()
    return bool(head) and head != FAIL_MSG
