# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 4: Parse QA output into tao-vl-reason-v1.0 task-format JSONs.

Reads ``step_3_qa/qa_output.jsonl`` and writes one JSON file per task type
under ``step_4_output/`` using the tao-vl-reason-v1.0 envelope:

    {
      "format": "tao-vl-reason-v1.0",
      "metadata": {"type": "annotation", "task": "<task>", "date": "...",
                   "description": "...", "license": "..."},
      "media_root": <video_root> | null,
      "items": [{"video_id": "...", "question": "...", "answer": "...",
                 "reasoning": "..."}, ...]
    }
"""

import json
import os
import re
from datetime import date as _date

from nvidia_tao_ds.auto_label.video_reasoning_annotation.io_utils import (
    get_entries_from_jsonl,
    parse_qa_output,
    permute_mcq,
)
from nvidia_tao_ds.core.logging.logging import logging as logger


_FORMAT_VERSION = "tao-vl-reason-v1.0"

# Prompt-key bucket -> output task name (file basename + metadata.task).
# MCQ and BCQ each fan out into two output tasks (short + open-ended);
# those derived names are added in _ALL_OUTPUT_TASKS below.
_OUTPUT_TASK = {
    "mcq": "mcq",
    "bcq": "bcq",
    "open_qa": "open_qa",
    "causal_linkage": "causal_linkage",
    "temporal_localization": "temporal_localization",
    "temporal_event_desc": "temporal_description",
    "scene_description": "scene_description",
    "event_summary": "video_summarization",
}

_ALL_OUTPUT_TASKS = sorted(set(_OUTPUT_TASK.values()) | {"mcq_openended", "bcq_openended"})

_PASSTHROUGH_BUCKETS = frozenset({
    "causal_linkage",
    "temporal_event_desc",
    "scene_description",
    "event_summary",
})

_INSTRUCTIONS = {
    "mcq": "Answer with a single letter.",
    "mcq_openended": "Answer with the correct option letter followed by a brief explanation.",
    "bcq": "Answer with Yes or No.",
    "bcq_openended": "Answer with Yes or No followed by a brief explanation.",
    "temporal_localization": (
        'Answer with the start and end timestamps in the format '
        '{"start": "MM:SS", "end": "MM:SS"}.'
    ),
}

_TASK_DESCRIPTIONS = {
    "bcq": "Binary choice QA (Yes/No answer only).",
    "bcq_openended": "Binary choice QA with open-ended explanation (Yes/No + explanation).",
    "causal_linkage": "Causal linkage QA: relationship between two timestamped events.",
    "mcq": "Multiple-choice QA (single letter answer).",
    "mcq_openended": "Multiple-choice QA with open-ended explanation (letter + explanation).",
    "open_qa": "Open-ended QA (free-text answer).",
    "scene_description": "Scene description: describe the scene and its environment.",
    "temporal_description": "Temporal description: describe what happens in a given time window.",
    "temporal_localization": "Temporal localization: predict start and end time of the anomaly.",
    "video_summarization": "Video summarization: summarize events and anomalies observed.",
}

_TL_ANSWER_RE = re.compile(
    r"Start_Time:\s*((?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?),?\s*"
    r"End_Time:\s*((?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?)",
    re.I,
)


def _classify_prompt_key(prompt_key):
    """Map a prompt_key (e.g. 'anomaly_mcq') to a task bucket."""
    for task_type in _PASSTHROUGH_BUCKETS:
        if task_type in prompt_key:
            return task_type
    if "temporal_localization" in prompt_key:
        return "temporal_localization"
    if "mcq" in prompt_key:
        return "mcq"
    if "bcq" in prompt_key:
        return "bcq"
    return "open_qa"


def _derive_video_id(video_path, video_root):
    """Strip ``video_root + '/'`` from ``video_path`` when present."""
    if video_root:
        anchor = video_root.rstrip("/") + "/"
        if video_path.startswith(anchor):
            return video_path[len(anchor):]
    return video_path


def _bcq_short_answer(answer):
    """Extract just Yes/No from a full BCQ answer string."""
    stripped = answer.strip().rstrip(".").lower()
    if stripped.startswith("yes"):
        return "Yes"
    if stripped.startswith("no"):
        return "No"
    return answer


def _mcq_openended_answer(qa):
    """Build mcq_openended answer: '<letter>. <option text>'."""
    raw = qa["answer"].strip()
    if not raw:
        return raw
    # If the answer already starts with '<letter>. <text>', keep it as-is.
    if re.match(r"^[A-Z]\.\s+\S", raw):
        return raw
    letter = raw.upper()[:1]
    for c in qa["choices"]:
        if c.startswith(f"{letter}."):
            return c.strip()
    return raw


def _process_mcq(qa, video_id):
    """Build mcq + mcq_openended items from a parsed MCQ entry."""
    if not qa["choices"]:
        return None, None
    full_q = qa["question"] + "\n" + "\n".join(qa["choices"])
    letter = qa["answer"].strip().upper()[:1]
    if not letter:
        return None, None
    mcq_item = {
        "video_id": video_id,
        "question": f"{full_q}\n{_INSTRUCTIONS['mcq']}",
        "answer": letter,
        "reasoning": qa["reasoning"],
    }
    mcq_oe_item = {
        "video_id": video_id,
        "question": f"{full_q}\n{_INSTRUCTIONS['mcq_openended']}",
        "answer": _mcq_openended_answer(qa),
        "reasoning": qa["reasoning"],
    }
    return mcq_item, mcq_oe_item


def _process_bcq(qa, video_id):
    """Build bcq + bcq_openended items from a parsed BCQ entry."""
    full_answer = qa["answer"]
    short_answer = _bcq_short_answer(full_answer)
    if short_answer not in ("Yes", "No"):
        return None, None
    bcq_item = {
        "video_id": video_id,
        "question": f"{qa['question']}\n{_INSTRUCTIONS['bcq']}",
        "answer": short_answer,
        "reasoning": qa["reasoning"],
    }
    bcq_oe_item = {
        "video_id": video_id,
        "question": f"{qa['question']}\n{_INSTRUCTIONS['bcq_openended']}",
        "answer": full_answer,
        "reasoning": qa["reasoning"],
    }
    return bcq_item, bcq_oe_item


def _process_temporal_localization(qa, video_id):
    """Build temporal_localization item with structured JSON answer."""
    m = _TL_ANSWER_RE.search(qa["answer"])
    if m:
        answer = json.dumps({"start": m.group(1), "end": m.group(2)})
    else:
        answer = qa["answer"]
    return {
        "video_id": video_id,
        "question": f"{qa['question']}\n{_INSTRUCTIONS['temporal_localization']}",
        "answer": answer,
        "reasoning": qa["reasoning"],
    }


def _process_passthrough(qa, video_id):
    """Build a plain item without modifying question or answer."""
    return {
        "video_id": video_id,
        "question": qa["question"],
        "answer": qa["answer"],
        "reasoning": qa["reasoning"],
    }


def _build_envelope(task, items, license_str, description_extra, media_root):
    """Wrap a task's items in the tao-vl-reason-v1.0 envelope."""
    description = _TASK_DESCRIPTIONS.get(task, "")
    if description_extra:
        description = f"{description} {description_extra}".strip()
    return {
        "format": _FORMAT_VERSION,
        "metadata": {
            "type": "annotation",
            "task": task,
            "date": str(_date.today()),
            "description": description,
            "license": license_str,
        },
        "media_root": media_root or None,
        "items": items,
    }


def run(vcfg, results_dir):
    """Run step 4: parse raw QA output into tao-vl-reason-v1.0 task JSONs.

    Reads ``step_3_qa/qa_output.jsonl``, parses each entry, and writes
    one ``<task>.json`` file per non-empty task type under
    ``step_4_output/``.

    Args:
        vcfg (object): Video reasoning annotation sub-config; must expose
            ``data.video_root`` and may expose top-level ``license`` and
            ``description_extra``.
        results_dir (str): Root results directory for all pipeline outputs.
    """
    input_file = os.path.join(results_dir, "step_3_qa", "qa_output.jsonl")
    if not os.path.exists(input_file):
        logger.warning("Step 4: No input from step 3 at %s", input_file)
        return

    output_dir = os.path.join(results_dir, "step_4_output")
    os.makedirs(output_dir, exist_ok=True)

    entries = get_entries_from_jsonl(input_file)
    if not entries:
        logger.info("Step 4: No entries to parse.")
        return

    video_root = getattr(vcfg.data, "video_root", "") or ""
    license_str = getattr(vcfg, "license", "") or ""
    description_extra = getattr(vcfg, "description_extra", "") or ""
    media_root = video_root or None

    items_by_task = {task: [] for task in _ALL_OUTPUT_TASKS}
    stats = {"total": 0, "ok": 0, "fail": 0, "skipped": 0}

    for entry in entries:
        stats["total"] += 1
        video = entry.get("video", "")
        prompt_key = entry.get("prompt_key", "")
        qa_text = entry.get("qa_output", "")

        bucket = _classify_prompt_key(prompt_key)
        video_id = _derive_video_id(video, video_root)

        parsed = parse_qa_output(qa_text)
        if not parsed:
            stats["fail"] += 1
            logger.warning(
                "Step 4: Failed to parse [%s] for %s",
                prompt_key, os.path.basename(video),
            )
            continue

        for qa in parsed:
            if qa["type"] == "mcq":
                qa = permute_mcq(qa)

            if bucket == "mcq":
                mcq_item, mcq_oe_item = _process_mcq(qa, video_id)
                if mcq_item is None:
                    stats["skipped"] += 1
                    continue
                items_by_task["mcq"].append(mcq_item)
                items_by_task["mcq_openended"].append(mcq_oe_item)
            elif bucket == "bcq":
                bcq_item, bcq_oe_item = _process_bcq(qa, video_id)
                if bcq_item is None:
                    stats["skipped"] += 1
                    continue
                items_by_task["bcq"].append(bcq_item)
                items_by_task["bcq_openended"].append(bcq_oe_item)
            elif bucket == "temporal_localization":
                items_by_task["temporal_localization"].append(
                    _process_temporal_localization(qa, video_id)
                )
            else:
                items_by_task[_OUTPUT_TASK.get(bucket, "open_qa")].append(
                    _process_passthrough(qa, video_id)
                )
            stats["ok"] += 1

    written = 0
    for task in _ALL_OUTPUT_TASKS:
        items = items_by_task[task]
        if not items:
            continue
        out_path = os.path.join(output_dir, f"{task}.json")
        envelope = _build_envelope(
            task=task,
            items=items,
            license_str=license_str,
            description_extra=description_extra,
            media_root=media_root,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
        logger.info("Step 4: [%s] %d items -> %s", task, len(items), out_path)
        written += 1

    logger.info(
        "Step 4: Stats — total: %d, parsed ok: %d, parse fail: %d, "
        "skipped: %d, files written: %d",
        stats["total"], stats["ok"], stats["fail"], stats["skipped"], written,
    )
