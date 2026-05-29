# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSONL I/O, QA parsing, and formatting utilities."""

import json
import os
import random
import re
import threading

from nvidia_tao_ds.core.logging.logging import logging as logger

_write_lock = threading.Lock()


def save_result_to_jsonl(result, output_file):
    """Thread-safe append of a single result dict to a JSONL file.

    Creates parent directories if they do not exist.

    Args:
        result (dict): Data to serialize as a single JSON line.
        output_file (str): Path to the target JSONL file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


def get_entries_from_jsonl(jsonl_path):
    """Read all valid JSON entries from a JSONL file.

    Lines that cannot be parsed are silently skipped.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        list[dict]: Parsed JSON objects, one per valid line.
    """
    entries = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found: %s", jsonl_path)
        return entries
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def format_chunk_captions(chunk_captions):
    """Format a list of chunk caption dicts into a human-readable string.

    Each chunk is rendered as ``Chunk <idx> (<start>-<end>s): <caption>``
    and chunks are separated by horizontal rules.

    Args:
        chunk_captions (list[dict]): Dicts with keys ``chunk_index``,
            ``timestamp_start``, ``timestamp_end``, and ``caption``.

    Returns:
        str: Formatted multi-line string suitable for inclusion in prompts.
    """
    formatted = []
    for c in chunk_captions:
        start = c.get("timestamp_start", "N/A")
        end = c.get("timestamp_end", "N/A")
        if isinstance(start, (int, float)):
            start = f"{start:.1f}"
        if isinstance(end, (int, float)):
            end = f"{end:.1f}"
        formatted.append(
            f"Chunk {c.get('chunk_index', '?')} ({start}-{end}s): {c.get('caption', '')}"
        )
    return "\n\n---\n\n".join(formatted)


def _new_qa_entry(q_type, question, choices, answer, reasoning):
    """Create a new QA entry dict."""
    return {
        "type": q_type,
        "question": question,
        "choices": choices,
        "answer": answer,
        "reasoning": reasoning,
    }


def parse_qa_output(text):
    """Parse raw QA text into structured QA entries.

    The input is expected to be delimited by lines of five or more ``=``
    characters, with sections labelled ``Question:``, ``Answer:``, and
    ``Reasoning:``.

    Args:
        text (str): Raw QA text produced by the LLM.

    Returns:
        list[dict]: Parsed QA entries. Each dict contains:
            ``type`` (``"mcq"`` | ``"binary"`` | ``"openended"``),
            ``question`` (str), ``choices`` (list[str]),
            ``answer`` (str), ``reasoning`` (str).
    """
    if not text:
        return []
    normalized = re.sub(r"\n?={5,}\n?", "\n<<<SEP>>>\n", text.strip())
    parts = [p.strip() for p in normalized.split("<<<SEP>>>") if p.strip()]
    results = []
    current = _new_qa_entry("", "", [], "", "")
    has_current = False
    for part in parts:
        m = re.match(
            r"^(?:\d+\.\s*)?(Multiple-Choice Question|Open-ended Question|Question):\s*(.*)",
            part,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            if has_current:
                results.append(current)
            tl = m.group(1).lower()
            q_type = (
                "mcq"
                if "multiple" in tl
                else ("openended" if "open" in tl else "binary")
            )
            q_text = m.group(2).strip()
            choices = []
            inline_answer = ""
            inline_reasoning = ""
            cs = re.search(r"\nA\.", q_text)
            if cs:
                cr = q_text[cs.start():]
                q_text = q_text[: cs.start()].strip()
                choices = [
                    f"{letter}. {t.strip()}"
                    for letter, t in re.findall(
                        r"^([A-Z])\.\s+(.+)", cr, re.MULTILINE
                    )
                ]
                am = re.search(
                    r"^Answer:\s*(.+)", cr, re.MULTILINE | re.IGNORECASE
                )
                if am:
                    inline_answer = am.group(1).strip()
                rm = re.search(
                    r"^Reasoning:\s*(.*)",
                    cr,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )
                if rm:
                    inline_reasoning = rm.group(1).strip()
            current = _new_qa_entry(
                q_type, q_text, choices, inline_answer, inline_reasoning,
            )
            has_current = True
        elif re.match(r"^Answer:", part, re.IGNORECASE) and has_current:
            answer_text = re.sub(
                r"^Answer:\s*", "", part, flags=re.IGNORECASE
            ).strip()
            rm2 = re.search(
                r"\nReasoning:\s*(.*)", answer_text, re.IGNORECASE | re.DOTALL
            )
            if rm2:
                if not current["reasoning"]:
                    current["reasoning"] = rm2.group(1).strip()
                answer_text = answer_text[: rm2.start()].strip()
            current["answer"] = answer_text
        elif re.match(r"^Reasoning:", part, re.IGNORECASE) and has_current:
            current["reasoning"] = re.sub(
                r"^Reasoning:\s*", "", part, flags=re.IGNORECASE
            ).strip()
    if has_current:
        results.append(current)
    return results


def permute_mcq(qa):
    """Shuffle MCQ option order and update the answer letter.

    Uses a deterministic seed derived from the question text so the same
    question always produces the same permutation.

    Args:
        qa (dict): QA entry with keys ``question`` (str),
            ``choices`` (list[str] of ``"X. text"`` items), and
            ``answer`` (str starting with the correct letter).

    Returns:
        dict: A shallow copy of *qa* with ``choices`` and ``answer``
            updated to reflect the new option ordering. Returns the
            original *qa* unchanged if fewer than two choices are present
            or the answer letter is not found among choices.
    """
    choices_dict = {}
    for c in qa["choices"]:
        m = re.match(r"^([A-Z])\.\s*(.*)", c)
        if m:
            choices_dict[m.group(1)] = m.group(2).strip()
    if len(choices_dict) < 2:
        return qa

    answer_letter = qa["answer"].strip().upper()[:1]
    if answer_letter not in choices_dict:
        return qa

    correct_text = choices_dict[answer_letter]
    texts = list(choices_dict.values())
    rng = random.Random(qa["question"])
    rng.shuffle(texts)

    letters = [chr(ord("A") + i) for i in range(len(texts))]
    new_choices, new_answer = [], ""
    for letter, text in zip(letters, texts):
        new_choices.append(f"{letter}. {text}")
        if text == correct_text:
            new_answer = letter

    return {**qa, "choices": new_choices, "answer": new_answer}


def build_question_str(qa):
    """Build the full question string from a QA entry.

    For MCQ entries, appends the choices and a short instruction to
    answer with the option letter.

    Args:
        qa (dict): QA entry with keys ``question`` (str), ``type`` (str),
            and ``choices`` (list[str]).

    Returns:
        str: Formatted question string ready for evaluation prompts.
    """
    q = qa["question"]
    if qa["type"] == "mcq":
        if qa["choices"]:
            q += "\n" + "\n".join(qa["choices"])
        q += "\nAnswer with the option's letter from the given choices directly."
    return q


def resolve_mode(entry, global_mode, mode_map=None):
    """Resolve the effective processing mode for a video entry.

    Priority: ``entry["mode"]`` > ``mode_map[video_path]`` > *global_mode*.
    When *global_mode* is ``"auto"``, falls back to ``"normal"`` if no
    per-video mode is found.

    Args:
        entry (dict): Video entry dict, may contain ``"mode"``,
            ``"video"``, or ``"video_path"`` keys.
        global_mode (str): Pipeline-wide mode (``"anomaly"``,
            ``"normal"``, or ``"auto"``).
        mode_map (dict | None): Optional mapping of video paths to
            modes, typically from step 0 classification.

    Returns:
        str: ``"anomaly"`` or ``"normal"``.
    """
    m = entry.get("mode")
    if m in ("anomaly", "normal"):
        return m
    if mode_map:
        video = entry.get("video") or entry.get("video_path", "")
        m = mode_map.get(video)
        if m in ("anomaly", "normal"):
            return m
    if global_mode == "auto":
        return "normal"
    return global_mode


def load_mode_map(results_dir):
    """Build a video-path-to-mode mapping from step 0 filter results.

    Args:
        results_dir (str): Root results directory containing
            ``step_0_filter/filter_results.jsonl``.

    Returns:
        dict[str, str]: Mapping of video paths to ``"anomaly"`` or
            ``"normal"``. Empty if the step 0 output does not exist.
    """
    step0_file = os.path.join(results_dir, "step_0_filter", "filter_results.jsonl")
    mode_map = {}
    if not os.path.exists(step0_file):
        return mode_map
    for entry in get_entries_from_jsonl(step0_file):
        video = entry.get("video") or entry.get("video_path", "")
        mode = entry.get("mode")
        if video and mode in ("anomaly", "normal"):
            mode_map[video] = mode
    return mode_map


def get_processed_videos(output_file):
    """Collect video paths that have already been processed.

    Args:
        output_file (str): Path to a JSONL file where each line has a
            ``"video"`` field.

    Returns:
        set[str]: Video paths found in the file. Empty if the file
            does not exist.
    """
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add(data.get("video"))
                except Exception:
                    pass
    return processed


def get_processed_video_prompt_keys(output_file):
    """Collect (video, prompt_key) pairs that have already been processed.

    Args:
        output_file (str): Path to a JSONL file where each line has
            ``"video"`` and ``"prompt_key"`` fields.

    Returns:
        set[tuple[str, str]]: Pairs of ``(video_path, prompt_key)``
            found in the file. Empty if the file does not exist.
    """
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed.add((data.get("video"), data.get("prompt_key")))
                except Exception:
                    pass
    return processed


def get_videos_from_dir(root_dir):
    """Recursively discover video files under a directory.

    Recognized extensions: ``.mp4``, ``.avi``, ``.mov``, ``.mkv``.

    Args:
        root_dir (str): Root directory to walk.

    Returns:
        list[str]: Absolute paths to discovered video files.
    """
    video_paths = []
    if not os.path.exists(root_dir):
        logger.warning("Directory not found: %s", root_dir)
        return video_paths
    for root, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_paths.append(os.path.join(root, f))
    return video_paths


def get_videos_from_jsonl(jsonl_path, filter_field=None):
    """Read video paths from a JSONL file, with optional boolean filtering.

    Each line is expected to have a ``"video_path"`` or ``"video"`` field.
    When *filter_field* is given, entries where that field is falsy are
    skipped.

    Args:
        jsonl_path (str): Path to the JSONL file.
        filter_field (str | None): Name of a boolean field to filter on.
            Only entries where ``entry[filter_field]`` is truthy are
            included.

    Returns:
        list[str]: Video paths extracted from qualifying entries.
    """
    video_paths = []
    if not os.path.exists(jsonl_path):
        logger.warning("JSONL file not found: %s", jsonl_path)
        return video_paths
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                video_path = entry.get("video_path") or entry.get("video", "")
                if video_path:
                    if filter_field and not entry.get(filter_field, False):
                        continue
                    video_paths.append(video_path)
            except json.JSONDecodeError:
                continue
    return video_paths
