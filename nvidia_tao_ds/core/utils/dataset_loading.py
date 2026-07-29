# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset annotation loading: COCO/KITTI parsing and per-image annotation lookup."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

_log = logging.getLogger(__name__)


class AnnotationFormat(str, Enum):
    """A supported detection-annotation format."""

    KITTI = "kitti"  # a directory of per-image ``.txt`` label files
    COCO = "coco"    # a single COCO-style ``.json`` file


def load_annotations(path: str, fmt: AnnotationFormat) -> Tuple[dict, dict, dict, dict]:
    """Parse a detection-annotation source into per-image class and box lookups.

    Args:
        path: Path to a KITTI label directory or a COCO ``.json`` file.
        fmt: The annotation format (``AnnotationFormat.KITTI`` or ``AnnotationFormat.COCO``),
            declared explicitly by the caller — it is never inferred from the path.

    Returns:
        A ``(fp_to_classes, bn_to_classes, fp_to_boxes, bn_to_boxes)`` tuple, where the
        ``*_to_classes`` maps go path -> set of class names and the ``*_to_boxes`` maps go
        path -> list of ``(class_name, [x, y, w, h])``. The ``fp_*`` maps are keyed by full
        image path and the ``bn_*`` maps by basename (empty for KITTI, which has no full paths).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a COCO file does not contain a valid JSON object.
    """
    return _FORMAT_LOADERS[fmt](path)


def load_scored_annotations(path: str, fmt: AnnotationFormat) -> Tuple[dict, dict, dict, dict]:
    """Like :func:`load_annotations` but box tuples include a confidence score.

    Identical to :func:`load_annotations` in every respect except that the
    ``*_to_boxes`` maps contain ``(class_name, [x, y, w, h], score_or_None)``
    3-tuples instead of 2-tuples. ``score_or_None`` is ``None`` when no score
    is present in the source (e.g. ground-truth KITTI/COCO files).

    Args:
        path: Path to a KITTI label directory or a COCO ``.json`` file.
        fmt: The annotation format, declared explicitly by the caller.

    Returns:
        A ``(fp_to_classes, bn_to_classes, fp_to_boxes, bn_to_boxes)`` tuple
        where ``*_to_boxes`` values are lists of ``(class_name, [x, y, w, h], score_or_None)``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a COCO file is invalid.
    """
    return _FORMAT_LOADERS[fmt](path, with_scores=True)


def _load_kitti(label_dir: str, with_scores: bool = False) -> Tuple[dict, dict, dict, dict]:
    """
    Parse a KITTI label directory. Each .txt file corresponds to one image
    (filename stem matches image basename stem). Each line format:
      class truncated occluded alpha x1 y1 x2 y2 h w l tx ty tz ry [score]
    Bbox is [x1,y1,x2,y2] — converted to [x,y,w,h] on return.

    Since KITTI has no absolute image paths, only basename (stem) matching is
    populated; fp_to_* dicts are always empty. Malformed lines within a label file
    are skipped; an empty directory (no ``.txt`` files) raises.

    When ``with_scores`` is True, box entries are ``(class, bbox, score_or_None)``
    3-tuples; otherwise they are ``(class, bbox)`` 2-tuples.

    Raises:
        FileNotFoundError: If the directory contains no ``.txt`` label files.
        OSError: If a label file cannot be read.
    """
    bn_to_classes: dict = {}
    bn_to_boxes: dict = {}

    txt_files = sorted(Path(label_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt label files found in KITTI directory: {label_dir}. "
            f"Expected a directory of per-image KITTI label files."
        )

    for txt_file in txt_files:
        bn = txt_file.stem  # e.g. "000042" for "000042.txt"
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cat_name = parts[0]
                if cat_name.lower() == "dontcare":
                    continue
                try:
                    x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                except ValueError:
                    continue
                bbox = [x1, y1, x2 - x1, y2 - y1]  # convert to [x, y, w, h]
                bn_to_classes.setdefault(bn, set()).add(cat_name)
                if with_scores:
                    score: Optional[float] = None
                    if len(parts) >= 16:
                        try:
                            score = float(parts[15])
                        except ValueError:
                            _log.warning(
                                "Malformed score field %r in %s — treating as unscored",
                                parts[15], txt_file,
                            )
                    bn_to_boxes.setdefault(bn, []).append((cat_name, bbox, score))
                else:
                    bn_to_boxes.setdefault(bn, []).append((cat_name, bbox))

    return {}, bn_to_classes, {}, bn_to_boxes


def _load_coco(coco_file: str, with_scores: bool = False) -> Tuple[dict, dict, dict, dict]:
    """
    Parse a COCO JSON file in one pass.
    Returns (fp_to_classes, bn_to_classes, fp_to_boxes, bn_to_boxes) where:
    - fp/bn_to_classes: path → set of class names
    - fp/bn_to_boxes:   path → list of (class_name, [x, y, w, h])

    When ``with_scores`` is True, box entries are ``(class, bbox, score_or_None)``
    3-tuples; ``score_or_None`` is the annotation's ``"score"`` field or ``None``
    when absent (standard for ground-truth COCO files).

    Raises ValueError if the file is not a valid JSON object or is missing the
    'images' / 'annotations' keys (i.e. is not a COCO annotation file).
    """
    try:
        with open(coco_file, encoding="utf-8") as f:
            coco = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid COCO JSON at {coco_file}: {exc}") from exc

    if not isinstance(coco, dict):
        raise ValueError(
            f"COCO file {coco_file} must contain a JSON object, got {type(coco).__name__}"
        )
    if "images" not in coco or "annotations" not in coco:
        raise ValueError(
            f"COCO file {coco_file} is missing 'images'/'annotations' — not a valid COCO annotation file."
        )

    image_id_to_fp = {}
    for img in coco.get("images", []):
        # file_name / filepath / file_path are all seen in the wild
        fp = img.get("file_name") or img.get("filepath") or img.get("file_path") or ""
        image_id = img.get("id")
        if fp and image_id is not None:
            image_id_to_fp[image_id] = fp

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    fp_to_classes: dict = {}
    bn_to_classes: dict = {}
    fp_to_boxes: dict = {}
    bn_to_boxes: dict = {}
    bn_to_fps: dict = {}  # basename -> set of distinct full paths (collision detection)

    for ann in coco.get("annotations", []):
        fp = image_id_to_fp.get(ann.get("image_id"))
        cat_name = cat_id_to_name.get(ann.get("category_id"))
        if not fp or not cat_name:
            continue
        bn = Path(fp).name
        fp_to_classes.setdefault(fp, set()).add(cat_name)
        bn_to_classes.setdefault(bn, set()).add(cat_name)
        bn_to_fps.setdefault(bn, set()).add(fp)
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            if with_scores:
                raw_score = ann.get("score")
                score: Optional[float] = None
                if raw_score is not None:
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        _log.warning(
                            "Invalid score value %r in COCO file — treating as unscored",
                            raw_score,
                        )
                entry = (cat_name, bbox, score)
            else:
                entry = (cat_name, bbox)
            fp_to_boxes.setdefault(fp, []).append(entry)
            bn_to_boxes.setdefault(bn, []).append(entry)

    # A basename shared by multiple distinct image paths is ambiguous — its
    # basename-keyed annotations merge unrelated images. Drop those from the
    # basename fallback maps so a basename-only lookup can't return the union of
    # unrelated images' classes; full-path lookup still resolves these images.
    for bn, fps in bn_to_fps.items():
        if len(fps) > 1:
            bn_to_classes.pop(bn, None)
            bn_to_boxes.pop(bn, None)

    return fp_to_classes, bn_to_classes, fp_to_boxes, bn_to_boxes


_FORMAT_LOADERS = {
    AnnotationFormat.KITTI: _load_kitti,
    AnnotationFormat.COCO: _load_coco,
}


def coco_lookup(fp: str, full_dict: dict, base_dict: dict, default: Any) -> Any:
    """Look up an image's annotation value by full path, falling back to basename then stem.

    Tries ``full_dict[fp]`` first, then ``base_dict`` keyed by the path's basename and finally
    its stem. The stem fallback handles KITTI, whose keys are stored without a file extension.

    Args:
        fp: Image filepath to look up.
        full_dict: Map keyed by full image path.
        base_dict: Map keyed by basename (and effectively stem) for fallback lookups.
        default: Value returned when no full-path, basename, or stem key matches.

    Returns:
        The matched value from ``full_dict`` or ``base_dict``, otherwise ``default``.
    """
    if fp in full_dict:
        return full_dict[fp]
    p = Path(fp)
    return base_dict.get(p.name) or base_dict.get(p.stem) or default
