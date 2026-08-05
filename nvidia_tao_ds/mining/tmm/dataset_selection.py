# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-image class membership for class-stratified mining: flag images by rare/target class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

import cudf

from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.dataset_loading import coco_lookup, load_annotations

if TYPE_CHECKING:
    from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat


def identify_rare_images(
    df: "cudf.DataFrame", detection_file: Optional[str],
    rare_class_list: Set[str], filepath_column_name: str = "filepath",
    *, fmt: AnnotationFormat,
) -> "cudf.Series":
    """Return a boolean Series: True if the image contains at least one rare class."""
    if detection_file is None or not rare_class_list:
        return cudf.Series([False] * len(df), index=df.index)

    fp_to_cls, bn_to_cls, _, _ = load_annotations(detection_file, fmt)
    rare_lower = {cls.lower() for cls in rare_class_list}

    is_rare = [
        bool({c.lower() for c in coco_lookup(fp, fp_to_cls, bn_to_cls, set())} & rare_lower)
        for fp in df[filepath_column_name].to_pandas()
    ]
    return cudf.Series(is_rare, index=df.index)


def identify_images_by_class(
    df: "cudf.DataFrame", detection_file: Optional[str],
    class_name: str, filepath_column_name: str = "filepath",
    *, fmt: AnnotationFormat,
) -> "cudf.Series":
    """Return a boolean Series: True if the image contains the given class."""
    if detection_file is None or not class_name:
        return cudf.Series([False] * len(df), index=df.index)

    fp_to_cls, bn_to_cls, _, _ = load_annotations(detection_file, fmt)
    class_lower = class_name.lower()

    has_class = []
    match_count = 0
    for fp in df[filepath_column_name].to_pandas():
        found = class_lower in {c.lower() for c in coco_lookup(fp, fp_to_cls, bn_to_cls, set())}
        if found:
            match_count += 1
        has_class.append(found)

    logger.info(f"  {class_name} class filtering: {match_count}/{len(df)} images contain {class_name}")
    return cudf.Series(has_class, index=df.index)
