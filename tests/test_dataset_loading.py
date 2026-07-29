# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nvidia_tao_ds/core/utils/dataset_loading.py (COCO annotation loading).

CPU-only: exercises the pure-Python COCO parser, no RAPIDS/GPU required.
"""

import json
from pathlib import Path

from nvidia_tao_ds.core.utils.dataset_loading import _load_coco


def _write_coco(path, images, categories, annotations):
    """Write a minimal COCO JSON.

    images: list of (id, file_name); categories: list of (id, name);
    annotations: list of (image_id, category_id, [x, y, w, h]).
    """
    coco = {
        "images": [{"id": i, "file_name": fn} for i, fn in images],
        "categories": [{"id": i, "name": n} for i, n in categories],
        "annotations": [
            {"id": k, "image_id": im, "category_id": c, "bbox": b}
            for k, (im, c, b) in enumerate(annotations)
        ],
    }
    Path(path).write_text(json.dumps(coco), encoding="utf-8")
    return str(path)


def test_load_coco_drops_ambiguous_basename(tmp_path):
    """Two images sharing a basename don't merge — the basename fallback drops the collision."""
    coco = _write_coco(
        tmp_path / "ann.json",
        images=[(0, "dirA/frame.png"), (1, "dirB/frame.png")],
        categories=[(1, "person"), (2, "car")],
        annotations=[(0, 1, [0, 0, 1, 1]), (1, 2, [0, 0, 1, 1])],
    )
    fp_to_classes, bn_to_classes, _, _ = _load_coco(coco)

    # Full-path lookups stay precise...
    assert fp_to_classes["dirA/frame.png"] == {"person"}
    assert fp_to_classes["dirB/frame.png"] == {"car"}
    # ...but the ambiguous basename is dropped, so it can't return {person, car}.
    assert "frame.png" not in bn_to_classes


def test_load_coco_keeps_unambiguous_basename(tmp_path):
    """A basename owned by a single image is retained for fallback lookup."""
    coco = _write_coco(
        tmp_path / "ann.json",
        images=[(0, "only/cat.png")], categories=[(1, "dog")],
        annotations=[(0, 1, [0, 0, 1, 1])],
    )
    _, bn_to_classes, _, _ = _load_coco(coco)
    assert bn_to_classes["cat.png"] == {"dog"}
