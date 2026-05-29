# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert ODVG annotations to COCO format"""

import os
import json

from tqdm.auto import tqdm


def xyxy_to_xywh(bbox):
    """Convert xywh to xyxy."""
    x1, y1, x2, y2 = bbox
    x1 = round(x1, 2)
    y1 = round(y1, 2)
    w = round(x2 - x1, 2)
    h = round(y2 - y1, 2)
    return [x1, y1, w, h]


def process_odvg(metas, add_background=False):
    """Process ODVG jsonl file"""
    img_id, ann_id = 0, 0
    imgs, anns = [], []
    for meta in tqdm(metas, total=len(metas)):
        imgs.append({
            "file_name": meta["file_name"],
            "height": meta["height"],
            "width": meta["width"],
            "id": img_id
        })

        for instance in meta["detection"]["instances"]:
            x1, y1, x2, y2 = instance["bbox"]
            if any([x1 > x2, y1 > y2]):
                print(f"Invalid annotation at object with coordinates [{x1}, {y1}, {x2}, {y2}]. Skipping this object.")
                continue
            x, y, w, h = xyxy_to_xywh(instance["bbox"])
            area = w * h
            category_id = instance["label"] + 1 if add_background else instance["label"]

            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "area": area,
                "bbox": [x, y, w, h],
                "iscrowd": 0
            })
            ann_id += 1
        img_id += 1
    return imgs, anns


def convert_odvg_to_coco(cfg, verbose=False):
    """Function to convert ODVG annotations to COCO format.

    Args:
        cfg (dataclass): Hydra Config.
        verbose (bool): verbosity. Default is False.
    """
    odvg_jsonl_path = cfg.odvg.ann_file
    add_background = cfg.coco.add_background
    coco_json_path = os.path.join(cfg.results_dir, os.path.basename(odvg_jsonl_path).replace(".jsonl", ".json"))

    labelmap_path = cfg.odvg.labelmap_file
    if labelmap_path:
        is_grounding = False
    else:
        is_grounding = True

    if not is_grounding:
        with open(odvg_jsonl_path, mode="r", encoding="utf-8") as f:
            metas = [json.loads(line) for line in f]

        with open(labelmap_path, mode="r", encoding="utf-8") as f:
            labelmap = json.load(f)

        categories = []
        if add_background:
            categories.append(
                {
                    "supercategory": "background",  # supercategory info is lost in ODVG
                    "id": 0,  # detection id needs to start from 1
                    "name": "background"
                }
            )
        for label_id, label in labelmap.items():
            categories.append(
                {
                    "supercategory": label,  # supercategory info is lost in ODVG
                    "id": int(label_id) + 1,  # detection id needs to start from 1
                    "name": label
                }
            )

        images, annotations = process_odvg(metas, add_background=add_background)

        result = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        with open(coco_json_path, mode="w", encoding="utf-8") as f:
            json.dump(result, f)

    print(f"COCO annotation file is stored at {coco_json_path}")
