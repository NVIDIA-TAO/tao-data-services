# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO utility functions"""

import cv2
import json
import os
import torch
from PIL import ImageDraw, ImageFont


def get_json_result(anns, ann_type="grounding"):
    """Process dictionary into JSONL element format."""
    regions = []
    if ann_type == "grounding":
        h, w = anns["size"]
        for box, label in zip(anns["boxes"], anns["labels"]):
            regions.append({
                "bbox": box,
                "phrase": label[:-6]
            })
        result = {
            "file_name": os.path.basename(anns["image_name"]),
            "height": h,
            "width": w,
            "grounding": {
                "caption": anns["caption"],
                "regions": regions
            }
        }
    elif ann_type == "detection":
        h, w = anns["size"]
        for box, label, label_id in zip(anns["boxes"], anns["labels"], anns["label_ids"]):
            regions.append({
                "bbox": box,
                "category": label,
                "label": label_id
            })
        result = {
            "file_name": os.path.basename(anns["image_name"]),
            "height": h,
            "width": w,
            "detection": {
                "instances": regions
            }
        }
    else:
        # Second iteration of closed-set detection
        h, w = anns["size"]
        category = anns["caption"].split(" . ")
        category[-1] = category[-1].replace(" .", "")

        for box, label in zip(anns["boxes"], anns["labels"]):
            label = label[:-6]
            regions.append({
                "bbox": box,
                "category": label,
                "label": category.index(label)
            })

        result = {
            "file_name": os.path.basename(anns["image_name"]),
            "height": h,
            "width": w,
            "detection": {
                "instances": regions
            }
        }
    return result


def plot_boxes_to_image(image_pil, tgt):
    """Plot bounding boxes on a image."""
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), f"len of boxes: {len(boxes)} & labels: {len(labels)}"

    draw = ImageDraw.Draw(image_pil)

    outs = []
    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # random color
        color = tuple(torch.randint(0, 255, size=(3,)))
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=20)

        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        draw.rectangle(bbox, fill=color)

        draw.text((x0, y0), str(label), fill="white", font=font)

        outs.append([x0, y0, x1, y1])

    return image_pil


def dump_jsonlines(results, output_path):
    """Store files as JSONL format."""
    with open(output_path, mode="w", encoding="utf-8") as writer:
        for result in results:
            writer.write(f"{json.dumps(result)}\n")


def load_jsonlines(jsonl_path):
    """Load JSONL file."""
    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    return results


def save_results(results, remaining_results, fabric, results_dir, multigpu=False):
    """Save results to JSONL format. Dump individual JSONL file for each rank."""
    if multigpu:
        dump_jsonlines(results,
                       os.path.join(results_dir, f"autolabelled.rank{fabric.local_rank}.jsonl"))
        dump_jsonlines(remaining_results,
                       os.path.join(results_dir, f"remaining.rank{fabric.local_rank}.jsonl"))

        # Wait for all the processes to dump the file
        fabric.barrier()

        # Aggregate the file at rank 0
        with fabric.rank_zero_first(local=False):
            final_results, final_images, final_remaining = [], [], []
            for i in range(fabric.world_size):
                rank_result = load_jsonlines(os.path.join(results_dir, f"autolabelled.rank{i}.jsonl"))

                # DDP dataloader has to set the uniform batch so may duplicate data across each rank
                for rr in rank_result:
                    if rr["file_name"] not in final_images:
                        final_images.append(rr["file_name"])
                        final_results.append(rr)
                    else:
                        fabric.print(f"Skipping {rr['file_name']} from rank {i} due to being a duplicate.")

                final_remaining.extend(load_jsonlines(os.path.join(results_dir, f"remaining.rank{i}.jsonl")))

            dump_jsonlines(final_results,
                           os.path.join(results_dir, "autolabelled.jsonl"))
            dump_jsonlines(final_remaining,
                           os.path.join(results_dir, "remaining.jsonl"))
    else:
        dump_jsonlines(results,
                       os.path.join(results_dir, "autolabelled.jsonl"))
        dump_jsonlines(remaining_results,
                       os.path.join(results_dir, "remaining.jsonl"))
