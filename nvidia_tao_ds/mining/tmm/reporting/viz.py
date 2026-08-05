# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization for unique_neighbor_matching."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from nvidia_tao_ds.core.utils.dataset_loading import coco_lookup, load_annotations

if TYPE_CHECKING:
    from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat


def _load_label_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Try DejaVu Sans Bold (present on the mining container's Ubuntu base); fall back to default."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_bboxes_on_image(img_path: str, boxes: list, highlight_class: Optional[str] = None) -> "Image.Image":
    """Open image and draw bboxes with class labels.

    Focal class: red box (width 4); others: blue box (width 2).
    Label is drawn above the box on a filled background;
    if near the image top, the label drops just inside the box.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    focal = highlight_class.lower() if highlight_class else None
    font = _load_label_font(max(14, int(min(img.size) / 30)))
    pad = 2

    for cat_name, bbox in boxes:
        x, y, w, h = bbox
        is_focal = focal is not None and cat_name.lower() == focal
        color = "red" if is_focal else "blue"
        width = 4 if is_focal else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

        # Measure text; place label above box, or inside top if no room.
        try:
            tb = draw.textbbox((0, 0), cat_name, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = draw.textsize(cat_name, font=font)

        label_y = y - th - 2 * pad if y - th - 2 * pad >= 0 else y + pad
        draw.rectangle([x, label_y, x + tw + 2 * pad, label_y + th + 2 * pad], fill=color)
        draw.text((x + pad, label_y + pad), cat_name, fill="white", font=font)
    return img


def visualize_target_to_mined(
    output_dir: str,
    rare_classes: Set[str],
    source_detection_file: Optional[str],
    target_detection_file: Optional[str],
    subset_name_for_class: Dict[str, str],
    n_targets_per_class: int = 5,
    n_mined_per_target: int = 3,
    seed: int = 42,
    *, fmt: AnnotationFormat,
) -> None:
    """
    For each rare class, save a grid PNG: rows = sampled target images,
    columns = (target, top1, top2, … topN mined), bboxes drawn on each.
    """
    if not rare_classes or not source_detection_file or not target_detection_file:
        print("  [viz] Skipping per-class viz (rare_classes or detection files missing)")
        return

    plt.switch_backend("Agg")  # headless rendering; set at call time, not as an import side effect
    output_path = Path(output_dir)
    rng = random.Random(seed)
    _, _, target_fp_to_boxes, target_bn_to_boxes = load_annotations(target_detection_file, fmt)
    _, _, source_fp_to_boxes, source_bn_to_boxes = load_annotations(source_detection_file, fmt)

    for class_name in sorted(subset_name_for_class):
        subset = subset_name_for_class.get(class_name)
        if not subset:
            continue
        is_non_rare = class_name.lower() == "non_rare"

        # Class-pool iterations first so they win the dedup; fallback files fill gaps
        # for targets that only got matches via the full-source fallback path.
        # "non_rare" has no fallback path.
        iter_files = sorted(output_path.glob(f"{subset}_iteration_*.parquet"))
        if not is_non_rare:
            iter_files += sorted(output_path.glob(f"{subset}_fallback_iteration_*.parquet"))
        if not iter_files:
            print(f"  [viz] No iteration files for '{class_name}' (looked for {subset}_iteration_*.parquet); skipping")
            continue

        target_to_topn: Dict[str, list] = {}
        for itf in iter_files:
            df = pd.read_parquet(itf)
            top_cols = sorted(
                [c for c in df.columns if c.startswith("top") and c.endswith("_source_filepath")],
                key=lambda c: int(c[3:].split("_")[0]),
            )
            for _, row in df.iterrows():
                tgt = row.get("target_filepath")
                if pd.isna(tgt) or tgt in target_to_topn:
                    continue
                target_to_topn[tgt] = [row.get(col) for col in top_cols[:n_mined_per_target] if pd.notna(row.get(col))]

        if is_non_rare:
            candidates = list(target_to_topn.keys())
        else:
            candidates = [
                tgt for tgt in target_to_topn
                if any(c.lower() == class_name.lower()
                       for c, _ in coco_lookup(tgt, target_fp_to_boxes, target_bn_to_boxes, []))
            ]
        if not candidates:
            print(f"  [viz] No candidate targets for '{class_name}'; skipping")
            continue

        sampled = rng.sample(candidates, min(n_targets_per_class, len(candidates)))
        highlight = None if is_non_rare else class_name
        n_cols = 1 + n_mined_per_target
        fig, axes = plt.subplots(len(sampled), n_cols, figsize=(n_cols * 4, len(sampled) * 4), squeeze=False)

        for r, tgt_fp in enumerate(sampled):
            tgt_boxes = coco_lookup(tgt_fp, target_fp_to_boxes, target_bn_to_boxes, [])
            try:
                axes[r][0].imshow(_draw_bboxes_on_image(tgt_fp, tgt_boxes, highlight_class=highlight))
            except Exception as e:
                print(f"  [viz] Could not load target {tgt_fp}: {e}")
                axes[r][0].text(0.5, 0.5, "load failed", ha="center", va="center")
            axes[r][0].set_title(f"target: {Path(tgt_fp).name}", fontsize=8)
            axes[r][0].axis("off")

            mined_fps = target_to_topn.get(tgt_fp, [])
            for c in range(n_mined_per_target):
                ax = axes[r][c + 1]
                if c < len(mined_fps):
                    m_fp = mined_fps[c]
                    m_boxes = coco_lookup(m_fp, source_fp_to_boxes, source_bn_to_boxes, [])
                    try:
                        ax.imshow(_draw_bboxes_on_image(m_fp, m_boxes, highlight_class=highlight))
                        ax.set_title(f"top{c + 1}: {Path(m_fp).name}", fontsize=8)
                    except Exception as e:
                        print(f"  [viz] Could not load mined {m_fp}: {e}")
                        ax.text(0.5, 0.5, "load failed", ha="center", va="center")
                else:
                    ax.text(0.5, 0.5, "(no match)", ha="center", va="center")
                ax.axis("off")

        suptitle = (
            "Non-rare pool  (all bboxes shown in blue)" if is_non_rare
            else f"Class: {class_name}  (red = {class_name} bbox, blue = other)"
        )
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        out_file = output_path / f"viz_class_{class_name}.png"
        fig.savefig(out_file, dpi=100)
        plt.close(fig)
        print(f"  [viz] Saved: {out_file}")


def plot_per_class_counts(
    target_per_class: Dict[str, Dict[str, int]],
    resultant_per_class: Dict[str, Dict[str, int]],
    output_dir: Path,
) -> None:
    """Save grouped bar plots comparing target vs resultant per-class counts."""
    classes = sorted(set(target_per_class) | set(resultant_per_class))
    if not classes:
        return

    plt.switch_backend("Agg")  # headless rendering; set at call time, not as an import side effect

    for metric, fname in [
        ("image_count", "image_count_per_class.png"),
        ("instance_count", "instance_count_per_class.png"),
    ]:
        target_vals = [target_per_class.get(c, {}).get(metric, 0) for c in classes]
        resultant_vals = [resultant_per_class.get(c, {}).get(metric, 0) for c in classes]
        x = np.arange(len(classes))
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(classes) + 4), 5))
        ax.bar(x - width / 2, target_vals, width, label="target", color="#4C72B0")
        ax.bar(x + width / 2, resultant_vals, width, label="resultant", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ').title()} per class")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Plot saved to: {out_path}")
