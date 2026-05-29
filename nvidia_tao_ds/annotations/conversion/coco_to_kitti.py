# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert COCO annotations to KITTI format"""

import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask


def find_corners_2d(np_array):
    """Find corners in a 2D binary mask."""
    result = np.where(np_array == np.amax(np_array))
    x1 = np.min(result[0])
    x2 = np.max(result[0])
    y1 = np.min(result[1])
    y2 = np.max(result[1])
    return y1, x1, y2, x2


def convert_coco_to_kitti(cfg=None,
                          annotations_file=None,
                          output_dir=None,
                          refine_box=False,
                          verbose=False):
    """Function to convert COCO annotations to KITTI format.

    Args:
        cfg (dataclass): Hydra Config.
        annotations_file (string): Path to the COCO annotation file.
        output_dir (string): Directory to output the KITTI files.
        refine_box (bool): Whether to refine boxes for segmentation. Default to False.
        verbose (bool): verbosity. Default is False.
    """
    if cfg is not None:
        annotations_file = cfg.coco.ann_file
        output_dir = cfg.results_dir
        refine_box = cfg.get("refine_box", False)

    if not os.path.isfile(annotations_file):
        raise FileNotFoundError("Annotation file does not exist. Please check the path.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    coco = COCO(annotations_file)
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)

    category_id_dict = {}
    for category in categories:
        category_id_dict[category['id']] = category['name']
    for img, ann in coco.imgs.items():
        annotation_ids = coco.getAnnIds(imgIds=[img], catIds=category_ids)
        if len(annotation_ids) > 0:
            img_fname = ann['file_name']
            label_fname = os.path.basename(img_fname).split('.')[0]

            with open(os.path.join(output_dir, f'{label_fname}.txt'), 'w', encoding='utf-8') as label_file:
                annotations = coco.loadAnns(annotation_ids)
                for annotation in annotations:
                    bbox = annotation['bbox']
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    if refine_box and annotation.get('segmentation', None):
                        rle = annotation['segmentation']
                        binary_mask = mask.decode(rle)
                        bbox = find_corners_2d(binary_mask)
                    bbox = [str(b) for b in bbox]
                    catname = category_id_dict[annotation['category_id']]
                    out_str = [catname.replace(" ", "") + ' ' + ' '.join(['0'] * 3) + ' ' + ' '.join(list(bbox)) + ' ' + ' '.join(['0'] * 7) + '\n']
                    label_file.write(out_str[0])
