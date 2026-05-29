# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert KITTI labels to COCO format"""

from collections import OrderedDict
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ujson
import yaml
from tqdm.contrib.concurrent import process_map
import csv
import imagesize


def read_kitti_labels(label_file):
    """Utility function to read a KITTI label file.

    Args:
        label_file (string): Label file path.

    Returns:
        label_list (list): List of labels.
    """
    label_list = []
    if not os.path.exists(label_file):
        raise ValueError(f"Labelfile : {label_file} does not exist")
    with open(label_file, 'r', encoding='utf-8') as lf:
        for row in csv.reader(lf, delimiter=' '):
            label_list.append(row)
    lf.closed
    return label_list


def check_bbox_coordinates(coord, img_h, img_w):
    """Utility function to validate the bounding box coordinates.

    Args:
        coord (tuple): Bounding box coordinates in KITTI format.
        img_h (int): Image height.
        img_w (int): Image widith.
        label_file (string): Label file path.

    Returns:
        Bounding box coordinates
    """
    x1, y1, x2, y2 = coord
    x1 = min(max(x1, 0), img_w)
    x2 = min(max(x2, 0), img_w)
    y1 = min(max(y1, 0), img_h)
    y2 = min(max(y2, 0), img_h)
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None


def convert_xyxy_to_xywh(coord):
    """Utility function to convert bounding box coordinates from KITTI format to COCO.

    Args:
        coord (tuple): Bounding box coordinates in KITTI format.
        img_h (int): Image height.
        img_w (int): Image widith.
        label_file (string): Label file path.

    Returns:
        Bounding box coordinates in COCO format
    """
    x1, y1, x2, y2 = coord
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


def get_categories(cat_map):
    """Function to convert the category map to COCO annotation format.

    Args:
        cat_map (dictionary): Category map.

    Returns:
        categories_list (list): COCO annotation format of the category map.
    """
    categories_list = []
    for i, class_name in enumerate(cat_map):
        category = {
            'id': i + 1,
            'name': class_name
        }
        categories_list.append(category)
    return categories_list


def construct_category_map(label_dir, mapping=None):
    """Function to create a category map for the given dataset.

    Args:
        label_dir (str): Label directory.
        mapping (str): Mapping file.

    Returns:
        cat_map (dictionary): Category mapping
    """
    cat_map = OrderedDict()
    if mapping is not None:
        with open(mapping, "r", encoding='utf-8') as f:
            try:
                cat_map_list = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
            for i in cat_map_list:
                k, v = list(i.items())[0]
                cat_map[k] = v
    else:
        for img in os.listdir(label_dir):
            labels = read_kitti_labels(os.path.join(label_dir, f"{img[:-4]}.txt"))
            df = pd.DataFrame(labels)
            for _, row_p in df.iterrows():
                if row_p[0] not in cat_map:
                    cat_map[row_p[0]] = [row_p[0]]

    return cat_map


def _convert(args):
    """Function to process single image and label for multiprocessing.

    Args:
        args (tuple): Contains image directory, image file name, label directory,
            category mapping, and category to id mapping.

    Returns:
        img_dict (dict): image dictionary.
        annot_list (list): list of annotation dictionary.
        include_image (bool): whether to include this image or not.
    """
    img_dir, img, label_dir, labels2cat, cat2index, preserve_hierarchy = args
    if preserve_hierarchy:
        # Last two dirs + img name.
        # In KITTI: training/image_2/0.jpg
        # In Astro: 20ft_45_degrees_4.mp4/images_final_hres/0.jpg
        filename = os.path.join(*Path(img_dir).parts[-2:], img)
    else:
        filename = img
    include_image = False

    if str(Path(img).suffix).lower() not in [".jpg", ".png", ".jpeg"]:
        return {"file_name": filename}, [], include_image

    labels = os.path.join(label_dir, f"{img[:-4]}.txt")
    # skip images w/o annotations
    if os.path.getsize(labels) <= 0:
        return {"file_name": filename}, [], include_image
    width, height = imagesize.get(os.path.join(img_dir, img))

    # update image list
    img_dict = {
        "file_name": filename,
        "height": height,
        "width": width,
    }

    # process labels
    bboxes = read_kitti_labels(labels)
    df = pd.DataFrame(bboxes)
    df = df.drop_duplicates()

    annot_list = []
    # update annotation list
    for _, row_p in df.iterrows():
        mapped = labels2cat.get(row_p[0], None)

        if not mapped:
            continue

        bbox = np.array(row_p[4:8])
        bbox = bbox.astype(float)

        coord = check_bbox_coordinates(bbox.tolist(), height, width)
        if not coord:
            continue
        include_image = True
        coord = convert_xyxy_to_xywh(coord)
        area = coord[2] * coord[3]
        annot_dict = {
            "bbox": coord,
            "image_id": img,
            "iscrowd": 0,
            "area": area,
            "category_id": cat2index[mapped],
        }
        annot_list.append(annot_dict)

    return (img_dict, annot_list, include_image)


def convert_kitti_to_coco(cfg=None,
                          img_dir=None,
                          label_dir=None,
                          output_dir=None,
                          mapping=None,
                          name=None,
                          no_skip=False,
                          preserve_hierarchy=False,
                          verbose=False):
    """Function to convert KITTI annotations to COCO format.

    Args:
        cfg (dataclass): Hydra Config.
        img_dir (string): Directory containing the images.
        label_dir (string): Directory containing the KITTI labels.
        output_dir (string): Directory to output the COCO annotation file.
        mapping (str): Mapping file.
        name (str): Name of the output JSON file.
        no_skip (bool): If True, do not skip images without any valid annotations
        preserve_hierarchy (bool): If True, preserve the KITTI folder structure
        verbose (bool): verbosity. Default is False.
    """
    if cfg is not None:
        img_dir = cfg.kitti.image_dir
        label_dir = cfg.kitti.label_dir
        output_dir = cfg.results_dir
        mapping = cfg.kitti.mapping
        name = cfg.kitti.project
        no_skip = cfg.kitti.no_skip
        preserve_hierarchy = cfg.kitti.preserve_hierarchy

    annot_list, img_list, skipped_list = [], [], []
    img_id, obj_id = 0, 0
    img_dir = str(Path(img_dir))
    label_dir = str(Path(label_dir))
    project = name or img_dir.split('/')[-2]

    os.makedirs(output_dir, exist_ok=True)

    cat_map = construct_category_map(label_dir, mapping)
    categories = get_categories(cat_map)
    labels2cat = {label: k for k, v in cat_map.items() for label in v}
    cat2index = {c['name']: c['id'] for c in categories}

    if verbose:
        print("category to id mapping:")
        print("*********************")
        print(cat2index)
        print("*********************")

    # Prepare arguments for multiprocessing
    args = []
    for img in os.listdir(img_dir):
        args.append((img_dir, img, label_dir, labels2cat, cat2index, preserve_hierarchy))

    # Run multiprocessing
    results = process_map(_convert, args, chunksize=1)

    # Iterate again to update the ids
    for img_dict, anns, include_image in results:
        fname = img_dict['file_name']

        # Skip files that are not image
        if str(Path(fname).suffix).lower() not in [".jpg", ".png", ".jpeg"]:
            skipped_list.append(fname)
            continue

        # Skip files if invalid and no_skip=True
        if not no_skip and not include_image:
            skipped_list.append(fname)
            continue

        img_id += 1
        img_dict['id'] = img_id
        img_list.append(img_dict)

        for ann in anns:
            obj_id += 1
            ann['id'] = obj_id
            ann['image_id'] = img_id
            annot_list.append(ann)

    final_dict = {
        "annotations": annot_list,
        "images": img_list,
        "categories": categories
    }
    save_path = os.path.join(output_dir, f"{project}.json")

    with open(save_path, "w", encoding='utf-8') as f:
        ujson.dump(final_dict, f)

    if skipped_list:
        with open(os.path.join(output_dir, 'skipped_files.txt'), 'w', encoding='utf-8') as g:
            g.write('\n'.join(str(fname) for fname in skipped_list))
