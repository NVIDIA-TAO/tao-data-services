# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle coco operations."""

import pandas
from pycocotools.coco import COCO
import os
import json

from nvidia_tao_ds.core.logging.logging import logger


def correct_data(coco_obj, output_dir):
    """
    Correct the invalid coco annotations.
    Correction criteria :
        set bounding box values = 0 if their values are less than 0.
        set x_max=img_width if x_max>img_width.
        set y_max=img_height if y_max>img_height.
        swap inverted bouding box coordinates.
    Args:
        coco_obj (DataFormat): object of coco data format.
        output_dir (str): output directory.

    Return:
        No explicit returns.
    """
    coco = COCO(coco_obj.ann_path)
    for annots in coco.anns.values():
        bbox = annots['bbox']
        bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        image_id = annots['image_id']
        image_data = coco.loadImgs([image_id])[0]
        height = image_data["height"]
        width = image_data['width']
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin = max(xmin, 0)
        xmax = max(xmax, 0)
        ymin = max(ymin, 0)
        ymax = max(ymax, 0)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        xmax = min(xmax, width)
        ymax = min(ymax, height)
        annots['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]

    final_dict = {"annotations": list(coco.anns.values()),
                  "images": list(coco.imgs.values()),
                  "categories": list(coco.cats.values())}
    # save the corrected coco annotation file.
    basename = os.path.basename(coco_obj.ann_path)
    save_path = os.path.join(output_dir, f"{basename}")

    with open(save_path, "w", encoding='utf-8') as f:
        json.dump(final_dict, f)
    logger.info(f"Corrected coco file is available at {save_path}")


def get_image_data(coco_obj):
    """ Get image width and height.

    Args:
        coco_obj (DataFormat): object of coco data format.

    Returns:
        image_data(dict): Dictionary of image name.
            mapped to image width and height.
    """
    coco = COCO(coco_obj.ann_path)
    image_dir = coco_obj.image_dir
    image_data = {}
    for img in coco.imgs.values():
        img_fname = img['file_name']
        width = img['width']
        height = img['height']
        image_data[img_fname] = [width, height, os.path.join(image_dir, img_fname)]
    return image_data


def create_image_dataframe(image_data):
    """ Create image data frame.

    Args:
        image_data(Dict): image data dictionary.
    Returns:
        No explicit returns.
    """
    image_df = pandas.DataFrame.from_dict(image_data, orient='index',
                                          columns=['img_width', 'img_height', 'path'])
    image_df['size'] = image_df['img_width'] * image_df['img_height']
    return image_df


def is_valid(bbox, width, height):
    """ Check if bbox coordinates are valid.

    Args:
        bbox(list): bbox coordinates.
        width(float): image width.
        height(float): image height.
    Returns:
        Bool: True if coordinates are valid else False.
        reason: list of reason for invalidaity.
    """
    reason = []
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
        reason.append("INVALID_OUT_OF_BOX_COORD")
    if (xmax - xmin) == 0 or (ymax - ymin) == 0:
        reason.append("INVALID_ZERO_COORD")
    if xmax < xmin or ymax < ymin:
        reason.append("INVALID_INVERTED_COORD")
    if (xmax > width or ymax > height):
        if "INVALID_OUT_OF_BOX_COORD" not in reason:
            reason.append("INVALID_OUT_OF_BOX_COORD")
    if len(reason) > 0:
        return False, reason
    return True, None


def create_dataframe(coco_obj):
    """Create DataFrame from coco annotation file.

    Args:
        coco_obj (DataFormat): object of coco data format.

    Returns:
        valid_df (Pandas DataFrame): output valid dataframe of kitti data.
        invalid_df (Pandas DataFrame): output invalid dataframe of kitti data.

    """
    coco = COCO(coco_obj.ann_path)
    valid_data_list = []
    invalid_data_list = []
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    image_dir = ""
    if coco_obj.image_dir:
        image_dir = coco_obj.image_dir
    category_id_dict = {}
    for category in categories:
        category_id_dict[category['id']] = category['name']

    for annots in coco.anns.values():
        bbox = annots['bbox']
        bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        image_id = annots['image_id']
        category_id = annots['category_id']
        category_name = category_id_dict[category_id]
        image_data = coco.loadImgs([image_id])[0]
        image_name = image_data['file_name']
        height = image_data["height"]
        width = image_data['width']
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        valid, reason = is_valid(bbox, width, height)
        image_path = os.path.join(image_dir, image_name)
        if valid:
            valid_data_list.append([category_name, bbox[0], bbox[1], bbox[2], bbox[3], image_name,
                                    width, height, bbox_area, image_path])
        else:
            out_of_box, zero_area, inverted_coord = 'False', 'False', 'False'
            if "INVALID_OUT_OF_BOX_COORD" in reason:
                out_of_box = 'True'
            if "INVALID_ZERO_COORD" in reason:
                zero_area = 'True'
            if "INVALID_INVERTED_COORD" in reason:
                inverted_coord = 'True'
            invalid_data_list.append([category_name, bbox[0], bbox[1], bbox[2], bbox[3], image_name,
                                      width, height, out_of_box, zero_area, inverted_coord, bbox_area, image_path])

    valid_df = pandas.DataFrame(valid_data_list, columns=['type', 'bbox_xmin',
                                                          'bbox_ymin', 'bbox_xmax',
                                                          'bbox_ymax', 'image_name',
                                                          'img_width', 'img_height',
                                                          'bbox_area', 'img_path'])

    invalid_df = pandas.DataFrame(invalid_data_list, columns=['type', 'bbox_xmin',
                                                              'bbox_ymin', 'bbox_xmax',
                                                              'bbox_ymax', 'image_name',
                                                              'img_width', 'img_height',
                                                              'out_of_box_coordinates',
                                                              'zero_area_bounding_box',
                                                              'inverted_coordinates',
                                                              'bbox_area', 'img_path'])

    return valid_df, invalid_df
