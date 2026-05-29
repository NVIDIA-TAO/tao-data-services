# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle different data formats."""
import numpy as np
from collections import OrderedDict


class DataFormat:
    """Class data format"""

    def __init__(self, image_dir, ann_path, data_source="ground_truth"):
        """Initialize a data object.

        Args:
            ann_path(str): path to annotation file.
            image_dir(str): path to image dir.
            data_source(str): data source(ground truth or predicted)

        """
        self.ann_path = ann_path
        self.image_dir = image_dir
        self.data_source = data_source
        self.ids = None


class CocoData(DataFormat):
    """COCO data format"""

    def get_columns(self):
        """ Return columns for COCO data file. """
        columns = OrderedDict([('type', str),
                               ('bbox_xmin', np.dtype('float')),
                               ('bbox_ymin', np.dtype('float')),
                               ('bbox_xmax', np.dtype('float')),
                               ('bbox_ymax', np.dtype('float')),
                               ('img_name', str),
                               ('img_width', np.dtype('float')),
                               ('img_height', np.dtype('float')),
                               ('img_full_path', str),
                               ('bbox_area', np.dtype('float'))])
        if self.data_source == "predicted":
            columns['conf_score'] = np.dtype('float')
        return columns


class KittiData(DataFormat):
    """KITTI data format"""

    def __init__(self, image_dir, ann_path, data_source="ground_truth") -> None:
        """Initialize a KITTI object.

        Args:
            image_dir(str): path to image dir.
            ann_path(str): path to label dir.
            data_source(str): data source(ground truth or predicted)

        """
        super().__init__(image_dir, ann_path, data_source)
        self.image_paths = None
        self.label_paths = None

    def get_columns(self):
        """ Return columns of kitti data file. """
        columns = OrderedDict([('type', str), ('truncated', np.dtype('float')),
                               ('occluded', np.dtype('int32')),
                               ('alpha', np.dtype('float')),
                               ('bbox_xmin', np.dtype('float')),
                               ('bbox_ymin', np.dtype('float')),
                               ('bbox_xmax', np.dtype('float')),
                               ('bbox_ymax', np.dtype('float')),
                               ('dim_height', np.dtype('float')),
                               ('dim_width', np.dtype('float')),
                               ('dim_length', np.dtype('float')),
                               ('loc_x', np.dtype('float')),
                               ('loc_y', np.dtype('float')),
                               ('loc_z', np.dtype('float')),
                               ('rotation_y', np.dtype('float'))])
        if self.data_source == "predicted":
            columns['conf_score'] = np.dtype('float')
        return columns


def create_data_object(data_format, ann_path, image_dir, data_source="ground_truth"):
    """ Initialize and return coco or kitti data object based on the data_format.
    Args:
        data_format(str): format of data.
        image_dir(str): path to image dir.
        ann_path(str): path to label dir.
        data_source(str): data source(ground truth or predicted)
    Return:
        kitti or coco data object.
    """
    if data_format == "KITTI":
        return KittiData(image_dir, ann_path, data_source)
    if data_format == "COCO":
        return CocoData(image_dir, ann_path, data_source)
    raise ValueError(data_format)
