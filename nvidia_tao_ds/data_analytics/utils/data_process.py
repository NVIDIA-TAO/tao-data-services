# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle processing of different data formats."""

from abc import ABC, abstractmethod
import glob
import pandas as pd
import os
from pycocotools.coco import COCO


def create_data_process_object(data_format, data_obj):
    """ Initialize and return coco or kitti data process object based on the data_format.
    Args:
        data_format(str): format of data.
        data_obj(DataFormat obj): data format object.
    Return:
        kitti or coco data processing object.
    """
    if data_format == "KITTI":
        return KittiDataProcess(data_obj)
    if data_format == "COCO":
        return CoCoDataProcess(data_obj)
    raise ValueError(data_format)


class DataProcess(ABC):
    """Class data process"""

    @abstractmethod
    def create_dataframe(self, Data, data_source):
        """Create Dataframe from Data object. """
        pass


class CoCoDataProcess(DataProcess):
    """Class COCO data process"""

    def __init__(self, data_object):
        """Initialize a Coco data process object.

        Args:
            data_object(DataFormat): object of DataFormat class.
        """
        self._data_obj = data_object

    def create_dataframe(self):
        """Create DataFrame from coco annotation file.

        Returns:
            df (Pandas DataFrame): output dataframe of COCO data.

        """
        ann_file = self._data_obj.ann_path
        coco = COCO(ann_file)
        valid_data_list = []
        category_ids = coco.getCatIds()
        categories = coco.loadCats(category_ids)
        image_dir = ""
        if self._data_obj.image_dir:
            image_dir = self._data_obj.image_dir
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
            image_name = image_data['file_name'].split(".")[0]
            height = image_data["height"]
            width = image_data['width']
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            image_path = os.path.join(image_dir, image_name)
            d_list = [category_name, bbox[0], bbox[1], bbox[2],
                      bbox[3], image_name, width, height, image_path, bbox_area]
            if self._data_obj.data_source == "predicted":
                d_list.append(annots['score'])
            valid_data_list.append(d_list)
        df_columns = list(self._data_obj.get_columns().keys())
        df = pd.DataFrame(valid_data_list, columns=df_columns)
        return df


class KittiDataProcess(DataProcess):
    """Class KITTI data process"""

    def __init__(self, data_object):
        """Initialize a KITTI data process object.

        Args:
            data_object(DataFormat): object of DataFormat class.
        """
        self._data_obj = data_object

    def get_label_paths(self):
        """Return all kitti label file paths. """
        label_dir = self._data_obj.ann_path
        # List label files.
        labels = glob.glob(os.path.join(label_dir, '**/*.txt'), recursive=True)
        labels = sorted(labels)
        return labels, [os.path.basename(label).replace(".txt", "") for label in labels]

    def create_dataframe(self):
        """Create DataFrame from kitti label files.

        Returns:
            df (Pandas DataFrame): output dataframe of KITTI data.

        """
        dtype = self._data_obj.get_columns()
        name_list = list(dtype.keys())
        kitti_file_paths, ids = self.get_label_paths()
        self._data_obj.label_paths = kitti_file_paths
        df_list = [pd.read_csv(filepath, sep=' ', names=name_list, dtype=dtype,
                               index_col=False).assign(img_name=os.path.basename(filepath).split(".")[0]
                                                       ).astype(str) for filepath in kitti_file_paths]
        df = pd.concat(df_list)
        return df, ids
