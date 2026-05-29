# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kitti loader."""
import os
import numpy as np

from nvidia_tao_ds.augmentation.utils import kitti
from nvidia_tao_ds.augmentation.utils.file_handlers import load_file
from nvidia_tao_ds.augmentation.utils.helper import encode_str


class KittiInputCallable:
    """KITTI loader for DALI pipeline."""

    def __init__(self, image_dir, label_dir, batch_size,
                 include_masks=False, shard_id=0, num_shards=1):
        """Init."""
        assert os.path.isdir(label_dir)
        self.include_masks = include_masks
        self.image_paths, self.label_paths = kitti.list_files(image_dir, label_dir)
        assert len(self.image_paths) == len(self.label_paths)
        self.labels = [kitti.parse_label_file(lbl) for lbl in self.label_paths]

        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.samples_per_iter = batch_size * num_shards
        self.size = (self.samples_per_iter - 1 + len(self.image_paths)) // self.samples_per_iter * self.samples_per_iter

        self.shard_size = self.size // num_shards
        self.shard_offset = self.shard_size * shard_id
        self.full_iterations = self.shard_size // batch_size

    def __call__(self, sample_info):
        """Call."""
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        sample_idx = sample_info.idx_in_epoch + self.shard_offset
        sample_idx = min(sample_idx, len(self.image_paths) - 1)

        image_path = self.image_paths[sample_idx]
        label_path = self.label_paths[sample_idx]
        label = self.labels[sample_idx]
        boxes = []
        if len(label) == 0:  # fake box for an empty annotation file
            boxes = [0, 0, 0, 0]
        else:
            for annotation in label:
                boxes.append(annotation.box)
        boxes = np.float32(boxes)
        encoded_img = load_file(image_path)

        return encoded_img, boxes, np.uint8(encode_str(image_path)), np.uint8(encode_str(label_path))
