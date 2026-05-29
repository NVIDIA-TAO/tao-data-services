# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DALI pipeline runner."""
import os
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image
from tqdm import tqdm

from nvidia_tao_ds.augmentation.utils import kitti
from nvidia_tao_ds.augmentation.utils.helper import decode_str


def save_image(img, path):
    """Write PIL image to path."""
    img.save(path)


def process_augmented_coco(image_id, image, boxes_per_image, masks_per_image,
                           coco, config):
    """Process augmented COCO data."""
    ann_load = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    img_info = coco.loadImgs(ids=image_id)[0]
    img_name = img_info['file_name']
    out_image_path = os.path.join(config.results_dir, 'images', img_name)
    ann_per_image = []
    for j in range(len(ann_load)):
        x1, y1, x2, y2 = list(map(lambda x: float(x), list(boxes_per_image[j])))
        ann_load[j]['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        if config.data.include_masks:
            # mask --> RLE
            encoded_mask = maskUtils.encode(
                np.asfortranarray(masks_per_image[..., j].astype(np.uint8)))
            encoded_mask['counts'] = encoded_mask['counts'].decode('ascii')
            ann_load[j]['segmentation'] = encoded_mask
        ann_per_image.append(ann_load[j])
    save_image(Image.fromarray(image), out_image_path)
    return ann_per_image


def process_augmented_kitti(image, boxes_per_image,
                            encoded_image_path,
                            encoded_label_path,
                            config):
    """Process augmented KITTI data."""
    image_path = decode_str(encoded_image_path)
    image_name = os.path.basename(image_path)
    label_path = decode_str(encoded_label_path)
    label_name = os.path.basename(label_path)

    output_image_dir = os.path.join(config.results_dir, 'images')
    output_label_dir = os.path.join(config.results_dir, 'labels')
    # save augmented image
    save_image(Image.fromarray(image), os.path.join(output_image_dir, image_name))
    # dump kitti file with augmented labels
    with open(os.path.join(output_label_dir, label_name), "w", encoding='utf-8') as f:
        annotations = kitti.parse_label_file(label_path)
        if annotations:
            for j in range(boxes_per_image.shape[0]):
                annotation = annotations[j]
                annotation.box = boxes_per_image[j]
                f.write(str(annotation))
                f.write('\n')
        else:
            f.write('')


class DALIPipeIter():
    """Dali pipe iterator."""

    def __init__(self, pipe):
        """Initialization of the pipeline iterator.

        Args:
            pipe (dali.pipeline.Pipeline): Dali pipeline object.

        Returns:
            DALIPipeIter object class.
        """
        self.pipe = pipe

    def __iter__(self):
        """Return interator."""
        return self

    def __next__(self):
        """Next method for the DALI iterator.

        This method runs the DALI pipeline and generates the
        pipes outputs.
        """
        return self.pipe.run()


def run(pipe, data_callable, config):
    """Run pipeline."""
    if config.data.dataset_type.lower() == 'coco':
        ann_dump = []
        img_id_set = set()
        with tqdm(total=data_callable.size) as pbar:
            for images, boxes, img_ids, masks in DALIPipeIter(pipe):

                images = images.as_cpu()
                img_ids = img_ids.as_array().flatten()
                for i, img_id in enumerate(img_ids):
                    img_id = int(img_id)
                    if img_id not in img_id_set:
                        img_id_set.add(img_id)
                        ann_dump.extend(
                            process_augmented_coco(
                                img_id,
                                images.at(i),
                                boxes.at(i),
                                masks.at(i),
                                data_callable.coco,
                                config))
                pbar.update(data_callable.samples_per_iter)
        return ann_dump

    if config.data.dataset_type.lower() == 'kitti':
        with tqdm(total=data_callable.size) as pbar:
            for images, boxes, img_paths, lbl_paths in DALIPipeIter(pipe):
                images = images.as_cpu()

                for i in range(len(images)):
                    process_augmented_kitti(
                        images.at(i),
                        boxes.at(i),
                        img_paths.at(i),
                        lbl_paths.at(i),
                        config)
                pbar.update(data_callable.samples_per_iter)
        return 0
    raise ValueError("Only `kitti` and `coco` are supported in dataset_type.")
