# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO dataset used for auto-labeling."""

import os
from PIL import Image

import torch
from torch.utils.data import DataLoader

from torch.utils.data.dataset import Dataset
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import collate_fn
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.coco import ODPredictDataset
from nvidia_tao_ds.auto_label.grounding_dino.utils import load_jsonlines


class AutolabelDataset(Dataset):
    """Base Object Detection Predict Dataset Class."""

    def __init__(self, root, anno, transforms=None):
        """Initialize the Object Detetion Dataset Class for inference.

        Unlike ODDataset, this class does not require COCO JSON file.

        Args:
            dataset_list (list): list of dataset directory.
            captions (list): list of captions.
            transforms: augmentations to apply.

        Raises:
            FileNotFoundErorr: If provided classmap, sequence, or image extension does not exist.
        """
        self.root = root
        self.transforms = transforms
        self._load_metas(anno)
        self.get_dataset_info()

    def _load_metas(self, anno):
        """Load ODVG jsonl file"""
        self.metas = load_jsonlines(anno)

    def get_dataset_info(self):
        """print dataset info."""
        print(f"  == total images: {len(self)}")

    def _load_image(self, img_path: int) -> Image.Image:
        """Load image given image path.

        Args:
            img_path (str): image path to load.

        Returns:
            Loaded PIL.Image.
        """
        return_output = (Image.open(img_path).convert("RGB"), img_path)

        return return_output

    def cleanup_nouns(self, noun_chunks):
        """Remove whitespaces and duplicates."""
        return list({nc.strip() for nc in noun_chunks})

    def __getitem__(self, index: int):
        """Get image, target, image_path given index.

        Args:
            index (int): index of the image id to load.

        Returns:
            (image, target, image_path): pre-processed image, target and image_path for the model.
        """
        meta = self.metas[index]
        img_path = os.path.join(self.root, meta['file_name'])
        caption = meta['caption']
        noun_chunks = meta['noun_chunks']
        noun_chunks = self.cleanup_nouns(noun_chunks)

        image, image_path = self._load_image(img_path)

        width, height = image.size
        target = {}
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])
        target["caption"] = ' . '.join(noun_chunks) + ' .'
        target["cat_list"] = noun_chunks
        target["full_caption"] = caption

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.metas)


def setup_dataloader(root_dir, augmentation, batch_size, num_workers, json_file=None, captions=None):
    """Setup dataloader for depending on the task."""
    transforms = build_transforms(augmentation, subtask_config=None, dataset_mode='infer')
    if json_file:
        dataset = AutolabelDataset(root_dir, json_file, transforms)
    elif captions:
        dataset = ODPredictDataset([root_dir], captions, transforms)
    else:
        raise NotImplementedError("Either dataset.class_names or dataset.noun_chunk_path must be passed.")

    sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
        collate_fn=collate_fn)
    return dataloader
