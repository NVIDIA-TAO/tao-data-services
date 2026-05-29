# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle kitti annotations."""

from pathlib import Path
import glob
import os

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def list_files(image_dir, label_dir):
    """List images and labels file in the dataset directory.

    Args:
        root_dir (str): Unix path to the dataset root.

    Returns:
        output_image(str), output_label (str): List of output images and output labels
    """
    images = []
    # List image files.
    for ext in IMAGE_EXTENSIONS:
        images.extend(
            glob.glob(
                os.path.join(image_dir, f'*{ext}'),
            )
        )
    # List label files.
    labels = glob.glob(os.path.join(label_dir, '*.txt'))

    image_names = list(map(lambda x: Path(x).stem, images))
    label_names = list(map(lambda x: Path(x).stem, labels))
    image_dict = dict(zip(image_names, images))
    label_dict = dict(zip(label_names, labels))
    common = set(image_names).intersection(set(label_names))
    return zip(*[(image_dict[i], label_dict[i]) for i in common])


class Annotation:
    """Label annotation object corresponding to a single line in the kitti object."""

    def __init__(self, *args):
        """Initialize a kitti label object.

        Args:
            args[list]: List of kitti labels.

        """
        self.category = args[0]
        self.truncation = float(args[1])
        self.occlusion = int(args[2])
        self.observation_angle = float(args[3])
        self.box = [float(x) for x in args[4:8]]
        hwlxyz = [float(x) for x in args[8:14]]
        self.world_bbox = hwlxyz[3:6] + hwlxyz[0:3]
        self.world_bbox_rot_y = float(args[14])

    def __str__(self):
        """String representation of annotation object."""
        world_box_str = "{3:.2f} {4:.2f} {5:.2f} {0:.2f} {1:.2f} {2:.2f}".format(*self.world_bbox)  # noqa pylint: disable=C0209
        box_str = "{:.2f} {:.2f} {:.2f} {:.2f}".format(*self.box)  # noqa pylint: disable=C0209
        return "{0} {1:.2f} {2} {3:.2f} {4} {5} {6:.2f}".format(  # noqa pylint: disable=C0209
            self.category, self.truncation, self.occlusion, self.observation_angle,
            box_str, world_box_str, self.world_bbox_rot_y)


def parse_label_file(label_file):
    """Parse a label file.

    Args:
        label_file (str): Unix path to the kitti label file.

    Returns:
        annotations (list): List of parsed kitti labels.
    """
    with open(label_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        if line[-1] == '\n':
            line = line[:-1]
            # skip empty line
            if not line:
                continue
        tokens = line.split(" ")
        annotations.append(Annotation(*tokens))
    return annotations
