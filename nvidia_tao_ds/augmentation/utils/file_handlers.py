# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities functions for file handling."""

import numpy as np


def load_file(path):
    """Get a numpy array from a loaded file.

    Args:
        path (str): Unix path to the file.

    Returns:
        np.array of loaded file.
    """
    return np.fromfile(path, dtype=np.uint8)


def file_fetcher(paths, batch_size):
    """Fetcher for a batch of file.

    Args:
        path (list): List of Unix paths to files.
        batch_size (int): Number of files per batch.

    Return:
        f (function pointer): Function pointer to the batch.
    """
    def f(i):
        start = batch_size * i

        end = min(len(paths), start + batch_size)
        if end <= start:
            raise StopIteration()
        batch = paths[start:end]
        if len(batch) < batch_size:
            # pad with last sample
            batch += [batch[-1]] * (batch_size - len(batch))
        return [load_file(path) for path in batch]
    return f


def box_fetcher(labels, batch_size):
    """Fetcher for a batch of file.

    Args:
        labels (list): List of kitti read annotation objects.
        batch_size (int): Number of files per batch.

    Return:
        f (function pointer): Function pointer to the batch.
    """
    def f(i):
        start = batch_size * i

        end = min(len(labels), start + batch_size)
        if end <= start:
            raise StopIteration()

        batch = []
        for j in range(start, end):
            boxes = []
            for annotation in labels[j]:
                boxes.append(annotation.box)
            batch.append(np.float32(boxes))
        if len(batch) < batch_size:
            # pad with last sample
            batch += [batch[-1]] * (batch_size - len(batch))
        return batch
    return f
