# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle image operations."""
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import re
import random
from PIL import Image, ImageDraw
from tqdm import tqdm

from nvidia_tao_ds.core.logging.logging import logger


def write_to_image(filepaths, output_image_folder, df, object_color_dict, dataformat=None):
    """
    Draw Bounding boxes in images.
    Args:
        df(Pandas dataframe): valid dataframe used to draw bboxes.
        filepaths(list): list of image files to draw.
        output_image_folder(str): output directory.
        object_color_dict(dict): dictiory of object names to their unique color.
        dataformat(str): input data format.
    Return:
        No explicit return.

    """
    for filepath in filepaths:
        basename = os.path.basename(filepath).split(".")[0]
        # read image
        im = Image.open(filepath)
        if dataformat == "KITTI":
            pattern = '.*' + basename + '.txt'
            pattern = re.compile(pattern)
            temp_df = df.loc[df['path'].str.contains(pattern), :]

        else:
            pattern = os.path.basename(filepath)
            pattern = re.compile(pattern)
            temp_df = df.loc[df['image_name'].str.contains(pattern), :]

        for _, row in temp_df.iterrows():
            bbox = row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']
            draw = ImageDraw.Draw(im)
            obj_type = row['type']
            color = object_color_dict[obj_type]
            draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), fill=None, outline=color)
            draw.text((bbox[0], bbox[1]), obj_type)
        output_filepath = os.path.join(output_image_folder, os.path.basename(filepath))
        im.save(output_filepath)


def assign_object_colors(objects_names):
    """
    Assign color to each object.
    Args:
        object_names(List): list of object names.
    Return:
        color_dict(dict): dictiory of object names to their unique color.

    """
    color_list = []
    objects_names = list(objects_names)
    for _ in range(0, len(objects_names)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        while ((r, g, b) in color_list):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
        color_list.append((r, g, b))
    color_dict = {}
    for i in range(0, len(objects_names)):
        color_dict[objects_names[i]] = color_list[i]
    return color_dict


def generate_images_with_bounding_boxes(df, image_data, output_dir, image_sample_size, workers, dataformat=None):
    """
    Draw Bounding boxes in images.
    Args:
        df(Pandas dataframe): valid dataframe used to draw bboxes.
        image_data(Dict): Dictionary to hold image data.
        output_dir(str): output directory.
        workers(int): number of workers for multiprocessing.
        dataformat(str): input data format.
    Return:
        No explicit return

    """
    output_image_folder = os.path.join(output_dir, "image_with_bounding_boxes")
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    all_image_paths = [i_data[2] for i_data in image_data.values()]
    if len(all_image_paths) > image_sample_size:
        all_image_paths = random.sample(set(all_image_paths), image_sample_size)
    logger.info(f"Total image files- {len(all_image_paths)}")
    object_color_dict = assign_object_colors(df['type'].unique())
    if dataformat == "KITTI":
        df = df.drop(["truncated", "occluded", "alpha", "rotation_y", "loc_x",
                      "loc_y", "loc_z", "dim_height", "dim_width", "dim_length"], axis=1)
    tq = tqdm(total=len(all_image_paths), position=0, leave=True)
    with ProcessPoolExecutor(workers) as executor:
        futures = []
        if len(all_image_paths) < workers:
            chunksize = 1
        else:
            chunksize = round(len(all_image_paths) / workers)
        # split the operations into chunks
        for i in range(0, len(all_image_paths), chunksize):
            # select a chunk of filenames
            filepaths = all_image_paths[i:(i + chunksize)]

            if dataformat == "COCO":
                patterns = [os.path.basename(filepath) for filepath in filepaths]
                patterns = re.compile('|'.join(patterns))
                temp_df = df.loc[df['image_name'].str.contains(patterns), :]
            else:
                patterns = [os.path.basename(filepath).split(".")[0] + ".txt" for filepath in filepaths]
                patterns = re.compile('|'.join(patterns))
                temp_df = df.loc[df['path'].str.contains(patterns), :]
            # submit the task
            future = executor.submit(write_to_image, filepaths, output_image_folder, temp_df, object_color_dict, dataformat)
            futures.append(future)

        for future in as_completed(futures):
            tq.update(chunksize)
    tq.close()
    logger.info(f"Images with bounding boxes are available at {output_image_folder}")
