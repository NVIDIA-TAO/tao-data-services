# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle kitti operations."""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed, wait
import glob
import os
import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image

from nvidia_tao_ds.core.logging.logging import logger
from nvidia_tao_ds.data_analytics.utils.constant import COMMON_FILE_NAMES

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def list_files(kitti_obj):
    """ List image and label files.

    Args:
        kitti_obj(DataFormat): object of kitti data format.
    Returns:
        No explicit returns.
    """
    image_dir = kitti_obj.image_dir
    label_dir = kitti_obj.ann_path

    images = []
    # List image files.
    for ext in IMAGE_EXTENSIONS:
        images.extend(
            glob.glob(
                os.path.join(image_dir, f'**/*{ext}'),
                recursive=True
            )
        )
    # List label files.
    labels = glob.glob(os.path.join(label_dir, '**/*.txt'), recursive=True)
    images = sorted(images)
    labels = sorted(labels)
    if len(images) == 0:
        kitti_obj.image_paths = None
        kitti_obj.label_paths = labels
        return
    image_names = [x[x.rfind('/'):-4] for x in images]
    label_names = [x[x.rfind('/'):-4] for x in labels]

    out_img = []
    out_lbl = []

    i = 0
    j = 0
    while i < len(image_names) and j < len(label_names):
        if image_names[i] < label_names[j]:
            i += 1
        elif image_names[i] > label_names[j]:
            j += 1
        else:
            out_img.append(images[i])
            out_lbl.append(labels[j])
            i += 1
            j += 1

    kitti_obj.image_paths = out_img
    kitti_obj.label_paths = out_lbl


def create_image_dataframe(image_data):
    """ Create image data frame.

    Args:
        image_data(Dict): image data dictionary.
    Returns:
        No explicit returns.
    """
    image_df = pd.DataFrame.from_dict(image_data, orient='index',
                                      columns=['img_width', 'img_height', 'path'])
    image_df['size'] = image_df['img_width'] * image_df['img_height']
    return image_df


def create_dataframe(valid_kitti_file_path, invalid_kitti_file_path, image_data):
    """Create DataFrame from kitti files.

    Args:
        paths (str): Unix path to the kitti files.
        image_data(dict): Dictionary of image data corresponding
        to kitti data.(None if no images are given).

    Returns:
        df (Pandas DataFrame): output dataframe of kitti data.

    """
    dtype = {'type': str, 'truncated': np.dtype('float'), 'occluded': np.dtype('int32'),
             'alpha': np.dtype('float'), 'bbox_xmin': np.dtype('float'),
             'bbox_ymin': np.dtype('float'), 'bbox_xmax': np.dtype('float'),
             'bbox_ymax': np.dtype('float'), 'dim_height': np.dtype('float'),
             'dim_width': np.dtype('float'), 'dim_length': np.dtype('float'),
             'loc_x': np.dtype('float'), 'loc_y': np.dtype('float'),
             'loc_z': np.dtype('float'), 'rotation_y': np.dtype('float'),
             'path': str}
    name_list = ['type', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin',
                 'bbox_xmax', 'bbox_ymax', 'dim_height', 'dim_width', 'dim_length',
                 'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'path']

    if image_data is not None:
        name_list += ['img_width', 'img_height', 'img_path']
        dtype['img_width'] = np.dtype('float')
        dtype['img_height'] = np.dtype('float')
        dtype['img_path'] = str

    df_list = [pd.read_csv(filepath, sep=' ', names=name_list, dtype=dtype, index_col=False) for filepath in valid_kitti_file_path]
    valid_df = pd.concat(df_list)
    valid_df['bbox_area'] = (valid_df['bbox_xmax'] - valid_df['bbox_xmin']) * (valid_df['bbox_ymax'] - valid_df['bbox_ymin'])

    dtype['out_of_box_coordinates'] = np.dtype('bool')
    dtype['zero_area_bounding_box'] = np.dtype('bool')
    dtype['inverted_coordinates'] = np.dtype('bool')

    name_list += ['out_of_box_coordinates', 'zero_area_bounding_box', 'inverted_coordinates']

    invalid_df = pd.read_csv(invalid_kitti_file_path, sep=' ', names=name_list, dtype=dtype, index_col=False)
    invalid_df['bbox_area'] = (invalid_df['bbox_xmax'] - invalid_df['bbox_xmin']) * (invalid_df['bbox_ymax'] - invalid_df['bbox_ymin'])
    return valid_df, invalid_df


def validate_kitti_row(kitti_row, img_height=None, img_width=None):
    """ Validate kitti row.

    Args:
        kitti_row (str): kitti row.
        img_height (int): corresponding image height.
        img_width (int): corresponding image width.

    Returns:
        Bool,str: boolean status if valid or not , description
    """
    columns = kitti_row.strip().split(' ')

    # data is not in kitti format if kitti columns are 15 or 16
    # data is invalid in case of out of bound bbox and inverted coordinates.

    if len(columns) not in (15, 16):
        return False, "NOT_KITTI"
    try:
        truncated, occluded,  = float(columns[1]), int(columns[2])
        x_min, y_min = float(columns[4]), float(columns[5])
        x_max, y_max = float(columns[6]), float(columns[7])
    except Exception:
        return False, "NOT_KITTI"
    return_status = True
    invalid_reason_list = []
    if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
        return_status = False
        invalid_reason_list.append("INVALID_OUT_OF_BOX_COORD")
    if (x_max - x_min) == 0 or (y_max - y_min) == 0:
        return_status = False
        invalid_reason_list.append("INVALID_ZERO_COORD")
    if x_max < x_min or y_max < y_min:
        return_status = False
        invalid_reason_list.append("INVALID_INVERTED_COORD")
    if img_width and img_height and (x_max > img_width or y_max > img_height):
        return_status = False
        if "INVALID_OUT_OF_BOX_COORD" not in invalid_reason_list:
            invalid_reason_list.append("INVALID_OUT_OF_BOX_COORD")
    if not return_status:
        return return_status, invalid_reason_list
    if truncated < 0 or truncated > 1 or occluded not in (0, 1, 2, 3):
        return False, "GOOD_TO_CORRECT"
    return True, "VALID_DATA"


def validate_and_merge(filepaths, lock, i, not_kitti_file, invalid_kitti_data,
                       good_to_correct_kitti_data, image_data=None):
    """
    Validate and merge kitti files in one file.

    Args:
        filepaths (list): List of kitti filepaths.
        lock (object): Multiprocessing lock object.
        i (int): Suffix for merged file to be created (thread_multi_{i}).
        not_kitti_file (str): Txt file Path where details will be logged for
        files that are not kitti or images.
        invalid_kitti_data (str): File Path where invalid kitti rows will be
        logged.
        good_to_correct_kitti_data(str): File Path where good to correct kitti
        rows will be logged.
        image_data(dict): Dictionary of image data corresponding to kitti data.
        (None if no images are given).

    Returns:
        mergedfilepath (str): Path of text file in which kitti files are
        merged.

    """
    filename = f"thread_multi_{str(i)}.txt"
    merged_file_path = os.path.join(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER'],
                                    filename)
    not_kitti_files_list = set()
    invalid_kitti_row = []
    good_to_correct_kitti_row = []

    with open(merged_file_path, 'w+', encoding='utf-8') as handle1:
        for filepath in filepaths:
            basename = os.path.basename(filepath).split(".")[0]

            try:
                with open(filepath, 'r', encoding='utf-8') as handle2:
                    content = ''
                    line_number = 0
                    for line in handle2:
                        if not line.strip():
                            continue
                        columns = line.strip().split(' ')
                        if image_data:
                            img_width, img_height, img_path = (image_data[basename][0],
                                                               image_data[basename][1], image_data[basename][2])
                            columns = columns[0:15] + [filepath, str(img_width),
                                                       str(img_height), img_path]
                        else:
                            columns = columns[0:15] + [filepath]
                            img_height = None
                            img_width = None

                        valid, desc = validate_kitti_row(line, img_height,
                                                         img_width)

                        if not valid:
                            if not isinstance(desc, list):
                                if desc == "NOT_KITTI":
                                    not_kitti_files_list.add(filepath + " " +
                                                             str(line_number))
                                    line_number += 1
                                elif desc == "GOOD_TO_CORRECT":
                                    good_to_correct_kitti_row.append(" ".join(columns).lower())
                            else:
                                out_of_box, zero_area, inverted_coord = 'False', 'False', 'False'
                                if "INVALID_OUT_OF_BOX_COORD" in desc:
                                    out_of_box = 'True'
                                if "INVALID_ZERO_COORD" in desc:
                                    zero_area = 'True'
                                if "INVALID_INVERTED_COORD" in desc:
                                    inverted_coord = 'True'
                                columns = columns + [out_of_box, zero_area, inverted_coord]
                                invalid_kitti_row.append(" ".join(columns).lower())
                            continue

                        content += " ".join(columns).lower() + "\n"
                        line_number += 1
                    handle1.write(content)
            except Exception:
                not_kitti_files_list.add(filepath)
    lock.acquire()
    with open(not_kitti_file, 'a+', encoding='utf-8') as handle:
        handle.write("\n")
        handle.write('\n'.join(not_kitti_files_list))
    with open(invalid_kitti_data, 'a+', encoding='utf-8') as handle:
        handle.write("\n")
        handle.write('\n'.join(invalid_kitti_row))

    with open(good_to_correct_kitti_data, 'a+', encoding='utf-8') as handle:
        handle.write("\n")
        handle.write('\n'.join(good_to_correct_kitti_row))

    lock.release()
    return merged_file_path


def validate_and_merge_kitti_files(files, output_dir, num_workers, image_data=None):
    """Validate and Merge Kitti Files using multiprocessing.

    Args:
        files (str): Unix path to the kitti files.
        output_dir (str): Path to output directory.
        num_workers (int): Number of workers.
        image_data(dict): Dictionary of image data corresponding
        to kitti data.(None if no images are given).

    Returns:
        mergedfilenames(list): List of all merged file paths.
    """
    merged_file_names = []
    workers = int(num_workers)
    not_kitti_file = os.path.join(output_dir, 'not_kitti.txt')

    if not os.path.exists(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER']):
        os.makedirs(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER'])
    invalid_kitti_data_file = os.path.join(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER'],
                                           COMMON_FILE_NAMES['INVALID_KITTI'])
    good_to_correct_kitti_data_file = os.path.join(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER'],
                                                   COMMON_FILE_NAMES['GOOD_TO_CORRECT_KITTI'])

    with open(not_kitti_file, 'w+', encoding='utf-8'):
        pass
    with open(invalid_kitti_data_file, 'w+', encoding='utf-8'):
        pass
    with open(good_to_correct_kitti_data_file, 'w+', encoding='utf-8'):
        pass
    # create the process pool
    with ProcessPoolExecutor(workers) as executor:
        futures = []
        m = multiprocessing.Manager()
        lock = m.Lock()
        if len(files) < workers:
            chunksize = 1
        else:
            chunksize = round(len(files) / workers)
        # split the merge operations into chunks
        for i in range(0, len(files), chunksize):
            # select a chunk of filenames
            filepaths = files[i:(i + chunksize)]
            # submit the task
            future = executor.submit(validate_and_merge, filepaths, lock,
                                     i, not_kitti_file, invalid_kitti_data_file,
                                     good_to_correct_kitti_data_file, image_data)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            filename = future.result()
            merged_file_names.append(filename)
    merged_file_names.append(good_to_correct_kitti_data_file)
    return merged_file_names


def get_image_data(kitti_obj):
    """ Get image width and height .

    Args:
        kitti_obj (DataFormat): object of kitti data format.

    Returns:
        image_data(dict): Dictionary of image name.
            mapped to image width and height .
    """
    filepaths = kitti_obj.image_paths
    image_data = {}
    for filepath in filepaths:
        basename = os.path.basename(filepath).rsplit(".", 1)[0]
        img = Image.open(filepath)
        width, height = img.size
        image_data[basename] = [width, height, filepath]
    return image_data


def correct_data(invalid_df):
    """
    Correct the invalid kitti dataframe.
    correction criteria :
        set bounding box values = 0 if their values are less than 0.
        set x_max=img_width if x_max>img_width.
        set y_max=img_height if y_max>img_height.
        swap inverted bouding box coordinates.
    Args:
        invalid_df(Pandas Dataframe): invalid kitti dataframe.

    Return:
        valid kitti dataframe.
    """
    if 'img_width' in invalid_df.columns and 'img_height' in invalid_df.columns:

        invalid_df.loc[invalid_df['bbox_xmax'] > invalid_df['img_width'], 'bbox_xmax'] = invalid_df['img_width']
        invalid_df.loc[invalid_df['bbox_ymax'] > invalid_df['img_height'], 'bbox_ymax'] = invalid_df['img_height']

    invalid_df.loc[invalid_df['bbox_xmax'] < 0, 'bbox_xmax'] = 0
    invalid_df.loc[invalid_df['bbox_xmin'] < 0, 'bbox_xmin'] = 0
    invalid_df.loc[invalid_df['bbox_ymin'] < 0, 'bbox_ymin'] = 0
    invalid_df.loc[invalid_df['bbox_ymax'] < 0, 'bbox_ymax'] = 0

    temp_rows = invalid_df['bbox_xmax'] < invalid_df['bbox_xmin']

    invalid_df.loc[temp_rows, ['bbox_xmax', 'bbox_xmin']] = (invalid_df.loc[temp_rows, ['bbox_xmin', 'bbox_xmax']].values)
    temp_rows = invalid_df['bbox_ymax'] < invalid_df['bbox_ymin']

    invalid_df.loc[temp_rows, ['bbox_ymax', 'bbox_ymin']] = (invalid_df.loc[temp_rows, ['bbox_ymin', 'bbox_ymax']].values)

    return invalid_df


def write_to_csv(paths, kitti_folder, df):
    """
    write dataframe to csv's.
    Args:
        paths(List): path to csv files.
        kitti_folder(str): path to folder to save kitti files.
        df(Pandas dataframe): dataframe.
    Return:
        No explicit return.
    """
    for filepath in paths:
        temp_df = df.loc[df['path'] == filepath, :]
        temp_df = temp_df.drop(['path', 'bbox_area', 'out_of_box_coordinates',
                                'zero_area_bounding_box', 'inverted_coordinates'], axis=1)
        if 'img_height' in temp_df.columns:
            temp_df = temp_df.drop(['img_height', 'img_width', 'img_path'], axis=1)
        basename = os.path.basename(filepath)
        kitti_path = os.path.join(kitti_folder, basename)
        temp_df.to_csv(kitti_path, header=None, index=None, sep=' ', mode='w+')


def create_correct_kitti_files(df, corrected_df, output_dir, workers):
    """
    Create corrected kitti files back from dataframe.
    Args:
        df(Pandas dataframe): valid dataframe used to create csv.
        corrected_df(Pandas dataframe): valid dataframe used to get filenames to rewrite.
        output_dir(str): output directory.
        workers(int): number of workers for multiprocessing.
    Return:
        No explicit return.

    """
    kitti_folder = os.path.join(output_dir, "corrected_kitti_files")
    if not os.path.exists(kitti_folder):
        os.makedirs(kitti_folder)

    kitti_files = corrected_df['path'].unique()
    logger.info(f"Total kitti files to be corrected - {len(kitti_files)}")
    with ProcessPoolExecutor(workers) as executor:
        futures = []
        if len(kitti_files) < workers:
            chunksize = 1
        else:
            chunksize = round(len(kitti_files) / workers)

        # split the operations into chunks
        for i in range(0, len(kitti_files), chunksize):
            # select a chunk of filenames
            filepaths = kitti_files[i:(i + chunksize)]
            temp_df = df.loc[df['path'].isin(filepaths), :]
            # submit the task
            future = executor.submit(write_to_csv, filepaths, kitti_folder, temp_df)
            futures.append(future)
        wait(futures)
        logger.info(f"Corrected kitti files are available at {kitti_folder}")
