# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint script to run TAO validate."""

import os
import time
import sys
import pandas as pd

from nvidia_tao_ds.config.analytics.default_config import ExperimentConfig
from nvidia_tao_ds.data_analytics.utils import kitti, coco, data_format
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logger
from nvidia_tao_ds.data_analytics.utils.constant import COMMON_FILE_NAMES


def class_balance_info(df):
    """Print class balance summary.

    Args:
        df (Pandas Dataframe): Dataframe for valid kitti rows.

    Return:
        No explicit return.
    """
    count_df = df['type'].value_counts(ascending=False, normalize=True).rename(
        'count_num').reset_index()
    count_df['count_num'] = count_df['count_num'] * 100
    count_df = count_df.rename(columns={"index": "Object_tags", "count_num": "Percentage"})
    logger.info("Review below table to find if data for object tags is imbalanced")
    logger.info(count_df.to_string())


def validate_summary(valid_df, invalid_df, data_format):
    """Show validation summary.

    Args:
        valid_df (Pandas Dataframe): Valid kitti dataframe.
        invalid_df (Pandas Dataframe): invalid kitti dataframe.
        data_format (str): Input data format
    Return:
        No explicit returns.
    """
    total_rows = len(valid_df) + len(invalid_df)
    num_of_invalid_rows = len(invalid_df)

    invalid_percentage = (num_of_invalid_rows / total_rows) * 100
    invalid_percentage = round(invalid_percentage, 2)
    logger.info(f"Number of total annotations : {total_rows}\
        \nNumber of invalid annotations : {num_of_invalid_rows}")
    if invalid_percentage > 5:
        logger.warning(f"WARNING: Number of invalid annotations are {invalid_percentage}"
                       " of total annotations , It is advisable"
                       " to correct the data before training.")
    else:
        logger.info(f"Number of invalid annotations are {invalid_percentage}"
                    " of total annotations,")

    if 'img_width' in invalid_df.columns:
        oob_condition = ((invalid_df['bbox_xmin'] < 0) |
                         (invalid_df['bbox_ymin'] < 0) |
                         (invalid_df['bbox_ymax'] < 0) |
                         (invalid_df['bbox_xmax'] < 0) |
                         (invalid_df['bbox_xmax'] > invalid_df['img_width']) |
                         (invalid_df['bbox_ymax'] > invalid_df['img_height']))
    else:
        oob_condition = ((invalid_df['bbox_xmin'] < 0) |
                         (invalid_df['bbox_ymin'] < 0) |
                         (invalid_df['bbox_ymax'] < 0) |
                         (invalid_df['bbox_xmax'] < 0))

    out_of_bound_count = len(invalid_df[oob_condition])

    inverted_cord_count = len(invalid_df[(invalid_df['bbox_ymax'] > 0) &
                                         (invalid_df['bbox_xmax'] > 0) &
                                         (invalid_df['bbox_ymin'] > 0) &
                                         (invalid_df['bbox_xmin'] > 0) &
                                         ((invalid_df['bbox_xmax'] < invalid_df['bbox_xmin']) |
                                         (invalid_df['bbox_ymax'] < invalid_df['bbox_ymin']))])

    out_of_bound_percentage = round((out_of_bound_count / total_rows) * 100, 2)
    inverted_cord_percentage = round((inverted_cord_count / total_rows) * 100, 2)
    logger.info("Number and Percent of annotations with out of bound "
                f"coordinates {out_of_bound_count}, {out_of_bound_percentage}%")

    logger.info("Number and Percent of annotations with inverted "
                f"coordinates {inverted_cord_count}, {inverted_cord_percentage}%")

    class_balance_info(valid_df)


@monitor_status(mode='KITTI validation')
def validate_dataset_kitti(config):
    """TAO KITTI dataset validate.

    Args:
        config (Hydra config): Config element of the analyze config.
    """
    start_time = time.perf_counter()
    kitti_obj = data_format.create_data_object("KITTI",
                                               ann_path=config.data.ann_path,
                                               image_dir=config.data.image_dir)
    if not os.path.isdir(config.data.ann_path):
        logger.info("Please provide path of kitti label directory in config data.ann_path.")
        sys.exit(1)
    kitti.list_files(kitti_obj)

    if kitti_obj.image_paths is None:
        logger.info("Image Directory not found.Processing only label files")
        image_data = None
    else:
        # Get image data (image width and height)
        image_data = kitti.get_image_data(kitti_obj)

    if kitti_obj.label_paths is None:
        logger.info("kitti files not Found. Exiting ")
        sys.exit(1)
    valid_kitti_filepaths = kitti.validate_and_merge_kitti_files(kitti_obj.label_paths,
                                                                 config.results_dir,
                                                                 config.workers,
                                                                 image_data)
    invalid_filepath = os.path.join(COMMON_FILE_NAMES['INTERMEDIATE_KITTI_FOLDER'],
                                    COMMON_FILE_NAMES['INVALID_KITTI'])
    # Dataframe creation for valid and invalid kitti data
    valid_df, invalid_df = kitti.create_dataframe(valid_kitti_filepaths,
                                                  invalid_filepath,
                                                  image_data)

    validate_summary(valid_df, invalid_df, config.data.input_format)
    if config.apply_correction:
        corrected_df = kitti.correct_data(invalid_df)
        df = pd.concat([valid_df, corrected_df])
        kitti.create_correct_kitti_files(df, corrected_df, config.results_dir, config.workers)

    logger.debug(f"Total time taken : {time.perf_counter() - start_time}")


@monitor_status(mode='COCO validation')
def validate_dataset_coco(config):
    """TAO COCO dataset validate.

    Args:
        config (Hydra config): Config element of the analyze config.
    """
    start_time = time.perf_counter()
    if not os.path.isfile(config.data.ann_path):
        logger.info("Please provide path of coco annotation file in config data.ann_path.")
        sys.exit(1)
    coco_obj = data_format.create_data_object("COCO",
                                              ann_path=config.data.ann_path,
                                              image_dir=config.data.image_dir)
    # Dataframe creation for valid and invalid kitti data
    valid_df, invalid_df = coco.create_dataframe(coco_obj)
    validate_summary(valid_df, invalid_df, config.data.input_format)
    if config.apply_correction:
        # correct the coco file and write into output_dir
        coco.correct_data(coco_obj, config.results_dir)

    logger.debug(f"Total time taken : {time.perf_counter() - start_time}")


spec_root = os.path.dirname(os.path.abspath(__file__))


@hydra_runner(
    config_path=os.path.join(spec_root, "../experiment_specs"),
    config_name="validate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """TAO Validate main wrapper function."""
    try:
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir)
        if cfg.data.input_format == "COCO":
            validate_dataset_coco(cfg)
        elif cfg.data.input_format == "KITTI":
            validate_dataset_kitti(cfg)
        else:
            logger.info(f"Data format {cfg.data.input_format} is not supported.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Aborting execution.")
        sys.exit(1)
    except RuntimeError as e:
        logger.info(f"Validate run failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.info(f"Validate run failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
