# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint script to run TAO image correction."""

import os
import time
import glob
import sys
import shutil
import imghdr

from nvidia_tao_ds.config.image.default_config import ExperimentConfig
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logger


@monitor_status(mode='Remove Corrupted images')
def remove_corruption_images(config):
    """Remove images that are corrupt.

    Args:
        config (Hydra config): Config element of the validate config.
    """
    start_time = time.perf_counter()
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_dir = config.data.image_dir
    results_dir = config.results_dir
    if not os.path.exists(image_dir):
        logger.info("Please provide path of image folder.")
        sys.exit(1)
    for file_name in glob.glob(os.path.join(image_dir, "**/*"), recursive=True):
        file_extension = file_name[file_name.rfind('.'):].lower()
        if file_extension in valid_extensions:
            destination_path = os.path.join(results_dir + file_name.replace(image_dir, ""))
            if not os.path.exists(os.path.dirname(destination_path)):
                os.makedirs(os.path.dirname(destination_path))
            if imghdr.what(file_name):
                if not config.in_place:  # Valid image and in_place correction is False then we copy valid image
                    shutil.copy(file_name, destination_path)
            else:
                if config.in_place:  # Invalid image and in_place correction is True then we move invalid image
                    shutil.move(file_name, destination_path)
                logger.warning(f"{file_name} is corrupted")
    logger.debug(f"Total time taken : {time.perf_counter() - start_time}")

spec_root = os.path.dirname(os.path.abspath(__file__))


@hydra_runner(
    config_path=os.path.join(spec_root, "../experiment_specs"),
    config_name="validate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """TAO Image validation wrapper function."""
    try:
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir)
        remove_corruption_images(cfg)
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
