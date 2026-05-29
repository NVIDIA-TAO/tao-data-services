# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routines for connecting with Weights and Biases client."""

from datetime import datetime

from pytorch_lightning.utilities import rank_zero_only
import wandb
from wandb import AlertLevel
import os

from nvidia_tao_ds.core.logging.logging import logger
DEFAULT_WANDB_CONFIG = "~/.netrc"

_WANDB_INITIALIZED = False


@rank_zero_only
def alert(title, text, duration=300, level=0):
    """Send alert."""
    alert_levels = {
        0: AlertLevel.INFO,
        1: AlertLevel.WARN,
        2: AlertLevel.ERROR
    }
    if is_wandb_initialized():
        wandb.alert(
            title=title,
            text=text,
            level=alert_levels[level],
            wait_duration=duration
        )


def is_wandb_initialized():
    """Check if wandb has been initialized."""
    global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
    return _WANDB_INITIALIZED


def check_wandb_logged_in():
    """Check if weights and biases have been logged in."""
    wandb_logged_in = False
    try:
        wandb_api_key = os.getenv("WANDB_API_KEY", None)
        if wandb_api_key is not None or os.path.exists(os.path.expanduser(DEFAULT_WANDB_CONFIG)):
            wandb_logged_in = wandb.login(key=wandb_api_key)
            return wandb_logged_in
    except wandb.errors.UsageError:
        logger.warning("WandB wasn't logged in.")
    return False


def initialize_wandb(output_dir,
                     project="TAO Data Analytics",
                     entity=None,
                     save_code=False,
                     name=None,
                     notes=None,
                     tags=None,
                     wandb_logged_in=False,
                     ):
    """Function to initialize wandb client with the weights and biases server.

    If wandb initialization fails, then the function just catches the exception
    and prints an error log with the reason as to why wandb.init() failed.

    Args:
        output_dir (str): Output directory of the experiment.
        project (str): Name of the project to sync data with.
        entity (str): Name of the wanbd entity.
        save_code (bool): save the main script or notebook to W&B
        notes (str): One line description about the wandb job.
        tags (list(str)): List of tags about the job.
        name (str): Name of the task running.
        wandb_logged_in (bool): Boolean flag to check if wandb was logged in.
    Returns:
        No explicit return.
    """
    logger.info("Initializing wandb.")
    try:
        assert wandb_logged_in, (
            "WandB client wasn't logged in. Please make sure to set "
            "the WANDB_API_KEY env variable or run `wandb login` in "
            "over the CLI and copy the ~/.netrc file to the container."
        )
        start_time = datetime.now()
        time_string = start_time.strftime("%d/%y/%m_%H:%M:%S")
        if name is None:
            wandb_name = f"run_{time_string}"
        else:            
            wandb_name = name
        wandb.init(
            project=project,
            entity=entity,
            save_code=save_code,
            name=wandb_name,
            notes=notes,
            tags=tags,
            dir=output_dir
        )
        global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
        _WANDB_INITIALIZED = True
    except Exception as e:
        logger.warning("Wandb logging failed with error %s", e)
