# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Lightning logging backend required by TAO training."""

import pytest
from pytorch_lightning.loggers import TensorBoardLogger


def test_training_logger_writes_events(tmp_path):
    """Importing the train module alone does not verify its lazy logger backend."""
    # The TensorBoard backend is optional: it ships in the Data Services container
    # (docker/requirements-pip.txt pins tensorboardX), but the CI functional-test
    # stage only installs this package plus tao-core, so neither `tensorboard` nor
    # `tensorboardX` is present there. Lightning raises ModuleNotFoundError from the
    # constructor in that case -- skip on exactly that, and re-raise anything else so
    # the guard cannot mask a genuine breakage.
    try:
        logger = TensorBoardLogger(save_dir=tmp_path, version=1, name="lightning_logs")
    except ModuleNotFoundError as error:
        if "tensorboard" not in str(error).lower():
            raise
        pytest.skip("requires a TensorBoard logging backend (tensorboard or tensorboardX)")
    try:
        logger.log_metrics({"train_loss": 0.5}, step=1)
    finally:
        logger.finalize("success")
    events = list((tmp_path / "lightning_logs" / "version_1").glob("events.out.tfevents.*"))
    assert events and all(path.stat().st_size > 0 for path in events)
