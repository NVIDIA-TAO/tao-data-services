# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Lightning logging backend required by TAO training."""

from pytorch_lightning.loggers import TensorBoardLogger


def test_training_logger_writes_events(tmp_path):
    """Importing the train module alone does not verify its lazy logger backend."""
    logger = TensorBoardLogger(save_dir=tmp_path, version=1, name="lightning_logs")
    try:
        logger.log_metrics({"train_loss": 0.5}, step=1)
    finally:
        logger.finalize("success")
    events = list((tmp_path / "lightning_logs" / "version_1").glob("events.out.tfevents.*"))
    assert events and all(path.stat().st_size > 0 for path in events)
