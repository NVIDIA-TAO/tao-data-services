# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for configuration."""


def update_config(cfg):
    """Update Hydra config."""
    # mask threshold
    if len(cfg.train.mask_thres) == 1:
        # this means to repeat the same threshold three times
        # all scale objects are sharing the same threshold
        cfg.train.mask_thres = [cfg.train.mask_thres[0] for _ in range(3)]
    assert len(cfg.train.mask_thres) == 3

    # frozen_stages
    if len(cfg.model.frozen_stages) == 1:
        cfg.model.frozen_stages = [0, cfg.model.frozen_stages[0]]
    assert len(cfg.model.frozen_stages) == 2
    assert len(cfg.train.margin_rate) == 2
    return cfg
