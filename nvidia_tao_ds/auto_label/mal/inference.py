# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAL inference."""

import os
from pytorch_lightning import Trainer
from nvidia_tao_pytorch.cv.mal.models.mal import MALPseudoLabels
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule
from nvidia_tao_ds.auto_label.mal.utils.config_utils import update_config


def run_mal_inference(experiment_config, results_dir):
    """Generate segmentation from boxes using MAL."""
    os.makedirs(results_dir, exist_ok=True)

    gpu_ids = [int(gpu) for gpu in os.environ['TAO_VISIBLE_DEVICES'].split(',')]
    num_workers = experiment_config.num_workers
    batch_size = experiment_config.batch_size

    cfg = experiment_config.mal
    cfg = update_config(cfg)
    cfg.train.lr = 0
    cfg.train.min_lr = 0

    # override validation path
    cfg.results_dir = results_dir
    cfg.dataset.val_ann_path = cfg.inference.ann_path
    cfg.dataset.val_img_dir = cfg.inference.img_dir
    cfg.dataset.load_mask = cfg.inference.load_mask
    cfg.inference.batch_size = batch_size
    cfg.train.batch_size = batch_size
    cfg.evaluate.use_mixed_model_test = False
    cfg.evaluate.use_teacher_test = False
    cfg.evaluate.comp_clustering = False
    cfg.evaluate.use_flip_test = False

    dm = WSISDataModule(
        num_workers=num_workers,
        cfg=cfg)
    dm.setup(stage="predict")

    # Phase 2: Generating pseudo-labels
    model = MALPseudoLabels.load_from_checkpoint(cfg.checkpoint,
                                                 map_location='cpu',
                                                 cfg=cfg,
                                                 categories=dm.val_dataset.coco.dataset['categories'])

    trainer = Trainer(
        strategy='auto',
        devices=gpu_ids,
        default_root_dir=results_dir,
        precision='16-mixed',
        check_val_every_n_epoch=1,
    )
    trainer.predict(model, dataloaders=dm)
