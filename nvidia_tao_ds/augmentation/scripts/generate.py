# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint script to run TAO augmentation."""

from copy import deepcopy
import gc
import glob
import json
import os
import sys
import time

from nvidia_tao_ds.config.augmentation.default_config import ExperimentConfig
from nvidia_tao_ds.annotations.conversion.kitti_to_coco import convert_kitti_to_coco
from nvidia_tao_ds.annotations.conversion.coco_to_kitti import convert_coco_to_kitti
from nvidia_tao_ds.augmentation.pipeline import runner
from nvidia_tao_ds.augmentation.utils import callable_dict, pipeline_dict
from nvidia_tao_ds.augmentation.utils.distributed_utils import MPI_local_rank

from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logger


@monitor_status(mode='Augment')
def run_augment(config):
    """TAO augmentation pipeline for OD datasets.

    Args:
        config (Hydra config): Config element of the augmentation config.
    """
    logger.info("Data augmentation started.")
    start_time = time.time()
    gpu_ids = [int(gpu) for gpu in os.environ['TAO_VISIBLE_DEVICES'].split(',')]

    image_dir = config.data.image_dir
    ann_path = config.data.ann_path
    batch_size = config.data.batch_size
    is_fixed_size = config.data.output_image_width and config.data.output_image_height
    dataset_type = config.data.dataset_type.lower()
    output_dir = config.results_dir
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    if MPI_local_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    # prepare dataloader and DALI pipeline
    data_callable = callable_dict[dataset_type](
        image_dir, ann_path, batch_size,
        include_masks=config.data.include_masks,
        shard_id=MPI_local_rank(), num_shards=len(gpu_ids))

    pipe = pipeline_dict[dataset_type](
        data_callable,
        is_fixed_size,
        config,
        batch_size,
        device_id=MPI_local_rank()
    )

    ann_dump = runner.run(pipe, data_callable, config)
    time_elapsed = time.time() - start_time
    logger.info(f"Data augmentation (rank #{MPI_local_rank()}) finished in {time_elapsed:.2f}s.")
    # extra process for COCO dump
    if config.data.dataset_type.lower() == 'coco' and ann_dump:
        logger.info("Saving output annotation JSON...")
        out_label_path = os.path.join(output_dir, "annotations.json")
        tmp_label_path = out_label_path + f'.part{MPI_local_rank()}'
        with open(tmp_label_path, "w", encoding='utf-8') as f:
            json.dump(ann_dump, f)
        from mpi4py import MPI  # noqa pylint: disable=C0415
        MPI.COMM_WORLD.Barrier()  # noqa pylint: disable=I1101

        if MPI_local_rank() == 0:
            tmp_files = glob.glob(out_label_path + '.part*')
            ann_final = []
            for tmp_i in tmp_files:
                with open(tmp_i, "r", encoding='utf-8') as g:
                    ann_i = json.load(g)
                ann_final.extend(ann_i)

            # write final json
            coco_output = deepcopy(data_callable.coco.dataset)
            coco_output['annotations'] = ann_final
            # modify image size if not output dim is specified
            if is_fixed_size:
                for image_info in coco_output['images']:
                    image_info['height'] = config.data.output_image_height
                    image_info['width'] = config.data.output_image_width
            with open(out_label_path, "w", encoding='utf-8') as o:
                json.dump(coco_output, o)
            # remove tmp files
            for tmp_i in tmp_files:
                if os.path.exists(tmp_i):
                    os.remove(tmp_i)
            logger.info(f"The annotation JSON is saved at {out_label_path}.")


def check_gt_cache(cfg, is_kitti=False, gt_cache=None):
    """Generate COCO cache file."""
    if not os.path.exists(gt_cache):
        root = os.path.abspath(cfg.results_dir)
        if MPI_local_rank() == 0:
            logger.info(f"Mask cache file ({gt_cache}) is not found.")
            if is_kitti:
                project_name = os.path.basename(root)
                tmp_ann_file = os.path.join(root, 'coco', f'{project_name}.json')
                if not os.path.isfile(tmp_ann_file):
                    # convert kitti to coco
                    logger.info("Converting KITTI labels into COCO format...")
                    convert_kitti_to_coco(
                        img_dir=cfg.data.image_dir,
                        label_dir=cfg.data.ann_path,
                        output_dir=os.path.join(root, 'coco'),
                        name=project_name)
                    logger.info("COCO format conversion completed.")
            else:
                tmp_ann_file = cfg.data.ann_path
            # generate masks
            logger.error(f"You need to generate pseudo-masks for `{tmp_ann_file}` with TAO auto-label tool first.")
        sys.exit()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="kitti", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """TAO Augment main wrapper function."""
    try:
        is_kitti = cfg.data.dataset_type.lower() == 'kitti'
        refine_box_enabled = cfg.spatial_aug.rotation.refine_box.enabled
        if refine_box_enabled:
            gt_cache = cfg.spatial_aug.rotation.refine_box.gt_cache
            check_gt_cache(cfg, is_kitti, gt_cache)
            # Update Hydra config
            cfg.data.dataset_type = 'coco'
            cfg.data.ann_path = gt_cache
            cfg.data.include_masks = True
            from mpi4py import MPI  # noqa pylint: disable=C0415
            MPI.COMM_WORLD.Barrier()  # noqa pylint: disable=I1101
        # run augmention
        run_augment(cfg)
        if is_kitti and refine_box_enabled and MPI_local_rank() == 0:
            logger.info("Converting COCO json into KITTI format...")
            # convert coco to kitti
            convert_coco_to_kitti(
                annotations_file=os.path.join(cfg.results_dir, 'annotations.json'),
                output_dir=os.path.join(cfg.results_dir, 'labels'),
                refine_box=refine_box_enabled
            )
            logger.info("KITTI conversion is complete.")
        gc.collect()
    except KeyboardInterrupt:
        logger.info("Interrupting augmentation.")
        sys.exit()
    except RuntimeError as e:
        logger.info(f"Augmentation run failed with error: {e}")
        sys.exit()


if __name__ == '__main__':
    main()
