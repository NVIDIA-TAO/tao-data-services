# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DALI pipeline."""

import numpy as np
from nvidia import dali
from nvidia.dali import fn, pipeline_def
import nvidia.dali.fn.transforms as tr

from nvidia_tao_ds.core.logging.logging import logger


def random_pick(values):
    """Randomly pick from values or range."""
    if len(values) == 0:
        value = 0
    elif len(values) == 1:
        value = values[0]
    elif len(values) == 2:
        value = np.random.uniform(low=values[0], high=values[1])
    else:
        value = np.random.choice(values)
    return value


def spatial_transform(config, input_size, output_size):
    """Define a spatial transform pipe.

    Args:
        config (Hydra config): Spatial augumentation config element.
        input_size (tuple): Size of the input image.
        output_size (tuple): Size of the output image.

    Return:
        m (2x3): Spatial transformation matrix.
    """
    # DALI cannot conditionally apply a transform, so we have to pass neutral parameters.
    # Since we're only manipulating 2x3 matrix, it's not a problem.

    angle = config.rotation.angle
    angle = random_pick(angle)
    if config.rotation.units == "radians":
        angle *= 180 / np.pi

    center = input_size / 2
    shear_ratio_x = random_pick(config.shear.shear_ratio_x)
    shear_ratio_y = random_pick(config.shear.shear_ratio_y)
    shear = np.float32([shear_ratio_x, shear_ratio_y])

    translate_x = random_pick(config.translation.translate_x)
    translate_y = random_pick(config.translation.translate_y)
    offset = np.float32(
        [translate_x, translate_y]
    )

    flip = np.int32([config.flip.flip_horizontal, config.flip.flip_vertical])

    m = tr.rotation(angle=angle, center=center)
    # flip operator is scale with negative axis.
    m = tr.scale(m, scale=1 - flip * 2.0, center=center)
    m = tr.shear(m, shear=shear, center=center)
    m = tr.translation(m, offset=offset)
    m = tr.scale(m, scale=output_size / input_size)
    return m


def apply_color_transform(images, config):
    """Apply color transform to the image.

    Args:
        images (list): Array of images.
        config (Hydra config): Color configuration element.

    Returns:
        images (list): Batch of images.
    """
    logger.info("Applying color augmentation to the images.")
    H = config.hue.hue_rotation_angle
    H = -random_pick(H)
    S = config.saturation.saturation_shift
    S = random_pick(S)
    C = 1.0  # TODO(@yuw): verify
    C += random_pick(config.contrast.contrast)
    center = random_pick(config.contrast.center)
    B = random_pick(config.brightness.offset) / 255

    # applying hue rotation
    images = fn.hsv(
        images,
        hue=H,
        saturation=S,
        dtype=dali.types.FLOAT
    )
    # applying contrast adjustment.
    images = fn.brightness_contrast(
        images,
        contrast=C,
        contrast_center=center,
        brightness_shift=B * C,
        dtype=dali.types.UINT8)
    return images


def transform_boxes(boxes, matrix, out_w, out_h, fmt="xyxy"):
    """Apply color transform to the image.

    Args:
        boxes (list): batch of boxes
        matrix (np.array): transformation matrix
        out_w (int): output width
        out_h (int): output height
    """
    # boxes' shape: num_boxes x 4
    box_x0 = fn.slice(boxes, 0, 1, axes=[1])
    box_y0 = fn.slice(boxes, 1, 1, axes=[1])
    if fmt == "xyxy":
        box_x1 = fn.slice(boxes, 2, 1, axes=[1])
        box_y1 = fn.slice(boxes, 3, 1, axes=[1])
    elif fmt == "xywh":
        box_x1 = fn.slice(boxes, 2, 1, axes=[1]) + box_x0
        box_y1 = fn.slice(boxes, 3, 1, axes=[1]) + box_y0
    else:
        raise ValueError("Only `xyxy` (KITTI) format and `xywh` (COCO) format are supported.")
    corners = fn.stack(
        fn.cat(box_x0, box_y0, axis=1),
        fn.cat(box_x1, box_y0, axis=1),
        fn.cat(box_x0, box_y1, axis=1),
        fn.cat(box_x1, box_y1, axis=1),
        axis=1
    )
    # corners' shape: nboxes x 4 x 2
    corners = fn.coord_transform(corners, MT=matrix)
    lo = fn.reductions.min(corners, axes=1)
    hi = fn.reductions.max(corners, axes=1)

    # I really DO wish DALI had broadcasting :(
    # hi, lo shape: nboxes x 2
    lohi = fn.stack(lo, hi, axis=1)
    # lohi shape: nboxes x 2 x 2
    lohix = dali.math.clamp(fn.slice(lohi, 0, 1, axes=[2]), 0, out_w)
    lohiy = dali.math.clamp(fn.slice(lohi, 1, 1, axes=[2]), 0, out_h)
    lohi = fn.stack(lohix, lohiy, axis=2)
    return fn.reshape(lohi, shape=[-1, 4])


def apply_blur(images, blur_config):
    """Apply blur operator on an image.

    Args:
        images (list): Batch of images.
        blur_config (Hydra config): Config element for the blur operator.

    Returns:
        images (batch): Batch of images blurred.
    """
    sigma = random_pick(blur_config.std) or 0
    size = random_pick(blur_config.size) or 0
    if sigma == size == 0:
        return images
    logger.info("Applying Gaussian blur operator to the images.")
    return fn.gaussian_blur(images, sigma=sigma, window_size=size)


def build_coco_pipeline(coco_callable, is_fixed_size, config,
                        batch_size=1, device_id=0):
    """Build DALI pipeline for COCO format dataset."""

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=device_id,
                  py_num_workers=1, py_start_method='spawn', seed=config.random_seed)
    def sharded_coco_pipeline(coco_callable, is_fixed_size, config):
        logger.debug("Defining COCO pipeline.")
        raw_files, boxes, img_ids, masks = fn.external_source(
            source=coco_callable,
            num_outputs=4, batch=False,
            parallel=True,
            dtype=[dali.types.UINT8, dali.types.FLOAT, dali.types.INT32, dali.types.UINT8])

        shapes = fn.peek_image_shape(raw_files)
        images = fn.image_decoder(raw_files, device="mixed")

        in_h = fn.slice(shapes, 0, 1, axes=[0])
        in_w = fn.slice(shapes, 1, 1, axes=[0])

        if is_fixed_size:
            out_w = config.data.output_image_width
            out_h = config.data.output_image_height
            out_size = [out_h, out_w]
        else:
            out_w = in_w
            out_h = in_h
            out_size = fn.cat(out_h, out_w)

        mt = spatial_transform(
            config.spatial_aug,
            input_size=fn.cat(in_w, in_h),
            output_size=dali.types.Constant(
                [out_w, out_h]) if is_fixed_size else fn.cat(out_w, out_h)
        )
        images = fn.warp_affine(
            images, matrix=mt,
            size=out_size,
            fill_value=0, inverse_map=False
        )
        if config.data.include_masks:
            masks = fn.warp_affine(
                masks, matrix=mt,
                size=out_size,
                fill_value=0, inverse_map=False
            )

        orig_boxes = boxes  # noqa pylint: disable=W0612
        boxes = transform_boxes(boxes, mt, out_w, out_h, fmt="xywh")
        images = apply_color_transform(images, config.color_aug)
        images = apply_blur(images, config.blur_aug)

        return images, boxes, img_ids, masks
    logger.info("Building COCO pipeline.")
    pipe = sharded_coco_pipeline(
        coco_callable, is_fixed_size, config)
    pipe.build()
    return pipe


def build_kitti_pipeline(kitti_callable, is_fixed_size, config,
                         batch_size=1, device_id=0):
    """Build DALI pipeline for KITTI format dataset."""

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=device_id,
                  py_num_workers=1, py_start_method='spawn', seed=config.random_seed)
    def sharded_kitti_pipeline(kitti_callable, is_fixed_size, config):
        logger.debug("Defining KITTI pipeline.")
        raw_files, boxes, img_paths, lbl_paths = fn.external_source(
            source=kitti_callable,
            num_outputs=4, batch=False,
            parallel=True,
            dtype=[dali.types.UINT8, dali.types.FLOAT, dali.types.UINT8, dali.types.UINT8])

        shapes = fn.peek_image_shape(raw_files)
        images = fn.image_decoder(raw_files, device="mixed")

        in_h = fn.slice(shapes, 0, 1, axes=[0])
        in_w = fn.slice(shapes, 1, 1, axes=[0])
        if is_fixed_size:
            out_w = config.data.output_image_width
            out_h = config.data.output_image_height
            out_size = [out_h, out_w]
        else:
            out_w = in_w
            out_h = in_h
            out_size = fn.cat(out_h, out_w)

        mt = spatial_transform(
            config.spatial_aug,
            input_size=fn.cat(in_w, in_h),
            output_size=dali.types.Constant(
                [out_w, out_h]) if is_fixed_size else fn.cat(out_w, out_h)
        )
        images = fn.warp_affine(
            images, matrix=mt,
            size=out_size,
            fill_value=0, inverse_map=False
        )

        orig_boxes = boxes  # noqa pylint: disable=W0612
        boxes = transform_boxes(boxes, mt, out_w, out_h, fmt="xyxy")
        images = apply_color_transform(images, config.color_aug)
        images = apply_blur(images, config.blur_aug)

        return images, boxes, img_paths, lbl_paths

    logger.info("Building KITTI pipeline.")
    pipe = sharded_kitti_pipeline(
        kitti_callable, is_fixed_size, config)
    pipe.build()
    return pipe
