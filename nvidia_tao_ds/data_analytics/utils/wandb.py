# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routines for connecting with Weights and Biases client."""

import os
import random
import wandb
import pandas as pd

from nvidia_tao_ds.core.logging.logging import logger
from nvidia_tao_ds.core.mlops.wandb import (
    check_wandb_logged_in,
    initialize_wandb,
    is_wandb_initialized
)


def login_and_initialize_wandb(wandb_config, output_dir):
    """ Login and initialize wandb.
    Args:
        wandb_config (DictConfig): wandb config.
        output_dir (str): output directory.
    Return:
        No explicit return
    """
    logged_in = check_wandb_logged_in()
    if not is_wandb_initialized():
        initialize_wandb(output_dir, project=wandb_config.project if wandb_config.project else None,
                         entity=wandb_config.entity if wandb_config.entity else None,
                         save_code=wandb_config.save_code if wandb_config.save_code else None,
                         name=wandb_config.name if wandb_config.name else None,
                         notes=wandb_config.notes if wandb_config.notes else None,
                         tags=wandb_config.tags if wandb_config.tags else None,
                         wandb_logged_in=logged_in)


def create_barplot(data, title, name):
    """ Create barplot in wandb.
    Args:
        data (Pandas dataframe): data to create barplot.
        title (str): barplot title.
        name (str): wandb plot log name.
    Return:
        No explicit return
    """
    table = wandb.Table(data=data)
    barplot = wandb.plot.bar(table, data.columns[0], data.columns[1], title=title)
    wandb.log({name: barplot})


def create_lineplot(data, title, name):
    """ Create lineplot in wandb.
    Args:
        data (Pandas dataframe): data to create barplot.
        title (str): lineplot title.
        name (str): wandb plot log name.
    Return:
        No explicit return
    """
    table = wandb.Table(data=data)
    lineplot = wandb.plot.line(table, data.columns[0], data.columns[1], title=title)
    wandb.log({name: lineplot})


def create_table(data, name):
    """ Create table in wandb.
    Args:
        data (Pandas dataframe): data to create table.
        name (str): wandb table log name.
    Return:
        No explicit return
    """
    table = wandb.Table(data=data)
    wandb.log({name: table})


def plot_PR_curve(result):
    """
    Generate PR curve.
    Args:
        result (Pandas dataframe): Pandas dataframe with KPI values.
        output_dir (str): Output directory.
    Return:
        No explicit return
    """
    for _, metric in result.iterrows():
        seqname = metric['Sequence Name']
        classname = metric['class_name']
        prec = metric['precision']
        recall = metric['recall']

        # Skip summary row
        if seqname == "Summary" and prec is None and recall is None:
            continue

        df = pd.DataFrame({'recall': recall, 'precision': prec})
        create_lineplot(df, f"{seqname} {classname} PR curve ", f"{seqname} {classname}_PR")


def generate_images_with_bounding_boxes(df, wandb_config, output_dir, image_sample_size):
    """
    Generate images with Bounding boxes in wandb.
    Args:
        df (Pandas dataframe): valid dataframe used to draw images with bboxes.
        wandb_config (dict): wandb config.
        output_dir (str): output directory.
    Return:
        No explicit return

    """
    table = wandb.Table(columns=['Name', 'Images'])
    classes = df['type'].unique()
    class_id_to_label = {}
    class_label_to_id = {}
    i = 0
    for classname in classes:
        class_id_to_label[i] = classname
        class_label_to_id[classname] = i
        i += 1

    images = df['img_path'].unique()
    if len(images) > image_sample_size:
        images = random.sample(set(images), image_sample_size)
    for img in images:
        temp_df = df.loc[df['img_path'].str.contains(img), :]
        box_data = []
        for _, box in temp_df.iterrows():
            xmin = box["bbox_xmin"]
            ymin = box["bbox_ymin"]
            xmax = box["bbox_xmax"]
            ymax = box["bbox_ymax"]
            class_id = class_label_to_id[box["type"]]

            box_data.append({"position": {
                             "minX": xmin,
                             "minY": ymin,
                             "maxX": xmax,
                             "maxY": ymax
                             },
                             "class_id": class_id,
                             "domain": "pixel"
                             })
        box_img = wandb.Image(img, boxes={
                              "ground_truth": {"box_data": box_data,
                                               "class_labels": class_id_to_label
                                               }})
        table.add_data(os.path.basename(img), box_img)
    logger.info("It might take some time to log images in wandb.Please wait...")
    wandb.log({"image_table": table})
    # run.finish()
