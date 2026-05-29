# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint script to run TAO analyze."""

import pandas as pd
import os
import time
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from nvidia_tao_ds.config.analytics.default_config import ExperimentConfig
from nvidia_tao_ds.data_analytics.utils import kitti, data_format, coco, image, wandb
from nvidia_tao_ds.data_analytics.utils.constant import COMMON_FILE_NAMES
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.mlops.wandb import is_wandb_initialized
import nvidia_tao_ds.core.logging.logging as status_logging
from nvidia_tao_ds.core.logging.logging import logger


def configure_subgraph(axn, xlabel=None, ylabel=None, xtickrotate=None,
                       ytickrotate=None, xticklabelformat=None,
                       yticklabelformat=None, xticks=None, yticks=None):
    """ Configure graph properties.

    Args:
        axn = graph axis object,
        xlabel (str): Label for x axis,
        ylabel (str): Label for y axis,
        xtickrotate (str): Rotation for x axis label,
        ytickrotate (str): Rotation for y axis label,
        xticklabelformat (str): Format of x axis label,
        yticklabelformat (str): Format of y axis label,
        xticks (str): set x ticks,
        yticks (str): set y ticks

    Return:
        No explicit Return.
    """
    if xlabel:
        axn.set_xlabel(xlabel)
    if ylabel:
        axn.set_ylabel(ylabel)
    if xticklabelformat:
        axn.ticklabel_format(style=xticklabelformat, axis='x')
    if yticklabelformat:
        axn.ticklabel_format(style=yticklabelformat, axis='y')
    if xticks:
        axn.set_xticks(xticks)
    if yticks:
        axn.set_yticks(yticks)
    if xtickrotate:
        axn.tick_params(axis='x', labelrotation=xtickrotate[0])
    if ytickrotate:
        axn.tick_params(axis='y', labelrotation=ytickrotate[0])


def object_count_visualize(valid_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Count.

    Args:
        valid_df (Pandas DataFrame): valid kitti dataframe
        output_dir (str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes
    Return:
        No explicit Return
    """
    # per Object Count
    count_df = valid_df['type'].value_counts(ascending=False).rename(
        'count_num').reset_index().rename(columns={'index': 'type'})
    figuresizes = (graph_attr.width, graph_attr.height)
    show_all = True
    if not graph_attr.show_all and len(count_df) > 100:
        show_all = False
    if not show_all:
        graph_data = count_df.head(100)
    else:
        graph_data = count_df

    s_logger = status_logging.get_status_logger()
    try:
        # Write KPI dict to status callback
        kpi_dict = graph_data.to_dict()
        kpi_dict['analyze_type'] = 'object_count'
        logger.info(f"Object Count KPI Dict: {kpi_dict}")
        s_logger.kpi = kpi_dict
        s_logger.write()
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        logger.error(f"Unable to write KPI dict to status logger: {str(e)}")
    # Create graph for object count
    if wandb_attr.visualize:
        graph_data = graph_data.rename(columns={'type': 'Object Name', 'count_num': 'Count'})
        wandb.create_barplot(graph_data, "Object Name Vs Count", "object_count_chart1")
    else:
        pdf = PdfPages(os.path.join(output_dir, 'Object_count.pdf'))
        fig = plt.figure(figsize=figuresizes)
        if max(count_df['count_num']) - min(count_df['count_num']) > 10:
            binrange = range(min(count_df['count_num']),
                             max(count_df['count_num']),
                             int((max(count_df['count_num']) - min(count_df['count_num'])) / 10))
        else:
            binrange = range(min(count_df['count_num']), max(count_df[
                             'count_num']))

        ax = plt.gca()
        sns.barplot(y='count_num', x='type', data=graph_data, width=0.2)
        txt = "The bar plot below describes the count of each object"\
            " available in valid kitti data."
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        configure_subgraph(ax, xlabel="Object Name", ylabel="Count",
                           xtickrotate=(90, "right"), yticklabelformat="plain",
                           yticks=binrange)
        pdf.savefig(fig)
        plt.close()
    # Create graph for object count(%)
    count_sum = count_df["count_num"].sum()
    count_df['percent'] = (count_df["count_num"] / count_sum) * 100
    count_df['percent'] = count_df['percent'].round(decimals=2)
    graph_data = count_df if show_all else count_df.head(100)

    if wandb_attr.visualize:
        graph_data = graph_data.drop("count_num", axis=1)
        graph_data = graph_data.rename(columns={'type': 'Object Name', 'percent': 'Count(%)'})
        wandb.create_barplot(graph_data, "Object Name Vs Count Percentage", "object_count_chart2")
    else:
        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)
        sns.barplot(y='percent', x='type', data=graph_data, width=0.2)
        txt = "The bar plot below describes the count percentage of each "\
            "object available in valid kitti data ."
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        configure_subgraph(ax, xlabel="Object Name", ylabel="Count (%)",
                           xtickrotate=(90, "right"), yticklabelformat="plain",
                           yticks=binrange)

        pdf.savefig(fig)
        plt.close()
    # Create stats table for object count
    count_df_desc = count_df['count_num'].describe()
    count_stat = pd.DataFrame({'Value': count_df_desc})
    count_stat = count_stat.reset_index()
    count_stat = count_stat.rename(columns={'index': 'Statistic'})
    if not wandb_attr.visualize:
        fig = plt.figure(figsize=figuresizes)
        fig.clf()
        plt.axis('off')
        txt = "The table below shows object count statistics"
        plt.text(0.05, 0.90, txt, transform=fig.transFigure, size=10)
        table = plt.table(cellText=count_stat.values, edges='closed',
                          loc='center', colLoc='right')
        for coord, cell in table.get_celld().items():
            if (coord[1] == 0):
                font_property = FontProperties(weight='bold')
                cell.set_text_props(fontproperties=font_property)
        pdf.savefig(fig)
        plt.close()
    else:
        wandb.create_table(count_stat, "Count statistics")

    # Create summary table for object count per tag
    if not wandb_attr.visualize:
        fig = plt.figure(figsize=figuresizes)
        fig.clf()
        plt.axis('off')
        txt = 'The table below shows object count per object.'
        plt.text(0.05, 0.90, txt, transform=fig.transFigure, size=10)
        count_df['percent'] = count_df['percent'].round(decimals=4)
        graph_data = count_df if show_all else count_df.head(100)

        table = plt.table(cellText=graph_data.values, edges='closed',
                          loc='center', colLoc='right',
                          colLabels=['Object Name', 'Count', 'Percentage'])
        for coord, cell in table.get_celld().items():
            if (coord[0] == 0):
                font_property = FontProperties(weight='bold')
                cell.set_text_props(fontproperties=font_property)
        pdf.savefig(fig)
        plt.close()
        pdf.close()


def occlusion_visualize(valid_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Count.

    Args:
        valid_df (Pandas DataFrame): valid kitti dataframe
        output_dir(str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes
    Return:
        No explicit Return
    """
    # Object Occlusion
    occluded_df = valid_df.groupby(['occluded'])['type'].count().rename('occ_count').reset_index()
    occluded_df['count_per'] = (occluded_df['occ_count'] / occluded_df[
        'occ_count'].sum()) * 100
    occluded_df['count_per'] = occluded_df['count_per'] .round(decimals=3)

    # Occlusion per object
    per_object_occluded_df = valid_df.groupby(['type', 'occluded'])[
        'truncated'].count().reset_index()
    figuresizes = (graph_attr.width, graph_attr.height)
    show_all = True
    if not graph_attr.show_all and len(per_object_occluded_df) > 100:
        show_all = False
    if not wandb_attr.visualize:
        pdf = PdfPages(os.path.join(output_dir, 'Occlusion.pdf'))
        # Create Occlusion Barplot
        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)
        gph = sns.barplot(occluded_df, y='count_per', x='occluded', width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="Occlusion", ylabel="Count (%)",
                           yticks=binrange)
        txt = "The bar plot below describes the count percentage of each "\
              "occlusion level in valid kitti data .\n0: Fully visible, "\
              "1: Partly occluded, 2: Largely occluded,"\
              " 3: Unknown, Any other integer: Unknown occlusion tag"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()
        sumvalue = per_object_occluded_df["truncated"].sum()
        # Draw a nested barplot for occlusion level per object Tag
        per_object_occluded_df["truncated"] = (per_object_occluded_df["truncated"] / sumvalue) * 100
        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        graph_data = per_object_occluded_df if show_all else per_object_occluded_df.head(100)

        gph = sns.barplot(data=graph_data, x="type", y="truncated",
                          hue="occluded", errorbar="sd", palette="dark")
        txt = "The bar plot below describes the count percentage of occlusion"\
              " per object, available in valid kitti data .\n0: Fully "\
              "Visible, 1: Partly occluded, 2: Largely occluded, 3: Unknown, "\
              "Any other integer: Unknown occlusion tag"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        configure_subgraph(ax, xlabel="Object Name", ylabel="Count(%)",
                           xtickrotate=(90, "right"), yticklabelformat="plain",
                           yticks=binrange)
        fig.legend(fontsize=5)
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    else:
        occluded_df['occluded'] = np.where((occluded_df['occluded'] < 0) | (occluded_df['occluded'] > 3),
                                           "Unknown occlusion value", occluded_df['occluded'])
        occluded_df['occluded'] = occluded_df['occluded'].replace(["0", "1", "2", "3"],
                                                                  ['Fully visible',
                                                                   'Partly occluded',
                                                                   'Largely occluded',
                                                                   'Unknown'])

        occluded_df = occluded_df.drop("occ_count", axis=1)
        occluded_df = occluded_df.rename(columns={'occluded': 'Occlusion', 'count_per': 'Count(%)'})
        wandb.create_barplot(occluded_df, "Occlusion Vs Count(%)", "Occlusion_chart1")


def bbox_area_visualize(valid_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Bounding box area.

    Args:
        valid_df (Pandas DataFrame): valid kitti data dataframe.
        output_dir(str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes
    Return:
        No explicit Return
    """
    figuresizes = (graph_attr.width, graph_attr.height)
    area_mean = valid_df.groupby('type')['bbox_area']
    area_mean = area_mean.describe()['mean'].reset_index()
    if not graph_attr.show_all and len(area_mean) > 100:
        graph_data = area_mean.head(100)
    else:
        graph_data = area_mean

    s_logger = status_logging.get_status_logger()
    try:
        # Write KPI dict to status callback
        kpi_dict = graph_data.to_dict()
        kpi_dict['analyze_type'] = 'bbox_area'
        logger.info(f"Bbox area KPI Dict: {kpi_dict}")
        s_logger.kpi = kpi_dict
        s_logger.write()
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        logger.error(f"Unable to write KPI dict to status logger: {str(e)}")

    area_stats = valid_df['bbox_area'].describe()
    area_stat = pd.DataFrame({'Value': area_stats})
    area_stat['Value'] = area_stat['Value'].round(decimals=4)
    area_stat = area_stat.reset_index()
    area_stat = area_stat.rename(columns={'index': 'Area Statistic'})
    area_stat_per_type = valid_df.groupby('type')['bbox_area'].describe()
    if not graph_attr.show_all and len(area_stat_per_type) > 100:
        graph_data_per_type = area_stat_per_type.head(100)
    else:
        graph_data_per_type = area_stat_per_type
    if not wandb_attr.visualize:
        pdf = PdfPages(os.path.join(output_dir, 'Area.pdf'))
        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        minv = min(area_mean['mean'])
        maxv = max(area_mean['mean'])
        if maxv - minv > 10:
            binrange = range(round(minv), round(maxv), int((maxv - minv) // 10))
        else:
            binrange = range(round(minv), round(maxv))

        sns.scatterplot(graph_data, x='type', y='mean')
        configure_subgraph(ax, xlabel="Object Name", ylabel="mean bbox area ",
                           xtickrotate=(90, "right"), yticklabelformat="plain",
                           yticks=binrange)
        txt = "The scatter plot below describes the mean bounding box size "\
              "of different objects."
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        fig.clf()
        plt.axis('off')
        txt = 'The table below shows bounding box area statistics.'
        plt.text(0.05, 0.90, txt, transform=fig.transFigure, size=10)
        table = plt.table(cellText=area_stat.values, edges='closed',
                          loc='center', colLoc='right',
                          colLabels=area_stat.columns)
        for coord, cell in table.get_celld().items():
            if (coord[0] == 0 or coord[1] == 0):
                font_property = FontProperties(weight='bold')
                cell.set_text_props(fontproperties=font_property)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        fig.clf()
        plt.axis('off')
        txt = 'The table below shows per object bounding box area statistics.'
        plt.text(0.05, 0.90, txt, transform=fig.transFigure, size=10)
        table = plt.table(cellText=graph_data_per_type.values, edges='closed',
                          loc='center', colLoc='right',
                          colLabels=graph_data_per_type.columns,
                          rowLabels=graph_data_per_type.index)
        for coord, cell in table.get_celld().items():
            if (coord[0] == 0 or coord[1] == -1):
                font_property = FontProperties(weight='bold')
                cell.set_text_props(fontproperties=font_property)
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    else:
        graph_data = graph_data.rename(columns={'type': 'Object Name', 'mean': 'Mean bbox area'})
        wandb.create_barplot(graph_data, "Object Name Vs Mean bbox area", "bbox_area_chart1")
        wandb.create_table(area_stat, "Area statistics")
        graph_data_per_type = graph_data_per_type.reset_index(level=0)
        graph_data_per_type = graph_data_per_type.rename(columns={'type': 'Object Name'})
        wandb.create_table(graph_data_per_type, "Area statistics per object.")


def truncation_visualize(valid_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Truncation.

    Args:
        valid_df (Pandas DataFrame): valid kitti dataframe.
        output_dir(str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes

    Return:
        No explicit Return.
    """
    # Truncation
    valid_df['truncated'] = valid_df['truncated'].round(decimals=1)
    truncated_df = valid_df['truncated']
    truncated_df = truncated_df.value_counts(normalize=True)
    truncated_df = truncated_df.rename('count_per').reset_index().rename(columns={'index': 'truncated'})
    truncated_df_count = valid_df['truncated']
    truncated_df_count = truncated_df_count.value_counts().rename('count')
    truncated_df_count = truncated_df_count.reset_index().rename(columns={'index': 'truncated'})
    truncated_df['truncated'] = truncated_df['truncated'] * 100
    truncated_df['count_per'] = truncated_df['count_per'] * 100
    truncated_df['count_per'] = truncated_df['count_per'].round(decimals=2)
    figuresizes = (graph_attr.width, graph_attr.height)
    if not wandb_attr.visualize:
        pdf = PdfPages(os.path.join(output_dir, 'Truncation.pdf'))
        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        c_min = min(truncated_df_count['count'])
        c_max = max(truncated_df_count['count'])
        if c_max - c_min > 10:
            binrange = range(c_min, c_max, int((c_max - c_min) / 10))
        else:
            binrange = range(c_min, c_max)
        gph = sns.barplot(data=truncated_df_count, y='count', x='truncated',
                          width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="Trucation(%)", ylabel="Object Count",
                           yticks=binrange, yticklabelformat="plain")
        txt = "The bar plot below describes the object count against "\
              "truncation percentage."
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)
        gph = sns.barplot(data=truncated_df, y='count_per', x='truncated',
                          width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="Trucation(%)", ylabel="Object Count(%)",
                           yticks=binrange)
        txt = "The bar plot below describes the object count percentage "\
              "against truncation percentage"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    else:
        truncated_df_count = truncated_df_count.rename(columns={'count': 'Count', 'truncated': 'Truncation(%)'})
        wandb.create_barplot(truncated_df_count, "Truncation(%) Vs Count", "truncation_chart1")


def invalid_data_visualize(valid_df, invalid_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Truncation.

    Args:
        valid_df (Pandas DataFrame): valid kitti data
        invalid_df (Pandas DataFrame): invalid kitti data
        output_dir(str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes
    Return:
        No explicit Return
    """
    count_df = valid_df['type']
    count_df = count_df.value_counts().rename('count_num')
    count_df = count_df.reset_index().rename(columns={'index': 'type'})

    # invalid Obejct tag kitti row
    invalid_count_df = invalid_df['type']
    var = 'invalid_count_num'
    invalid_count_df = invalid_count_df.value_counts().rename(var)
    invalid_count_df = invalid_count_df.reset_index().rename(columns={'index': 'type'})

    valid_invalid_count_df = count_df.merge(invalid_count_df, on='type', how='outer')
    cols = {"count_num": "valid_count", "invalid_count_num": "invalid_count"}
    valid_invalid_count_df = valid_invalid_count_df.rename(columns=cols)
    valid_invalid_count_df = valid_invalid_count_df.fillna(0)

    xmin = invalid_df['bbox_xmin']
    xmax = invalid_df['bbox_xmax']
    ymin = invalid_df['bbox_ymin']
    ymax = invalid_df['bbox_ymax']
    if 'img_height' in invalid_df.columns:
        oob_condition = ((xmin < 0) | (ymin < 0) | (ymax < 0) |
                         (xmax < 0) | (ymax > invalid_df['img_height']) |
                         (xmax > invalid_df['img_width']))
    else:
        oob_condition = ((xmin < 0) | (ymin < 0) | (ymax < 0) |
                         (xmax < 0))

    out_of_bound_bbox = invalid_df[oob_condition]
    inverted_cord = invalid_df[(ymax > 0) &
                               (xmax > 0) &
                               (ymin > 0) &
                               (xmin > 0) &
                               ((xmax < xmin) |
                               (ymax < ymin))]
    coord_df = pd.DataFrame({"Coordinates": ["valid", "inverted",
                                             "out_of_bound"],
                            "count": [len(valid_df), len(inverted_cord),
                                      len(out_of_bound_bbox)]})
    figuresizes = (graph_attr.width, graph_attr.height)
    if len(inverted_cord) > 0:
        temp_df = inverted_cord['type']
        var = "inverted_cord_count"
        temp_df = temp_df.value_counts(ascending=True).rename(var)
        temp_df = temp_df.reset_index().rename(columns={"index": "type"})
        if not graph_attr.show_all and len(temp_df) > 100:
            graph_data_inverted = temp_df.head(100)
        else:
            graph_data_inverted = temp_df

    if len(out_of_bound_bbox) > 0:
        temp_df = out_of_bound_bbox['type']
        var = "out_of_bound_bbox_count"
        temp_df = temp_df.value_counts(ascending=True).rename(var)
        temp_df = temp_df.reset_index().rename(columns={"index": "type"})
        if not graph_attr.show_all and len(temp_df) > 100:
            graph_data_oob = temp_df.head(100)
        else:
            graph_data_oob = temp_df

    if not wandb_attr.visualize:
        pdf = PdfPages(os.path.join(output_dir, 'Invalid_data.pdf'))

        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        gph = sns.barplot(data=coord_df, x="Coordinates", y="count",
                          errorbar="sd", palette="dark")
        gph.bar_label(gph.containers[0])
        txt = "The bar plot below describes the count of valid, "\
              "inverted and out of bound coordinates."
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        if len(inverted_cord) > 0:
            fig = plt.figure(figsize=figuresizes)
            ax = plt.gca()
            gph = sns.barplot(y='inverted_cord_count', x='type', data=graph_data_inverted,
                              width=0.2)
            gph.bar_label(gph.containers[0])
            configure_subgraph(ax, xlabel="Object Name",
                               ylabel="inverted_cord_count",
                               xtickrotate=(90, "right"),
                               yticklabelformat="plain")
            txt = "The bar plot below describes the inverted coordinates "\
                  "count per object."
            plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
            pdf.savefig(fig)
            plt.close()

        if len(out_of_bound_bbox) > 0:
            fig = plt.figure(figsize=figuresizes)
            ax = plt.gca()

            gph = sns.barplot(y='out_of_bound_bbox_count', x='type',
                              data=graph_data_oob, width=0.2)
            gph.bar_label(gph.containers[0])

            configure_subgraph(ax, xlabel="Object Name",
                               ylabel="out of bound coordinates count",
                               xtickrotate=(90, "right"),
                               yticklabelformat="plain")
            txt = "The bar plot below describes the out of bound coordinates "\
                  "count per object."
            plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
            pdf.savefig(fig)
            plt.close()
        pdf.close()
    else:
        wandb.create_barplot(coord_df, "Count Vs bbox Coordinates", "coordinates_chart1")
        if len(inverted_cord) > 0:
            graph_data_inverted = graph_data_inverted.rename(columns={'type': 'Object Name',
                                                                      'inverted_cord_count': 'Inverted coordinates count'})

            wandb.create_barplot(graph_data_inverted, "Object Name Vs Inverted coordinates count", "coordinates_chart2")
        if len(out_of_bound_bbox) > 0:
            graph_data_oob = graph_data_oob.rename(columns={'out_of_bound_bbox_count': 'out of bound coordinates count',
                                                            'type': 'Object Name'})

            wandb.create_barplot(graph_data_oob, "Object Name Vs Out of bound coordinates count", "coordinates_chart3")


def image_visualize(image_df, output_dir, graph_attr, wandb_attr):
    """ Create Graphs for Object Truncation.

    Args:
        image_df (Pandas DataFrame): Image Dataframe
        output_dir(str): Result Directory.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes
    Return:
        No explicit Return
    """
    # Image stats
    size_df = image_df['size'].value_counts(ascending=True,
                                            normalize=True).rename('count_per').reset_index().rename(columns={'index': 'size'})

    size_df['count_per'] = size_df['count_per'] * 100

    width_df = image_df['img_width']
    width_df = width_df.value_counts(ascending=True, normalize=True)
    width_df = width_df.rename('count_per').reset_index()

    width_df['count_per'] = width_df['count_per'] * 100

    height_df = image_df['img_height']
    height_df = height_df.value_counts(ascending=True, normalize=True)

    height_df = height_df.rename('count_per').reset_index()
    height_df['count_per'] = height_df['count_per'] * 100

    height_df['count_per'] = height_df['count_per'].round(decimals=3)
    width_df['count_per'] = width_df['count_per'].round(decimals=3)
    size_df['count_per'] = size_df['count_per'].round(decimals=3)
    image_stat = image_df.describe()
    figuresizes = (graph_attr.width, graph_attr.height)
    if not wandb_attr.visualize:
        pdf = PdfPages(os.path.join(output_dir, 'Image.pdf'))

        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)

        gph = sns.barplot(size_df, y='count_per', x='size', width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="image size", ylabel="Count Percentage",
                           yticks=binrange)
        txt = "The bar plot below decribes the count(%) of different image "\
              "sizes"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)
        gph = sns.barplot(width_df, y='count_per', x='img_width', width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="image width",
                           ylabel="Count Percentage", yticks=binrange)

        txt = "The bar plot below decribes the count(%) of different image "\
              "widths"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        ax = plt.gca()
        binrange = range(0, 100, 10)

        gph = sns.barplot(height_df, y='count_per', x='img_height', width=0.2)
        gph.bar_label(gph.containers[0])
        configure_subgraph(ax, xlabel="image height",
                           ylabel="Count Percentage",
                           yticks=binrange)
        txt = "The bar plot below decribes the count(%) of different image "\
              "heights"
        plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
        pdf.savefig(fig)
        plt.close()

        fig = plt.figure(figsize=figuresizes)
        fig.clf()
        plt.axis('off')
        txt = 'The table below shows image statistics.'
        plt.text(0.05, 0.90, txt, transform=fig.transFigure, size=10)
        table = plt.table(cellText=image_stat.values, edges='closed',
                          loc='center', colLoc='right',
                          colLabels=image_stat.columns,
                          rowLabels=image_stat.index)
        for coord, cell in table.get_celld().items():
            if (coord[0] == 0 or coord[1] == -1):
                font_property = FontProperties(weight='bold')
                cell.set_text_props(fontproperties=font_property)
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    else:

        size_df = size_df.rename(columns={'count_per': 'Count(%)', 'size': 'image area'})
        wandb.create_barplot(size_df, "image area vs count(%)", "image_chart1")
        width_df = width_df.rename(columns={'count_per': 'Count(%)', 'img_width': 'image width'})
        wandb.create_barplot(width_df, "image width vs count(%)", "image_chart2")
        height_df = height_df.rename(columns={'count_per': 'Count(%)', 'img_height': 'image height'})
        wandb.create_barplot(height_df, "image height vs count(%)", "image_chart3")
        image_stat = image_stat.reset_index(level=0)
        image_stat = image_stat.rename(columns={'index': 'Stat'})
        wandb.create_table(image_stat, "Image Statistics")


def create_csv(valid_df, invalid_df, output_dir):
    """ Create Graph and summary.

    Args:
        valid_df(Pandas Dataframe): Valid kitti dataframe
        invalid_df(Pandas Dataframe): invalid kitti dataframe
        output_dir(str): result Dir

    Return:
        No explicit returns.
    """
    csv_folder = os.path.join(output_dir, 'csv')
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    invalid_file = os.path.join(csv_folder, 'invalid_data.csv')
    valid_file = os.path.join(csv_folder, 'valid_data.csv')
    valid_df = valid_df.drop('bbox_area', axis=1)
    invalid_df = invalid_df.drop('bbox_area', axis=1)
    valid_df.to_csv(valid_file, columns=valid_df.columns, index=False)
    invalid_df.to_csv(invalid_file, columns=invalid_df.columns, index=False)


def summary_and_graph(valid_df, invalid_df, image_df, output_dir, data_format, graph_attr, wandb_attr):
    """ Create Graph and summary.

    Args:
        valid_df (Pandas Dataframe): Valid kitti dataframe
        invalid_df (Pandas Dataframe): invalid kitti dataframe
        image_df (Pandas Dataframe): image dataframe
        output_dir (str): result Dir
        data_format (str): input data format.
        graph_attr (DictConfig): graph attributes(
            height - to set graph height
            width - to set graph width
            show_all - to show all object on graph,
            by default maximum 100 object will be shown.)
        wandb_attr (DictConfig): wandb attributes

    Return:
        No explicit returns.
    """
    # Create CSV for valid and invalid data
    create_csv(valid_df, invalid_df, output_dir)
    # Create visualizations
    output_dir = os.path.join(output_dir, 'graphs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    object_count_visualize(valid_df, output_dir, graph_attr, wandb_attr)
    bbox_area_visualize(valid_df, output_dir, graph_attr, wandb_attr)
    if image_df is not None:
        image_visualize(image_df, output_dir, graph_attr, wandb_attr)
    if data_format == "KITTI":
        occlusion_visualize(valid_df, output_dir, graph_attr, wandb_attr)
        truncation_visualize(valid_df, output_dir, graph_attr, wandb_attr)

    invalid_data_visualize(valid_df, invalid_df, output_dir, graph_attr, wandb_attr)


def visualize_on_wandb(config, valid_df, invalid_df, image_df, len_image_data):
    """ Visualize on wandb.

    Args:
        config (Hydra config): Config element of the analyze config.
        valid_df (Pandas Dataframe): Valid kitti dataframe
        invalid_df (Pandas Dataframe): invalid kitti dataframe
        image_df (Pandas Dataframe): image dataframe
        len_image_data (int): len of image data dict.
    Return:
        No explicit returns.
    """
    wandb.login_and_initialize_wandb(config.wandb, config.results_dir)
    if config.graph.generate_summary_and_graph:
        summary_and_graph(valid_df, invalid_df, image_df, config.results_dir,
                          config.data.input_format, config.graph, config.wandb)
    if not is_wandb_initialized():
        logger.info("Not able to login or initialize wandb.Exiting..")
        sys.exit(1)
    if config.image.generate_image_with_bounding_box:
        if len_image_data == 0:
            logger.info("Skipping visualizing images with Bounding boxes.Please provide correct path in data.image_dir .")
        else:
            wandb.generate_images_with_bounding_boxes(valid_df, config.wandb, config.results_dir, config.image.sample_size)


def visualize_on_desktop(config, valid_df, invalid_df, image_df, image_data):
    """ Visualize and save locally.

    Args:
        config (Hydra config): Config element of the analyze config.
        valid_df (Pandas Dataframe): Valid kitti dataframe
        invalid_df (Pandas Dataframe): invalid kitti dataframe
        image_df (Pandas Dataframe): image dataframe
        image_data (Dict): Dict containing image info (image_width,
                            image_height, image_path)
    Return:
        No explicit returns.
    """
    if config.graph.generate_summary_and_graph:
        summary_and_graph(valid_df, invalid_df, image_df, config.results_dir,
                          config.data.input_format, config.graph, config.wandb)
        logger.info(f"Created Graphs inside {config.results_dir} folder")
    # Generate Images with bounding boxes

    if config.image.generate_image_with_bounding_box:
        if len(image_data) == 0:
            logger.info("Skipping visualizing images with Bounding boxes.Please provide correct path in data.image_dir .")
        else:
            logger.info("Generating images with bounding boxes and labels.")
            image.generate_images_with_bounding_boxes(valid_df, image_data, config.results_dir,
                                                      config.image.sample_size, config.workers, config.data.input_format)


@monitor_status(mode='KITTI analysis')
def analyze_dataset_kitti(config):
    """Tao kitti analysis.

    Args:
        config (Hydra config): Config element of the analyze config.
    """
    start_time = time.perf_counter()
    kitti_obj = data_format.create_data_object("KITTI",
                                               image_dir=config.data.image_dir,
                                               ann_path=config.data.ann_path)
    if not os.path.isdir(config.data.ann_path):
        logger.info("Please provide path of kitti label directory in config data.ann_path.")
        sys.exit(1)
    kitti.list_files(kitti_obj)

    if kitti_obj.image_paths is None:
        logger.info("Image Directory not found.Processing only label files")
        image_data = None
        image_df = None
    else:
        # Get image data (image width and height)
        image_data = kitti.get_image_data(kitti_obj)
        image_df = kitti.create_image_dataframe(image_data)
    # Validate and create big merged kitti files
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
    if config.wandb.visualize:
        visualize_on_wandb(config, valid_df, invalid_df, image_df, len(image_data))
    else:
        visualize_on_desktop(config, valid_df, invalid_df, image_df, image_data)

    logger.debug(f"Total time taken : {time.perf_counter() - start_time}")


@monitor_status(mode='COCO analysis')
def analyze_dataset_coco(config):
    """Tao coco analysis.

    Args:
        config (Hydra config): Config element of the analyze config.
    """
    start = time.perf_counter()
    if not os.path.isfile(config.data.ann_path):
        logger.info("Please provide path of coco annotation file in config data.ann_path.")
        sys.exit(1)
    coco_obj = data_format.create_data_object("COCO",
                                              ann_path=config.data.ann_path,
                                              image_dir=config.data.image_dir)
    image_data = coco.get_image_data(coco_obj)
    image_df = None
    if image_data:
        image_df = coco.create_image_dataframe(image_data)
    # Dataframe creation for valid and invalid kitti data
    valid_df, invalid_df = coco.create_dataframe(coco_obj)
    if config.wandb.visualize:
        visualize_on_wandb(config, valid_df, invalid_df, image_df, len(image_data))
    else:
        visualize_on_desktop(config, valid_df, invalid_df, image_df, image_data)

    logger.debug(f"Total time taken : {time.perf_counter() - start}")


spec_root = os.path.dirname(os.path.abspath(__file__))


@hydra_runner(
    config_path=os.path.join(spec_root, "../experiment_specs"),
    config_name="analyze", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """TAO Analyze main wrapper function."""
    try:
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir)
        if cfg.data.input_format == "COCO":
            analyze_dataset_coco(cfg)
        elif cfg.data.input_format == "KITTI":
            analyze_dataset_kitti(cfg)
        else:
            logger.info(f"Data format {cfg.data.input_format} is not supported.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupting dataset analysis.")
        sys.exit(1)
    except RuntimeError as e:
        logger.info(f"Analysis run failed with runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.info(f"Analysis run failed with other error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
