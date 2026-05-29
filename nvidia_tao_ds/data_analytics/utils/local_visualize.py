# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities to handle visualization on desktop."""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os


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


def plot_PR_curve(result, output_dir):
    """
    Generate PR curve.
    Args:
        result (Pandas dataframe): Pandas dataframe with KPI values.
        output_dir (str): Output directory.
    Return:
        No explicit return
    """
    pdf = PdfPages(os.path.join(output_dir, 'PR_curves.pdf'))
    for _, metric in result.iterrows():
        seqname = metric['Sequence Name']
        classname = metric['class_name']
        prec = metric['precision']
        recall = metric['recall']
        df = pd.DataFrame({'precision': prec, 'recall': recall})
        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca()
        configure_subgraph(ax, xlabel="Recall", ylabel="Precision")
        plt.plot(df['recall'], df['precision'])
        plt.title(f'{seqname} {classname} -  Precision x Recall curve ')
        pdf.savefig(fig)
        plt.close()
    pdf.close()
