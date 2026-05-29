# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint script to run TAO kpi analyze."""

import os
import sys
from prettytable import PrettyTable
import pandas as pd

from nvidia_tao_ds.config.analytics.default_config import ExperimentConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logger
from nvidia_tao_ds.data_analytics.utils import data_format, data_process, kpi, wandb, local_visualize
from nvidia_tao_ds.annotations.conversion.kitti_to_coco import construct_category_map


def analyze(cfg):
    """TAO KPI analysis.

    Args:
        config (Hydra config): Config element of the KPI analyze config.
    """
    input_format = cfg.data.input_format
    mapping = cfg.data.mapping
    iou_threshold = cfg.kpi.iou_threshold
    conf_threshold = cfg.kpi.conf_threshold
    ignore_sqwidth = cfg.kpi.ignore_sqwidth
    num_recall_points = cfg.kpi.num_recall_points

    column_names = ["Sequence", "Class", "TP", "FP", "FN", "TN", "Precision", "Recall", "Accuracy", "AP"]
    resultTable = PrettyTable(column_names)
    df_lists = []
    for idx, data_source in enumerate(cfg.data.kpi_sources):
        # check keys in the data_source dict
        for k in ["image_dir", "ground_truth_ann_path", "inference_ann_path"]:
            assert k in data_source, f"{k} not found in kpi_sources. Please check your spec file."

        image_dir = data_source["image_dir"]
        ground_truth_ann_path = data_source["ground_truth_ann_path"]
        inference_ann_path = data_source["inference_ann_path"]

        logger.info(f"[{idx + 1}/{len(cfg.data.kpi_sources)}]: {image_dir}")

        # Creating object for ground-truth and predicted data.
        gt_data_obj = data_format.create_data_object(input_format,
                                                     ann_path=ground_truth_ann_path,
                                                     image_dir=image_dir,
                                                     data_source="ground_truth")
        pred_data_obj = data_format.create_data_object(input_format,
                                                       ann_path=inference_ann_path,
                                                       image_dir=image_dir,
                                                       data_source="predicted")
        # Creating objecting to process the data objects.
        gt_data_service_obj = data_process.create_data_process_object(input_format, gt_data_obj)
        pred_data_service_obj = data_process.create_data_process_object(input_format, pred_data_obj)

        # Creation of dataframes to use for analysis.
        gt_df, gt_ids = gt_data_service_obj.create_dataframe()
        gt_data_obj.df, gt_data_obj.ids = gt_df, gt_ids
        pred_df, pred_ids = pred_data_service_obj.create_dataframe()
        pred_data_obj.df, pred_data_obj.ids = pred_df, pred_ids

        # Evaluate Results
        cat_map = None
        if input_format == "KITTI":
            cat_map = construct_category_map(ground_truth_ann_path, mapping)

        statistics, MAP = kpi.evaluate(gt_data_obj, pred_data_obj, cat_map, iou_threshold,
                                       conf_threshold, ignore_sqwidth, num_recall_points)

        # Add sequence Name
        statistics['Sequence Name'] = image_dir.split('/')[-2]
        statistics['Tag'] = cfg.visualize.tag

        for _, result_dict in statistics.iterrows():
            resultTable.add_row([image_dir.split('/')[-2], result_dict['class_name'],
                                result_dict['TP'], result_dict['FP'],
                                result_dict['FN'], result_dict['TN'],
                                round(result_dict['Pr'], 4), round(result_dict['Re'], 4),
                                round(result_dict['Acc'], 4), round(result_dict['AP'], 4)])

        if cfg.kpi.is_internal:
            # For intenal KPI, drop non-person class
            statistics = statistics[statistics.class_name == 'person']

        # Sort columns
        kpi_df = statistics[["Tag", "Sequence Name", "class_name",
                             "TP", "FP", "FN", "TN", "Pr", "Re", "Acc", "AP",
                             "precision", "recall"]]
        df_lists.append(kpi_df)
        logger.info(f"mAP: {MAP}")

    # Aggregate per-sequence results
    final_kpi_df = pd.concat(df_lists)

    if cfg.kpi.is_internal:
        # Get summary
        tp, fp = final_kpi_df.TP.sum(), final_kpi_df.FP.sum()
        fn, tn = final_kpi_df.FN.sum(), final_kpi_df.TN.sum()
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        # TODO: Check if this AP is correct calculation
        ap = final_kpi_df.AP.mean()
        final_kpi_df.loc[len(final_kpi_df)] = [cfg.visualize.tag, "Summary", "person",
                                               tp, fp, fn, tn, prec, rec, acc, ap,
                                               None, None]

    # Dump KPI into csv
    output_csv_path = os.path.join(cfg.results_dir, "kpi_calc.csv")
    final_kpi_df[["Sequence Name", "TP", "FP", "FN", "TN", "Pr", "Re", "Acc", "AP"]].to_csv(output_csv_path, index=False)

    logger.info(resultTable)

    # Visualize results locally or on wandb.
    if cfg.visualize.platform == "local":
        local_visualize.plot_PR_curve(statistics, cfg.results_dir)
    elif cfg.visualize.platform == "wandb":
        wandb.login_and_initialize_wandb(cfg.wandb, cfg.results_dir)
        if not wandb.is_wandb_initialized():
            logger.info("Not able to login or initialize wandb. Skipping Wandb Initialization.")
        wandb.plot_PR_curve(final_kpi_df)

        final_kpi_df = final_kpi_df.drop(columns=["precision", "recall"])
        if cfg.kpi.is_internal:
            # For internal, only person class is used
            final_kpi_df = final_kpi_df.drop(columns=["class_name"])
        else:
            # Rename column
            final_kpi_df = final_kpi_df.rename(columns={"class_name": "Class Name"})
        wandb.create_table(final_kpi_df, "Statistics")


spec_root = os.path.dirname(os.path.abspath(__file__))


@hydra_runner(
    config_path=os.path.join(spec_root, "../experiment_specs"),
    config_name="kpi_analyze", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """TAO KPI Analyzer main wrapper function."""
    try:
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir)
        if cfg.data.input_format in ("COCO", "KITTI"):
            analyze(cfg)
        else:
            logger.info(f"Data format {cfg.data.input_format} is not supported.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupting dataset KPI analysis.")
        sys.exit(1)
    except RuntimeError as e:
        logger.info(f"KPI Analysis run failed with error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.info(f"KPI Analysis run failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
