# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative unique-assignment k-NN mining (unique_neighbor_matching).

The core matching is a greedy, one-to-one assignment: each source image is matched
to at most one target. Targets are processed in order, and each claims its ``topn``
nearest source images that are still unclaimed. So if target ``t1`` takes its three
nearest sources ``s1, s2, s3``, a later target ``t2`` whose nearest also include
``s3`` cannot reuse it — it skips the claimed sources and falls through to its
next-nearest free ones (``s4, s5, s6``). A target that cannot fill ``topn`` free
sources releases its tentative picks (leaving them available to other targets) and
is retried with a widened candidate pool, iterating until ``desired_unique_count``
unique sources are collected.

Two allocation policies:
  - ``global``: match ``desired_unique_count`` unique source images resembling
    the whole target set via iterative unique assignment.
  - ``class_stratified``: split into per-rare-class pools plus a ``non_rare`` pool
    using COCO/KITTI annotations, apportion the budget by target class ratio
    (largest-remainder), and match each class in turn with a non-rare-pool fallback.
    Each source is owned by the first class that matches it: every pass excludes
    already-matched files, so classes never reselect one another's sources.

Writes a directory of outputs: ``final_unique_files.parquet``, ``summary.json``,
per-iteration parquets, and (with ``visualize``) per-class visualization PNGs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cudf
import pandas as pd

from nvidia_tao_ds.config.mining.tmm.unique_neighbor_matching import UniqueNeighborMatchingConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.dataset_loading import AnnotationFormat
from nvidia_tao_ds.core.utils.path_utils import ensure_dir
from nvidia_tao_ds.mining.tmm.parquet_helpers import load_datasets
from nvidia_tao_ds.mining.tmm.vector_preparation import extract_column_matrix
from nvidia_tao_ds.mining.tmm.selection import iterative_tmm_retrieval, split_datasets_by_class
from nvidia_tao_ds.mining.tmm.reporting.stats import (
    calculate_distance_stats_per_class,
    calculate_distribution_stats,
    calculate_per_class_counts,
)
from nvidia_tao_ds.mining.tmm.reporting.viz import plot_per_class_counts, visualize_target_to_mined

if TYPE_CHECKING:
    import cupy

spec_root = Path(__file__).resolve().parent


def _allocate_quotas(group_ratios: dict[str, float], desired: int) -> dict[str, int]:
    """Apportion ``desired`` across groups by ratio using the largest-remainder method.

    Returns an integer quota per group that is non-negative and sums to EXACTLY
    ``desired`` (no over-allocation, no negative non_rare quota). Each
    group gets ``floor(desired * ratio / total_ratio)``; the leftover units go
    one each to the groups with the largest fractional remainders (ties broken by
    group name, so the result is invariant to input ordering).
    """
    total_ratio = sum(group_ratios.values())
    if desired <= 0 or total_ratio <= 0:
        return {group: 0 for group in group_ratios}

    ideal = {group: desired * ratio / total_ratio for group, ratio in group_ratios.items()}
    quota = {group: int(value) for group, value in ideal.items()}  # floor
    leftover = desired - sum(quota.values())
    by_remainder = sorted(group_ratios, key=lambda s: (ideal[s] - quota[s], s), reverse=True)
    for group in by_remainder[:leftover]:
        quota[group] += 1
    return quota


def _resolve_detection_format(cfg: UniqueNeighborMatchingConfig) -> Optional[AnnotationFormat]:
    """Resolve the user-declared detection format to an ``AnnotationFormat``.

    Returns None when no detection file is in play (format irrelevant). Raises when a
    detection file is provided but ``detection_format`` was not declared — the format is
    required from the user, never inferred from the path.
    """
    if not (cfg.source_detection_file or cfg.target_detection_file):
        return None
    if not cfg.detection_format:
        raise ValueError(
            "detection_format is required (coco or kitti) when a detection file is provided"
        )
    return AnnotationFormat(cfg.detection_format)


class UniqueNeighborMatchingMining:
    """Iterative unique-assignment k-NN mining (global + class_stratified policies)."""

    def run(self, cfg: UniqueNeighborMatchingConfig) -> None:
        """Match similar source items via iterative k-NN; dispatch on allocation policy."""
        ensure_dir(cfg.output_dir)

        if not cfg.desired_unique_count:
            raise ValueError("desired_unique_count is required")
        rare_classes = {c.strip().lower() for c in (cfg.rare_class_list or "").split(",") if c.strip()}

        if cfg.allocation_policy == "class_stratified":
            self._validate_class_stratified_prereqs(cfg, rare_classes)
            self._run_class_stratified(cfg, rare_classes)
        else:
            self._run_global(cfg, rare_classes)

    @staticmethod
    def _validate_class_stratified_prereqs(cfg: UniqueNeighborMatchingConfig, rare_classes: set[str]) -> None:
        """class_stratified needs rare classes and both detection files."""
        if not rare_classes:
            raise ValueError("rare_class_list is required when allocation_policy is class_stratified")
        if not cfg.source_detection_file:
            raise ValueError("source_detection_file is required when allocation_policy is class_stratified")
        if not cfg.target_detection_file:
            raise ValueError("target_detection_file is required when allocation_policy is class_stratified")
        if not cfg.detection_format:
            raise ValueError("detection_format (coco or kitti) is required when allocation_policy is class_stratified")

    # ── global policy ──────────────────────────────────────────────────────────

    def _run_global(self, cfg: UniqueNeighborMatchingConfig, rare_classes: set[str]) -> None:
        """Global allocation: match ``desired_unique_count`` unique sources against the whole target set.

        No class stratification — a single iterative unique-assignment pass over every target,
        then write final_unique_files.parquet and summary.json (plus per-class stats and optional
        visualizations when a detection file is provided).
        """
        source_path = cfg.source_path
        target_path = cfg.target_path
        output_path = Path(cfg.output_dir)
        distance_metric = cfg.distance_metric
        source_embedding_column = cfg.source_embedding_column
        target_embedding_column = cfg.target_embedding_column
        source_filepath_column = cfg.source_filepath_column
        target_filepath_column = cfg.target_filepath_column
        desired_unique_count = cfg.desired_unique_count

        logger.info("Global iterative TMM matching (whole set); desired unique count: %d", desired_unique_count)

        df_source, df_target = load_datasets(source_path, target_path, cfg.exclude_path, source_filepath_column)
        source_embeddings = extract_column_matrix(df_source, source_embedding_column)
        target_embeddings = extract_column_matrix(df_target, target_embedding_column)

        retrieved_filepaths = iterative_tmm_retrieval(
            target_embeddings, source_embeddings, df_source, df_target,
            desired_unique_count, distance_metric,
            source_filepath_column, target_filepath_column,
            candidate_expansion_factor=cfg.candidate_expansion_factor,
            output_dir=cfg.output_dir, subset_name="global",
            save_embeddings=cfg.save_embeddings,
        )

        final_output_path = output_path / "final_unique_files.parquet"
        pd.DataFrame({source_filepath_column: sorted(retrieved_filepaths)}).to_parquet(final_output_path, index=False)
        coverage = len(retrieved_filepaths) / desired_unique_count * 100 if desired_unique_count else 0.0
        logger.info("Matched %d/%d unique files (%.1f%% coverage) -> %s",
                    len(retrieved_filepaths), desired_unique_count, coverage, final_output_path)

        summary = {
            "allocation_policy": cfg.allocation_policy,
            "desired_unique_count": desired_unique_count,
            "retrieved_unique_count": len(retrieved_filepaths),
            "coverage_pct": round(coverage, 2),
            "target_queries": len(df_target),
            "rare_classes": sorted(rare_classes),
        }
        if cfg.target_detection_file or cfg.source_detection_file:
            fmt = _resolve_detection_format(cfg)
            target_filepaths = set(df_target[target_filepath_column].to_pandas())
            target_stats = calculate_distribution_stats(
                target_filepaths, cfg.target_detection_file, rare_classes, target_filepath_column, fmt=fmt
            )
            resultant_stats = calculate_distribution_stats(
                retrieved_filepaths, cfg.source_detection_file, rare_classes, source_filepath_column, fmt=fmt
            )
            summary["target_dataset"] = {
                "total_images": target_stats["total_files"],
                "rare_images": target_stats["rare_count"],
                "common_images": target_stats["common_count"],
                "rare_instances": target_stats["rare_instance_count"],
                "per_class": calculate_per_class_counts(target_filepaths, cfg.target_detection_file, fmt=fmt),
            }
            summary["resultant_dataset"] = {
                "total_images": resultant_stats["total_files"],
                "rare_images": resultant_stats["rare_count"],
                "common_images": resultant_stats["common_count"],
                "rare_instances": resultant_stats["rare_instance_count"],
                "per_class": calculate_per_class_counts(retrieved_filepaths, cfg.source_detection_file, fmt=fmt),
            }
        summary_path = output_path / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary saved to: %s", summary_path)

        if cfg.visualize and "target_dataset" in summary and "resultant_dataset" in summary:
            plot_per_class_counts(
                summary["target_dataset"]["per_class"], summary["resultant_dataset"]["per_class"], output_path
            )

    # ── class_stratified policy ────────────────────────────────────────────────

    def _run_class_stratified(self, cfg: UniqueNeighborMatchingConfig, rare_classes: set[str]) -> None:
        """Per-class matching with largest-remainder quota allocation and a non-rare fallback pool;
        each source is owned by the first class that claims it (later classes exclude what's
        already matched).
        """
        logger.info("Class-stratified iterative TMM matching; desired unique count: %d | rare classes: %s",
                    cfg.desired_unique_count, sorted(rare_classes))

        fmt = _resolve_detection_format(cfg)
        df_source, df_target = load_datasets(
            cfg.source_path, cfg.target_path, cfg.exclude_path, cfg.source_filepath_column,
        )
        class_subsets = split_datasets_by_class(
            df_source, df_target, cfg.source_detection_file, cfg.target_detection_file, rare_classes,
            cfg.source_filepath_column, cfg.target_filepath_column, fmt=fmt,
        )

        total_targets = len(df_target)
        common_source, common_target = class_subsets.get("non_rare", (cudf.DataFrame(), cudf.DataFrame()))
        class_ratios = {
            cn: (len(ct) / total_targets if total_targets else 0.0)
            for cn, (_, ct) in class_subsets.items() if cn != "non_rare"
        }
        common_ratio = len(common_target) / total_targets if total_targets else 0.0
        quotas = _allocate_quotas({**class_ratios, "non_rare": common_ratio}, cfg.desired_unique_count)
        common_source_embeddings = (
            extract_column_matrix(common_source, cfg.source_embedding_column) if len(common_source) > 0 else None
        )

        # Ownership: each pass excludes everything matched so far, so a source is claimed by the
        # first class that matches it (the non-rare pool matches last).
        all_retrieved: set = set()
        for class_name in sorted(rare_classes):
            if class_name not in class_subsets:
                continue
            all_retrieved |= self._retrieve_for_class(
                cfg, class_name, class_subsets[class_name], quotas.get(class_name, 0),
                common_source, common_source_embeddings, all_retrieved,
            )
        all_retrieved |= self._retrieve_non_rare(
            cfg, common_source, common_source_embeddings, common_target,
            quotas.get("non_rare", 0), all_retrieved,
        )

        self._write_class_stratified_outputs(
            cfg, rare_classes, df_target, class_subsets, class_ratios, common_ratio,
            common_target, quotas, all_retrieved,
        )

    def _retrieve_for_class(
        self,
        cfg: UniqueNeighborMatchingConfig,
        class_name: str,
        subset: tuple[cudf.DataFrame, cudf.DataFrame],
        quota: int,
        common_source: cudf.DataFrame,
        common_source_embeddings: Optional[cupy.ndarray],
        already_retrieved: set[str],
    ) -> set[str]:
        """Match up to ``quota`` sources for one rare class: its own class pool first, then a
        non-rare fallback when the class pool comes up short.

        Both passes exclude ``already_retrieved`` so a source is never reused across classes.
        """
        class_source, class_target = subset
        if not len(class_target) or quota <= 0:
            return set()
        class_target_embeddings = extract_column_matrix(class_target, cfg.target_embedding_column)

        retrieved: set = set()
        if len(class_source) > 0:
            logger.info("[%s] quota=%d; searching class pool (%d images)", class_name, quota, len(class_source))
            retrieved = iterative_tmm_retrieval(
                class_target_embeddings, extract_column_matrix(class_source, cfg.source_embedding_column),
                class_source, class_target, quota, cfg.distance_metric,
                cfg.source_filepath_column, cfg.target_filepath_column,
                candidate_expansion_factor=cfg.candidate_expansion_factor,
                excluded_filepaths=already_retrieved,
                output_dir=cfg.output_dir, subset_name=class_name, save_embeddings=cfg.save_embeddings,
            )

        if len(retrieved) < quota and common_source_embeddings is not None:
            logger.info("[%s] class pool gave %d/%d; refilling from non-rare pool", class_name, len(retrieved), quota)
            retrieved |= iterative_tmm_retrieval(
                class_target_embeddings, common_source_embeddings, common_source, class_target,
                quota - len(retrieved), cfg.distance_metric,
                cfg.source_filepath_column, cfg.target_filepath_column,
                candidate_expansion_factor=cfg.candidate_expansion_factor,
                excluded_filepaths=already_retrieved | retrieved,
                output_dir=cfg.output_dir, subset_name=f"{class_name}_fallback",
                save_embeddings=cfg.save_embeddings,
            )
        return retrieved

    def _retrieve_non_rare(
        self,
        cfg: UniqueNeighborMatchingConfig,
        common_source: cudf.DataFrame,
        common_source_embeddings: Optional[cupy.ndarray],
        common_target: cudf.DataFrame,
        quota: int,
        already_retrieved: set[str],
    ) -> set[str]:
        """Match up to ``quota`` sources for the non-rare pool."""
        if len(common_target) == 0 or quota <= 0 or common_source_embeddings is None:
            return set()
        logger.info("[non_rare] quota=%d; searching non-rare pool", quota)
        return iterative_tmm_retrieval(
            extract_column_matrix(common_target, cfg.target_embedding_column),
            common_source_embeddings, common_source, common_target, quota, cfg.distance_metric,
            cfg.source_filepath_column, cfg.target_filepath_column,
            candidate_expansion_factor=cfg.candidate_expansion_factor,
            excluded_filepaths=already_retrieved,
            output_dir=cfg.output_dir, subset_name="non_rare", save_embeddings=cfg.save_embeddings,
        )

    def _write_class_stratified_outputs(
        self,
        cfg: UniqueNeighborMatchingConfig,
        rare_classes: set[str],
        df_target: cudf.DataFrame,
        class_subsets: dict[str, tuple[cudf.DataFrame, cudf.DataFrame]],
        class_ratios: dict[str, float],
        common_ratio: float,
        common_target: cudf.DataFrame,
        quotas: dict[str, int],
        all_retrieved: set[str],
    ) -> None:
        """Write final_unique_files.parquet, summary.json, and optional visualizations."""
        output_path = Path(cfg.output_dir)
        desired = cfg.desired_unique_count
        fmt = _resolve_detection_format(cfg)
        pd.DataFrame({cfg.source_filepath_column: sorted(all_retrieved)}).to_parquet(
            output_path / "final_unique_files.parquet", index=False,
        )
        coverage = round(len(all_retrieved) / desired * 100, 2) if desired else 0.0
        logger.info("Matched %d/%d unique files (%s%% coverage)", len(all_retrieved), desired, coverage)

        target_filepaths = set(df_target[cfg.target_filepath_column].to_pandas())
        target_stats = calculate_distribution_stats(
            target_filepaths, cfg.target_detection_file, rare_classes, cfg.target_filepath_column, fmt=fmt
        )
        resultant_stats = calculate_distribution_stats(
            all_retrieved, cfg.source_detection_file, rare_classes, cfg.source_filepath_column, fmt=fmt
        )
        total_rare_targets = sum(len(class_subsets[c][1]) for c in rare_classes if c in class_subsets)
        subset_name_for_class = {**{c: c for c in rare_classes}, "non_rare": "non_rare"}

        summary = {
            "allocation_policy": cfg.allocation_policy,
            "desired_unique_count": desired,
            "retrieved_unique_count": len(all_retrieved),
            "coverage_pct": coverage,
            "rare_classes": sorted(rare_classes),
            "target_queries": {"rare": total_rare_targets, "non_rare": len(common_target)},
            "allocation": {
                "per_class": {
                    cn: {
                        "target_count": len(class_subsets[cn][1]) if cn in class_subsets else 0,
                        "ratio_pct": round(class_ratios.get(cn, 0.0) * 100, 2),
                        "desired_count": quotas.get(cn, 0),
                    }
                    for cn in sorted(class_ratios)
                },
                "non_rare": {
                    "target_count": len(common_target),
                    "ratio_pct": round(common_ratio * 100, 2),
                    "desired_count": quotas.get("non_rare", 0),
                },
                "total_desired": sum(quotas.values()),
            },
            "target_dataset": {
                "total_images": target_stats["total_files"],
                "rare_images": target_stats["rare_count"],
                "common_images": target_stats["common_count"],
                "rare_instances": target_stats["rare_instance_count"],
                "per_class": calculate_per_class_counts(target_filepaths, cfg.target_detection_file, fmt=fmt),
            },
            "resultant_dataset": {
                "total_images": resultant_stats["total_files"],
                "rare_images": resultant_stats["rare_count"],
                "common_images": resultant_stats["common_count"],
                "rare_instances": resultant_stats["rare_instance_count"],
                "per_class": calculate_per_class_counts(all_retrieved, cfg.source_detection_file, fmt=fmt),
            },
            "distance_stats": {
                "metric": cfg.distance_metric,
                "based_on": "top1_distance",
                "per_class": calculate_distance_stats_per_class(
                    output_dir=cfg.output_dir,
                    subset_name_for_class=subset_name_for_class,
                ),
            },
        }
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary saved to: %s", output_path / "summary.json")

        if cfg.visualize:
            plot_per_class_counts(
                summary["target_dataset"]["per_class"], summary["resultant_dataset"]["per_class"], output_path
            )
            visualize_target_to_mined(
                output_dir=cfg.output_dir, rare_classes=rare_classes,
                source_detection_file=cfg.source_detection_file,
                target_detection_file=cfg.target_detection_file,
                subset_name_for_class=subset_name_for_class, fmt=fmt,
            )


@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="unique_neighbor_matching",
    schema=UniqueNeighborMatchingConfig,
)
def main(cfg: UniqueNeighborMatchingConfig) -> None:
    """Entrypoint: delegates to UniqueNeighborMatchingMining.run()."""
    UniqueNeighborMatchingMining().run(cfg)


if __name__ == "__main__":
    main()
