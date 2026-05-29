# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-accelerated k-NN mining for SDA data augmentation.

Given SigLIP embeddings of *source* and *target* images, finds the
``topn`` nearest source neighbours for each target query.  When
``--filter-by-label`` is enabled and both parquets carry a ``label``
column, source-target pairs with mismatched labels are dropped so
only same-class samples are mined.

Outputs:
    ``cfg.output_parquet``: a parquet with a single ``filepath``
        column listing the unique mined source images (target × topn
        candidates, deduplicated, with cross-label pairs removed
        when label filtering is active).
    ``mining_summary.txt`` (next to ``cfg.output_parquet``):
        plaintext run summary — target query count, topn, total
        candidates, duplicates removed, unique items saved, and
        (when label filtering ran) kept-vs-dropped pair counts.
"""

import logging
from os import getenv
from pathlib import Path
from typing import Final

import cudf
import cuml.neighbors
import pandas as pd

from nvidia_tao_ds.config.mining.tmm.nearest_neighbors import NearestNeighborsConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner

logger = logging.getLogger(__name__)

MINING_SUMMARY_TXT: Final[str] = "mining_summary.txt"

spec_root = Path(__file__).resolve().parent


@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="nearest_neighbors",
    schema=NearestNeighborsConfig
)
def main(cfg: NearestNeighborsConfig):
    """Find the top-N most similar source images for each target query."""
    _log_level = getattr(logging, getenv("TAO_LOGGING_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    source_parquet = cfg.source_parquet
    target_parquet = cfg.target_parquet
    output_parquet = cfg.output_parquet
    topn = cfg.topn
    knn_metric = cfg.knn_metric
    source_embed_column_name = cfg.source_embed_column_name
    target_embed_column_name = cfg.target_embed_column_name
    filter_by_label = cfg.filter_by_label

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Load embeddings onto GPU via cuDF.
    logger.info("Loading target data from: %s", target_parquet)
    df_target = cudf.read_parquet(target_parquet)
    logger.info("Target data shape: %s", df_target.shape)

    logger.info("Loading source data from: %s", source_parquet)
    df_source = cudf.read_parquet(source_parquet)
    logger.info("Source data shape: %s", df_source.shape)

    # cuDF stores variable-length lists; reshape into dense 2-D
    # arrays for cuML.
    source_embeddings = df_source[source_embed_column_name].list.leaves.values.reshape(
        len(df_source), -1
    )
    target_embeddings = df_target[target_embed_column_name].list.leaves.values.reshape(
        len(df_target), -1
    )
    logger.info("Source embeddings shape: %s", source_embeddings.shape)
    logger.info("Target embeddings shape: %s", target_embeddings.shape)

    # -- GPU k-NN: fit on source, query with targets.
    logger.info("Performing k-NN search (metric=%s, k=%d)...", knn_metric, topn)
    knn = cuml.neighbors.NearestNeighbors(n_neighbors=topn, metric=knn_metric)
    knn.fit(source_embeddings)
    distances, indices = knn.kneighbors(target_embeddings)
    logger.info("k-NN search completed!")

    # -- Collect mined filepaths, optionally filtering label mismatches.
    # When filter_by_label is enabled and both parquets carry a label
    # column, drop source-target pairs where the labels disagree so
    # only same-class samples are mined.
    source_filepaths = df_source['filepath']
    label_filter_requested = str(filter_by_label).lower() in ("true", "1", "yes")
    do_label_filter = (
        label_filter_requested
        and 'label' in df_source.columns
        and 'label' in df_target.columns
    )

    # Surface the case where the user asked for label filtering but
    # one or both parquets are missing the column — otherwise the
    # filter silently no-ops and the mined output looks fine but
    # contains cross-label pairs.
    if label_filter_requested and not do_label_filter:
        missing = [
            role for role, df in (("source", df_source), ("target", df_target))
            if 'label' not in df.columns
        ]
        logger.warning(
            "filter_by_label is enabled but the 'label' column is missing "
            "from: %s. Skipping label filtering — mined output will include "
            "cross-label pairs.",
            ", ".join(missing),
        )

    if do_label_filter:
        source_labels = df_source['label'].to_pandas()
        target_labels = df_target['label'].to_pandas()
        logger.info("Label columns found — filtering cross-label mismatches")

    all_neighbor_filepaths = []
    label_mismatch_count = 0
    for i in range(len(target_embeddings)):
        for j in range(topn):
            neighbor_idx = indices[i, j]
            if do_label_filter:
                src_label = str(source_labels.iloc[int(neighbor_idx)]).strip().upper()
                tgt_label = str(target_labels.iloc[i]).strip().upper()
                if src_label != tgt_label:
                    label_mismatch_count += 1
                    continue
            all_neighbor_filepaths.append(source_filepaths[neighbor_idx])

    label_filter_line = None
    if do_label_filter:
        total_pairs = len(target_embeddings) * topn
        label_filter_line = (
            f"Label filtering: kept {len(all_neighbor_filepaths)}/{total_pairs} pairs, "
            f"dropped {label_mismatch_count} mismatches"
        )
        logger.info(label_filter_line)

    # TODO: in the future, we can have other forms of filtering here
    # like embeddings distances, talk to @jkalra

    logger.info("Saving results...")
    output_df = pd.DataFrame({"filepath": all_neighbor_filepaths})
    total_before = len(output_df)
    output_df = output_df.drop_duplicates(subset=["filepath"])
    total_after = len(output_df)
    output_df.to_parquet(output_path, index=False)

    summary_lines = [
        "Summary:",
        f"  Target queries: {len(df_target)}",
        f"  Similar items per query: {topn}",
        f"  Total candidates: {total_before}",
        f"  Duplicates removed: {total_before - total_after}",
        f"  Unique items saved: {total_after}",
        f"  Results saved to: {output_parquet}",
    ]
    if label_filter_line is not None:
        summary_lines.append(label_filter_line)
        summary_lines.append("")
    summary_text = "\n".join(summary_lines) + "\n"

    logger.info("\n%s", summary_text.strip())

    summary_path = output_path.parent / MINING_SUMMARY_TXT
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Summary written to: %s", summary_path)

if __name__ == "__main__":
    main()
