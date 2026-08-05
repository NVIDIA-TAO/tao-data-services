# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet readers for TMM mining: load source/target datasets and apply exclude filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple

import cudf
import pandas as pd

from nvidia_tao_ds.core.logging.logging import logging as logger


def load_parquet_data(parquet_path: str) -> "cudf.DataFrame":
    """Load parquet data from a single ``.parquet`` file or a directory of parquet files.

    A directory reads and concatenates every ``*.parquet`` file in sorted filename order; all
    files must share the same schema or the concatenation raises (a missing/extra column across
    shards is a common cause).

    Args:
        parquet_path: Path to a ``.parquet`` file or a directory containing parquet files.

    Returns:
        A cuDF DataFrame with the loaded (and, for directories, concatenated) rows.

    Raises:
        FileNotFoundError: If the path does not exist, or a directory has no parquet files.
        ValueError: If the path is a file without a ``.parquet`` suffix, is neither file nor dir,
            or directory shards have mismatched schemas. (cuDF raises directly for an
            unreadable/corrupt parquet file.)
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
    if path.is_file():
        if not path.suffix == '.parquet':
            raise ValueError(f"Expected a .parquet file, got: {parquet_path}")
        return cudf.read_parquet(str(path))
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        frames = [cudf.read_parquet(str(pf)) for pf in parquet_files]
        try:
            return cudf.concat(frames, ignore_index=True)
        except Exception as exc:
            raise ValueError(
                f"Failed to concatenate parquet files in {parquet_path} "
                f"(do all files share the same schema?): {exc}"
            ) from exc
    raise ValueError(f"Path is neither a file nor a directory: {parquet_path}")


def build_exclude_set(exclude_parquet_path: str) -> Set[str]:
    """Build a set of files to exclude from a parquet, expanding each entry to path and basename.

    Every value in the 'filepath' column contributes two keys (the value itself and its
    basename), kept together in one set so a single membership test excludes by either full
    path or filename.

    Args:
        exclude_parquet_path: Path to a parquet file containing a 'filepath' column.

    Returns:
        A set of strings holding both the full filepaths and their basenames.

    Raises:
        ValueError: If the parquet has no 'filepath' column.
    """
    df = pd.read_parquet(exclude_parquet_path)
    if "filepath" not in df.columns:
        raise ValueError(f"Exclude parquet must have 'filepath' column. Columns: {list(df.columns)}")
    files_to_exclude: Set[str] = set()
    for v in df["filepath"].astype(str).tolist():
        files_to_exclude.add(v)
        files_to_exclude.add(Path(v).name)
    return files_to_exclude


def filter_source_by_filepath(
    df_source: "cudf.DataFrame",
    files_to_exclude: Set[str],
    filepath_column_name: str,
) -> "cudf.DataFrame":
    """Drop source rows whose filepath or basename is in ``files_to_exclude``."""
    # Only the filepath column crosses GPU->CPU (negligible next to the embeddings); pandas does
    # the basename mapping + membership test, then the boolean mask is pushed back to cuDF.
    paths_pd = df_source[filepath_column_name].to_pandas()
    keep = ~paths_pd.isin(files_to_exclude) & ~paths_pd.map(lambda x: Path(x).name).isin(files_to_exclude)
    return df_source[cudf.from_pandas(keep)].reset_index(drop=True)


def load_datasets(
    source_parquet: str, target_parquet: str,
    exclude_parquet: Optional[str], source_filepath_column_name: str,
) -> Tuple["cudf.DataFrame", "cudf.DataFrame"]:
    """Load source and target parquet datasets, filtering the source by an exclude list if given.

    Args:
        source_parquet: Path to a source parquet file or a directory of parquet files.
        target_parquet: Path to a target parquet file or a directory of parquet files.
        exclude_parquet: Optional path to a parquet with a 'filepath' column; matching source
            rows (by full path or basename) are dropped. If given, the file must exist.
        source_filepath_column_name: Name of the filepath column in the source dataset, used for
            exclude matching.

    Returns:
        A ``(df_source, df_target)`` tuple of cuDF DataFrames, with the exclude filter already
        applied to ``df_source``.

    Raises:
        FileNotFoundError: If ``exclude_parquet`` is given but does not exist.
    """
    logger.info(f"Loading target data from: {target_parquet}")
    df_target = load_parquet_data(target_parquet)
    logger.info(f"Target data shape: {df_target.shape}")

    logger.info(f"Loading source data from: {source_parquet}")
    df_source = load_parquet_data(source_parquet)
    logger.info(f"Source data shape: {df_source.shape}")

    if exclude_parquet:
        if not Path(exclude_parquet).is_file():
            raise FileNotFoundError(f"Exclude parquet not found: {exclude_parquet}")
        files_to_exclude = build_exclude_set(exclude_parquet)
        n_before = len(df_source)
        df_source = filter_source_by_filepath(df_source, files_to_exclude, source_filepath_column_name)
        logger.info(
            f"Excluded {n_before - len(df_source)} source rows from "
            f"{exclude_parquet}; source shape: {df_source.shape}"
        )

    return df_source, df_target
