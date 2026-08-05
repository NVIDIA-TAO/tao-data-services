# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector preparation for k-NN: list-column reshape, embedding normalization, and conversion helpers."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

# cupy is imported eagerly (used at runtime): this is a single-process module and the base
# image always ships RAPIDS, so there's no CPU-only import path to keep light — unlike the
# multi-GPU thresholded workers, which must set CUDA_VISIBLE_DEVICES before importing RAPIDS.
# cudf/numpy below are referenced only in type hints.
import cupy

if TYPE_CHECKING:
    import cudf
    import numpy as np


def l2_normalize(x: cupy.ndarray, eps: float = 1e-12) -> cupy.ndarray:
    """L2 normalize embeddings on GPU."""
    norms = cupy.maximum(cupy.linalg.norm(x, axis=1, keepdims=True), eps)
    return x / norms


def _embeddings_to_lists(embeddings: cupy.ndarray | np.ndarray) -> list[Any]:
    """Convert an embedding array (cupy or numpy, 1-D or 2-D) to Python lists."""
    if isinstance(embeddings, cupy.ndarray):
        return embeddings.get().tolist()
    return embeddings.tolist() if hasattr(embeddings, "tolist") else [list(e) for e in embeddings]


def extract_column_matrix(df: cudf.DataFrame, column: str) -> cupy.ndarray:
    """Reshape a cuDF list-typed column into a dense 2-D array (e.g. embeddings).

    cuDF stores per-row equal-width vectors (e.g. embeddings) as a list column,
    but downstream GPU code (e.g. cuML) expects a dense ``(n_rows, width)`` array.
    ``.list.leaves`` flattens every list into one contiguous buffer and ``.values``
    exposes it as a CuPy array, reshaped to one row per dataframe row.

    A module-level free function, so it stays picklable for multiprocessing workers.

    Args:
        df: A cuDF DataFrame.
        column: Name of a list-typed column whose rows are equal-width vectors (e.g. embeddings).

    Returns:
        A 2-D array of shape ``(len(df), width)``.
    """
    return df[column].list.leaves.values.reshape(len(df), -1)
