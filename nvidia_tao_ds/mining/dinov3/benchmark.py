# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed identity checks for a declared DINOv3 held-out benchmark."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Reported by preflight; raise it when the isolation guarantee changes.
ISOLATION_CONTRACT = 1


class BenchmarkGuard:
    """Match sample IDs, acquisition units and optional content SHA-256 values."""

    def __init__(self, path: str | Path, acquisition_unit_column: str):
        """Read the identity sidecar, never the evaluator-specific manifest."""
        frame = pd.read_parquet(path, pre_buffer=False)
        if frame.empty:
            raise ValueError("Benchmark identity manifest must not be empty")
        self.columns = list(dict.fromkeys(["sample_id", acquisition_unit_column]))
        if "content_sha256" in frame:
            # Declaring hashes makes them mandatory on every checked row.
            self.columns.append("content_sha256")
        self.identities = {
            column: set(self._values(frame, column, "Benchmark sidecar"))
            for column in self.columns
        }

    @staticmethod
    def _values(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
        """Return exact, case-sensitive identities with surrounding whitespace removed."""
        if column not in frame:
            raise ValueError(f"{label} lacks benchmark identity column {column!r}")
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{label} has null or empty benchmark identity {column!r}")
        if isinstance(values.dtype, pd.CategoricalDtype):
            # Parquet keeps dictionary encoding; check the values, not the codes.
            values = values.astype(values.cat.categories.dtype)
        # Floats, bytes and booleans have no stable text form to compare against.
        if pd.api.types.infer_dtype(values) not in {"string", "integer", "empty"}:
            raise ValueError(f"{label} identity {column!r} must be string or integer")
        values = values.astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"{label} has null or empty benchmark identity {column!r}")
        if column == "content_sha256":
            values = values.str.lower().str.removeprefix("sha256:")
            if not values.str.fullmatch(r"[0-9a-f]{64}").all():
                raise ValueError(f"{label} has invalid content_sha256 values")
        return values

    def matches(self, frame: pd.DataFrame, *, label: str) -> pd.Series:
        """Return an OR across all declared identities; missing metadata fails."""
        overlap = pd.Series(False, index=frame.index)
        if len(frame):
            for column in self.columns:
                overlap |= self._values(frame, column, label).isin(self.identities[column])
        return overlap

    def reject(self, frame: pd.DataFrame, *, label: str) -> None:
        """Reject an intersection before publishing or using training data."""
        overlap = self.matches(frame, label=label)
        if overlap.any():
            examples = frame.loc[overlap, "sample_id"].astype(str).head(10).tolist()
            raise ValueError(
                f"{label} overlaps the sealed benchmark: "
                f"count={int(overlap.sum())}, examples={examples}"
            )

    def reject_parquet(self, path: str | Path, *, label: str) -> None:
        """Check identity columns in bounded batches, without loading embeddings."""
        parquet = pq.ParquetFile(path)
        columns = [name for name in self.columns if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=65536, columns=columns):
            self.reject(batch.to_pandas(ignore_metadata=True), label=label)
