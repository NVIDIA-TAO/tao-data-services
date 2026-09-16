# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish immutable cumulative training manifests without copying images."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from .contracts import (
    ArtifactManifest,
    file_identity,
    file_sha256,
    require_uncommitted_output,
)


LOCATOR_COLUMNS = {"storage_type", "path"}


def _normalize_sample_ids(frame: pd.DataFrame, *, id_column: str, label: str) -> None:
    """Validate identities before converting them to the canonical string type."""
    if id_column not in frame:
        raise ValueError(f"{label} has no {id_column!r} column")
    values = frame[id_column]
    if values.isnull().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains null or empty sample IDs")
    frame[id_column] = values.astype(str)
    if frame[id_column].duplicated().any():
        raise ValueError(f"{label} contains duplicate sample IDs")


def _attach_query_provenance(
    delta: pd.DataFrame,
    *,
    query_path: str | Path | None,
    balance_column: str | None,
    id_column: str,
) -> pd.DataFrame:
    """Attach the task that caused each neighbor to enter the training set."""
    if query_path is None and balance_column is None:
        return delta
    if query_path is None or balance_column is None:
        raise ValueError("query_path and balance_column must be configured together")
    if "query_id" not in delta:
        raise ValueError("Balanced materialization requires delta.query_id")
    queries = pd.read_parquet(query_path, pre_buffer=False)
    required = {id_column, "task"}
    if missing := required.difference(queries.columns):
        raise ValueError(f"Query manifest is missing columns: {sorted(missing)}")
    if queries[id_column].astype(str).duplicated().any():
        raise ValueError("Query manifest contains duplicate sample IDs")
    mapping = queries[[id_column, "task"]].copy()
    mapping[id_column] = mapping[id_column].astype(str)
    mapping["task"] = mapping["task"].astype(str)
    mapping = mapping.rename(
        columns={id_column: "query_id", "task": balance_column}
    )
    enriched = delta.copy()
    enriched["query_id"] = enriched["query_id"].astype(str)
    if balance_column in enriched:
        existing = enriched[balance_column].astype(str)
        expected = enriched["query_id"].map(mapping.set_index("query_id")[balance_column])
        if expected.isnull().any() or not existing.equals(expected.astype(str)):
            raise ValueError("Delta task provenance conflicts with query manifest")
        return enriched
    enriched = enriched.merge(mapping, on="query_id", how="left", validate="many_to_one")
    if enriched[balance_column].isnull().any():
        raise ValueError("Some delta rows have no task provenance")
    return enriched


def _balanced_training_view(
    cumulative: pd.DataFrame, *, balance_column: str, id_column: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repeat minority provenance groups while retaining every unique sample."""
    if balance_column not in cumulative:
        raise ValueError(f"Training manifest has no {balance_column!r} column")
    if cumulative[balance_column].isnull().any():
        raise ValueError("Balanced training provenance contains null values")
    tasks = cumulative[balance_column].astype(str)
    groups = {
        task: cumulative.loc[tasks == task].sort_values(
            id_column, kind="mergesort"
        )
        for task in sorted(tasks.unique())
    }
    if not groups:
        empty = cumulative.copy()
        empty["replay_repeat"] = pd.Series(dtype="int64")
        return empty, {
            "column": balance_column,
            "policy": "oversample_each_task_to_largest_group_v1",
            "source_rows_by_task": {},
            "training_rows_by_task": {},
        }
    target_rows = max(len(group) for group in groups.values())
    parts = []
    for task, group in groups.items():
        original = group.copy()
        original["replay_repeat"] = 0
        parts.append(original)
        missing = target_rows - len(group)
        if missing:
            indexes = [index % len(group) for index in range(missing)]
            repeated = group.iloc[indexes].copy()
            repeated["replay_repeat"] = [
                1 + index // len(group) for index in range(missing)
            ]
            parts.append(repeated)
    view = pd.concat(parts, ignore_index=True, sort=False)
    return view, {
        "column": balance_column,
        "policy": "oversample_each_task_to_largest_group_v1",
        "source_rows_by_task": {
            task: int(len(group)) for task, group in groups.items()
        },
        "training_rows_by_task": {task: int(target_rows) for task in groups},
    }


def _canonicalize_legacy_file_locators(
    frame: pd.DataFrame, *, label: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve an omitted storage type only for unambiguous local paths."""
    if "storage_type" in frame or frame.empty or "path" not in frame:
        return frame, {}
    if frame["path"].isnull().any():
        return frame, {}

    def is_local(value: Any) -> bool:
        text = str(value)
        parsed = urlparse(text)
        if parsed.scheme:
            return parsed.scheme == "file" and Path(parsed.path).is_absolute()
        return Path(text).is_absolute()

    if not frame["path"].map(is_local).all():
        raise ValueError(
            f"{label} has no storage_type and contains non-absolute paths"
        )
    normalized = frame.copy()
    normalized["storage_type"] = "file"
    return normalized, {
        "storage_type": {
            "value": "file",
            "rows": int(len(normalized)),
            "rule": "absolute_local_path_v1",
        }
    }


def validate_locator_frame(frame: pd.DataFrame, *, label: str) -> None:
    """Validate the canonical, train-readable image locator contract."""
    missing = LOCATOR_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing locator columns: {sorted(missing)}")
    if frame[list(LOCATOR_COLUMNS)].isnull().any().any():
        raise ValueError(f"{label} locator columns contain null values")
    storage_types = frame["storage_type"].astype(str)
    unsupported = set(storage_types).difference({"file", "tar", "zip"})
    if unsupported:
        raise ValueError(f"{label} has unsupported storage types: {sorted(unsupported)}")
    archive_rows = storage_types.isin({"tar", "zip"})
    if archive_rows.any() and (
        "member" not in frame or frame.loc[archive_rows, "member"].isnull().any()
    ):
        raise ValueError(f"{label} archive rows require member")
    invalid_paths = []
    for value in frame["path"]:
        text = str(value)
        parsed = urlparse(text)
        if parsed.scheme:
            valid = all(
                (
                    parsed.scheme == "file",
                    parsed.netloc in {"", "localhost"},
                    not parsed.params,
                    not parsed.query,
                    not parsed.fragment,
                    Path(unquote(parsed.path)).is_absolute(),
                )
            )
        else:
            valid = Path(text).is_absolute()
        if not valid:
            invalid_paths.append(text)
    if invalid_paths:
        raise ValueError(
            f"{label} contains non-absolute local paths: {invalid_paths[:3]}"
        )


def materialize_manifest(
    *,
    delta_path: str | Path,
    output_dir: str | Path,
    previous_path: str | Path | None = None,
    query_path: str | Path | None = None,
    balance_column: str | None = None,
    id_column: str = "sample_id",
    overlap_policy: str = "reject",
) -> dict:
    """Merge a novel delta into an immutable, exactly deduplicated manifest."""
    if overlap_policy not in {"reject", "drop_existing"}:
        raise ValueError("overlap_policy must be 'reject' or 'drop_existing'")
    delta = pd.read_parquet(delta_path, pre_buffer=False)
    _normalize_sample_ids(delta, id_column=id_column, label="Delta manifest")
    if delta.empty:
        # A valid empty delta is a first-class no-op. Keep a train-readable
        # typed schema even on the first round, without requiring fake rows.
        for name in ("storage_type", "path", "member"):
            if name not in delta:
                delta[name] = pd.Series(dtype="string")
        if balance_column is not None and balance_column not in delta:
            delta[balance_column] = pd.Series(dtype="string")
    else:
        delta = _attach_query_provenance(
            delta,
            query_path=query_path,
            balance_column=balance_column,
            id_column=id_column,
        )
    delta, locator_inference = _canonicalize_legacy_file_locators(
        delta, label="Delta manifest"
    )
    validate_locator_frame(delta, label="Delta manifest")

    if previous_path:
        previous = pd.read_parquet(previous_path, pre_buffer=False)
        _normalize_sample_ids(previous, id_column=id_column, label="Previous manifest")
        validate_locator_frame(previous, label="Previous manifest")
        overlap = set(previous[id_column]).intersection(delta[id_column])
        if overlap and overlap_policy == "reject":
            raise ValueError(f"Delta reselects {len(overlap)} existing sample IDs")
        if overlap:
            delta = delta.loc[~delta[id_column].isin(overlap)].reset_index(drop=True)
        cumulative = pd.concat([previous, delta], ignore_index=True, sort=False)
    else:
        previous = None
        cumulative = delta
        overlap = set()

    if cumulative[id_column].duplicated().any():
        raise RuntimeError("Cumulative manifest contains duplicate sample IDs")

    destination = require_uncommitted_output(output_dir)
    manifest_path = destination / "training_manifest.parquet"
    with tempfile.NamedTemporaryFile(
        dir=destination,
        prefix=".training_manifest.",
        suffix=".tmp.parquet",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    cumulative.to_parquet(temporary, index=False)
    file_publications = [(temporary, manifest_path)]
    training_view = None
    balance = None
    view_temporary = None
    if balance_column is not None:
        training_view, balance = _balanced_training_view(
            cumulative, balance_column=balance_column, id_column=id_column
        )
        view_path = destination / "balanced_training_manifest.parquet"
        with tempfile.NamedTemporaryFile(
            dir=destination,
            prefix=".balanced_training_manifest.",
            suffix=".tmp.parquet",
            delete=False,
        ) as stream:
            view_temporary = Path(stream.name)
        training_view.to_parquet(view_temporary, index=False)
        file_publications.append((view_temporary, view_path))
    payload = {
        "manifest_uri": manifest_path.resolve().as_uri(),
        "row_count": int(len(cumulative)),
        "delta_rows": int(len(delta)),
        "replayed_rows": int(len(overlap)),
        "previous_rows": 0 if previous is None else int(len(previous)),
        "id_column": id_column,
        "locator_schema": "storage_type_path_member_v1",
        "locator_inference": locator_inference,
        "manifest_sha256": file_identity(temporary)["sha256"],
    }
    if training_view is not None:
        training_view_identity = file_identity(view_temporary)
        training_view_identity["uri"] = view_path.resolve().as_uri()
        payload.update(
            {
                "training_view": training_view_identity,
                "training_view_rows": int(len(training_view)),
                "balance": balance,
            }
        )
    artifact = ArtifactManifest(
        artifact_type="training_manifest",
        producer={
            "action": "materialize_manifest",
            "version": "1.0",
            "implementation_sha256": file_sha256(Path(__file__)),
        },
        inputs=[
            file_identity(delta_path, role="delta"),
            *(
                [file_identity(previous_path, role="previous")]
                if previous_path
                else []
            ),
            *(
                [file_identity(query_path, role="query_manifest")]
                if query_path
                else []
            ),
        ],
        payload=payload,
    )
    try:
        artifact.commit(destination, file_publications=file_publications)
    finally:
        temporary.unlink(missing_ok=True)
        if view_temporary is not None:
            view_temporary.unlink(missing_ok=True)
    return artifact.to_dict()
