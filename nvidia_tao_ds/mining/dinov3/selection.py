# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic target-selection strategies for refinement workflows."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ("sample_id", "task")


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[list(columns)].isnull().any().any():
        raise ValueError("Required target columns contain null values")


def _midrank_ecdf(values: pd.Series) -> pd.Series:
    """Return tie-aware ordinal ranks in the open interval (0, 1)."""
    return (values.rank(method="average") - 0.5) / len(values)


def allocate_multitask_budgets(
    tasks: list[str],
    *,
    total: int,
    task_weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Allocate a fixed target budget with deterministic largest remainders."""
    ordered = sorted(map(str, tasks))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("tasks must be non-empty and unique")
    if total < len(ordered):
        raise ValueError("total must provide at least one target per task")
    supplied = {
        str(task): float(weight) for task, weight in (task_weights or {}).items()
    }
    unknown = set(supplied).difference(ordered)
    if unknown:
        raise ValueError(f"task_weights contains unknown tasks: {sorted(unknown)}")
    weights = {task: supplied.get(task, 1.0) for task in ordered}
    if not all(
        math.isfinite(weight) and weight > 0.0 for weight in weights.values()
    ):
        raise ValueError("task weights must be finite and positive")

    weight_sum = sum(weights.values())
    exact = {task: total * weights[task] / weight_sum for task in ordered}
    budgets = {task: int(math.floor(exact[task])) for task in ordered}
    for task in sorted(ordered, key=lambda item: (-(exact[item] % 1.0), item))[
        : total - sum(budgets.values())
    ]:
        budgets[task] += 1

    # A configured task remains represented even under an extreme preference.
    for task in ordered:
        if budgets[task] > 0:
            continue
        donors = [candidate for candidate in ordered if budgets[candidate] > 1]
        if not donors:
            raise ValueError("could not allocate at least one target per task")
        donor = sorted(donors, key=lambda item: (-budgets[item], item))[0]
        budgets[donor] -= 1
        budgets[task] = 1
    return budgets


def select_grit_targets(
    frame: pd.DataFrame,
    *,
    fraction: float,
) -> pd.DataFrame:
    """Select the highest GRIT fraction independently within every domain."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    _require(frame, {*IDENTITY_COLUMNS, "grit_score"})
    scored = frame
    parts = []
    for task, group in scored.groupby("task", sort=True):
        count = int(math.ceil(len(group) * fraction))
        selected = group.sort_values(
            ["grit_score", "sample_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(count).copy()
        selected["target_rank"] = np.arange(1, len(selected) + 1)
        selected["weakness_score"] = selected["grit_score"].astype(float)
        selected["strategy"] = "grit_score"
        selected["task"] = str(task)
        parts.append(selected)
    if not parts:
        empty = scored.head(0).copy()
        empty["target_rank"] = pd.Series(dtype="int64")
        empty["weakness_score"] = pd.Series(dtype="float64")
        empty["strategy"] = pd.Series(dtype="object")
        return empty
    return pd.concat(parts, ignore_index=True, sort=False)


def select_multitask_targets(
    frame: pd.DataFrame,
    *,
    per_task: int | None = None,
    total: int | None = None,
    task_weights: Mapping[str, float] | None = None,
    configured_tasks: list[str] | None = None,
    preserve_unfilled_budget: bool = False,
    score_column: str = "weakness_score",
) -> pd.DataFrame:
    """Select weighted, normalized weakness budgets from each active task.

    A sample may be weak for multiple tasks, but is emitted once. Selection
    advances one pick per active task per cycle, scanning past identities that
    another task already claimed. Task ordering and ties are stable, making
    resumes reproducible without allowing the first task to consume its entire
    budget before the others receive a pick.
    """
    if (per_task is None) == (total is None):
        raise ValueError("configure exactly one of per_task or total")
    if per_task is not None and per_task <= 0:
        raise ValueError("per_task must be positive")
    if total is not None and total <= 0:
        raise ValueError("total must be positive")
    _require(frame, {*IDENTITY_COLUMNS, score_column})
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Multi-task inputs contain duplicate sample/task identities")

    scored = frame.copy()
    scored["normalized_weakness"] = scored.groupby(
        "task", sort=True, group_keys=False
    )[score_column].transform(_midrank_ecdf)
    ranked_by_task = {
        str(task): group.sort_values(
            ["normalized_weakness", score_column, "sample_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for task, group in scored.groupby("task", sort=True)
    }
    if not ranked_by_task:
        empty = scored.head(0).copy()
        empty["target_rank"] = pd.Series(dtype="int64")
        empty["strategy"] = pd.Series(dtype="object")
        return empty
    configured = (
        sorted(map(str, configured_tasks))
        if configured_tasks is not None
        else sorted(ranked_by_task)
    )
    if not configured or len(configured) != len(set(configured)):
        raise ValueError("configured_tasks must be non-empty and unique")
    unknown_tasks = set(ranked_by_task).difference(configured)
    if unknown_tasks:
        raise ValueError(
            f"Scores contain tasks outside configured_tasks: {sorted(unknown_tasks)}"
        )
    weight_contract = configured if configured_tasks is not None else ranked_by_task
    unknown_weights = set(task_weights or {}).difference(weight_contract)
    if unknown_weights:
        raise ValueError(
            f"Task weights contain unknown tasks: {sorted(unknown_weights)}"
        )
    budget_tasks = configured if preserve_unfilled_budget else sorted(ranked_by_task)
    total_budget = (
        int(per_task) * len(budget_tasks) if per_task is not None else int(total)
    )
    active_weights = (
        {task: float(task_weights[task]) for task in budget_tasks if task in task_weights}
        if task_weights
        else None
    )
    budgets = allocate_multitask_budgets(
        budget_tasks, total=total_budget, task_weights=active_weights
    )
    selected_rows = []
    positions = {task: 0 for task in ranked_by_task}
    counts = {task: 0 for task in ranked_by_task}
    used: set[str] = set()
    active = list(ranked_by_task)
    while active:
        next_active = []
        for task in active:
            ranked = ranked_by_task[task]
            position = positions[task]
            while position < len(ranked):
                row = ranked.iloc[position].copy()
                position += 1
                sample_id = str(row["sample_id"])
                if sample_id in used:
                    continue
                counts[task] += 1
                row["target_rank"] = counts[task]
                row["strategy"] = "multi_task_round_robin"
                row["task"] = task
                selected_rows.append(row)
                used.add(sample_id)
                break
            positions[task] = position
            if counts[task] < budgets[task] and position < len(ranked):
                next_active.append(task)
        active = next_active
    if not selected_rows:
        empty = scored.head(0).copy()
        empty["target_rank"] = pd.Series(dtype="int64")
        empty["strategy"] = pd.Series(dtype="object")
        return empty
    return pd.DataFrame(selected_rows).reset_index(drop=True)
