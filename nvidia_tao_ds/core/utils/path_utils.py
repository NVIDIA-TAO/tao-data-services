# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str) -> None:
    """Create ``path`` (and any parents) if it does not already exist; a no-op for an empty/None path."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
