# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share stable file digests within one DINOv3 workflow execution."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
from time import time_ns


_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
_HASH_CACHE = ContextVar("dinov3_hash_cache", default=None)
# POSIX metadata may have coarser effective resolution than its ns interface.
# Never memoize files within the current timestamp tick (up to one second),
# with an additional second of margin. Future timestamps also bypass caching.
_SETTLE_NS = 2_000_000_000


@contextmanager
def snapshot_cache():
    """Reuse unchanged file digests locally, never across execute/resume calls."""
    token = _HASH_CACHE.set({})
    try:
        yield
    finally:
        _HASH_CACHE.reset(token)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(int(getattr(value, name)) for name in _STAT_FIELDS)


def _settled(value: os.stat_result) -> bool:
    return time_ns() - max(value.st_mtime_ns, value.st_ctime_ns) >= _SETTLE_NS


def _read_digest(stream, *, capture_bytes: bool = False) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    chunks = [] if capture_bytes else None
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
    return "sha256:" + digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def stable_file_snapshot(
    path: str | Path, *, capture_bytes: bool = False
) -> tuple[Path, os.stat_result, bytes | None, str]:
    """Read a stable descriptor snapshot or reuse its unchanged local digest."""
    resolved = Path(path).expanduser().resolve()
    cache = _HASH_CACHE.get()
    stat = resolved.stat()
    key = (str(resolved), _stat_identity(stat))
    if not capture_bytes and cache is not None and key in cache and _settled(stat):
        return resolved, stat, None, cache[key]
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        settled_before = _settled(before)
        checksum, raw = _read_digest(stream, capture_bytes=capture_bytes)
        after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError(f"Input changed while it was read: {resolved}")
        if not settled_before or not _settled(after):
            # Same-size writes can leave identical metadata within one coarse
            # timestamp tick. Verify recent files with a second content read.
            stream.seek(0)
            verified_checksum, _ = _read_digest(stream)
            if verified_checksum != checksum:
                raise ValueError(f"Input changed while it was read: {resolved}")
            after = os.fstat(stream.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"Input changed while it was read: {resolved}")
    if _stat_identity(resolved.stat()) != _stat_identity(after):
        raise ValueError(f"Input path changed while it was read: {resolved}")
    if raw is not None and len(raw) != after.st_size:
        raise ValueError(f"Input size changed while it was read: {resolved}")
    if cache is not None and settled_before and _settled(after):
        cache[(str(resolved), _stat_identity(after))] = checksum
    return resolved, after, raw, checksum


def file_sha256(path: str | Path) -> str:
    """Hash a workflow input using the current execution's stable-file cache."""
    return stable_file_snapshot(path)[3]
