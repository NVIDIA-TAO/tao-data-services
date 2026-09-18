# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check execution-scoped digest reuse without weakening mutation detection."""

import hashlib
import os
from pathlib import Path
import time

import pytest

from nvidia_tao_ds.mining.dinov3.workflow import controller, native_actions, snapshots


def _advance_filesystem_clock(reference: os.stat_result, directory: Path) -> None:
    """Block until the filesystem clock is strictly past ``reference``.

    Linux stamps inodes from a clock whose resolution is the filesystem's
    timestamp granularity.  Kernels without fine-grained timestamps (< 6.13)
    advance that clock only once per timer tick -- 4-10 ms depending on
    ``CONFIG_HZ`` -- so two writes issued inside the same tick are recorded with
    byte-identical ``st_mtime_ns`` and ``st_ctime_ns``.

    The mutation tests below rewrite ``checkpoint`` in place with a replacement
    of exactly the same length, which leaves ``st_dev``, ``st_ino`` and
    ``st_size`` untouched.  When the timestamps also collide, the whole stat
    identity is unchanged and ``snapshots`` correctly reports the file as
    unmodified -- there is no stat field left that could reveal the edit.  That
    made both ``[rewrite]`` cases pass on fine-grained workstations and fail on
    the coarse-grained kernel CI runs on.

    Waiting for the next tick makes the mutation observable without weakening
    what the tests assert.  The ``[replace]`` cases never needed this because
    ``Path.replace`` installs a new inode and therefore changes ``st_ino``.
    """
    probe = directory / ".timestamp-probe"
    floor = max(reference.st_mtime_ns, reference.st_ctime_ns)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        probe.write_bytes(b"")
        stamp = probe.stat()
        probe.unlink()
        if stamp.st_mtime_ns > floor and stamp.st_ctime_ns > floor:
            return
        time.sleep(0.001)
    raise AssertionError(
        "filesystem timestamp clock did not advance; cannot make the mutation observable"
    )


@pytest.fixture
def counted_input(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    path.write_bytes(b"original")
    original = Path.open
    reads = []

    def counted(self, *args, **kwargs):
        if self == path and args == ("rb",):
            reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    return path, reads


def test_controller_and_native_actions_share_only_current_execution(counted_input):
    path, reads = counted_input
    for expected_reads in (1, 2):
        with controller._snapshot_cache():
            expected = controller._sha256(path)
            assert native_actions.file_sha256(str(path)) == expected
            assert native_actions.file_sha256(path) == expected
            assert len(reads) == expected_reads
    assert controller._sha256(path) == expected
    assert len(reads) == 3


@pytest.mark.parametrize("mutation", ["rewrite", "replace"])
def test_same_size_and_mtime_does_not_hide_changed_content(counted_input, mutation):
    path, reads = counted_input
    with snapshots.snapshot_cache():
        original = snapshots.file_sha256(path)
        stat = path.stat()
        _advance_filesystem_clock(stat, path.parent)
        if mutation == "replace":
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b"modified")
            replacement.replace(path)
        else:
            path.write_bytes(b"modified")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert snapshots.file_sha256(path) != original
        assert len(reads) == 2


def test_captured_bytes_always_come_from_the_hashed_read(counted_input):
    path, reads = counted_input
    with snapshots.snapshot_cache():
        expected = snapshots.file_sha256(path)
        _, stat, raw, digest = snapshots.stable_file_snapshot(path, capture_bytes=True)
        assert raw == b"original"
        assert stat.st_size == len(raw)
        assert digest == expected == "sha256:" + hashlib.sha256(raw).hexdigest()
        assert len(reads) == 2
        assert snapshots.file_sha256(path) == expected
        assert len(reads) == 2


def test_nested_scope_restores_outer_cache_and_exception_clears_it(counted_input):
    path, reads = counted_input
    with pytest.raises(RuntimeError, match="test failure"):
        with snapshots.snapshot_cache():
            expected = snapshots.file_sha256(path)
            with snapshots.snapshot_cache():
                assert snapshots.file_sha256(path) == expected
            assert snapshots.file_sha256(path) == expected
            assert len(reads) == 2
            raise RuntimeError("test failure")
    assert snapshots.file_sha256(path) == expected
    assert len(reads) == 3


@pytest.mark.parametrize("mutation", ["rewrite", "replace"])
def test_mutation_during_read_fails_and_is_not_cached(counted_input, monkeypatch, mutation):
    path, reads = counted_input
    original = os.fstat
    calls = []

    def mutate(fd):
        calls.append(fd)
        if len(calls) == 2:
            _advance_filesystem_clock(path.stat(), path.parent)
            if mutation == "replace":
                replacement = path.with_suffix(".new")
                replacement.write_bytes(b"modified")
                replacement.replace(path)
            else:
                path.write_bytes(b"modified")
        return original(fd)

    monkeypatch.setattr(snapshots.os, "fstat", mutate)
    with snapshots.snapshot_cache():
        with pytest.raises(ValueError, match="changed while it was read"):
            snapshots.file_sha256(path)
        assert snapshots.file_sha256(path) == "sha256:" + hashlib.sha256(b"modified").hexdigest()
        assert len(reads) == 2


def test_deleted_cached_file_is_not_accepted(counted_input):
    path, _ = counted_input
    with snapshots.snapshot_cache():
        snapshots.file_sha256(path)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            snapshots.file_sha256(path)
