# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check execution-scoped digest reuse without weakening mutation detection."""

import hashlib
import os
from pathlib import Path

import pytest

from nvidia_tao_ds.mining.dinov3.workflow import controller, native_actions, snapshots


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
