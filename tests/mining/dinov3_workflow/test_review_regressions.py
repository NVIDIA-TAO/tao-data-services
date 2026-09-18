# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for review-discovered persistence and recovery failures."""

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from nvidia_tao_ds.mining.dinov3.workflow import state as state_module
from nvidia_tao_ds.mining.dinov3.workflow.execution import LocalRunner, StageRequest, StageResult
from nvidia_tao_ds.mining.dinov3.workflow.state import StateStore
from nvidia_tao_ds.mining.dinov3.workflow.controller import _sha256, _snapshot_cache


def _request(tmp_path):
    return StageRequest(
        client_job_id="review-recovery", run_id="test", round_index=1, stage="score",
        command=[sys.executable, "-c", "pass"], workdir=str(tmp_path),
        results_dir=str(tmp_path), environment={}, resources={}, execution_contract={},
    )


def test_failed_state_publication_does_not_advance_generation(tmp_path, monkeypatch):
    store = StateStore(tmp_path)
    state = store.initialize(run_id="test", config_digest="test")
    previous = dict(state)

    def fail(*_):
        raise OSError("simulated full disk")

    monkeypatch.setattr(state_module, "write_json_atomic", fail)
    with pytest.raises(OSError, match="full disk"):
        store.save(state)
    assert state == previous == store.load()


def test_lock_liveness_is_read_only(tmp_path):
    store = StateStore(tmp_path)
    assert not store.controller_active()
    assert not tmp_path.joinpath(".controller.lock").exists()
    with store.controller_lock():
        before = store.lock_path.read_bytes()
        assert store.controller_active()
        assert store.lock_path.read_bytes() == before
    assert not store.controller_active()


def test_unset_gpu_allocation_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    runner = LocalRunner(tmp_path / "jobs")
    with pytest.raises(RuntimeError, match="platform-pinned"):
        runner.run(replace(_request(tmp_path), resources={"gpus": 1}))
    assert not runner.jobs_dir.exists()


def test_retry_limit_is_enforced_and_attempts_are_recorded(tmp_path):
    runner = LocalRunner(tmp_path / "jobs", max_attempts=2)
    request = replace(_request(tmp_path), command=[sys.executable, "-c", "raise SystemExit(7)"])
    assert runner.run(request).attempt == 1
    assert runner.run(request).attempt == 2
    assert json.loads((runner.jobs_dir / "review-recovery.attempt-2.json").read_text())["retry"]
    with pytest.raises(RuntimeError, match="exhausted max_attempts"):
        runner.run(request)


def test_killed_leader_is_observed_then_retried(tmp_path):
    runner = LocalRunner(tmp_path / "jobs")
    runner.jobs_dir.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        start = runner._process_start_time(process.pid)
        result = StageResult(state="RUNNING", client_job_id="review-recovery",
                             backend_ref=f"pid:{process.pid}:start:{start}",
                             return_code=None, log_path=None, native_state="RUNNING")
        runner._write_record(runner.jobs_dir / "review-recovery.json", result)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        assert runner.status("review-recovery")["native_state"] == "ORPHANED"
        terminal = runner.run(_request(tmp_path))
        assert terminal.state == "COMPLETE"
        assert terminal.attempt == 2
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_snapshot_cache_invalidates_when_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"old")
    original = Path.open
    reads = []

    def counted(self, *args, **kwargs):
        if self == path and args == ("rb",):
            reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    with _snapshot_cache():
        first = _sha256(path)
        assert _sha256(path) == first
        assert len(reads) == 1
        path.write_bytes(b"changed")
        assert _sha256(path) != first
        assert len(reads) == 2
    assert _sha256(path) != first
    assert len(reads) == 3
