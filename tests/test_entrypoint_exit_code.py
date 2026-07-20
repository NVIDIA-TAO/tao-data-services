# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for exit-code propagation in the shared DS entrypoint.

These guard against a CLI error-signaling bug where a failed subtask (e.g.
auto_label failing because Gemini credentials were missing) wrote FAILURE to
its status.json yet the ``launch()`` dispatcher returned a zero process exit
code, silently breaking CI/automation that keys off ``$?``.
"""

import pytest

from nvidia_tao_ds.core.entrypoint import entrypoint


def _write_runner(tmp_path, name, body):
    """Create a tiny runner script the dispatcher will exec as a subprocess."""
    runner = tmp_path / name
    runner.write_text(body)
    return str(runner)


def _make_args(tmp_path, results_dir):
    """Build a minimal valid args/spec pair for launch()."""
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"results_dir: {results_dir}\n")
    return {
        "subtask": "generate",
        "experiment_spec_file": str(spec),
        "results_dir": str(results_dir),
    }


@pytest.fixture(autouse=True)
def _isolate_launch(monkeypatch):
    """Stub out GPU discovery and telemetry so launch() runs offline."""
    monkeypatch.setattr(
        entrypoint.subprocess, "check_output", lambda *a, **k: b"GPU 0: UUID(GPU-xxxx)"
    )
    monkeypatch.setattr(entrypoint, "send_telemetry_data", lambda *a, **k: None)
    monkeypatch.setattr(entrypoint, "get_device_details", lambda: [])


def test_launch_exits_nonzero_on_subprocess_failure(tmp_path):
    """A subtask that exits non-zero must make launch() exit non-zero."""
    runner = _write_runner(tmp_path, "failing_runner.py", "import sys\nsys.exit(1)\n")
    subtasks = {"generate": {"module_name": "x", "runner_path": runner}}
    args = _make_args(tmp_path, tmp_path)

    with pytest.raises(SystemExit) as exc:
        entrypoint.launch(args, [], subtasks, network="auto_label")

    assert exc.value.code == 1


def test_launch_exits_nonzero_when_status_reports_failure(tmp_path):
    """Even if the subtask exits 0, a FAILURE in status.json must propagate."""
    runner = _write_runner(tmp_path, "ok_runner.py", "import sys\nsys.exit(0)\n")
    (tmp_path / "status.json").write_text(
        '{"date": "1/1/2026", "time": "0:0:0", "status": "STARTED"}\n'
        '{"date": "1/1/2026", "time": "0:0:1", "status": "FAILURE", "message": "boom"}\n'
    )
    subtasks = {"generate": {"module_name": "x", "runner_path": runner}}
    args = _make_args(tmp_path, tmp_path)

    with pytest.raises(SystemExit) as exc:
        entrypoint.launch(args, [], subtasks, network="auto_label")

    assert exc.value.code == 1


def test_launch_passes_when_subprocess_succeeds(tmp_path):
    """A clean run (exit 0, no FAILURE status) returns True and does not exit."""
    runner = _write_runner(tmp_path, "ok_runner.py", "import sys\nsys.exit(0)\n")
    (tmp_path / "status.json").write_text(
        '{"date": "1/1/2026", "time": "0:0:0", "status": "STARTED"}\n'
        '{"date": "1/1/2026", "time": "0:0:1", "status": "RUNNING", "message": "done"}\n'
    )
    subtasks = {"generate": {"module_name": "x", "runner_path": runner}}
    args = _make_args(tmp_path, tmp_path)

    assert entrypoint.launch(args, [], subtasks, network="auto_label") is True


def test_status_reports_failure_helper(tmp_path):
    """The status-file helper reads the last record and tolerates junk."""
    # No directory / no file -> not a failure.
    assert entrypoint._status_reports_failure(None) is False
    assert entrypoint._status_reports_failure(str(tmp_path)) is False

    status_file = tmp_path / "status.json"
    # A later RUNNING record supersedes an earlier FAILURE.
    status_file.write_text(
        '{"status": "FAILURE"}\n'
        'not-json\n'
        '{"status": "RUNNING"}\n'
    )
    assert entrypoint._status_reports_failure(str(tmp_path)) is False

    # Final record is FAILURE -> failure.
    status_file.write_text('{"status": "RUNNING"}\n{"status": "FAILURE"}\n')
    assert entrypoint._status_reports_failure(str(tmp_path)) is True


def test_resolve_results_dir_precedence(tmp_path):
    """CLI override beats args, which beats the spec file; '???' is ignored."""
    spec = tmp_path / "spec.yaml"
    spec.write_text("results_dir: /from/spec\n")
    args = {"experiment_spec_file": str(spec), "results_dir": "/from/args"}

    # CLI override wins.
    assert entrypoint._resolve_results_dir(
        args, ["results_dir=/from/cli"]
    ) == "/from/cli"

    # No CLI override -> args wins.
    assert entrypoint._resolve_results_dir(args, []) == "/from/args"

    # No args results_dir -> spec file wins.
    assert entrypoint._resolve_results_dir(
        {"experiment_spec_file": str(spec)}, []
    ) == "/from/spec"

    # Hydra mandatory placeholder is not a usable path.
    spec.write_text("results_dir: ???\n")
    assert entrypoint._resolve_results_dir(
        {"experiment_spec_file": str(spec)}, []
    ) is None
