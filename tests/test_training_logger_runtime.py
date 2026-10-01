# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Lightning logging backend required by TAO training."""

from importlib.util import find_spec
import os
import subprocess
import sys
import textwrap

import pytest
from pytorch_lightning.loggers import TensorBoardLogger


def test_training_logger_writes_events(tmp_path, deft_runtime_unavailable):
    """Importing the train module alone does not verify its lazy logger backend."""
    # Missing packages are optional in source-only CI, not container validation.
    # Installed but broken/unsupported backends must fail at construction or use.
    if find_spec("tensorboard") is None and find_spec("tensorboardX") is None:
        deft_runtime_unavailable("requires a TensorBoard logging backend (tensorboard or tensorboardX)")
    logger = TensorBoardLogger(save_dir=tmp_path, version=1, name="lightning_logs")
    try:
        logger.log_metrics({"train_loss": 0.5}, step=1)
    finally:
        logger.finalize("success")
    events = list((tmp_path / "lightning_logs" / "version_1").glob("events.out.tfevents.*"))
    assert events and all(path.stat().st_size > 0 for path in events)


@pytest.mark.parametrize("required", [False, True])
def test_runtime_requirement_policy(monkeypatch, pytestconfig, deft_runtime_unavailable, required):
    """The same missing dependency must fail in required-runtime mode."""
    monkeypatch.setattr(pytestconfig.option, "require_deft_runtime", required)
    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(expected, match="missing runtime"):
        deft_runtime_unavailable("missing runtime")


@pytest.mark.parametrize("name", ["tensorboardX.writer", "google.protobuf"])
def test_installed_backend_import_failure_is_not_skipped(monkeypatch, tmp_path, name):
    """Broken backend internals are regressions, not optional dependencies."""
    monkeypatch.setitem(test_training_logger_writes_events.__globals__, "find_spec", lambda _: object())

    def broken_logger(**kwargs):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setitem(test_training_logger_writes_events.__globals__, "TensorBoardLogger", broken_logger)
    with pytest.raises(ModuleNotFoundError, match=name):
        test_training_logger_writes_events(tmp_path, lambda reason: pytest.fail("unexpected runtime gate"))


def test_missing_backend_uses_runtime_policy(monkeypatch, tmp_path):
    """Only genuinely absent packages are delegated to the runtime policy."""
    monkeypatch.setitem(test_training_logger_writes_events.__globals__, "find_spec", lambda _: None)
    with pytest.raises(pytest.fail.Exception, match="requires a TensorBoard"):
        test_training_logger_writes_events(tmp_path, pytest.fail)


def test_preflight_rejects_real_rank_filtered_logger(deft_runtime_unavailable):
    """A real nonzero-rank Lightning no-op must fail controller preflight."""
    if find_spec("tensorboard") is None and find_spec("tensorboardX") is None:
        deft_runtime_unavailable("requires a TensorBoard logging backend")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent('''
            from types import SimpleNamespace
            from pytorch_lightning.loggers import TensorBoardLogger
            from nvidia_tao_ds.mining.dinov3.workflow import cli

            # Only isolate unrelated model imports; use the real logger/decorators.
            cli.importlib = SimpleNamespace(
                import_module=lambda _: SimpleNamespace(TensorBoardLogger=TensorBoardLogger),
            )
            try:
                cli.main(["preflight"])
            except RuntimeError as error:
                assert "no nonempty event file" in str(error), str(error)
            else:
                raise AssertionError("rank-filtered logger incorrectly passed")
        ''')],
        env={**os.environ, "RANK": "1", "LOCAL_RANK": "1", "SLURM_PROCID": "1", "JSM_NAMESPACE_RANK": "1"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
