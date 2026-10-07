# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone package resources and minimal-runtime command checks."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nvidia_tao_ds.mining.dinov3.workflow import cli
from nvidia_tao_ds.mining.dinov3.workflow.containers import resolve_image


@pytest.mark.parametrize("recipe", ["grit-score", "multi-task-round-robin"])
def test_init_uses_packaged_recipes_without_bank(tmp_path, monkeypatch, recipe):
    monkeypatch.delenv("TAO_SKILL_BANK_PATH", raising=False)
    target = tmp_path / "run.yaml"
    assert cli.main(["init", "--recipe", recipe, "--output", str(target)]) == 0
    value = yaml.safe_load(target.read_text())
    assert value["execution"]["backend"] == "local"
    assert "container_images" not in value["execution"]
    assert value["actions"]["train"]["resources"] == {"nodes": 1, "gpus_per_node": 1}
    if recipe == "grit-score":
        assert value["actions"]["score"]["settings"]["neighbor_backend"] == "torch_exact"
    with pytest.raises(ValueError, match="overwrite"):
        cli.main(["init", "--recipe", recipe, "--output", str(target)])


def test_explicit_images_do_not_require_skill_bank():
    assert resolve_image("registry/ds@sha256:abc") == "registry/ds@sha256:abc"
    with pytest.raises(ValueError, match="explicit container image"):
        resolve_image("tao_toolkit.data_services")


def test_preflight_checks_all_installed_components(monkeypatch, capsys):
    imports = []
    logger_type = Mock()

    def write_event(*args, **kwargs):
        (Path(logger_type.call_args.kwargs["save_dir"]) / "events.out.tfevents.test").write_bytes(b"event")

    logger_type.return_value.log_metrics.side_effect = write_event

    def import_module(name):
        imports.append(name)
        return SimpleNamespace(TensorBoardLogger=logger_type)

    monkeypatch.setattr(cli.importlib, "import_module", import_module)
    monkeypatch.setattr(cli.metadata, "version", lambda _: "test")
    assert cli.main(["preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert imports == result["modules"] + ["pytorch_lightning.loggers"]
    assert result["cuda_verified"] is False
    assert result["contracts"] == {"benchmark_isolation": 1}
    logger_type.return_value.log_metrics.assert_called_once_with({"preflight": 0.0}, step=0)
    logger_type.return_value.finalize.assert_called_once_with("success")
    assert not Path(logger_type.call_args.kwargs["save_dir"]).exists()


@pytest.mark.parametrize("failure", ["construct", "write"])
def test_preflight_rejects_unusable_training_logger(monkeypatch, capsys, failure):
    logger_type = Mock()
    error = ModuleNotFoundError("Training logger backend unavailable")
    if failure == "construct":
        logger_type.side_effect = error
    else:
        logger_type.return_value.log_metrics.side_effect = error
    monkeypatch.setattr(
        cli.importlib, "import_module",
        lambda _: SimpleNamespace(TensorBoardLogger=logger_type),
    )
    with pytest.raises(ModuleNotFoundError, match="Training logger backend unavailable"):
        cli.main(["preflight"])
    assert capsys.readouterr().out == ""
    assert not Path(logger_type.call_args.kwargs["save_dir"]).exists()
    if failure == "write":
        logger_type.return_value.finalize.assert_called_once_with("success")


@pytest.mark.parametrize("empty_file", [False, True])
def test_preflight_rejects_silent_logger(monkeypatch, capsys, empty_file):
    """Rank-filtered no-ops and empty files cannot certify a working writer."""
    logger_type = Mock()

    def finalize(*args):
        if empty_file:
            (Path(logger_type.call_args.kwargs["save_dir"]) / "events.out.tfevents.test").touch()

    logger_type.return_value.finalize.side_effect = finalize
    monkeypatch.setattr(cli.importlib, "import_module", lambda _: SimpleNamespace(TensorBoardLogger=logger_type))
    with pytest.raises(RuntimeError, match="no nonempty event file"):
        cli.main(["preflight"])
    assert capsys.readouterr().out == ""
    assert not Path(logger_type.call_args.kwargs["save_dir"]).exists()


@pytest.mark.parametrize("command", ["run", "resume"])
def test_run_commands_install_sigterm_handoff(monkeypatch, capsys, command):
    workflow = object()
    calls = []
    monkeypatch.setattr(cli, "_workflow", lambda _path: workflow)
    monkeypatch.setattr(
        cli,
        "_execute_with_termination_handoff",
        lambda observed: calls.append(observed) or {"status": "complete"},
    )

    assert cli.main([command, "input.yaml"]) == 0
    assert calls == [workflow]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}
