# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone package resources and minimal-runtime command checks."""

import json

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
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: imports.append(name))
    monkeypatch.setattr(cli.metadata, "version", lambda _: "test")
    assert cli.main(["preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(imports) == 4 and imports == result["modules"]
    assert result["cuda_verified"] is False


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
