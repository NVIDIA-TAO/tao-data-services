# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO component routing and platform-boundary regression tests."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nvidia_tao_ds.mining.dinov3.workflow.containers import configure_containers, resolve_image, stage_image
from nvidia_tao_ds.mining.dinov3.workflow.execution import ExternalRunner, LocalRunner, StageRequest


def config():
    return {
        "execution": {"backend": "external", "capabilities": {"containers": True},
                      "container_images": {"pytorch": "registry/pyt:reviewed",
                                           "data_services": "registry/ds:reviewed"}},
        "actions": {name: {} for name in ("score", "train", "data", "search", "evaluate")},
    }


def test_component_routing_uses_bank_versions():
    value = config()
    configure_containers(value)
    images = value["execution"]["container_images"]
    for stage in ("score", "train"):
        assert stage_image(value["actions"], stage) == images["pytorch"]
    for stage in ("select_targets", "materialize", "search", "search_candidates"):
        assert stage_image(value["actions"], stage) == images["data_services"]
    assert stage_image(value["actions"], "evaluate") is None
    assert images["pytorch"] != images["data_services"]


@pytest.mark.parametrize("backend,capability", [("local", True), ("external", False)])
def test_container_config_requires_capable_external_runner(backend, capability):
    value = config()
    value["execution"].update(backend=backend, capabilities={"containers": capability})
    with pytest.raises(ValueError):
        configure_containers(value)


def test_evaluator_requires_explicit_image():
    value = config()
    value["actions"]["evaluate"]["command"] = ["customer_eval"]
    with pytest.raises(ValueError, match="nonempty"):
        configure_containers(value)


def test_ann_stage_overrides_are_preserved():
    value = config()
    value["actions"]["search"]["candidate"] = {"container_image": "registry/ann:reviewed"}
    configure_containers(value)
    assert stage_image(value["actions"], "search_candidates") == "registry/ann:reviewed"
    assert stage_image(value["actions"], "search") == value["execution"]["container_images"]["data_services"]


def test_unknown_key_fails_closed(tmp_path):
    (tmp_path / "versions.yaml").write_text("images: {}\n")
    with pytest.raises(ValueError, match="Unknown"):
        resolve_image("tao_toolkit.missing", tmp_path)


def request(tmp_path):
    return StageRequest("test", "run", 1, "train", ["unused"], str(tmp_path),
                        str(tmp_path), {}, {},
                        {"attempt_scope": "process", "container_image": "registry/tao:reviewed"})


def test_local_runner_cannot_ignore_image(tmp_path):
    with pytest.raises(RuntimeError, match="container request"):
        LocalRunner(tmp_path).run(request(tmp_path))


@pytest.mark.parametrize("reported", [None, "registry/wrong:tag", "registry/tao:reviewed"])
def test_external_runner_checks_image_attestation(tmp_path, reported):
    runner = ExternalRunner(command=["unused"], jobs_dir=tmp_path,
                            poll_seconds=0, call_timeout_seconds=1)
    status = {"state": "COMPLETE", "container_image": reported}
    with patch.object(runner, "_call", side_effect=[{}, status]):
        if reported == "registry/tao:reviewed":
            assert runner.run(request(tmp_path)).state == "COMPLETE"
        else:
            with pytest.raises(RuntimeError, match="attest"):
                runner.run(request(tmp_path))
