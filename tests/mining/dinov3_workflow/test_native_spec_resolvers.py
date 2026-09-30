# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check shipped-spec composition without resolver state leaking between tests."""

import functools
import subprocess
import sys
import textwrap

import pytest


# ``build_grit_spec`` / ``build_training_spec`` compose over the *installed* TAO
# DINOv3 structured schema (``native_actions._experiment_spec``), so they can only
# be exercised against the stacked DEFT runtime that ships in the Data Services
# container. An older ``nvidia_tao_pytorch.ssl.dinov3`` still imports cleanly but
# its ``ExperimentConfig`` has no ``grit_score`` section and no
# ``dataset.train_manifest`` field, which makes OmegaConf struct mode raise
# ``ConfigAttributeError``. A plain ImportError guard therefore does not catch it —
# probe the schema itself. The probe runs in a throwaway subprocess so importing
# TAO never registers the ``eval`` resolver in the pytest process, which would
# defeat the very resolver-isolation assertions this module makes.
_DEFT_SCHEMA_PROBE = textwrap.dedent('''
    import sys

    from omegaconf import OmegaConf

    try:
        from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
        import nvidia_tao_pytorch.ssl.dinov3  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name not in (
            "nvidia_tao_pytorch", "nvidia_tao_pytorch.config",
            "nvidia_tao_pytorch.config.dinov3",
            "nvidia_tao_pytorch.config.dinov3.default_config",
            "nvidia_tao_pytorch.ssl", "nvidia_tao_pytorch.ssl.dinov3",
        ):
            raise
        print(f"TAO DINOv3 runtime unavailable: {error}")
        sys.exit(77)

    # Compare against a plain dict: `in` on a struct-mode DictConfig has subtle
    # semantics, and this probe must never report "absent" for a good runtime.
    schema = OmegaConf.to_container(
        OmegaConf.structured(ExperimentConfig()), resolve=False, throw_on_missing=False,
    )
    dataset = schema.get("dataset") if isinstance(schema, dict) else None
    missing = []
    if not isinstance(schema, dict) or "grit_score" not in schema:
        missing.append("grit_score")
    if not isinstance(dataset, dict) or "train_manifest" not in dataset:
        missing.append("dataset.train_manifest")
    if missing:
        print("ExperimentConfig has no " + ", ".join(missing))
        sys.exit(77)
''')


@functools.lru_cache(maxsize=1)
def _deft_schema_unavailable_reason() -> str:
    """Return why the installed DINOv3 schema cannot compose DEFT specs, else ``""``."""
    probe = subprocess.run(
        [sys.executable, "-c", _DEFT_SCHEMA_PROBE],
        capture_output=True, text=True, timeout=120,
    )
    assert probe.returncode in (0, 77), probe.stdout + probe.stderr
    if probe.returncode == 0:
        return ""
    return (probe.stdout + probe.stderr).strip() or "DINOv3 schema probe failed"


@pytest.mark.parametrize("import_order", ["controller-first", "tao-first", "existing-resolver"])
@pytest.mark.parametrize("first_action", ["score", "train"])
def test_shipped_spec_resolvers_in_fresh_process(
    tmp_path, import_order, first_action, deft_runtime_unavailable,
):
    """Both native actions must resolve shipped learning rates in either order."""
    reason = _deft_schema_unavailable_reason()
    if reason:
        deft_runtime_unavailable(
            f"requires the stacked TAO PyTorch DINOv3 DEFT runtime ({reason})"
        )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent('''
            import importlib
            import math
            from pathlib import Path
            import sys

            from omegaconf import OmegaConf
            from omegaconf.errors import UnsupportedInterpolationType
            import pyarrow as pa
            import pyarrow.parquet as pq

            assert not OmegaConf.has_resolver("eval")
            if sys.argv[2] == "tao-first":
                importlib.import_module("nvidia_tao_pytorch.core.hydra.hydra_runner")
            elif sys.argv[2] == "existing-resolver":
                OmegaConf.register_new_resolver("eval", eval)

            from nvidia_tao_ds.mining.dinov3.workflow.native_actions import (
                build_grit_spec, build_training_spec,
            )
            if sys.argv[2] == "controller-first":
                assert not OmegaConf.has_resolver("eval")
                assert "nvidia_tao_pytorch.core.hydra.hydra_runner" not in sys.modules
            import nvidia_tao_pytorch.ssl.dinov3 as dinov3

            root = Path(sys.argv[1])
            base = Path(dinov3.__file__).parent / "experiment_specs/train_dinov3_vitb.yaml"
            original = base.read_bytes()
            checkpoint = root / "original.pth"
            checkpoint.write_bytes(b"unused-spec-composition-checkpoint")
            manifest = root / "training.parquet"
            pq.write_table(pa.table({
                "sample_id": ["sample"], "storage_type": ["file"], "path": ["/data/sample.jpg"],
            }), manifest)

            def check_score():
                score = OmegaConf.load(build_grit_spec(
                    base_spec=base, input_parquet=manifest, checkpoint=checkpoint,
                    output_dir=root / "score", settings={},
                ))
                for section in ("learning_rate", "last_layer_learning_rate"):
                    assert math.isclose(score.train.schedulers[section].val_base, 6.25e-6)

            if sys.argv[3] == "score":
                check_score()

            for nodes, gpus in ((1, 1), (2, 2)):
                spec_path, _, _ = build_training_spec(
                    base_spec=base, manifest=manifest, parent_checkpoint=checkpoint,
                    passes=1, output_dir=root / f"train-{nodes}-{gpus}",
                    num_nodes=nodes, gpus_per_node=gpus,
                    checkpoint_policy="base_checkpoint_each_round",
                )
                spec = OmegaConf.load(spec_path)
                expected = 5e-5 * math.sqrt(16 * nodes * gpus / 1024)
                for section in ("learning_rate", "last_layer_learning_rate"):
                    assert math.isclose(spec.train.schedulers[section].val_base, expected)
                assert "${eval:" not in spec_path.read_text()
                assert spec.train.results_dir == spec.results_dir

            check_score()
            assert OmegaConf.has_resolver("eval")
            # Importing the real training entrypoint afterwards must not re-register eval.
            if sys.argv[2] != "existing-resolver":
                importlib.import_module("nvidia_tao_pytorch.ssl.dinov3.scripts.train")
            else:
                # The controller preserves externally owned resolvers and does
                # not import training in-process; native actions run separately.
                assert "nvidia_tao_pytorch.core.hydra.hydra_runner" not in sys.modules
            invalid = OmegaConf.load(base)
            invalid.train.schedulers.learning_rate.val_base = "${unknown_resolver:1}"
            invalid_base = root / "invalid.yaml"
            OmegaConf.save(invalid, invalid_base)
            try:
                build_grit_spec(
                    base_spec=invalid_base, input_parquet=manifest, checkpoint=checkpoint,
                    output_dir=root / "invalid", settings={},
                )
            except UnsupportedInterpolationType:
                pass
            else:
                raise AssertionError("Unknown resolvers must still fail")
            assert not (root / "invalid" / "grit_score.yaml").exists()
            assert base.read_bytes() == original
        '''), str(tmp_path), import_order, first_action],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("code", [1, -11])
def test_schema_probe_errors_are_not_optional(monkeypatch, code):
    """Unexpected import failures and crashes must not become runtime skips."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], code, "", "unexpected runtime failure",
    ))
    with pytest.raises(AssertionError, match="unexpected runtime failure"):
        _deft_schema_unavailable_reason.__wrapped__()


def test_schema_probe_timeout_is_not_optional(monkeypatch):
    """A hung installed runtime must fail instead of reducing test coverage."""
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        _deft_schema_unavailable_reason.__wrapped__()


@pytest.mark.parametrize("code,reason", [(0, ""), (77, "unsupported schema")])
def test_schema_probe_explicit_outcomes(monkeypatch, code, reason):
    """Only the explicit unsupported-runtime exit status permits a skip."""
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], code, reason, "",
    ))
    assert _deft_schema_unavailable_reason.__wrapped__() == reason


@pytest.mark.parametrize("statement,optional", [
    ("raise ModuleNotFoundError('missing runtime', name='nvidia_tao_pytorch.ssl.dinov3')", True),
    ("raise ModuleNotFoundError('missing dependency', name='google.protobuf')", False),
    ("raise RuntimeError('broken runtime')", False),
    ("raise SyntaxError('broken runtime')", False),
])
def test_schema_probe_classifies_import_errors(monkeypatch, statement, optional):
    """Real subprocess imports distinguish absent TAO from a broken install."""
    probe = _DEFT_SCHEMA_PROBE.replace(
        "from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig",
        statement,
    )
    monkeypatch.setitem(_deft_schema_unavailable_reason.__wrapped__.__globals__, "_DEFT_SCHEMA_PROBE", probe)
    if optional:
        assert "missing runtime" in _deft_schema_unavailable_reason.__wrapped__()
    else:
        with pytest.raises(AssertionError):
            _deft_schema_unavailable_reason.__wrapped__()
