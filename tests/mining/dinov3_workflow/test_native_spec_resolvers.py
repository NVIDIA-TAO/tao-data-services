# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check shipped-spec composition without resolver state leaking between tests."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("import_order", ["controller-first", "tao-first"])
@pytest.mark.parametrize("first_action", ["score", "train"])
def test_shipped_spec_resolvers_in_fresh_process(tmp_path, import_order, first_action):
    """Both native actions must resolve shipped learning rates in either order."""
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
            importlib.import_module("nvidia_tao_pytorch.ssl.dinov3.scripts.train")
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
