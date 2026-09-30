# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate DINOv3 preparation contracts and the shipped image producer path."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from nvidia_tao_ds.mining.dinov3 import store
from nvidia_tao_ds.mining.dinov3.contracts import file_sha256
from nvidia_tao_ds.mining.dinov3.internal.refinement import _validated_query_contract
from nvidia_tao_ds.mining.dinov3.workflow.controller import _target_identity_set


ENCODER = {
    "name": "CLIP", "checkpoint_digest": "sha256:" + "a" * 64,
    "input_resolution": 16, "normalization": "clip-default",
}


@pytest.fixture
def prepared(tmp_path):
    """Create a committed two-dimensional source and a separate target file."""
    shards = tmp_path / "shards"
    shards.mkdir()
    frame = pd.DataFrame({
        "sample_id": ["s1", "s2"], "embedding": [[1., 0.], [0., 1.]],
        "path": [str(tmp_path / "1.png"), str(tmp_path / "2.png")],
        "storage_type": ["file", "file"],
    })
    frame.to_parquet(shards / "part.parquet", index=False)
    store.register_embedding_store(store_root=shards, output_dir=tmp_path / "source", encoder=ENCODER)
    targets = tmp_path / "targets.parquet"
    frame.to_parquet(targets, index=False)
    return {
        "targets": targets, "source_store_manifest": tmp_path / "source/embedding_store.json",
        "output_dir": tmp_path / "contract", "encoder": ENCODER,
    }


def test_contract_matches_existing_consumer(prepared):
    """The generated contract is accepted without manual JSON edits."""
    result = store.write_target_embedding_contract(**prepared)
    contract = prepared["output_dir"] / "target_embedding_contract.json"
    source = json.loads(prepared["source_store_manifest"].read_text())
    assert result["payload"]["source_store_manifest"]["inventory_digest"] == source["inventory_digest"]
    assert _validated_query_contract(str(contract), source)["sha256"] == file_sha256(contract)
    assert result["inputs"][0]["sha256"] == file_sha256(prepared["targets"])
    assert (prepared["output_dir"] / "_SUCCESS").read_text().strip() == result["artifact_id"]
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        store.write_target_embedding_contract(**prepared)


@pytest.mark.parametrize("vectors", [[], [[0., 0.]], [[float("nan"), 1.]],
                                     [[float("inf"), 1.]], [[1., 2., 3.]],
                                     [[1., 2.], [1.]], [["bad", "data"]]])
def test_invalid_target_vectors_do_not_publish(prepared, vectors):
    """Empty, nonnumeric, nonfinite, zero and mismatched vectors fail closed."""
    pd.DataFrame({"embedding": vectors}).to_parquet(prepared["targets"], index=False)
    with pytest.raises(ValueError):
        store.write_target_embedding_contract(**prepared)
    assert not (prepared["output_dir"] / "_SUCCESS").exists()


def test_missing_embedding_column(prepared):
    """A filepath-only manifest is not an embedding contract input."""
    pd.DataFrame({"filepath": ["/tmp/a.png"]}).to_parquet(prepared["targets"], index=False)
    with pytest.raises(ValueError, match="embedding column"):
        store.write_target_embedding_contract(**prepared)


@pytest.mark.parametrize("field,value", [
    ("name", "SigLIP"), ("checkpoint_digest", "other"),
    ("normalization", "other"), ("input_resolution", 32),
    ("name", " "), ("input_resolution", 0), ("input_resolution", True),
])
def test_encoder_identity_must_match(prepared, field, value):
    """Equal vector dimensions do not make different encoders compatible."""
    prepared["encoder"] = {**ENCODER, field: value}
    with pytest.raises(ValueError):
        store.write_target_embedding_contract(**prepared)
    assert not (prepared["output_dir"] / "_SUCCESS").exists()


def test_uncommitted_source_rejected(prepared):
    """The source must be a committed artifact, not a handwritten manifest."""
    prepared["source_store_manifest"].with_name("_SUCCESS").unlink()
    with pytest.raises(FileNotFoundError):
        store.write_target_embedding_contract(**prepared)


def test_later_batch_dimension_mismatch_rejected(prepared):
    """Validation scans beyond the first Arrow batch before publishing."""
    pd.DataFrame({"embedding": [[1., 2.]] * 16_384 + [[1., 2., 3.]]}).to_parquet(
        prepared["targets"], index=False,
    )
    with pytest.raises(ValueError, match="dimension differs"):
        store.write_target_embedding_contract(**prepared)
    assert not (prepared["output_dir"] / "_SUCCESS").exists()


def test_mutated_target_rejected(prepared, monkeypatch):
    """Changing the input between vector scan and hashing cannot publish."""
    original = store.file_identity

    def changed(path, **kwargs):
        if Path(path) == prepared["targets"]:
            pd.DataFrame({"embedding": [[1., 2., 3.]]}).to_parquet(path, index=False)
        return original(path, **kwargs)

    monkeypatch.setattr(store, "file_identity", changed)
    with pytest.raises(ValueError, match="Targets changed"):
        store.write_target_embedding_contract(**prepared)
    assert not (prepared["output_dir"] / "_SUCCESS").exists()


def _run(module, *args):
    result = subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("missing,optional", [
    ("torch", True), ("transformers", True), ("accelerate", True),
    ("nvidia_tao_pytorch", True), ("google.protobuf", False),
])
def test_producer_runtime_gate_is_narrow(monkeypatch, tmp_path, missing, optional):
    """Missing top-level runtimes use policy; broken transitive imports fail."""
    def blocked(name):
        if name == missing or missing == "google.protobuf":
            raise ModuleNotFoundError("missing test dependency", name=missing)
        return object()

    monkeypatch.setattr(importlib, "import_module", blocked)
    expected = pytest.fail.Exception if optional else ModuleNotFoundError
    with pytest.raises(expected):
        test_shipped_clip_producer_to_registered_store_and_target_contract(tmp_path, pytest.fail)


def test_shipped_clip_producer_to_registered_store_and_target_contract(tmp_path, deft_runtime_unavailable):
    """Run real local CLIP inference on dummy images, then both preparation CLIs."""
    for module in ("torch", "transformers", "accelerate", "nvidia_tao_pytorch"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name != module:
                raise
            deft_runtime_unavailable(f"CLIP producer smoke requires {module}")
    from PIL import Image
    import torch
    import yaml
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer

    torch.manual_seed(7)
    snapshot = tmp_path / "clip"
    snapshot.mkdir()
    config = CLIPConfig(
        text_config={"vocab_size": 2, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2},
        vision_config={"image_size": 16, "patch_size": 8, "hidden_size": 16,
                       "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2},
        projection_dim=8,
    )
    CLIPModel(config).save_pretrained(snapshot)
    (snapshot / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}))
    (snapshot / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(vocab_file=str(snapshot / "vocab.json"), merges_file=str(snapshot / "merges.txt"))
    CLIPProcessor(
        tokenizer=tokenizer,
        image_processor=CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}),
    ).save_pretrained(snapshot)
    payload_root = tmp_path / "images"
    payload_root.mkdir()
    shards = tmp_path / "shards"
    shards.mkdir()
    outputs = {}
    for population in ("source", "target"):
        paths = []
        for index, color in enumerate(((200, 40, 20), (30, 170, 90), (50, 20, 180))):
            path = payload_root / f"{population}-{index}.png"
            Image.new("RGB", (24, 24), color).save(path)
            paths.append(str(path))
        frame = pd.DataFrame({
            "filepath": paths, "path": paths, "storage_type": ["file"] * 3,
            "sample_id": [f"{population}-{i}" for i in range(3)],
            "task": ["dummy"] * 3, "role": ["query", "reference", "reference"],
        })
        input_path = tmp_path / f"{population}-input.parquet"
        frame.to_parquet(input_path, index=False)
        output = shards / "source.parquet" if population == "source" else tmp_path / "target.parquet"
        _run("nvidia_tao_ds.mining.embedding.scripts.image_embeddings",
             f"input_parquet={input_path}", f"output_parquet={output}",
             "model=CLIP", f"model_path={snapshot}", "batch_size=2", f"hydra.run.dir={tmp_path}/hydra-{population}")
        actual = pd.read_parquet(output)
        pd.testing.assert_frame_equal(actual[frame.columns], frame)
        assert len(actual) == 3
        assert all(len(vector) == 8 for vector in actual.embedding)
        outputs[population] = output
    payload_contract = tmp_path / "payload.json"
    payload_contract.write_text(json.dumps({
        "schema_version": "1.0", "immutability": "immutable",
        "datasets": [{"dataset_id": "dummy", "version": "v1", "root_uri": payload_root.as_uri()}],
    }))
    encoder_flags = ["--encoder-name", "CLIP", "--encoder-checkpoint-digest", file_sha256(snapshot / "model.safetensors"),
                     "--input-resolution", "16", "--normalization", "clip-default"]
    module = "nvidia_tao_ds.mining.dinov3.internal.refinement"
    _run(module, "register-store", "--store-root", shards, "--output-dir", tmp_path / "registered",
         "--source-payload-contract", payload_contract, *encoder_flags)
    source_manifest = tmp_path / "registered/embedding_store.json"
    _run(module, "write-target-contract", "--targets", outputs["target"],
         "--source-store-manifest", source_manifest, "--output-dir", tmp_path / "contract", *encoder_flags)
    contract = tmp_path / "contract/target_embedding_contract.json"
    source = json.loads(source_manifest.read_text())
    _validated_query_contract(str(contract), source)
    assert _target_identity_set(outputs["target"], "grit_score", 8) == {("target-0", "dummy")}
    _run(module, "exact-search", "--queries", outputs["target"],
         "--source-part", outputs["source"],
         "--source-store-manifest", source_manifest, "--query-embedding-contract", contract,
         "--top-k", "1", "--min-similarity", "-1", "--output-dir", tmp_path / "search")
    assert (tmp_path / "search/_SUCCESS").is_file()
    neighbors = pd.read_parquet(tmp_path / "search/neighbors.parquet")
    assert not neighbors.empty
    assert set(neighbors.sample_id) <= {f"source-{i}" for i in range(3)}
    assert set(neighbors.query_id) <= {f"target-{i}" for i in range(3)}
    assert neighbors.cosine_similarity.between(-1, 1).all()
    assert set(neighbors.source_inventory_digest) == {source["inventory_digest"]}
    workflow_module = "nvidia_tao_ds.mining.dinov3.workflow"
    run_config = tmp_path / "run.yaml"
    _run(workflow_module, "init", "--recipe", "grit-score", "--output", run_config)
    value = yaml.safe_load(run_config.read_text())
    base = tmp_path / "base.pth"
    base.write_bytes(b"placeholder; validate does not execute training")
    spec = tmp_path / "train.yaml"
    spec.write_text("model: {}\ndataset:\n  batch_size: 2\n")
    value["model"]["base_checkpoint"] = str(base)
    value["training"]["base_spec"] = str(spec)
    value["data"].pop("previous_training_manifest")
    value["data"].update({
        "target_manifest": str(outputs["target"]),
        "source_store_manifest": str(source_manifest),
        "source_payload_contract": str(payload_contract),
        "target_embedding_contract": str(contract),
    })
    value["output"]["run_dir"] = str(tmp_path / "run")
    run_config.write_text(yaml.safe_dump(value))
    result = _run(workflow_module, "validate", run_config)
    assert json.loads(result.stdout)["valid"] is True
