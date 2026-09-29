# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real embedding-store producers against sealed-inventory validation."""

import json
import os
import subprocess
import sys
import time

import pandas as pd
import pytest

from nvidia_tao_ds.mining.dinov3 import store
from nvidia_tao_ds.mining.dinov3.workflow.controller import (
    _validate_store_payload_binding,
    _verified_embedding_store,
)
from nvidia_tao_ds.mining.dinov3.contracts import ArtifactManifest, file_identity


@pytest.fixture
def source(tmp_path):
    """Create two real Parquet shards under a contracted local payload root."""
    shards = tmp_path / "shards"
    images = tmp_path / "images"
    shards.mkdir()
    images.mkdir()
    for index in range(2):
        image = images / f"{index}.jpg"
        image.write_bytes(b"immutable-test-payload")
        pd.DataFrame({
            "sample_id": [str(index)], "embedding": [[1.0, float(index)]],
            "storage_type": ["file"], "path": [str(image)],
        }).to_parquet(shards / f"{index}.parquet", index=False)
    contract = tmp_path / "payload.json"
    contract.write_text(json.dumps({
        "schema_version": "1.0", "immutability": "immutable",
        "datasets": [{"dataset_id": "dummy", "version": "v1", "root_uri": images.as_uri()}],
    }), encoding="utf-8")
    return shards, contract


@pytest.mark.parametrize("producer", ["register", "register-with-contract", "bind"])
def test_store_producer_satisfies_sealed_validator(tmp_path, source, producer):
    """Neither documented registration nor binding requires hand-edited seals."""
    shards, contract = source
    output = tmp_path / "registered"
    store.register_embedding_store(
        store_root=shards, output_dir=output, encoder={"name": "dummy"},
        source_payload_contract=contract if producer == "register-with-contract" else None,
    )
    if producer == "bind":
        store.bind_store_payload_contract(
            source_store_manifest=output / "embedding_store.json",
            output_dir=tmp_path / "bound", source_payload_contract=contract,
        )
        output = tmp_path / "bound"
    payload, artifact, paths, _ = _verified_embedding_store(
        output / "embedding_store.json", content_validation="sealed_inventory",
    )
    assert paths == sorted(shards.glob("*.parquet"))
    assert len(payload["content_verification"]["shards"]) == 2
    if producer != "register":
        _validate_store_payload_binding(
            payload, artifact, json.loads(contract.read_text()), file_identity(contract),
        )


def test_unhashed_registration_does_not_claim_verification(tmp_path, source):
    """A metadata-only inventory must not acquire a content-verification seal."""
    shards, _ = source
    output = tmp_path / "unhashed"
    result = store.register_embedding_store(
        store_root=shards, output_dir=output, encoder={"name": "dummy"}, hash_content=False,
    )
    assert "content_verification" not in result["payload"]
    with pytest.raises(ValueError, match="content-verification seal"):
        _verified_embedding_store(output / "embedding_store.json", content_validation="sealed_inventory")


def test_register_cli_emits_default_seal(tmp_path, source):
    """The documented CLI registers dummy shards ready for default validation."""
    shards, contract = source
    output = tmp_path / "registered"
    result = subprocess.run([
        sys.executable, "-m", "nvidia_tao_ds.mining.dinov3.internal.refinement", "register-store",
        "--store-root", str(shards), "--output-dir", str(output),
        "--encoder-name", "dummy", "--encoder-checkpoint-digest", "sha256:" + "a" * 64,
        "--input-resolution", "16", "--normalization", "imagenet",
        "--source-payload-contract", str(contract),
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload, artifact, _, _ = _verified_embedding_store(
        output / "embedding_store.json", content_validation="sealed_inventory",
    )
    _validate_store_payload_binding(
        payload, artifact, json.loads(contract.read_text()), file_identity(contract),
    )


def test_reregister_preserves_content_identity_but_refreshes_seal(tmp_path, source):
    """Unchanged bytes retain semantic identity while stale POSIX bindings fail."""
    shards, contract = source
    first = store.register_embedding_store(
        store_root=shards, output_dir=tmp_path / "first", encoder={"name": "dummy"},
        source_payload_contract=contract,
    )
    shard = shards / "0.parquet"
    stat = shard.stat()
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    with pytest.raises(ValueError, match="differs from its content seal"):
        _verified_embedding_store(tmp_path / "first/embedding_store.json", content_validation="sealed_inventory")
    second = store.register_embedding_store(
        store_root=shards, output_dir=tmp_path / "second", encoder={"name": "dummy"},
        source_payload_contract=contract,
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert first["audit_id"] != second["audit_id"]
    assert (first["payload"]["content_verification"]["shard_seal_digest"] ==
            second["payload"]["content_verification"]["shard_seal_digest"])
    _verified_embedding_store(tmp_path / "second/embedding_store.json", content_validation="sealed_inventory")


@pytest.mark.parametrize("mutation", ["replace", "inplace"])
def test_same_size_changed_shard_is_rejected(tmp_path, source, mutation):
    """Restoring mtime does not hide changed bytes from the inode/ctime seal."""
    shards, contract = source
    output = tmp_path / "registered"
    store.register_embedding_store(
        store_root=shards, output_dir=output, encoder={"name": "dummy"}, source_payload_contract=contract,
    )
    shard = shards / "0.parquet"
    stat = shard.stat()
    content = bytearray(shard.read_bytes())
    content[-1] ^= 1
    if mutation == "replace":
        replacement = shards / "replacement"
        replacement.write_bytes(content)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(shard)
        assert shard.stat().st_ino != stat.st_ino
    else:
        # Exercise ctime even on supported filesystems with one-second timestamps.
        time.sleep(1.1)
        shard.write_bytes(content)
        os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert shard.stat().st_ctime_ns != stat.st_ctime_ns
    with pytest.raises(ValueError, match="differs from its content seal"):
        _verified_embedding_store(output / "embedding_store.json", content_validation="sealed_inventory")
    with pytest.raises(ValueError, match="digest changed"):
        _verified_embedding_store(output / "embedding_store.json", content_validation="full_sha256")


def test_registration_hashes_each_shard_once(tmp_path, source, monkeypatch):
    """Seal generation reuses verified hashes rather than rereading the store."""
    shards, contract = source
    original_hash = store.file_sha256
    hashed = []

    def record_hash(path):
        if path.parent == shards:
            hashed.append(path)
        return original_hash(path)

    monkeypatch.setattr(store, "file_sha256", record_hash)
    store.register_embedding_store(
        store_root=shards, output_dir=tmp_path / "registered", encoder={"name": "dummy"},
        source_payload_contract=contract,
    )
    assert hashed == sorted(shards.glob("*.parquet"))


def test_registration_rejects_mutation_before_sealing(tmp_path, source, monkeypatch):
    """A shard changed during registration must not receive a committed seal."""
    shards, contract = source
    original_hash = store.file_sha256

    def mutate_after_hash(path):
        digest = original_hash(path)
        if path.parent == shards:
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        return digest

    monkeypatch.setattr(store, "file_sha256", mutate_after_hash)
    output = tmp_path / "registered"
    with pytest.raises(RuntimeError, match="changed during registration"):
        store.register_embedding_store(
            store_root=shards, output_dir=output, encoder={"name": "dummy"}, source_payload_contract=contract,
        )
    assert not (output / "_SUCCESS").exists()
    assert not (output / "embedding_store.json").exists()


@pytest.mark.parametrize(("damage", "message"), [
    ("missing", "does not cover every shard"),
    ("duplicate", "must be unique"),
    ("nonobject", "must be an object"),
    ("digest", "valid content-verification seal"),
    ("stat", "differs from its content seal"),
    ("null-stat", "changed: stat.device"),
    ("list-stat", "changed: stat.device"),
])
def test_malformed_seals_fail_closed(tmp_path, source, damage, message):
    """Even a committed manifest cannot bypass seal shape or filesystem checks."""
    shards, contract = source
    registered = store.register_embedding_store(
        store_root=shards, output_dir=tmp_path / "registered", encoder={"name": "dummy"},
        source_payload_contract=contract,
    )
    payload = registered["payload"]
    verification = payload["content_verification"]
    seals = verification["shards"]
    if damage == "missing":
        seals.pop()
    elif damage == "duplicate":
        seals[1] = seals[0]
    elif damage == "nonobject":
        seals[0] = None
    elif damage == "digest":
        verification["shard_seal_digest"] = "sha256:" + "0" * 64
    elif damage in ("null-stat", "list-stat"):
        seals[0]["stat"] = None if damage == "null-stat" else []
    else:
        seals[0]["stat"]["inode"] += 1
    output = tmp_path / "malformed"
    ArtifactManifest(artifact_type="embedding_store", producer={"action": "test"}, inputs=[], payload=payload).commit(
        output, json_payloads={"embedding_store.json": payload},
    )
    with pytest.raises(ValueError, match=message):
        _verified_embedding_store(output / "embedding_store.json", content_validation="sealed_inventory")
