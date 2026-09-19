# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register an existing partitioned embedding store without re-encoding it."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import (
    ArtifactManifest,
    artifact_content_id,
    canonical_digest,
    file_identity,
    file_posix_identity,
    file_sha256,
    require_uncommitted_output,
    shard_inventory_digest,
    vector_matrix,
)
from .materialize import validate_locator_frame


def _load_source_payload_contract(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    contract_path = Path(path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema_version") != "1.0":
        raise ValueError("Source payload contract schema_version must be 1.0")
    if contract.get("immutability") != "immutable":
        raise ValueError("Source payload contract must declare immutable payloads")
    datasets = contract.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Source payload contract requires at least one dataset")
    roots = []
    for dataset in datasets:
        if not isinstance(dataset, dict) or not all(
            isinstance(dataset.get(name), str) and dataset[name].strip()
            for name in ("dataset_id", "version", "root_uri")
        ):
            raise ValueError(
                "Each source dataset requires dataset_id, version, and root_uri"
            )
        parsed = urlparse(dataset["root_uri"])
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("Source dataset roots must be local file:// URIs")
        root_text = unquote(parsed.path)
        root = Path(root_text)
        if not root.is_absolute() or str(root) != root_text or ".." in root.parts:
            raise ValueError(
                "Source dataset roots must be canonical absolute paths"
            )
        roots.append(root)
    return (
        contract,
        file_identity(contract_path, role="source_payload_contract"),
        roots,
    )


def _local_locator(value: Any) -> Path:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Source locator is not a local file path: {text}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(text)
    if not path.is_absolute():
        raise ValueError(f"Source locator is not absolute: {text}")
    return path.resolve()


def _committed_store(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).resolve()
    artifact_path = manifest_path.with_name("artifact.json")
    success_path = manifest_path.with_name("_SUCCESS")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_id = artifact_content_id(artifact)
    if artifact.get("artifact_type") != "embedding_store":
        raise ValueError("Source artifact is not an embedding store")
    if artifact.get("artifact_id") != expected_id:
        raise ValueError("Source embedding-store artifact identity is invalid")
    if artifact.get("payload") != payload:
        raise ValueError("Source embedding-store manifest differs from its artifact")
    if success_path.read_text(encoding="utf-8").strip() != expected_id:
        raise ValueError("Source embedding-store success marker is invalid")
    return payload, artifact


def bind_store_payload_contract(
    *,
    source_store_manifest: str | Path,
    output_dir: str | Path,
    source_payload_contract: str | Path,
) -> dict[str, Any]:
    """Verify and bind a store inventory to immutable payload roots."""
    source_path = Path(source_store_manifest).resolve()
    source, source_artifact = _committed_store(source_path)
    contract, contract_identity, roots = _load_source_payload_contract(
        source_payload_contract
    )
    parsed = urlparse(str(source.get("root_uri", "")))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Source embedding store requires a local file:// root")
    store_root = Path(unquote(parsed.path)).resolve()
    shards = source.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Source embedding store has no shard inventory")
    if int(source.get("shard_count", -1)) != len(shards):
        raise ValueError("Source embedding store shard_count is invalid")
    if source.get("inventory_digest") not in {
        shard_inventory_digest(shards),
        canonical_digest(shards),  # pre-v2 compatibility
    }:
        raise ValueError("Source embedding store inventory digest is invalid")
    relative_paths: set[str] = set()
    declared_rows = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"Embedding shard {index} must be an object")
        relative_text = str(shard.get("relative_path", ""))
        relative = Path(relative_text)
        if any(
            (
                not relative_text,
                relative.is_absolute(),
                ".." in relative.parts,
                relative.as_posix() in relative_paths,
            )
        ):
            raise ValueError(f"Invalid or duplicate embedding shard: {relative_text}")
        relative_paths.add(relative.as_posix())
        for field in ("bytes", "rows"):
            value = shard.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Embedding shard {relative_text} has invalid {field}"
                )
        digest = str(shard.get("sha256", ""))
        digest = digest.removeprefix("sha256:")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"Embedding shard {relative_text} has invalid SHA-256"
            )
        declared_rows += int(shard["rows"])
    if declared_rows != int(source.get("row_count", -1)):
        raise ValueError("Source embedding store row_count is invalid")
    locator_count = 0
    content_seals = []
    resolved_roots = tuple(root.resolve() for root in roots)
    for shard in shards:
        relative = Path(str(shard["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Embedding shard escapes its root: {relative}")
        path = (store_root / relative).resolve()
        if not path.is_relative_to(store_root) or not path.is_file():
            raise ValueError(f"Embedding shard is missing or escapes its root: {path}")
        if path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"Embedding shard size changed: {path}")
        before = path.stat()
        actual_sha256 = file_sha256(path)
        after = path.stat()
        stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stat_fields):
            raise ValueError(f"Embedding shard changed while hashing: {path}")
        declared_sha256 = str(shard["sha256"])
        if not declared_sha256.startswith("sha256:"):
            declared_sha256 = f"sha256:{declared_sha256}"
        if actual_sha256 != declared_sha256:
            raise ValueError(f"Embedding shard digest changed: {path}")
        content_seals.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": after.st_size,
                "sha256": actual_sha256,
                "stat": {
                    "device": after.st_dev,
                    "inode": after.st_ino,
                    "mtime_ns": after.st_mtime_ns,
                    "ctime_ns": after.st_ctime_ns,
                },
            }
        )
        parquet = pq.ParquetFile(path)
        if int(parquet.metadata.num_rows) != int(shard["rows"]):
            raise ValueError(f"Embedding shard row count changed: {path}")
        columns = set(parquet.schema_arrow.names)
        selected = ["path"]
        if "storage_type" in columns:
            selected.append("storage_type")
        if "member" in columns:
            selected.append("member")
        for batch in parquet.iter_batches(batch_size=65536, columns=selected):
            frame = batch.to_pandas()
            locators = frame["path"]
            if locators.isnull().any():
                raise ValueError(f"Source shard contains null locators: {path}")
            locator_text = locators.astype(str)
            for locator in locator_text:
                resolved_locator = _local_locator(locator)
                if not any(
                    resolved_locator.is_relative_to(root) for root in resolved_roots
                ):
                    raise ValueError(
                        "Source locator resolves outside every contracted root: "
                        f"{locator}"
                    )
            if "storage_type" in frame:
                storage_types = frame["storage_type"].fillna("").astype(str)
                if not storage_types.eq("file").all():
                    invalid = storage_types[~storage_types.eq("file")].iloc[0]
                    raise ValueError(
                        "Source payload contracts currently require file locators; "
                        f"found {invalid!r}"
                    )
            elif source.get("locator_defaults", {}).get("storage_type") != "file":
                raise ValueError(
                    "Source payload contracts currently require file locators"
                )
            locator_count += len(frame)
        final_stat = path.stat()
        if any(
            getattr(after, name) != getattr(final_stat, name)
            for name in stat_fields
        ):
            raise ValueError(f"Embedding shard changed during locator audit: {path}")
    if locator_count != int(source.get("row_count", -1)):
        raise ValueError("Locator audit row count differs from the store manifest")
    payload = deepcopy(source)
    payload["content_verification"] = {
        "algorithm": "sha256_each_shard_with_posix_stat_v1",
        "inventory_digest": source["inventory_digest"],
        "shard_seal_digest": canonical_digest([
            {name: seal[name] for name in ("relative_path", "bytes", "sha256")}
            for seal in content_seals
        ]),
        "shards": content_seals,
    }
    payload["source_payload_contract"] = {
        "sha256": contract_identity["sha256"],
        "datasets_digest": canonical_digest(contract["datasets"]),
        "locator_audit": {
            "algorithm": "sealed_shard_inventory_and_canonical_root_prefix_v1",
            "inventory_digest": source["inventory_digest"],
            "row_count": locator_count,
        },
    }
    payload["parent_store_artifact_id"] = source_artifact["artifact_id"]
    destination = require_uncommitted_output(output_dir)
    artifact = ArtifactManifest(
        artifact_type="embedding_store",
        producer={
            "action": "bind_store_payload_contract",
            "version": "1.0",
            "implementation_sha256": file_sha256(Path(__file__)),
        },
        inputs=[
            {
                **file_identity(source_path, role="source_embedding_store"),
                "artifact_id": source_artifact["artifact_id"],
            },
            contract_identity,
        ],
        payload=payload,
    )
    artifact.commit(
        destination, json_payloads={"embedding_store.json": payload}
    )
    return artifact.to_dict()


def register_embedding_store(
    *,
    store_root: str | Path,
    output_dir: str | Path,
    encoder: dict[str, Any],
    embedding_column: str = "embedding",
    id_column: str = "sample_id",
    hash_content: bool = True,
    default_storage_type: str | None = None,
    source_payload_contract: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory Parquet shards and publish a versioned store manifest."""
    root = Path(store_root).resolve()
    parts = sorted(root.rglob("*.parquet"))
    if not parts:
        raise ValueError(f"No Parquet shards found under {root}")
    payload_contract = None
    payload_contract_identity = None
    payload_roots: list[Path] = []
    if source_payload_contract is not None:
        if not hash_content:
            raise ValueError("Source payload contracts require hash_content=True")
        (
            payload_contract,
            payload_contract_identity,
            payload_roots,
        ) = _load_source_payload_contract(source_payload_contract)
    resolved_payload_roots = [root.resolve() for root in payload_roots]

    destination = require_uncommitted_output(output_dir)
    identity_audit = destination / ".sample_ids.blake2b128.tmp"
    locator_hasher = hashlib.sha256()
    locator_count = 0
    locator_root_cache: dict[str, bool] = {}
    shards = []
    total_rows = 0
    embedding_dim: int | None = None
    try:
        with identity_audit.open("wb") as identity_stream:
            for path in parts:
                identity_before = file_posix_identity(path)
                parquet = pq.ParquetFile(path)
                names = set(parquet.schema_arrow.names)
                required = {id_column, embedding_column, "path"}
                if default_storage_type is None:
                    required.add("storage_type")
                missing = required.difference(names)
                if missing:
                    raise ValueError(f"{path} is missing columns: {sorted(missing)}")
                columns = list(required)
                if "storage_type" in names and "storage_type" not in columns:
                    columns.append("storage_type")
                if "member" in names:
                    columns.append("member")
                for batch in parquet.iter_batches(
                    batch_size=65536, columns=columns
                ):
                    frame = batch.to_pandas()
                    if "storage_type" not in frame:
                        frame["storage_type"] = default_storage_type
                    validate_locator_frame(frame, label=str(path))
                    if frame[id_column].isnull().any():
                        raise ValueError(f"{path} contains null sample IDs")
                    if payload_contract is not None:
                        members = (
                            frame["member"]
                            if "member" in frame
                            else [None] * len(frame)
                        )
                        for storage_type, locator, member in zip(
                            frame["storage_type"], frame["path"], members
                        ):
                            locator_text = str(locator)
                            belongs = locator_root_cache.get(locator_text)
                            if belongs is None:
                                resolved = _local_locator(locator_text)
                                belongs = any(
                                    resolved.is_relative_to(payload_root)
                                    for payload_root in resolved_payload_roots
                                )
                                locator_root_cache[locator_text] = belongs
                            if not belongs:
                                raise ValueError(
                                    "Source locator is outside every contracted root: "
                                    f"{locator_text}"
                                )
                            locator_record = {
                                "storage_type": str(storage_type),
                                "path": locator_text,
                                "member": (
                                    None if pd.isna(member) else str(member)
                                ),
                            }
                            encoded_locator = json.dumps(
                                locator_record,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            locator_hasher.update(
                                encoded_locator + b"\n"
                            )
                            locator_count += 1
                    vectors = vector_matrix(
                        frame[embedding_column], label=f"{path} embedding"
                    )
                    current_dim = int(vectors.shape[1])
                    if embedding_dim is None:
                        embedding_dim = current_dim
                    elif embedding_dim != current_dim:
                        raise ValueError(
                            f"Embedding dimension changed from {embedding_dim} to "
                            f"{current_dim} in {path}"
                        )
                    for value in frame[id_column]:
                        identity_stream.write(
                            hashlib.blake2b(
                                str(value).encode("utf-8"), digest_size=16
                            ).digest()
                        )
                rows = int(parquet.metadata.num_rows)
                if rows <= 0:
                    raise ValueError(f"Embedding shard is empty: {path}")
                total_rows += rows
                content_sha256 = file_sha256(path) if hash_content else None
                identity_after = file_posix_identity(path)
                if identity_before != identity_after:
                    raise RuntimeError(
                        f"Embedding shard changed during registration: {path}"
                    )
                shard = {
                    "relative_path": str(path.relative_to(root)),
                    "bytes": int(identity_after["bytes"]),
                    "rows": rows,
                    "posix_identity": identity_after,
                }
                if hash_content:
                    shard["sha256"] = content_sha256
                shards.append(shard)

        identities = np.memmap(identity_audit, mode="r+", dtype="V16")
        identities.sort()
        if len(identities) > 1 and np.any(identities[1:] == identities[:-1]):
            raise ValueError("Embedding store contains duplicate sample IDs")
        del identities
    finally:
        identity_audit.unlink(missing_ok=True)

    payload = {
        "root_uri": root.as_uri(),
        "id_column": id_column,
        "embedding_column": embedding_column,
        "encoder": encoder,
        "shards": shards,
        "shard_count": len(shards),
        "row_count": total_rows,
        "embedding_dim": embedding_dim,
        "locator_schema": "storage_type_path_member_v1",
        "locator_defaults": (
            {"storage_type": default_storage_type}
            if default_storage_type is not None
            else {}
        ),
        "identity_contract": "globally_unique_blake2b128_audit_v1",
        "embedding_contract": "finite_nonzero_cosine_v1",
        "fingerprint_method": "sha256" if hash_content else "parquet_inventory_v1",
        "inventory_digest": shard_inventory_digest(shards),
    }
    inputs = []
    if payload_contract is not None and payload_contract_identity is not None:
        inputs.append(payload_contract_identity)
        payload["source_payload_contract"] = {
            "sha256": payload_contract_identity["sha256"],
            "datasets_digest": canonical_digest(payload_contract["datasets"]),
            "locator_audit": {
                "algorithm": "sha256_canonical_jsonl_v1",
                "digest": "sha256:" + locator_hasher.hexdigest(),
                "row_count": locator_count,
            },
        }
    artifact = ArtifactManifest(
        artifact_type="embedding_store",
        producer={
            "action": "register_embedding_store",
            "version": "1.0",
            "implementation_sha256": file_sha256(Path(__file__)),
        },
        inputs=inputs,
        payload=payload,
    )
    artifact.commit(
        destination, json_payloads={"embedding_store.json": payload}
    )
    return artifact.to_dict()
