# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``mining/embedding/scripts/text_embeddings.py``.

Heavy-weight model loads (CLIP / SigLIP from Hugging Face, ``accelerate``
distributed state) are replaced with deterministic stubs so the tests
exercise the script's plumbing without any GPU work or model downloads.
"""

import sys
import types
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_ds.mining.embedding.scripts import text_embeddings as te
from nvidia_tao_ds.mining.embedding.scripts.text_embeddings import (
    _bounded_max_length,
    _resolve_text_max_length,
    get_batches,
    get_input_texts,
    main,
)


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _FakeTokenizerOutput(dict):
    """Mimics ``transformers.BatchEncoding`` (dict-like, supports ``.to``)."""

    def to(self, device):
        return self


class _FakeTokenizer:
    """Stand-in for ``CLIPTokenizer`` / ``AutoTokenizer``.

    Records the last call so tests can assert on tokenizer arguments, and
    returns a ``_FakeTokenizerOutput`` with deterministic ``input_ids``.
    """

    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length
        self.last_call_kwargs = {}

    def __call__(self, text, **kwargs):
        self.last_call_kwargs = {"text": text, **kwargs}
        n = len(text)
        max_length = kwargs.get("max_length", 4)
        return _FakeTokenizerOutput(
            {"input_ids": torch.zeros(n, max_length, dtype=torch.long)}
        )


class _FakeEncoder:
    """Stand-in for ``CLIPModel`` / ``AutoModel``.

    ``get_text_features`` returns a deterministic per-row numpy array so
    tests can assert exact embedding values without model weights.
    """

    def __init__(self, embed_dim=4):
        self.embed_dim = embed_dim
        self.config = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_text_features(self, **kwargs):
        n = kwargs["input_ids"].shape[0]
        return torch.arange(
            n * self.embed_dim, dtype=torch.float32,
        ).reshape(n, self.embed_dim)


class _FakeDistributedState:
    """Stub of ``accelerate.PartialState`` for non-distributed test runs."""

    is_main_process = True
    device = torch.device("cpu")

    @contextmanager
    def split_between_processes(self, items):
        yield items

    def wait_for_everyone(self):
        pass


def _identity_gather_object(obj):
    """``accelerate.utils.gather_object`` is a no-op without distribution."""
    return obj


# ---------------------------------------------------------------------------
# get_input_texts
# ---------------------------------------------------------------------------


class TestGetInputTexts:
    def test_reads_text_column(self, tmp_path):
        parquet = tmp_path / "input.parquet"
        pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]}).to_parquet(parquet)
        result = get_input_texts(str(parquet))
        assert result == ["hello", "world"]

    def test_returns_list(self, tmp_path):
        parquet = tmp_path / "input.parquet"
        pd.DataFrame({"text": ["a"]}).to_parquet(parquet)
        assert isinstance(get_input_texts(str(parquet)), list)

    def test_empty_parquet(self, tmp_path):
        parquet = tmp_path / "input.parquet"
        pd.DataFrame({"text": []}).to_parquet(parquet)
        assert get_input_texts(str(parquet)) == []


# ---------------------------------------------------------------------------
# get_batches
# ---------------------------------------------------------------------------


class TestGetBatches:
    def test_exact_multiple(self):
        batches = get_batches([1, 2, 3, 4], batch_size=2)
        assert batches == [[1, 2], [3, 4]]

    def test_partial_last_batch(self):
        batches = get_batches([1, 2, 3], batch_size=2)
        assert batches == [[1, 2], [3]]

    def test_batch_larger_than_input(self):
        batches = get_batches([1, 2], batch_size=10)
        assert batches == [[1, 2]]

    def test_empty_input(self):
        assert get_batches([], batch_size=4) == []

    def test_batch_size_one(self):
        assert get_batches(["a", "b", "c"], batch_size=1) == [["a"], ["b"], ["c"]]


# ---------------------------------------------------------------------------
# _bounded_max_length
# ---------------------------------------------------------------------------


class TestBoundedMaxLength:
    @pytest.mark.parametrize("value,expected", [
        (77, 77),
        ("64", 64),
        (1, 1),
        (999_999, 999_999),
    ])
    def test_valid_values(self, value, expected):
        assert _bounded_max_length(value) == expected

    @pytest.mark.parametrize("value", [0, -1, 1_000_000, 2_000_000])
    def test_out_of_range_returns_none(self, value):
        assert _bounded_max_length(value) is None

    @pytest.mark.parametrize("value", [None, "abc", object()])
    def test_non_numeric_returns_none(self, value):
        assert _bounded_max_length(value) is None


# ---------------------------------------------------------------------------
# _resolve_text_max_length
# ---------------------------------------------------------------------------


class TestResolveTextMaxLength:
    def test_clip_always_returns_77(self):
        tokenizer = _FakeTokenizer(model_max_length=999)
        encoder = _FakeEncoder()
        assert _resolve_text_max_length(tokenizer, encoder, "CLIP") == 77

    def test_siglip_uses_tokenizer_model_max_length(self):
        tokenizer = _FakeTokenizer(model_max_length=64)
        encoder = _FakeEncoder()
        assert _resolve_text_max_length(tokenizer, encoder, "SIGLIP") == 64

    def test_siglip_falls_back_to_encoder_config(self):
        tokenizer = _FakeTokenizer(model_max_length=1_000_001)  # sentinel → ignored
        encoder = _FakeEncoder()
        encoder.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(max_position_embeddings=128),
        )
        assert _resolve_text_max_length(tokenizer, encoder, "SIGLIP") == 128

    def test_siglip_default_fallback(self):
        tokenizer = _FakeTokenizer(model_max_length=1_000_001)  # sentinel → ignored
        encoder = _FakeEncoder()  # no config
        assert _resolve_text_max_length(tokenizer, encoder, "SIGLIP") == 64

    def test_siglip_init_kwargs_fallback(self):
        tokenizer = _FakeTokenizer(model_max_length=1_000_001)
        tokenizer.init_kwargs = {"model_max_length": 50}
        encoder = _FakeEncoder()
        assert _resolve_text_max_length(tokenizer, encoder, "SIGLIP") == 50


# ---------------------------------------------------------------------------
# main() — end-to-end with stubbed model loaders
# ---------------------------------------------------------------------------


class TestMain:
    """End-to-end tests of ``main()`` via ``main.__wrapped__``.

    All heavy machinery (HF model/tokenizer loads, accelerate distributed
    state) is replaced with deterministic stubs.
    """

    def _patch_distributed(self, monkeypatch):
        monkeypatch.setattr(te, "PartialState", lambda: _FakeDistributedState())
        monkeypatch.setattr(te, "gather_object", _identity_gather_object)

    def _patch_clip(self, monkeypatch, embed_dim=4):
        monkeypatch.setattr(
            te, "CLIPModel",
            types.SimpleNamespace(
                from_pretrained=lambda p: _FakeEncoder(embed_dim=embed_dim),
            ),
        )
        monkeypatch.setattr(
            te, "CLIPTokenizer",
            types.SimpleNamespace(from_pretrained=lambda p: _FakeTokenizer()),
        )

    def _patch_siglip(self, monkeypatch, embed_dim=4):
        monkeypatch.setattr(
            te, "AutoModel",
            types.SimpleNamespace(
                from_pretrained=lambda p: _FakeEncoder(embed_dim=embed_dim),
            ),
        )
        monkeypatch.setattr(
            te, "AutoTokenizer",
            types.SimpleNamespace(from_pretrained=lambda p: _FakeTokenizer()),
        )

    def _make_input_parquet(self, tmp_path, texts, extra=None):
        df = {"text": texts}
        if extra:
            df.update(extra)
        parquet = tmp_path / "input.parquet"
        pd.DataFrame(df).to_parquet(parquet)
        return str(parquet)

    def _make_cfg(
        self, input_parquet, output_parquet,
        model="CLIP", model_path="openai/clip-vit-base-patch32", batch_size=2,
    ):
        return OmegaConf.create({
            "input_parquet": input_parquet,
            "output_parquet": output_parquet,
            "model": model,
            "model_path": model_path,
            "batch_size": batch_size,
        })

    def test_clip_writes_embedding_column(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch, embed_dim=4)
        input_parquet = self._make_input_parquet(tmp_path, ["hello", "world", "foo"])
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet))
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert "embedding" in df.columns
        assert "text" in df.columns
        assert len(df) == 3
        assert all(len(e) == 4 for e in df["embedding"])

    def test_siglip_routing(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_siglip(monkeypatch, embed_dim=5)
        input_parquet = self._make_input_parquet(tmp_path, ["a", "b"])
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(
            input_parquet, str(output_parquet),
            model="SigLIP", model_path="google/siglip-base-patch16-224",
        )
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert len(df) == 2
        assert all(len(e) == 5 for e in df["embedding"])

    def test_siglip2_routing(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_siglip(monkeypatch, embed_dim=3)
        input_parquet = self._make_input_parquet(tmp_path, ["x"])
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(
            input_parquet, str(output_parquet),
            model="SigLIP2", model_path="google/siglip2-base-patch16-224",
        )
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert len(df) == 1
        assert len(df["embedding"].iloc[0]) == 3

    def test_invalid_model_raises(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        input_parquet = self._make_input_parquet(tmp_path, ["hello"])
        cfg = self._make_cfg(
            input_parquet, str(tmp_path / "out.parquet"),
            model="DINO", model_path="some/hf/path",
        )
        with pytest.raises(NotImplementedError):
            main.__wrapped__(cfg)

    def test_extra_metadata_columns_preserved(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch)
        input_parquet = self._make_input_parquet(
            tmp_path, ["a", "b", "c"], extra={"label": ["X", "Y", "Z"]},
        )
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet))
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert "label" in df.columns
        assert sorted(df["label"].tolist()) == ["X", "Y", "Z"]

    def test_output_directory_created(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch)
        input_parquet = self._make_input_parquet(tmp_path, ["hello"])
        output_parquet = tmp_path / "nested" / "dir" / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet))
        main.__wrapped__(cfg)

        assert output_parquet.exists()

    def test_text_order_preserved(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch, embed_dim=2)
        texts = ["first", "second", "third"]
        input_parquet = self._make_input_parquet(tmp_path, texts)
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet), batch_size=2)
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert df["text"].tolist() == texts

    def test_embedding_values_are_deterministic(self, tmp_path, monkeypatch):
        """Embeddings from _FakeEncoder are arange-based; verify exact values."""
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch, embed_dim=2)
        input_parquet = self._make_input_parquet(tmp_path, ["a", "b"])
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet), batch_size=4)
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        # _FakeEncoder returns arange(n * embed_dim).reshape(n, embed_dim)
        # for 2 texts with embed_dim=2: [[0, 1], [2, 3]]
        assert list(df["embedding"].iloc[0]) == pytest.approx([0.0, 1.0])
        assert list(df["embedding"].iloc[1]) == pytest.approx([2.0, 3.0])
