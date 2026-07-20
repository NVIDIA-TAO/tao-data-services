# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``mining/embedding/scripts/image_embeddings.py``.

Heavy-weight model loads (CLIP / SigLIP from Hugging Face, TAO checkpoints,
``accelerate`` distributed state) are replaced with deterministic stubs so
the tests exercise the script's plumbing without any GPU work, model
downloads, or checkpoint files.
"""

import sys
import types
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Stub nvidia_tao_pytorch.* before importing image_embeddings.
#
# The script imports CLIPPlModel and load_model_from_checkpoint at module
# load time. We don't want a hard dependency on that package in unit tests
# (no model ever gets touched), so we install minimal stand-ins into
# sys.modules here.
# ---------------------------------------------------------------------------


class _StubCLIPPlModel:
    """Stand-in for ``nvidia_tao_pytorch...CLIPPlModel`` (type hint only)."""


def _stub_load_model_from_checkpoint(*args, **kwargs):
    """Default impl. Tests that exercise the success path monkeypatch this."""
    raise AssertionError(
        "load_model_from_checkpoint must be patched in tests."
    )


def _install_pytorch_stub():
    # If the real nvidia_tao_pytorch subtree is importable, use it. This
    # populates sys.modules with the real packages so other tests in the
    # session (e.g. test_text2box.py importing nvidia_tao_pytorch.core)
    # see real packages — not bare ModuleType placeholders.
    try:
        from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (  # noqa: F401
            CLIPPlModel,
        )
        from nvidia_tao_pytorch.multimodal.clip.utils.utils import (  # noqa: F401
            load_model_from_checkpoint,
        )
        return
    except ImportError:
        pass

    pkgs = [
        "nvidia_tao_pytorch",
        "nvidia_tao_pytorch.multimodal",
        "nvidia_tao_pytorch.multimodal.clip",
        "nvidia_tao_pytorch.multimodal.clip.model",
        "nvidia_tao_pytorch.multimodal.clip.utils",
    ]
    for name in pkgs:
        sys.modules.setdefault(name, types.ModuleType(name))

    pl_name = "nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model"
    pl_module = types.ModuleType(pl_name)
    pl_module.CLIPPlModel = _StubCLIPPlModel
    sys.modules.setdefault(pl_name, pl_module)

    utils_name = "nvidia_tao_pytorch.multimodal.clip.utils.utils"
    utils_module = types.ModuleType(utils_name)
    utils_module.load_model_from_checkpoint = _stub_load_model_from_checkpoint
    sys.modules.setdefault(utils_name, utils_module)


_install_pytorch_stub()


import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from nvidia_tao_ds.mining.embedding.scripts import image_embeddings as ie
from nvidia_tao_ds.mining.embedding.scripts.image_embeddings import (
    extract_hf_embeddings,
    extract_tao_embeddings,
    load_tao_checkpoint_model,
    main,
    preprocess_tao_images,
)


# ---------------------------------------------------------------------------
# Test stubs
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """Mimics ``transformers.BatchEncoding`` enough for ``**unpack`` + ``.to``."""

    def to(self, device):
        return self


class _FakeProcessor:
    """Stand-in for ``CLIPProcessor`` / ``SiglipProcessor``.

    Returns a ``_FakeBatch`` with deterministic ``pixel_values`` so callers
    can pass it straight to a stub encoder.
    """

    def __call__(self, images, return_tensors=None, **kwargs):
        n = len(list(images))
        return _FakeBatch({"pixel_values": torch.zeros(n, 3, 4, 4)})


class _FakeEncoder:
    """Stand-in for ``CLIPModel`` / ``SiglipModel``.

    ``get_image_features`` returns a deterministic per-row tensor so tests
    can assert exact embedding values without model weights.
    """

    def __init__(self, embed_dim=4):
        self.embed_dim = embed_dim

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, **kwargs):
        n = kwargs["pixel_values"].shape[0]
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


def _make_image(path, size=(4, 4), color=(255, 0, 0), mode="RGB"):
    Image.new(mode, size, color=color).save(path)


# ---------------------------------------------------------------------------
# model_path routing
# ---------------------------------------------------------------------------


class TestModelPathRouting:
    """Tests for deciding whether model_path is a TAO checkpoint."""

    @pytest.mark.parametrize("model_path", ["model.pth", "nested/model.CKPT"])
    def test_tao_checkpoint_extensions(self, model_path):
        assert ie.is_tao_checkpoint_path(model_path)

    @pytest.mark.parametrize(
        "model_path",
        ["openai/clip-vit-base-patch32", "google/siglip-base-patch16-224", "model.pt"],
    )
    def test_non_tao_checkpoint_paths(self, model_path):
        assert not ie.is_tao_checkpoint_path(model_path)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessTaoImages:
    """Tests for the TAO preprocessing batcher (uses stub preprocess_fn)."""

    def test_empty_returns_none_batch(self):
        batch, names = preprocess_tao_images(
            [], lambda x: x, torch.device("cpu"),
        )
        assert batch is None
        assert names == []

    def test_dict_outputs_stacked_per_key(self, tmp_path):
        paths = []
        for i in range(2):
            p = tmp_path / f"img_{i}.png"
            _make_image(p)
            paths.append(str(p))

        def fn(_image):
            return {
                "pixel_values": torch.ones(3, 4, 4),
                "mask": torch.zeros(4, 4),
            }

        batch, names = preprocess_tao_images(paths, fn, torch.device("cpu"))
        assert isinstance(batch, dict)
        assert batch["pixel_values"].shape == (2, 3, 4, 4)
        assert batch["mask"].shape == (2, 4, 4)
        assert names == paths


# ---------------------------------------------------------------------------
# extract_hf_embeddings / extract_tao_embeddings
# ---------------------------------------------------------------------------


class TestExtractHfEmbeddings:
    """Tests for ``extract_hf_embeddings`` against stub encoder/processor."""

    @pytest.mark.parametrize("model", ["CLIP", "SigLIP"])
    def test_success(self, tmp_path, model):
        paths = []
        for i in range(2):
            path = tmp_path / f"img_{i}.png"
            _make_image(path)
            paths.append(str(path))

        embeds, names = extract_hf_embeddings(
            model, _FakeEncoder(embed_dim=2), _FakeProcessor(),
            paths, torch.device("cpu"),
        )

        assert embeds == [[0.0, 1.0], [2.0, 3.0]]
        assert names == paths

    def test_invalid_model_raises(self, tmp_path):
        path = tmp_path / "img.png"
        _make_image(path)
        with pytest.raises(NotImplementedError):
            extract_hf_embeddings(
                "DINO", _FakeEncoder(), _FakeProcessor(),
                [str(path)], torch.device("cpu"),
            )

    def test_empty_input(self):
        embeds, names = extract_hf_embeddings(
            "CLIP", _FakeEncoder(), _FakeProcessor(), [], torch.device("cpu"),
        )
        assert embeds == []
        assert names == []


class TestExtractTaoEmbeddings:
    """Tests for ``extract_tao_embeddings`` against stub TAO encoder."""

    class _DictEncoder:
        def model(self, image):
            n = image.shape[0]
            return {
                "image_features": torch.arange(
                    n * 3, dtype=torch.float32,
                ).reshape(n, 3),
            }

    class _TupleEncoder:
        def model(self, image):
            n = image.shape[0]
            features = torch.arange(
                n * 3, dtype=torch.float32,
            ).reshape(n, 3)
            return (features, "extra")

    def test_none_batch_returns_empty(self):
        encoder = self._DictEncoder()
        assert extract_tao_embeddings(encoder, None) == []

    def test_tuple_output_extraction(self):
        encoder = self._TupleEncoder()
        embeds = extract_tao_embeddings(encoder, torch.zeros(2, 3, 4, 4))
        assert embeds == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


# ---------------------------------------------------------------------------
# load_tao_checkpoint_model
# ---------------------------------------------------------------------------


class TestLoadTaoCheckpointModel:
    """Error paths and the success path (with stubbed loader)."""

    def test_success_loads_config_and_sets_eval(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"weights")
        cfg_path = tmp_path / "model_config.yaml"
        cfg_path.write_text("model:\n  type: tao_clip\n")

        class _LoadedModel:
            moved_to = None
            eval_called = False

            def to(self, device):
                self.moved_to = device
                return self

            def eval(self):
                self.eval_called = True
                return self

        loaded_model = _LoadedModel()
        captured = {}

        def _load_model_from_checkpoint(model_path, experiment_config, model_class):
            captured["model_path"] = model_path
            captured["experiment_config"] = experiment_config
            captured["model_class"] = model_class
            return loaded_model

        monkeypatch.setattr(
            ie, "load_model_from_checkpoint", _load_model_from_checkpoint,
        )

        result = load_tao_checkpoint_model(
            str(ckpt), str(cfg_path), torch.device("cpu"),
        )

        assert result is loaded_model
        assert loaded_model.moved_to == torch.device("cpu")
        assert loaded_model.eval_called
        assert captured["model_path"] == str(ckpt)
        assert captured["experiment_config"].model.type == "tao_clip"
        assert captured["model_class"] is ie.CLIPPlModel

    def test_missing_config_path_raises(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"")
        with pytest.raises(ValueError, match="model_config_path is required"):
            load_tao_checkpoint_model(str(ckpt), "", torch.device("cpu"))

    def test_missing_checkpoint_raises(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("model: {}\n")
        with pytest.raises(FileNotFoundError, match="TAO checkpoint path"):
            load_tao_checkpoint_model(
                str(tmp_path / "missing.pth"),
                str(cfg_path),
                torch.device("cpu"),
            )

    def test_missing_config_file_raises(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="TAO model_config_path"):
            load_tao_checkpoint_model(
                str(ckpt),
                str(tmp_path / "missing.yaml"),
                torch.device("cpu"),
            )

# ---------------------------------------------------------------------------
# main() — end-to-end with stubbed model loaders
# ---------------------------------------------------------------------------


class TestMain:
    """End-to-end tests of ``main()`` via ``main.__wrapped__``.

    All heavy machinery (HF model/processor loads, TAO checkpoint load,
    accelerate distributed state) is replaced with deterministic stubs.
    """

    def _patch_distributed(self, monkeypatch):
        monkeypatch.setattr(
            ie, "PartialState", lambda: _FakeDistributedState(),
        )
        monkeypatch.setattr(ie, "gather_object", _identity_gather_object)

    def _patch_clip(self, monkeypatch, embed_dim=4):
        monkeypatch.setattr(
            ie, "CLIPModel",
            types.SimpleNamespace(
                from_pretrained=lambda p: _FakeEncoder(embed_dim=embed_dim),
            ),
        )
        monkeypatch.setattr(
            ie, "AutoProcessor",
            types.SimpleNamespace(from_pretrained=lambda p: _FakeProcessor()),
        )

    def _patch_siglip(self, monkeypatch, embed_dim=4):
        monkeypatch.setattr(
            ie, "SiglipModel",
            types.SimpleNamespace(
                from_pretrained=lambda p: _FakeEncoder(embed_dim=embed_dim),
            ),
        )
        monkeypatch.setattr(
            ie, "SiglipProcessor",
            types.SimpleNamespace(from_pretrained=lambda p: _FakeProcessor()),
        )

    def _make_input_parquet(self, tmp_path, n=3, with_label=False):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        paths = []
        for i in range(n):
            p = img_dir / f"img_{i}.png"
            _make_image(p)
            paths.append(str(p))
        df = {"filepath": paths}
        if with_label:
            df["label"] = ["A", "B", "C", "D", "E"][:n]
        parquet = tmp_path / "input.parquet"
        pd.DataFrame(df).to_parquet(parquet)
        return str(parquet), paths

    def _make_cfg(
        self, input_parquet, output_parquet,
        model="CLIP", model_path="openai/clip-vit-base-patch32",
        model_config_path="", batch_size=2,
    ):
        return OmegaConf.create({
            "input_parquet": input_parquet,
            "output_parquet": output_parquet,
            "model": model,
            "model_path": model_path,
            "model_config_path": model_config_path,
            "batch_size": batch_size,
        })

    def test_extra_metadata_columns_preserved(self, tmp_path, monkeypatch):
        """Input columns beyond ``filepath`` (e.g. ``label``) flow through."""
        self._patch_distributed(monkeypatch)
        self._patch_clip(monkeypatch)
        input_parquet, _ = self._make_input_parquet(
            tmp_path, n=3, with_label=True,
        )
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(input_parquet, str(output_parquet))
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert "label" in df.columns
        assert sorted(df["label"].tolist()) == ["A", "B", "C"]

    def test_invalid_model_raises(self, tmp_path, monkeypatch):
        self._patch_distributed(monkeypatch)
        input_parquet, _ = self._make_input_parquet(tmp_path, n=1)
        cfg = self._make_cfg(
            input_parquet, str(tmp_path / "out.parquet"),
            model="DINO", model_path="some/hf/path",
        )
        with pytest.raises(NotImplementedError):
            main.__wrapped__(cfg)

    def test_siglip_hf_path(self, tmp_path, monkeypatch):
        """``main`` routes non-checkpoint SigLIP model_path through HF SigLIP."""
        self._patch_distributed(monkeypatch)
        self._patch_siglip(monkeypatch, embed_dim=5)
        input_parquet, _ = self._make_input_parquet(tmp_path, n=2)
        output_parquet = tmp_path / "nested" / "out.parquet"

        cfg = self._make_cfg(
            input_parquet, str(output_parquet),
            model="SigLIP", model_path="google/siglip-base-patch16-224",
        )
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert len(df) == 2
        assert all(len(e) == 5 for e in df["embedding"])

    def test_tao_checkpoint_path(self, tmp_path, monkeypatch):
        """``main`` routes a ``.pth`` model_path through the TAO loader."""
        self._patch_distributed(monkeypatch)

        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"")
        cfg_path = tmp_path / "model_config.yaml"
        cfg_path.write_text("model: {}\n")

        class _PreprocessFn:
            def __call__(self, image):
                return torch.ones(3, 4, 4)

        class _TaoEncoder:
            preprocess_val = _PreprocessFn()

            def model(self, image):
                n = image.shape[0]
                return {
                    "image_features": torch.arange(
                        n * 3, dtype=torch.float32,
                    ).reshape(n, 3),
                }

        monkeypatch.setattr(
            ie, "load_tao_checkpoint_model",
            lambda *a, **kw: _TaoEncoder(),
        )
        input_parquet, _ = self._make_input_parquet(tmp_path, n=2)
        output_parquet = tmp_path / "out.parquet"

        cfg = self._make_cfg(
            input_parquet, str(output_parquet),
            model="CLIP",  # ignored when model_path is .pth
            model_path=str(ckpt), model_config_path=str(cfg_path),
        )
        main.__wrapped__(cfg)

        df = pd.read_parquet(output_parquet)
        assert all(len(e) == 3 for e in df["embedding"])
        assert len(df) == 2
