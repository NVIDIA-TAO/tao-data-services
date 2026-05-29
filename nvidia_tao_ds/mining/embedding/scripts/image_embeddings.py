# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute image embeddings with CLIP or SigLIP.

Reads a parquet with a ``filepath`` column, produces embeddings
via the chosen model, and writes an output parquet with
``filepath`` and ``embedding`` columns.  Any extra metadata
columns from the input (e.g. ``label``) are preserved so
downstream steps can use them for filtering.
"""

import logging
from os import getenv
from pathlib import Path
from typing import List, Tuple

from accelerate import PartialState
from accelerate.utils import gather_object
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel, SiglipModel, SiglipProcessor
from tqdm import tqdm

from nvidia_tao_ds.config.mining.embedding.image_embeddings import ImageEmbeddingsConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.utils.utils import load_model_from_checkpoint

logger = logging.getLogger(__name__)

TAO_CHECKPOINT_EXTENSIONS = {".pth", ".ckpt"}


def get_input_filepaths(input_parquet: str) -> List[str]:
    """Get the input filepaths from the input parquet.
    Expects the input parquet to have a column `filepath`."""
    df_column = pd.read_parquet(input_parquet, columns=['filepath'])
    # Convert the pandas Series to a Python list
    return df_column['filepath'].tolist()


def get_batches(items, batch_size):
    """Split *items* into fixed-size chunks for batch processing."""
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def is_tao_checkpoint_path(model_path: str) -> bool:
    """Return whether model_path points to a TAO checkpoint file."""
    return Path(model_path).suffix.lower() in TAO_CHECKPOINT_EXTENSIONS


def load_tao_checkpoint_model(
    model_path: str,
    model_config_path: str,
    device: torch.device
) -> CLIPPlModel:
    """Load a TAO checkpoint and its experiment spec."""
    if not model_config_path:
        raise ValueError("model_config_path is required when model_path is a TAO checkpoint")

    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"TAO checkpoint path does not exist: {model_path}"
        )

    config_path = Path(model_config_path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"TAO model_config_path does not exist: {model_config_path}"
        )

    experiment_config = OmegaConf.load(config_path)
    encoder = load_model_from_checkpoint(
        model_path, experiment_config, CLIPPlModel
    )
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def load_pil_images(image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
    """Load image paths as RGB PIL images."""
    images = []
    image_names = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
        image_names.append(image_path)
    return images, image_names


def preprocess_tao_images(image_paths: List[str], preprocess_fn, device: torch.device):
    """Load and preprocess image paths with a TAO preprocessing function."""
    images, image_names = load_pil_images(image_paths)
    if not images:
        return None, image_names

    processed_images = [preprocess_fn(image) for image in images]
    if isinstance(processed_images[0], dict):
        batch = {
            key: torch.stack([image[key] for image in processed_images]).to(device)
            for key in processed_images[0]
        }
    else:
        batch = torch.stack(processed_images).to(device)
    return batch, image_names


def extract_hf_embeddings(model: str, encoder, processor, image_paths: List[str], device):
    """Extract image embeddings from a Hugging Face CLIP/SigLIP model."""
    images, image_names = load_pil_images(image_paths)
    if not images:
        return [], image_names

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        if model == "CLIP":
            image_features = encoder.get_image_features(**inputs)
        elif model == "SigLIP":
            image_features = encoder.get_image_features(**inputs)
        else:
            raise NotImplementedError(f"Embedding model {model} is not valid")
    return image_features.to("cpu").detach().numpy().tolist(), image_names


def extract_tao_embeddings(encoder: CLIPPlModel, batch) -> List[List[float]]:
    """Extract image embeddings from a TAO model batch."""
    if batch is None:
        return []

    with torch.no_grad():
        image_features = encoder.model(image=batch)
    if isinstance(image_features, dict):
        image_features = image_features["image_features"]
    else:
        image_features = image_features[0]
    return image_features.to("cpu").detach().numpy().tolist()


spec_root = Path(__file__).resolve().parent


@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="image_embeddings",
    schema=ImageEmbeddingsConfig
)
def main(cfg: ImageEmbeddingsConfig) -> None:
    """Takes in an input Parquet file with a `filepath` column and computes
    an embedding specified by the model parameter. It then stores
    the embedding in a new Parquet with 2 columns: `filepath`
    and `embedding`."""
    _log_level = getattr(logging, getenv("TAO_LOGGING_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    input_parquet = cfg.input_parquet
    output_parquet = cfg.output_parquet
    model = cfg.model
    model_path = cfg.model_path
    model_config_path = cfg.model_config_path
    batch_size = cfg.batch_size

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    input_files = get_input_filepaths(input_parquet)
    logger.info("Total images to process: %d", len(input_files))

    data_loader = get_batches(input_files, int(batch_size))

    distributed_state = PartialState()
    use_tao_checkpoint = is_tao_checkpoint_path(model_path)

    if use_tao_checkpoint:
        encoder = load_tao_checkpoint_model(
            model_path, model_config_path, distributed_state.device
        )
        processor = encoder.preprocess_val
    elif model == "CLIP":
        encoder = CLIPModel.from_pretrained(model_path).to(distributed_state.device)
        encoder.eval()
        processor = AutoProcessor.from_pretrained(model_path)
    elif model == "SigLIP":
        encoder = SiglipModel.from_pretrained(model_path).to(distributed_state.device)
        encoder.eval()
        processor = SiglipProcessor.from_pretrained(model_path)
    else:
        msg = f"Embedding model {model} is not valid"
        logger.error(msg)
        raise NotImplementedError(msg)

    output_df = pd.DataFrame(columns=["filepath", "embedding"])

    # Process images in batches, distributing work across GPUs
    # via accelerate.  Each process encodes its shard, then results
    # are gathered on the main process.
    for image_file_paths in tqdm(data_loader, total=len(data_loader)):
        with distributed_state.split_between_processes(image_file_paths) as imgs:
            if use_tao_checkpoint:
                inputs, image_names = preprocess_tao_images(
                    list(imgs), processor, distributed_state.device
                )
                image_embeds = extract_tao_embeddings(encoder, inputs)
            else:
                image_embeds, image_names = extract_hf_embeddings(
                    model, encoder, processor, list(imgs), distributed_state.device
                )

        distributed_state.wait_for_everyone()
        image_embeds = gather_object(image_embeds)
        image_names = gather_object(image_names)

        if distributed_state.is_main_process:
            batch_output_df = pd.DataFrame({
                "filepath": image_names,
                "embedding": image_embeds,
            })
            output_df = pd.concat([output_df, batch_output_df], ignore_index=True)

    if distributed_state.is_main_process:
        # Carry forward any extra metadata columns (e.g. label) from
        # the input parquet so downstream steps can use them.
        input_df = pd.read_parquet(input_parquet)
        extra_cols = [c for c in input_df.columns if c not in ("filepath", "embedding")]
        if extra_cols:
            output_df = output_df.merge(
                input_df[["filepath"] + extra_cols], on="filepath", how="left",
            )
        output_df.to_parquet(output_parquet)


if __name__ == "__main__":
    main()
