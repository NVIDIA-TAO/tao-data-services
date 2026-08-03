# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute text embeddings with CLIP or SigLIP.

Reads a parquet with a ``text`` column, produces embeddings via the
chosen model, and writes an output parquet with ``text`` and
``embedding`` columns.  Any extra metadata columns from the input
(e.g. ``label``) are preserved so downstream steps can use them
for filtering.
"""

import logging
from os import getenv
from pathlib import Path
from typing import List, Optional

from accelerate import PartialState
from accelerate.utils import gather_object
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTokenizer
from tqdm import tqdm

from nvidia_tao_ds.config.mining.embedding.text_embeddings import TextEmbeddingsConfig
from nvidia_tao_ds.core.decorators import experimental
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner

logger = logging.getLogger(__name__)


def get_input_texts(input_parquet: str) -> List[str]:
    """Get the input texts from the input parquet.
    Expects the input parquet to have a column `text`."""
    df_column = pd.read_parquet(input_parquet, columns=['text'])
    return df_column['text'].tolist()


def get_batches(items, batch_size):
    """Split *items* into fixed-size chunks for batch processing."""
    return [
        items[i : i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def _bounded_max_length(value) -> Optional[int]:
    """Return a usable tokenizer max length, ignoring HF sentinel values."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value >= 1_000_000:
        return None
    return value


def _resolve_text_max_length(tokenizer, encoder, model_key: str) -> int:
    """Resolve a concrete text length for tensorized batches.

    Some SigLIP/SigLIP2 tokenizers do not advertise model_max_length.
    Passing padding="max_length" without a concrete max_length silently
    disables padding in Transformers, which makes batched tensor creation fail
    when captions have different token counts.
    """
    if model_key == "CLIP":
        return 77

    candidates = [getattr(tokenizer, "model_max_length", None)]
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    candidates.append(init_kwargs.get("model_max_length"))

    config = getattr(encoder, "config", None)
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is None:
            continue
        for attr in ("max_position_embeddings", "max_sequence_length", "seq_length"):
            candidates.append(getattr(cfg, attr, None))

    for candidate in candidates:
        max_length = _bounded_max_length(candidate)
        if max_length is not None:
            return max_length

    # SigLIP/SigLIP2 checkpoints use short text contexts; 64 keeps batches
    # padded/truncated consistently when HF metadata is missing.
    return 64


spec_root = Path(__file__).resolve().parent


@experimental("Text embedding functionality is experimental")
@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="text_embeddings",
    schema=TextEmbeddingsConfig
)
def main(cfg: TextEmbeddingsConfig) -> None:
    """Takes in an input Parquet file with a `text` column and computes
    an embedding specified by the model parameter. It then stores
    the embedding in a new Parquet with 2 columns: `text`
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
    batch_size = cfg.batch_size

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    input_texts = get_input_texts(input_parquet)
    logger.info("Total texts to process: %d", len(input_texts))

    data_loader = get_batches(input_texts, int(batch_size))

    distributed_state = PartialState()

    model_key = str(model).upper().replace("-", "").replace("_", "")
    if model_key == "CLIP":
        encoder = CLIPModel.from_pretrained(model_path).to(distributed_state.device)
        encoder.eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_path)
    elif model_key in {"SIGLIP", "SIGLIP2"}:
        encoder = AutoModel.from_pretrained(model_path).to(distributed_state.device)
        encoder.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        msg = f"Embedding model {model} is not valid"
        logger.error(msg)
        raise NotImplementedError(msg)

    text_max_length = _resolve_text_max_length(tokenizer, encoder, model_key)
    logger.info("Tokenizing text with max_length=%d", text_max_length)

    output_df = pd.DataFrame(columns=["text", "embedding"])

    # Process texts in batches, distributing work across GPUs
    # via accelerate.  Each process encodes its shard, then results
    # are gathered on the main process.
    for text_batch in tqdm(data_loader, total=len(data_loader)):
        with distributed_state.split_between_processes(text_batch) as txts:
            text_list = list(txts)
            text_embeds = []
            # Final small batches can leave some ranks idle; HF tokenizers fail on empty lists.
            if text_list:
                inputs = tokenizer(
                    text=text_list,
                    padding="max_length",
                    truncation=True,
                    max_length=text_max_length,
                    return_tensors="pt",
                ).to(distributed_state.device)
                with torch.no_grad():
                    text_embeds = encoder.get_text_features(**inputs).to("cpu").detach().numpy()

        distributed_state.wait_for_everyone()
        text_embeds = gather_object(text_embeds)
        text_list = gather_object(text_list)

        if distributed_state.is_main_process:
            if len(text_embeds) != len(text_list):
                raise RuntimeError(
                    "Gathered text embedding count does not match gathered text count: "
                    f"{len(text_embeds)} embeddings vs {len(text_list)} texts"
                )
            batch_output_df = pd.DataFrame({
                "text": text_list,
                # list necessary since text_embeds is a 2-D np array
                "embedding": list(text_embeds),
            })
            output_df = pd.concat([output_df, batch_output_df], ignore_index=True)

    if distributed_state.is_main_process:
        # Carry forward any extra metadata columns from the input parquet
        # so downstream steps can use them.  Text strings can repeat, so
        # join positionally rather than by value.
        input_df = pd.read_parquet(input_parquet).reset_index(drop=True)
        extra_cols = [c for c in input_df.columns if c not in ("text", "embedding")]
        if extra_cols:
            output_df = pd.concat(
                [output_df.reset_index(drop=True), input_df[extra_cols]], axis=1,
            )
        output_df.to_parquet(output_parquet)


if __name__ == "__main__":
    main()
