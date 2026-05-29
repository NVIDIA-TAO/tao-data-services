# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO tokenization."""

import torch

from nvidia_tao_pytorch.cv.grounding_dino.model.bertwraper import (
    generate_masks_with_special_tokens_and_transfer_map
)


def tokenize_captions(captions, specical_tokens, tokenizer, pad_to_max=False, max_text_len=256):
    """Tokenize the captions through model tokeninzer."""
    if pad_to_max:
        padding = "max_length"
    else:
        padding = "longest"

    tokenized = tokenizer(captions, padding=padding, return_tensors="pt").to(
        "cuda"
    )
    one_hot_token = tokenized

    (
        text_self_attention_masks,
        position_ids,
        _,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, specical_tokens, tokenizer)

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : max_text_len, : max_text_len]

        position_ids = position_ids[:, : max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

    return tokenized, one_hot_token, position_ids, text_self_attention_masks


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except Exception:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except Exception:
                end_pos = None

        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map
