# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions."""


def encode_str(s):
    """Encode str to array."""
    return [ord(e) for e in s]


def decode_str(arr):
    """Decode array to str."""
    return ''.join([chr(e) for e in arr])
