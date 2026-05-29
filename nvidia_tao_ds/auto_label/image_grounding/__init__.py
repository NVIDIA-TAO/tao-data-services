# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image grounding-data-engine auto-labeling pipeline.

Generates referring expressions grounded to bounding boxes from
image-caption pairs using a VLM. Implements steps 0 (expression
extraction) and 1 (phrase grounding) of the 2d-data-engine pipeline.
"""
