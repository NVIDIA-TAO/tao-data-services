# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image referring-data-engine auto-labeling pipeline.

Generates grounding expressions from images with KITTI-format bounding
box labels via a four-step pipeline (region expression, image caption,
grounding expression, and optional double-check verification).
"""
