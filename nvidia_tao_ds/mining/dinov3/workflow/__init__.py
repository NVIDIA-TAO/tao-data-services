# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete DINOv3 SSL DEFT workflow."""

from .config import WorkflowConfig
from .controller import RefinementWorkflow

__all__ = ["RefinementWorkflow", "WorkflowConfig"]
__version__ = "0.1.0"
