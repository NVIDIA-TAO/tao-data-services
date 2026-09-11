# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve TAO component images without implementing a private platform launcher."""

from pathlib import Path
import os

import yaml


def resolve_image(reference: str, bank: Path | None = None) -> str:
    """Resolve the same images keys used by TAO model and data skills."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Container image must be a nonempty string")
    reference = reference.strip()
    if any(char.isspace() for char in reference):
        raise ValueError("Container image cannot contain whitespace")
    if "/" in reference or ":" in reference:
        return reference
    if bank is None:
        raise ValueError("Symbolic image keys require execution.skill_bank or TAO_SKILL_BANK_PATH; use explicit image references otherwise")
    value = yaml.safe_load((bank / "versions.yaml").read_text())["images"]
    for part in reference.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Unknown TAO image key: {reference}")
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"TAO image key does not resolve to an image: {reference}")
    if not ("/" in value or ":" in value) or any(c.isspace() for c in value):
        raise ValueError(f"TAO image key does not resolve to an image: {reference}")
    return value.strip()


def configure_containers(value: dict) -> None:
    """Bind image choices before config locking; native runners own execution."""
    execution = value["execution"]
    images = execution.get("container_images")
    if images is None:
        if any(action.get("container_image") for action in value["actions"].values()):
            raise ValueError("Action container overrides require execution.container_images")
        return  # Explicit local/test and pre-existing external configurations.
    if execution["backend"] != "external":
        raise ValueError("Container images require the external platform runner")
    if execution.get("capabilities", {}).get("containers") is not True:
        raise ValueError("Container execution requires runner capability containers")
    if not isinstance(images, dict) or set(images) - {"pytorch", "data_services"}:
        raise ValueError("container_images accepts pytorch and data_services only")
    bank_value = execution.get("skill_bank") or os.environ.get("TAO_SKILL_BANK_PATH")
    bank = Path(bank_value) if bank_value else None
    for role in ("pytorch", "data_services"):
        if role not in images:
            raise ValueError(f"container_images.{role} is required")
        images[role] = resolve_image(images[role], bank)
    actions = value["actions"]
    for name, role in (("score", "pytorch"), ("train", "pytorch"),
                       ("data", "data_services"), ("search", "data_services")):
        action = actions[name]
        action["container_image"] = resolve_image(
            action.get("container_image", images[role]), bank)
    evaluate = actions["evaluate"]
    if evaluate.get("command"):
        # A customer evaluator is not necessarily a PyTorch TAO action.
        evaluate["container_image"] = resolve_image(
            evaluate.get("container_image", ""), bank)
    for stage in ("candidate", "rerank"):
        action = actions["search"].get(stage)
        if action is not None:
            action["container_image"] = resolve_image(
                action.get("container_image", actions["search"]["container_image"]), bank)


def stage_image(actions: dict, stage: str) -> str | None:
    """Map internal stages to their model/data owner, including split ANN jobs."""
    name = {"select_targets": "data", "materialize": "data",
            "search_candidates": "search"}.get(stage, stage)
    action = actions.get(name, {})
    child = "candidate" if stage == "search_candidates" else "rerank"
    if name == "search":
        return action.get(child, {}).get("container_image", action.get("container_image"))
    return action.get("container_image")
