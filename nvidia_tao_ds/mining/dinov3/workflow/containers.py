# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve TAO component images without implementing a private platform launcher."""


def resolve_image(reference: str) -> str:
    """Require an explicit image; Skill Bank resolves its own symbolic keys."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Container image must be a nonempty string")
    reference = reference.strip()
    if any(char.isspace() for char in reference):
        raise ValueError("Container image cannot contain whitespace")
    if "/" not in reference and ":" not in reference:
        raise ValueError("Use an explicit container image, not a Skill Bank symbolic key")
    return reference


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
    capabilities = execution.get("capabilities", {})
    if capabilities.get("containers") is not True:
        raise ValueError("Container execution requires runner capability containers")
    if capabilities.get("shared_filesystem") is not True:
        raise ValueError(
            "Container execution requires runner capability shared_filesystem "
            "for cross-stage DINOv3 artifact paths"
        )
    if not isinstance(images, dict) or set(images) - {"pytorch", "data_services"}:
        raise ValueError("container_images accepts pytorch and data_services only")
    for role in ("pytorch", "data_services"):
        if role not in images:
            raise ValueError(f"container_images.{role} is required")
        images[role] = resolve_image(images[role])
    actions = value["actions"]
    for name, role in (("score", "pytorch"), ("train", "pytorch"),
                       ("data", "data_services"), ("search", "data_services")):
        action = actions[name]
        action["container_image"] = resolve_image(
            action.get("container_image", images[role]))
    evaluate = actions["evaluate"]
    if evaluate.get("command"):
        # A customer evaluator is not necessarily a PyTorch TAO action.
        evaluate["container_image"] = resolve_image(
            evaluate.get("container_image", ""))
    for stage in ("candidate", "rerank"):
        action = actions["search"].get(stage)
        if action is not None:
            action["container_image"] = resolve_image(
                action.get("container_image", actions["search"]["container_image"]))


def stage_image(actions: dict, stage: str) -> str | None:
    """Map internal stages to their model/data owner, including split ANN jobs."""
    name = {"select_targets": "data", "materialize": "data",
            "search_candidates": "search"}.get(stage, stage)
    action = actions.get(name, {})
    child = "candidate" if stage == "search_candidates" else "rerank"
    if name == "search":
        return action.get(child, {}).get("container_image", action.get("container_image"))
    return action.get("container_image")
