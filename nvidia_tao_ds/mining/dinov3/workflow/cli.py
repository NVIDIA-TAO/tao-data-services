#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layered CLI for the concrete DINOv3 SSL DEFT workflow."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import signal


from . import RefinementWorkflow, WorkflowConfig
from .execution import LocalRunner, StageRequest


PACKAGE_ROOT = Path(__file__).resolve().parent
RECIPES = {
    "grit-score": PACKAGE_ROOT / "recipes" / "grit_score.yaml",
    "multi-task-round-robin": PACKAGE_ROOT / "recipes" / "multi_task_round_robin.yaml",
}


def _execute_with_termination_handoff(workflow: RefinementWorkflow) -> dict:
    """Translate wrapper termination into the workflow cancellation protocol."""
    intent = workflow.run_dir / "cancel.requested"
    previous = signal.getsignal(signal.SIGTERM)

    def request_cancellation(_signum, _frame) -> None:
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.touch(exist_ok=True)

    signal.signal(signal.SIGTERM, request_cancellation)
    try:
        return workflow.execute()
    finally:
        signal.signal(signal.SIGTERM, previous)


def _workflow(config_path: str) -> RefinementWorkflow:
    return RefinementWorkflow(WorkflowConfig.from_file(config_path))


def build_parser() -> argparse.ArgumentParser:
    """Build the DS-owned workflow command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Verify the installed DS/PyTorch runtime")
    preflight.add_argument("config", nargs="?", help="Check recipe allocation before scanning datasets")
    preflight.add_argument("--gpu", action="store_true", help="Require a working CUDA allocation")

    init = commands.add_parser("init", help="Write a reviewed starter recipe")
    init.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    init.add_argument("--output", required=True)

    for name in ("validate", "plan", "adopt-training", "run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("config")
    for name in ("status", "cancel", "report"):
        command = commands.add_parser(name)
        command.add_argument("run_dir")
        command.add_argument("--config")
    logs = commands.add_parser("logs")
    logs.add_argument("run_dir")
    logs.add_argument("client_job_id")
    logs.add_argument("--cursor")
    logs.add_argument("--config")
    return parser


def _config_for_run(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    resolved = Path(args.run_dir) / "input.yaml"
    if not resolved.is_file():
        raise ValueError(f"Run has no input.yaml: {args.run_dir}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Execute one workflow command in the installed TAO runtime."""
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        if args.config:
            config = WorkflowConfig.from_file(args.config)
            if config.value["execution"]["backend"] == "local":
                runner = LocalRunner(config.run_dir / "jobs")
                for name, action in config.value["actions"].items():
                    if not action.get("command"):
                        continue
                    runner.preflight(StageRequest(
                        client_job_id="preflight", run_id="preflight", round_index=0,
                        stage=name, command=action["command"], workdir=".", results_dir=".",
                        resources=action.get("resources", {}),
                        environment=config.value["execution"].get("environment", {}),
                        execution_contract={"required_capabilities": (
                            ["gpu_faiss"] if name == "score" and
                            config.value["execution"].get("capabilities", {}).get("gpu_faiss") else []
                        )},
                    ))
        modules = (
            "nvidia_tao_ds.mining.dinov3.internal.refinement",
            "nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli",
            "nvidia_tao_pytorch.ssl.dinov3.scripts.grit_score",
            "nvidia_tao_pytorch.ssl.dinov3.scripts.train",
        )
        for module in modules:
            importlib.import_module(module)
        if args.gpu:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("DINOv3 SSL DEFT requires an allocated CUDA device")
            # Module presence and tensor allocation do not prove DINOv3 kernels work.
            layer = torch.nn.Conv2d(3, 384, kernel_size=16, stride=16).cuda()
            images = torch.randn(2, 3, 256, 256, device="cuda")
            for dtype in (torch.bfloat16, torch.float16):
                layer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    output = layer(images)
                    loss = output.float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
        versions = {}
        for name in ("nvidia-tao-ds", "nvidia-tao-pytorch", "nvidia-tao-core"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "source-overlay"
        print(json.dumps({"modules": list(modules), "versions": versions,
                          "cuda_verified": args.gpu}, indent=2, sort_keys=True))
        return 0
    if args.command == "init":
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"Refusing to overwrite {destination}")
        shutil.copyfile(RECIPES[args.recipe], destination)
        print(destination.resolve())
        return 0
    if args.command in {"validate", "plan", "adopt-training", "run", "resume"}:
        workflow = _workflow(args.config)
        if args.command == "validate":
            result = workflow.validate(require_paths=True)
        elif args.command == "plan":
            result = workflow.plan()
        elif args.command == "adopt-training":
            result = workflow.adopt_training()
        else:
            result = _execute_with_termination_handoff(workflow)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    config_path = _config_for_run(args)
    workflow = _workflow(str(config_path))
    if args.command == "status":
        result = workflow.status()
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "cancel":
        print(json.dumps(workflow.cancel(), indent=2, sort_keys=True))
    elif args.command == "logs":
        print(
            json.dumps(
                workflow.logs(args.client_job_id, args.cursor),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        report = Path(args.run_dir) / "report.html"
        if not report.is_file():
            raise ValueError(f"Run report does not exist: {report}")
        print(report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
