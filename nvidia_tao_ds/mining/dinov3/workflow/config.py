# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level configuration contract for DINOv3 SSL DEFT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .containers import configure_containers


SCHEMA_VERSION = "1.0"
STRATEGIES = {"grit_score", "multi_task_round_robin"}
STOP_REASONS = {
    "metric_patience",
    "max_rounds",
    "no_actionable_targets",
    "no_novel_samples",
    "pool_exhausted",
    "radius_exhausted",
    "search_budget_exhausted",
}
MULTINODE_RUNNER_CAPABILITIES = {
    "gang_scheduling",
    "gang_retry",
    "attempt_scoped_launch_id",
    "shared_filesystem",
}


def canonical_digest(value: Any) -> str:
    """Hash the canonical JSON representation of a value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _required(mapping: dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Missing required configuration field: {path}")
        current = current[part]
    return current


def multitask_budgets(multitask: dict[str, Any]) -> dict[str, int]:
    """Resolve a fixed total target budget into deterministic task quotas."""
    tasks = sorted(map(str, multitask["tasks"]))
    has_total = "targets_per_round" in multitask
    has_legacy = "targets_per_task" in multitask
    if has_total and has_legacy:
        raise ValueError(
            "Configure only multi_task.targets_per_round; targets_per_task is "
            "supported only for legacy recipes"
        )
    total = (
        int(multitask["targets_per_round"])
        if has_total
        else int(multitask.get("targets_per_task", 800)) * len(tasks)
    )
    if total < len(tasks):
        raise ValueError("multi_task target budget must provide at least one per task")
    raw_weights = multitask.get("task_weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("multi_task.task_weights must be a mapping")
    unknown = set(map(str, raw_weights)).difference(tasks)
    if unknown:
        raise ValueError(
            f"multi_task.task_weights contains unknown tasks: {sorted(unknown)}"
        )
    weights = {task: float(raw_weights.get(task, 1.0)) for task in tasks}
    if not all(
        math.isfinite(weight) and weight > 0.0 for weight in weights.values()
    ):
        raise ValueError("multi_task.task_weights must be finite and positive")
    weight_sum = sum(weights.values())
    exact = {task: total * weights[task] / weight_sum for task in tasks}
    budgets = {task: int(math.floor(exact[task])) for task in tasks}
    for task in sorted(tasks, key=lambda item: (-(exact[item] % 1.0), item))[
        : total - sum(budgets.values())
    ]:
        budgets[task] += 1
    for task in tasks:
        if budgets[task] > 0:
            continue
        donor = sorted(
            (candidate for candidate in tasks if budgets[candidate] > 1),
            key=lambda item: (-budgets[item], item),
        )[0]
        budgets[donor] -= 1
        budgets[task] = 1
    return budgets


def training_allocation(
    training: dict[str, Any],
    *,
    manifest_rows: int,
    batch_size_per_gpu: int,
    fixed_nodes: int = 1,
    fixed_gpus_per_node: int = 1,
) -> dict[str, int | str]:
    """Resolve a traceable training allocation from cumulative sample count."""
    if manifest_rows <= 0:
        raise ValueError("Training manifest must contain at least one row")
    if batch_size_per_gpu <= 0:
        raise ValueError("Training batch size per GPU must be positive")
    passes = int(training["passes_per_round"])
    scaling = training.get("node_scaling")
    if scaling is None:
        nodes = int(fixed_nodes)
        gpus_per_node = int(fixed_gpus_per_node)
        mode = "fixed"
        target_updates = 0
    else:
        nodes_allowed = list(map(int, scaling["allowed_nodes"]))
        gpus_per_node = int(scaling["gpus_per_node"])
        target_updates = int(scaling["target_optimizer_updates"])
        nodes = nodes_allowed[0]
        for candidate in nodes_allowed:
            samples_per_rank = math.ceil(
                manifest_rows / (candidate * gpus_per_node)
            )
            steps_per_pass = math.ceil(samples_per_rank / batch_size_per_gpu)
            if steps_per_pass * passes >= target_updates:
                nodes = candidate
        mode = "target_optimizer_updates"
    if nodes <= 0 or gpus_per_node <= 0:
        raise ValueError("Training nodes and GPUs per node must be positive")
    samples_per_rank = math.ceil(manifest_rows / (nodes * gpus_per_node))
    steps_per_pass = math.ceil(samples_per_rank / batch_size_per_gpu)
    return {
        "mode": mode,
        "manifest_rows": manifest_rows,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": nodes * gpus_per_node,
        "batch_size_per_gpu": batch_size_per_gpu,
        "steps_per_pass": steps_per_pass,
        "passes": passes,
        "total_optimizer_steps": steps_per_pass * passes,
        "target_optimizer_updates": target_updates,
        "effective_global_batch": batch_size_per_gpu * nodes * gpus_per_node,
    }


@dataclass(frozen=True)
class WorkflowConfig:
    """Validated resolved configuration, independent of a UI layer."""

    value: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "WorkflowConfig":
        """Load and validate a workflow configuration from YAML."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkflowConfig":
        """Normalize and validate the supplied workflow configuration."""
        value = deepcopy(raw)
        value.setdefault("schema_version", SCHEMA_VERSION)
        if str(value["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={value['schema_version']!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        strategy = str(_required(value, "workflow.strategy"))
        if strategy not in STRATEGIES:
            raise ValueError(f"workflow.strategy must be one of {sorted(STRATEGIES)}")
        workflow = value["workflow"]
        workflow.setdefault("initialization", "base_checkpoint")
        workflow.setdefault("start_round", 1)
        workflow.setdefault("max_rounds", 10)
        workflow.setdefault("persistent_target_rounds", 3)
        if workflow["initialization"] not in {
            "base_checkpoint",
            "parent_history",
        }:
            raise ValueError(
                "workflow.initialization must be base_checkpoint or "
                "parent_history"
            )
        if int(workflow["max_rounds"]) <= 0:
            raise ValueError("workflow.max_rounds must be positive")
        if not 1 <= int(workflow["start_round"]) <= int(workflow["max_rounds"]):
            raise ValueError("workflow.start_round must be within max_rounds")
        if int(workflow["persistent_target_rounds"]) <= 0:
            raise ValueError("workflow.persistent_target_rounds must be positive")
        early_stopping = workflow.get("early_stopping")
        if early_stopping is not None:
            if not isinstance(early_stopping, dict):
                raise ValueError("workflow.early_stopping must be a mapping")
            early_stopping.setdefault("patience", 2)
            early_stopping.setdefault("min_delta", 0.0)
            if int(early_stopping["patience"]) <= 0:
                raise ValueError(
                    "workflow.early_stopping.patience must be positive"
                )
            min_delta = float(early_stopping["min_delta"])
            if not math.isfinite(min_delta) or min_delta < 0.0:
                raise ValueError(
                    "workflow.early_stopping.min_delta must be finite and "
                    "non-negative"
                )
            monitored_metrics = early_stopping.get("metrics")
            if not isinstance(monitored_metrics, list) or not monitored_metrics:
                raise ValueError(
                    "workflow.early_stopping.metrics must be a non-empty list"
                )
            normalized_metrics = []
            identities = set()
            for metric in monitored_metrics:
                if not isinstance(metric, dict):
                    raise ValueError(
                        "workflow.early_stopping.metrics entries must be mappings"
                    )
                task = metric.get("task")
                name = metric.get("name")
                if not isinstance(task, str) or not task.strip():
                    raise ValueError(
                        "workflow.early_stopping.metrics entries require task"
                    )
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        "workflow.early_stopping.metrics entries require name"
                    )
                identity = (task, name)
                if identity in identities:
                    raise ValueError(
                        "workflow.early_stopping.metrics identities must be unique"
                    )
                identities.add(identity)
                weight = float(metric.get("weight", 1.0))
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        "workflow.early_stopping metric weights must be finite "
                        "and positive"
                    )
                normalized_metrics.append(
                    {"task": task, "name": name, "weight": weight}
                )
            early_stopping["metrics"] = normalized_metrics

        _required(value, "model.base_checkpoint")
        model = value["model"]
        if model.get("initial_scoring_checkpoint") is not None and not isinstance(
            model["initial_scoring_checkpoint"], str
        ):
            raise ValueError("model.initial_scoring_checkpoint must be a path string")
        _required(value, "data.target_manifest")
        _required(value, "data.source_payload_contract")
        _required(value, "output.run_dir")
        data = value["data"]
        data.setdefault("source_store_validation", "full_sha256")
        if data["source_store_validation"] not in {
            "full_sha256",
            "sealed_inventory",
        }:
            raise ValueError(
                "data.source_store_validation must be full_sha256 or sealed_inventory"
            )
        source_parts = data.get("source_parts")
        source_store = data.get("source_store_manifest")
        if bool(source_parts) == bool(source_store):
            raise ValueError(
                "Configure exactly one of data.source_store_manifest or data.source_parts"
            )
        if source_parts and not isinstance(source_parts, list):
            raise ValueError("data.source_parts must be a non-empty list")
        if source_store and not data.get("target_embedding_contract"):
            raise ValueError(
                "Registered-store search requires data.target_embedding_contract"
            )
        if data.get("previous_training_manifest") is not None and not isinstance(
            data["previous_training_manifest"], str
        ):
            raise ValueError("data.previous_training_manifest must be a path string")
        if (
            workflow["initialization"] == "parent_history" and
            not data.get("previous_training_manifest")
        ):
            raise ValueError(
                "parent_history initialization requires "
                "data.previous_training_manifest"
            )
        continuation = value.get("continuation")
        if continuation is None:
            if int(workflow["start_round"]) > 1:
                raise ValueError(
                    "workflow.start_round > 1 requires a continuation contract"
                )
        else:
            if not isinstance(continuation, dict):
                raise ValueError("continuation must be a mapping")
            required_continuation = {
                "previous_run_dir",
                "adopted_round",
                "training_contract",
                "training_commit",
                "success_marker",
                "previous_data_lock",
                "previous_release_lock",
            }
            if missing := required_continuation.difference(continuation):
                raise ValueError(
                    "continuation is missing required fields: "
                    f"{sorted(missing)}"
                )
            if int(continuation["adopted_round"]) != int(workflow["start_round"]):
                raise ValueError(
                    "continuation.adopted_round must equal workflow.start_round"
                )
            if not model.get("initial_scoring_checkpoint"):
                raise ValueError(
                    "continuation requires model.initial_scoring_checkpoint"
                )
            if not data.get("previous_training_manifest"):
                raise ValueError(
                    "continuation requires data.previous_training_manifest"
                )

        mining = value.setdefault("mining", {})
        mining.setdefault("top_k_per_target", 32)
        mining.setdefault("min_similarity", 0.80)
        mining.setdefault("hard_min_similarity", mining["min_similarity"])
        mining.setdefault("similarity_step", 0.02)
        mining.setdefault("duplicate_similarity", 1.0)
        mining.setdefault("candidate_multiplier", 10)
        mining.setdefault("parent_history_policy", "exclude")
        if int(mining["top_k_per_target"]) <= 0:
            raise ValueError("mining.top_k_per_target must be positive")
        hard_min = float(mining["hard_min_similarity"])
        initial_min = float(mining["min_similarity"])
        duplicate = float(mining["duplicate_similarity"])
        if not -1.0 <= hard_min <= initial_min <= 1.0:
            raise ValueError(
                "mining thresholds must satisfy -1 <= hard_min_similarity "
                "<= min_similarity <= 1"
            )
        if not initial_min < duplicate <= 1.0:
            raise ValueError(
                "mining.duplicate_similarity must exceed min_similarity and be <= 1"
            )
        if float(mining["similarity_step"]) <= 0:
            raise ValueError("mining.similarity_step must be positive")
        if int(mining["candidate_multiplier"]) <= 0:
            raise ValueError("mining.candidate_multiplier must be positive")
        if mining["parent_history_policy"] != "exclude":
            raise ValueError("mining.parent_history_policy must be exclude")
        if strategy == "grit_score":
            grit = value.setdefault("grit", {})
            grit.setdefault("target_fraction", 0.10)
            if not 0.0 < float(grit["target_fraction"]) <= 1.0:
                raise ValueError("grit.target_fraction must be in (0, 1]")
        else:
            multitask = value.setdefault("multi_task", {})
            tasks = multitask.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("multi_task.tasks must be a non-empty list")
            if len(set(map(str, tasks))) != len(tasks):
                raise ValueError("multi_task.tasks must be unique")
            multitask.setdefault("policy", "round_robin")
            if multitask["policy"] not in {"round_robin", "balanced"}:
                raise ValueError(
                    "multi_task.policy must be round_robin or balanced"
                )
            multitask_budgets(multitask)

        training = value.setdefault("training", {})
        _required(value, "training.base_spec")
        training.setdefault("passes_per_round", 12)
        training.setdefault("lr_scaling_rule", "preserve_base_spec")
        if training["lr_scaling_rule"] != "preserve_base_spec":
            raise ValueError(
                "training.lr_scaling_rule must be preserve_base_spec"
            )
        if "warm_start" in training:
            raise ValueError(
                "training.warm_start is unsupported; every candidate must start "
                "from model.base_checkpoint"
            )
        training.setdefault("checkpoint_policy", "base_checkpoint_each_round")
        if training["checkpoint_policy"] != "base_checkpoint_each_round":
            raise ValueError(
                "training.checkpoint_policy must be base_checkpoint_each_round; "
                "every candidate must start from model.base_checkpoint"
            )
        if int(training["passes_per_round"]) <= 0:
            raise ValueError("training.passes_per_round must be positive")
        node_scaling = training.get("node_scaling")
        if node_scaling is not None:
            if not isinstance(node_scaling, dict):
                raise ValueError("training.node_scaling must be a mapping")
            node_scaling.setdefault("mode", "target_optimizer_updates")
            if node_scaling["mode"] != "target_optimizer_updates":
                raise ValueError(
                    "training.node_scaling.mode must be target_optimizer_updates"
                )
            allowed_nodes = node_scaling.get("allowed_nodes")
            if not isinstance(allowed_nodes, list) or not allowed_nodes:
                raise ValueError(
                    "training.node_scaling.allowed_nodes must be a non-empty list"
                )
            normalized_nodes = sorted(set(map(int, allowed_nodes)))
            if normalized_nodes[0] <= 0:
                raise ValueError(
                    "training.node_scaling.allowed_nodes must be positive"
                )
            node_scaling["allowed_nodes"] = normalized_nodes
            for name in ("gpus_per_node", "target_optimizer_updates"):
                if int(node_scaling.get(name, 0)) <= 0:
                    raise ValueError(
                        f"training.node_scaling.{name} must be positive"
                    )

        execution = value.setdefault("execution", {})
        execution.setdefault("backend", "local")
        execution.setdefault("workdir", value["output"]["run_dir"])
        execution.setdefault("poll_seconds", 10)
        execution.setdefault("call_timeout_seconds", 30)
        if execution["backend"] not in {"local", "external"}:
            raise ValueError("execution.backend must be local or external")
        if execution["backend"] == "external" and not execution.get("runner_command"):
            raise ValueError("external execution requires execution.runner_command")
        if float(execution["call_timeout_seconds"]) <= 0:
            raise ValueError("execution.call_timeout_seconds must be positive")
        capabilities = execution.setdefault("capabilities", {})
        if not isinstance(capabilities, dict) or not all(
            isinstance(name, str) and isinstance(enabled, bool)
            for name, enabled in capabilities.items()
        ):
            raise ValueError(
                "execution.capabilities must be a mapping of names to booleans"
            )
        if (
            execution["backend"] == "external" and
            capabilities.get("shared_filesystem") is not True
        ):
            raise ValueError(
                "external execution requires capability shared_filesystem=true "
                "for workflow inputs and outputs"
            )

        actions = value.setdefault("actions", {})
        score = actions.setdefault("score", {})
        if strategy == "grit_score":
            score.setdefault(
                "command",
                ["dinov3", "grit_score", "-e", "{score_config}"],
            )
            score.setdefault("settings", {})
            if not isinstance(score["settings"], dict):
                raise ValueError("actions.score.settings must be a mapping")
        elif not score.get("command"):
            raise ValueError("multi-task runs require actions.score.command")
        score.setdefault("parameters", {})
        if not isinstance(score["parameters"], dict):
            raise ValueError("actions.score.parameters must be a mapping")
        data_action = actions.setdefault("data", {})
        data_action.setdefault(
            "command",
            [
                "python",
                "-m",
                "nvidia_tao_ds.mining.dinov3.entrypoint.refinement",
            ],
        )
        search = actions.setdefault("search", {})
        search.setdefault("command", [])
        search.setdefault("parameters", {})
        if not isinstance(search["parameters"], dict):
            raise ValueError("actions.search.parameters must be a mapping")
        search.setdefault(
            "backend", "custom" if search["command"] else "exact"
        )
        if search["backend"] not in {"exact", "custom", "audited_ann"}:
            raise ValueError(
                "actions.search.backend must be exact, custom, or audited_ann"
            )
        if search["backend"] == "custom" and not search["command"]:
            raise ValueError("Custom search requires actions.search.command")
        if search["backend"] == "exact" and search["command"]:
            raise ValueError(
                "Exact search uses actions.data.command; remove actions.search.command"
            )
        if search["backend"] == "audited_ann":
            if not source_store:
                raise ValueError(
                    "Audited ANN search requires data.source_store_manifest"
                )
            if not data.get("source_identity_audit"):
                raise ValueError(
                    "Audited ANN search requires data.source_identity_audit"
                )
            if search["command"]:
                raise ValueError(
                    "Audited ANN search uses candidate and rerank commands, not "
                    "actions.search.command"
                )
            for name in (
                "dense_store_manifest",
                "ann_index_manifest",
                "ann_audit_manifest",
                "n_probes",
                "ann_candidates",
            ):
                if name not in search:
                    raise ValueError(
                        f"Audited ANN search requires actions.search.{name}"
                    )
            if int(search["n_probes"]) <= 0:
                raise ValueError("actions.search.n_probes must be positive")
            minimum_candidates = (
                int(mining["top_k_per_target"]) *
                int(mining["candidate_multiplier"])
            )
            if int(search["ann_candidates"]) < minimum_candidates:
                raise ValueError(
                    "actions.search.ann_candidates must be at least "
                    "top_k_per_target * candidate_multiplier"
                )
            for stage in ("candidate", "rerank"):
                stage_config = search.setdefault(stage, {})
                if not stage_config.get("command"):
                    raise ValueError(
                        "Audited ANN search requires "
                        f"actions.search.{stage}.command"
                    )
                stage_config.setdefault("resources", {})
                if not isinstance(stage_config["resources"], dict):
                    raise ValueError(
                        f"actions.search.{stage}.resources must be a mapping"
                    )
                if int(stage_config["resources"].get("nodes", 1)) <= 0:
                    raise ValueError(
                        f"actions.search.{stage}.resources.nodes must be positive"
                    )
        train = actions.setdefault("train", {})
        if not train.get("command"):
            raise ValueError("actions.train.command is required")
        unsupported_train_outputs = sorted(
            set(train).intersection({"checkpoint", "contract"})
        )
        if unsupported_train_outputs:
            raise ValueError(
                "Native DINOv3 training owns fixed checkpoint/contract paths; "
                f"remove actions.train fields: {unsupported_train_outputs}"
            )
        evaluate = actions.setdefault("evaluate", {})
        evaluate.setdefault("command", [])
        evaluate.setdefault("parameters", {})
        evaluate.setdefault("scope", "sealed_benchmark")
        if evaluate["scope"] not in {"sealed_benchmark", "diagnostic_replay"}:
            raise ValueError(
                "actions.evaluate.scope must be sealed_benchmark or diagnostic_replay"
            )
        if not isinstance(evaluate["parameters"], dict):
            raise ValueError("actions.evaluate.parameters must be a mapping")
        if evaluate["command"]:
            _required(value, "data.benchmark_manifest")
            if evaluate["scope"] == "sealed_benchmark":
                _required(value, "data.benchmark_acquisition_units")
                value["data"].setdefault(
                    "acquisition_unit_column", "acquisition_unit_id"
                )
            evaluate.setdefault("metrics", "{output_dir}/metrics.json")
            evaluate.setdefault(
                "commit", "{output_dir}/evaluation_commit.json"
            )
        elif early_stopping is not None:
            raise ValueError(
                "workflow.early_stopping requires actions.evaluate.command"
            )

        configure_containers(value)
        legacy_train_backend = train.pop("backend", None)
        if legacy_train_backend is not None:
            legacy_mode = (
                "adapter_managed"
                if legacy_train_backend == "native_leaf"
                else legacy_train_backend
            )
            if (
                "execution_mode" in train and
                train["execution_mode"] != legacy_mode
            ):
                raise ValueError(
                    "actions.train.backend conflicts with execution_mode"
                )
            train["execution_mode"] = legacy_mode
        for action_name, action in (
            ("score", score),
            ("data", data_action),
            ("search", search),
            ("train", train),
            ("evaluate", evaluate),
        ):
            action.setdefault("implementation_files", [])
            if not isinstance(action["implementation_files"], list) or not all(
                isinstance(path, str) and path.strip()
                for path in action["implementation_files"]
            ):
                raise ValueError(
                    f"actions.{action_name}.implementation_files must be a list "
                    "of paths"
                )
            action.setdefault("execution_mode", "runner")
            if action["execution_mode"] not in {"runner", "adapter_managed"}:
                raise ValueError(
                    f"actions.{action_name}.execution_mode must be runner or "
                    "adapter_managed"
                )
            action.setdefault("resources", {})
            action.setdefault("wrapper_resources", {"nodes": 1})
            if not isinstance(action["resources"], dict):
                raise ValueError(f"actions.{action_name}.resources must be a mapping")
            local_scratch = action["resources"].get("local_scratch")
            if local_scratch is not None and (
                not isinstance(local_scratch, dict) or
                not isinstance(local_scratch.get("path_environment"), str) or
                not local_scratch["path_environment"].strip()
            ):
                raise ValueError(
                    f"actions.{action_name}.resources.local_scratch requires "
                    "path_environment"
                )
            if not isinstance(action["wrapper_resources"], dict):
                raise ValueError(
                    f"actions.{action_name}.wrapper_resources must be a mapping"
                )
            if int(action["resources"].get("nodes", 1)) <= 0:
                raise ValueError(
                    f"actions.{action_name}.resources.nodes must be positive"
                )
            if action["execution_mode"] == "adapter_managed" and int(
                action["wrapper_resources"].get("nodes", 1)
            ) != 1:
                raise ValueError(
                    f"actions.{action_name}.adapter_managed requires one "
                    "wrapper process"
                )

        score_command = list(map(str, score["command"]))
        uses_builtin_grit = score_command[:2] == ["dinov3", "grit_score"]
        train_command = list(map(str, train["command"]))
        uses_builtin_train = train_command[:2] == ["dinov3", "train"]
        if uses_builtin_grit and score["implementation_files"]:
            raise ValueError(
                "Built-in DINOv3 GRIT owns its implementation closure; remove "
                "actions.score.implementation_files"
            )
        if uses_builtin_train and train["implementation_files"]:
            raise ValueError(
                "Built-in DINOv3 training owns its implementation closure; remove "
                "actions.train.implementation_files"
            )
        if not uses_builtin_grit and not score["implementation_files"]:
            raise ValueError(
                "Custom actions.score requires implementation_files containing "
                "its approved dependency closure"
            )
        if not uses_builtin_train and not train["implementation_files"]:
            raise ValueError(
                "Custom actions.train requires implementation_files containing "
                "its approved dependency closure"
            )
        if evaluate["command"] and not evaluate["implementation_files"]:
            raise ValueError(
                "Custom actions.evaluate requires implementation_files containing "
                "its approved dependency closure"
            )

        score_settings = score.get("settings", {})
        neighbor_backend = str(score_settings.get("neighbor_backend", "auto"))
        neighbor_device = str(
            score_settings.get("neighbor_device", score_settings.get("device", "cuda"))
        )
        requires_gpu_faiss = (
            uses_builtin_grit and
            neighbor_backend in {"auto", "faiss_exact"} and
            neighbor_device.startswith("cuda")
        )
        if requires_gpu_faiss and capabilities.get("gpu_faiss") is not True:
            raise ValueError(
                "CUDA GRIT with auto/faiss_exact neighbors requires "
                "execution.capabilities.gpu_faiss=true"
            )

        runner_node_requests = []
        for action_name, action in (
            ("score", score),
            ("data", data_action),
            ("search", search),
            ("train", train),
            ("evaluate", evaluate),
        ):
            requested_nodes = int(action["resources"].get("nodes", 1))
            if action_name == "train" and node_scaling is not None:
                requested_nodes = max(
                    requested_nodes, *node_scaling["allowed_nodes"]
                )
            if action["execution_mode"] == "runner":
                runner_node_requests.append((action_name, requested_nodes))
                if action_name == "search":
                    for stage_name in ("candidate", "rerank"):
                        stage = action.get(stage_name, {})
                        runner_node_requests.append(
                            (
                                f"search.{stage_name}",
                                int(stage.get("resources", {}).get("nodes", 1)),
                            )
                        )
        multi_node_actions = sorted(
            name for name, nodes in runner_node_requests if nodes > 1
        )
        if multi_node_actions and execution["backend"] == "local":
            raise ValueError(
                "Multi-node runner actions require execution.backend=external; "
                f"local execution launches one process: {multi_node_actions}"
            )
        if multi_node_actions:
            missing_capabilities = sorted(
                name
                for name in MULTINODE_RUNNER_CAPABILITIES
                if capabilities.get(name) is not True
            )
            if missing_capabilities:
                raise ValueError(
                    "Multi-node external runner is missing required capabilities: "
                    f"{missing_capabilities}; actions={multi_node_actions}"
                )
        return cls(value)

    @property
    def strategy(self) -> str:
        """Return the selected acquisition strategy."""
        return str(self.value["workflow"]["strategy"])

    @property
    def run_dir(self) -> Path:
        """Return the resolved durable output directory."""
        return Path(self.value["output"]["run_dir"]).expanduser().resolve()

    @property
    def digest(self) -> str:
        """Return the immutable normalized configuration digest."""
        return canonical_digest(self.value)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent copy of the normalized configuration."""
        return deepcopy(self.value)

    def plan(self) -> dict[str, Any]:
        """Return a deterministic, non-side-effecting execution plan."""
        stages = ["score", "select_targets"]
        if self.value["actions"]["search"]["backend"] == "audited_ann":
            stages.append("search_candidates")
        stages.extend(["search", "materialize", "train"])
        if self.value["actions"]["evaluate"]["command"]:
            stages.append("evaluate")
        selection: dict[str, Any]
        if self.strategy == "multi_task_round_robin":
            quotas = multitask_budgets(self.value["multi_task"])
            selection = {
                "method": (
                    "balanced_within_task_normalized_weakness_round_robin"
                    if self.value["multi_task"]["policy"] == "balanced"
                    else "within_task_normalized_weakness_round_robin"
                ),
                "targets_per_round": sum(quotas.values()),
                "task_weights": self.value["multi_task"].get("task_weights", {}),
                "task_quotas": quotas,
                "unfilled_task_budget": (
                    "preserved"
                    if self.value["multi_task"]["policy"] == "balanced"
                    else "redistributed"
                ),
                "training_replay": (
                    "oversample each task provenance group to the largest group"
                    if self.value["multi_task"]["policy"] == "balanced"
                    else "uniform over the cumulative unique manifest"
                ),
            }
        else:
            selection = {
                "method": "within_domain_grit_rank",
                "target_fraction": float(self.value["grit"]["target_fraction"]),
            }
        data = self.value["data"]
        mining = self.value["mining"]
        evaluation = self.value["actions"]["evaluate"]
        checkpoint_policy = self.value["training"]["checkpoint_policy"]
        training_summary = (
            "initialize every candidate from the immutable base checkpoint -> "
            "train on the cumulative manifest"
        )
        action_execution = {
            name: {
                "execution_mode": action["execution_mode"],
                "container_image": action.get("container_image"),
                "resources": action.get("resources", {}),
                "wrapper_resources": action.get("wrapper_resources", {}),
            }
            for name, action in self.value["actions"].items()
            if isinstance(action, dict) and "execution_mode" in action
        }
        stop_conditions = [
            "max_rounds",
            "no_actionable_targets",
            "no_novel_samples",
            "pool_exhausted",
            "radius_exhausted",
            "search_budget_exhausted",
        ]
        if self.value["workflow"].get("early_stopping"):
            stop_conditions.insert(0, "metric_patience")
        return {
            "schema_version": SCHEMA_VERSION,
            "strategy": self.strategy,
            "initialization": (
                "seedless_from_base_checkpoint"
                if self.value["workflow"]["initialization"] == "base_checkpoint"
                else "base_checkpoint_with_parent_data_history"
            ),
            "round_stages": stages,
            "max_rounds": int(self.value["workflow"]["max_rounds"]),
            "start_round": int(self.value["workflow"]["start_round"]),
            "passes_per_round": int(self.value["training"]["passes_per_round"]),
            "training_checkpoint_policy": checkpoint_policy,
            "source_shards": (
                len(self.value["data"].get("source_parts", [])) or
                "registered_store"
            ),
            "execution_backend": self.value["execution"]["backend"],
            "container_images": self.value["execution"].get("container_images", {}),
            "execution_workdir": self.value["execution"]["workdir"],
            "execution_capabilities": self.value["execution"].get(
                "capabilities", {}
            ),
            "run_dir": str(self.run_dir),
            "config_digest": self.digest,
            "approval_contract": {
                "loop": {
                    "summary": (
                        "score -> select weak targets -> retrieve relevant source "
                        "neighbors -> publish cumulative manifest -> "
                        f"{training_summary} -> optional evaluate"
                    ),
                    "max_rounds": int(self.value["workflow"]["max_rounds"]),
                    "persistent_target_rounds": int(
                        self.value["workflow"]["persistent_target_rounds"]
                    ),
                    "early_stopping": self.value["workflow"].get(
                        "early_stopping"
                    ),
                },
                "data": {
                    "target_manifest": data["target_manifest"],
                    "source": data.get("source_store_manifest", data.get("source_parts")),
                    "source_payload_contract": data["source_payload_contract"],
                    "source_store_validation": data.get(
                        "source_store_validation"
                    ),
                    "parent_training_manifest": data.get("previous_training_manifest"),
                    "initial_scoring_checkpoint": self.value["model"].get(
                        "initial_scoring_checkpoint"
                    ),
                    "continuation": self.value.get("continuation"),
                    "training_manifest": (
                        "Per-round cumulative Parquet of unique source locators; it "
                        "references file or archive members and does not copy images."
                    ),
                },
                "selection": selection,
                "mining": {
                    "search_backend": self.value["actions"]["search"]["backend"],
                    "neighbors_per_target": int(mining["top_k_per_target"]),
                    "similarity_start": float(mining["min_similarity"]),
                    "similarity_floor": float(mining["hard_min_similarity"]),
                    "similarity_step": float(mining["similarity_step"]),
                    "duplicate_similarity": float(mining["duplicate_similarity"]),
                    "parent_history_policy": mining["parent_history_policy"],
                    "ann": (
                        {
                            "index_manifest": self.value["actions"]["search"][
                                "ann_index_manifest"
                            ],
                            "audit_manifest": self.value["actions"]["search"][
                                "ann_audit_manifest"
                            ],
                            "n_probes": int(
                                self.value["actions"]["search"]["n_probes"]
                            ),
                            "candidate_count": int(
                                self.value["actions"]["search"]["ann_candidates"]
                            ),
                            "decision_rule": "exact_float32_cosine_rerank",
                            "underfill_stop": "search_budget_exhausted",
                        }
                        if self.value["actions"]["search"]["backend"] ==
                        "audited_ann"
                        else None
                    ),
                },
                "training": {
                    "passes_per_round": int(self.value["training"]["passes_per_round"]),
                    "checkpoint_policy": checkpoint_policy,
                    "base_checkpoint": self.value["model"]["base_checkpoint"],
                    "resources": self.value["actions"]["train"].get("resources", {}),
                    "wrapper_resources": self.value["actions"]["train"].get(
                        "wrapper_resources", {}
                    ),
                    "execution_mode": self.value["actions"]["train"][
                        "execution_mode"
                    ],
                    "node_scaling": self.value["training"].get("node_scaling"),
                },
                "execution": {
                    "controller_backend": self.value["execution"]["backend"],
                    "workdir": self.value["execution"]["workdir"],
                    "actions": action_execution,
                },
                "cache_and_artifacts": {
                    "reused_read_only": "source and target embeddings plus source images",
                    "per_run": (
                        "scores, selections, neighbor tables, cumulative manifests, "
                        "checkpoints, metrics, logs, state, and report"
                    ),
                    "image_payload_copies": "none by the controller",
                    "score_scratch": {
                        "work_dir": self.value["actions"]["score"].get(
                            "settings", {}
                        ).get("work_dir"),
                        "resource": self.value["actions"]["score"].get(
                            "resources", {}
                        ).get("local_scratch"),
                        "capacity": (
                            "leaf preflight from manifest rows, model width, and "
                            "scratch_headroom_fraction"
                        ),
                    },
                    "root": str(self.run_dir),
                },
                "evaluation": {
                    "enabled": bool(evaluation["command"]),
                    "scope": evaluation["scope"] if evaluation["command"] else None,
                    "drives_selection": False,
                },
                "stop_conditions": stop_conditions,
                "monitoring": (
                    "Remain attached when requested; report stage and round transitions, "
                    "mining yield, KPI movement, retries, failures, and terminal reason."
                ),
            },
        }
