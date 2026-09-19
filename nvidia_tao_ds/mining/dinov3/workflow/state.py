# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked, event-backed state for the DINOv3 SSL DEFT controller."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterator

from ..contracts import write_json_atomic


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ControllerBusy(RuntimeError):
    """Another process owns the only state writer lock."""


class StateStore:
    """Single-controller state with atomic snapshots and append-only events."""

    def __init__(self, run_dir: str | Path):
        """Bind the workflow configuration or durable runtime paths."""
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".controller.lock"

    @contextmanager
    def controller_lock(self) -> Iterator[None]:
        """Acquire the exclusive controller writer lock for this run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerBusy(f"Another controller owns {self.run_dir}") from exc
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "acquired_at": _now()}))
            stream.flush()
            os.fsync(stream.fileno())
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def controller_active(self) -> bool:
        """Probe lock ownership without changing its contents or durable state."""
        if not self.lock_path.exists():
            return False
        with self.lock_path.open("r", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
        return False

    def load(self) -> dict[str, Any]:
        """Load persisted state, or return an empty mapping for a new run."""
        if not self.state_path.is_file():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def initialize(self, *, run_id: str, config_digest: str) -> dict[str, Any]:
        """Create initial run state or validate an existing configuration lock."""
        state = self.load()
        if state:
            if state["config_digest"] != config_digest:
                raise RuntimeError(
                    "Resolved configuration changed; fork the run instead of resuming it"
                )
            return state
        state = {
            "schema_version": "1.0",
            "run_id": run_id,
            "config_digest": config_digest,
            "status": "running",
            "generation": 0,
            "current_round": 0,
            "current_checkpoint": None,
            "current_training_manifest": None,
            "completed_stages": {},
            "completed_rounds": {},
            "active_jobs": {},
            "persistent_targets": {},
            "stop_reason": None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.save(state)
        self.append_event(state, round_index=0, stage="initialize", status="complete")
        return state

    def save(self, state: dict[str, Any]) -> None:
        """Atomically publish the next state generation."""
        snapshot = dict(state, generation=int(state.get("generation", 0)) + 1,
                        updated_at=_now())
        write_json_atomic(self.state_path, snapshot)
        state.update(snapshot)

    def append_event(
        self,
        state: dict[str, Any],
        *,
        round_index: int,
        stage: str,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append and flush a sequenced workflow event."""
        sequence = 1
        if self.events_path.is_file():
            with self.events_path.open("rb") as stream:
                sequence += sum(1 for line in stream if line.strip())
        event = {
            "sequence": sequence,
            "timestamp": _now(),
            "run_id": state["run_id"],
            "generation": state["generation"],
            "round": round_index,
            "stage": stage,
            "status": status,
        }
        if extra:
            event.update(extra)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def complete_stage(
        self,
        state: dict[str, Any],
        *,
        round_index: int,
        stage: str,
        outputs: dict[str, Any],
        job: dict[str, Any] | None = None,
    ) -> None:
        """Commit stage outputs and clear the corresponding active job."""
        key = f"round_{round_index:03d}/{stage}"
        state["completed_stages"][key] = {
            "completed_at": _now(),
            "outputs": outputs,
            "job": job,
        }
        state["active_jobs"].pop(key, None)
        self.save(state)
        self.append_event(
            state,
            round_index=round_index,
            stage=stage,
            status="complete",
            extra={"outputs": outputs, "job": job},
        )

    def complete_round(self, state: dict[str, Any], *, round_index: int) -> None:
        """Atomically publish the transition to the next refinement round."""
        key = f"round_{round_index:03d}"
        state.setdefault("completed_rounds", {})[key] = {"completed_at": _now()}
        self.save(state)
        self.append_event(
            state,
            round_index=round_index,
            stage="round_complete",
            status="complete",
        )

    def fail_stage(
        self,
        state: dict[str, Any],
        *,
        round_index: int,
        stage: str,
        error: str,
    ) -> None:
        """Record a failed stage without discarding its recovery history."""
        state["status"] = "failed"
        state["failure"] = {"round": round_index, "stage": stage, "error": error}
        self.save(state)
        self.append_event(
            state,
            round_index=round_index,
            stage=stage,
            status="error",
            extra={"error": error},
        )
