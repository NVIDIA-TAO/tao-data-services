# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-free execution boundary for concrete workflow stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol


TERMINAL = {"COMPLETE", "ERROR", "CANCELED"}
_GATED_EXEC = """
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Retain the session leader until its children have stopped. A caught handler
# is reset by exec in the action, unlike SIG_IGN which children would inherit.
signal.signal(signal.SIGTERM, lambda *_: None)
gate = int(sys.argv[1])
released = os.read(gate, 1)
os.close(gate)
if released != b"1":
    raise SystemExit(125)
record_path = Path(sys.argv[2])
return_code = subprocess.run(sys.argv[6:], check=False).returncode
# An action can leave descendants behind even after its direct child exits.
# Keep the owned leader available for identity-checked group cancellation.
while True:
    active = False
    for entry in Path('/proc').glob('[0-9]*/stat'):
        try:
            value = entry.read_text()
            fields = value[value.rfind(')') + 2:].split()
            if (int(entry.parent.name) != os.getpid() and
                    fields[0] != 'Z' and int(fields[2]) == os.getpid() and
                    int(fields[3]) == os.getpid()):
                active = True
                break
        except (OSError, ValueError, IndexError):
            continue
    if not active:
        break
    time.sleep(0.05)
try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(return_code)
if record.get("state") == "RUNNING":
    state = "COMPLETE" if return_code == 0 else "ERROR"
    record.update(
        state=state,
        native_state=state,
        return_code=return_code,
    )
    temporary = record_path.with_name(record_path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    temporary.replace(record_path)
raise SystemExit(return_code)
"""


@dataclass(frozen=True)
class StageRequest:
    """Portable request passed to a local or external four-verb runner."""

    client_job_id: str
    run_id: str
    round_index: int
    stage: str
    command: list[str]
    workdir: str
    results_dir: str
    environment: dict[str, str]
    resources: dict[str, Any]
    execution_contract: dict[str, Any]


@dataclass(frozen=True)
class StageResult:
    """Normalized terminal job result."""

    state: str
    client_job_id: str
    backend_ref: str
    return_code: int | None
    log_path: str | None
    native_state: str
    attempt: int = 1
    attempt_id: str | None = None
    container_image: str | None = None


class Runner(Protocol):
    """Execution interface consumed by the workflow controller."""

    def run(self, request: StageRequest) -> StageResult:
        """Submit or adopt a request and return its terminal result."""

    def cancel(self, client_job_id: str) -> dict[str, Any]:
        """Cancel a run-owned backend job."""

    def logs(self, client_job_id: str, cursor: str | None = None) -> dict[str, Any]:
        """Read logs without changing job state."""


def client_job_id(
    *, run_id: str, round_index: int, stage: str, command: list[str]
) -> str:
    """Derive a stable job identity from the run, stage and command."""
    payload = json.dumps(
        [run_id, round_index, stage, command], separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "d3deft-" + hashlib.sha256(payload).hexdigest()[:20]


class LocalRunner:
    """Synchronous subprocess runner for development and smoke tests."""

    def __init__(self, jobs_dir: str | Path):
        """Bind the workflow configuration or durable runtime paths."""
        self.jobs_dir = Path(jobs_dir)

    @staticmethod
    def _write_record(path: Path, result: StageResult) -> None:
        # Cancellation and the waiting controller can publish concurrently.
        # Separate temporary files avoid racing over one shared rename source.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f"{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            stream.write(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
            temporary = Path(stream.name)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _process_reference(backend_ref: str) -> tuple[int | None, int | None]:
        if not backend_ref.startswith("pid:"):
            return None, None
        try:
            parts = backend_ref.split(":")
            pid = int(parts[1])
            start_time = (
                int(parts[3])
                if len(parts) == 4 and parts[2] == "start"
                else None
            )
            return pid, start_time
        except ValueError:
            return None, None

    @staticmethod
    def _process_start_time(pid: int) -> int | None:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        fields_after_name = value[value.rfind(")") + 2:].split()
        try:
            return int(fields_after_name[19])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _alive(pid: int, start_time: int | None = None) -> bool:
        try:
            value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            return False
        except PermissionError:
            return True
        fields = value[value.rfind(")") + 2:].split()
        return fields[0] != "Z" and (
            start_time is None or int(fields[19]) == start_time
        )

    @staticmethod
    def _group_alive(pid: int) -> bool:
        # Read-only membership checks remain safe after the leader exits: never
        # use this alone to authorize signaling a possibly reused process group.
        for entry in Path("/proc").glob("[0-9]*/stat"):
            try:
                value = entry.read_text(encoding="utf-8")
                fields = value[value.rfind(")") + 2:].split()
                if fields[0] != "Z" and int(fields[2]) == pid and int(fields[3]) == pid:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    def run(self, request: StageRequest) -> StageResult:
        """Submit or adopt the stage and return its normalized terminal result."""
        if int(request.resources.get("nodes", 1)) > 1:
            raise RuntimeError(
                "LocalRunner cannot honor multi-node resources; use a "
                "gang-capable external runner"
            )
        if request.execution_contract.get("container_image"):
            raise RuntimeError("LocalRunner cannot execute a container request")
        required_capabilities = set(
            request.execution_contract.get("required_capabilities", [])
        )
        if "gpu_faiss" in required_capabilities:
            try:
                faiss = importlib.import_module("faiss")
                resources = faiss.StandardGpuResources()
                faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(1))
            except (ImportError, AttributeError, RuntimeError) as exc:
                raise RuntimeError(
                    "LocalRunner cannot satisfy required gpu_faiss capability"
                ) from exc
        requested_gpus = int(
            request.resources.get(
                "gpus_per_node", request.resources.get("gpus", 0)
            )
        )
        allocated_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        requested_visible = request.environment.get("CUDA_VISIBLE_DEVICES")
        if requested_gpus and allocated_visible is None:
            raise RuntimeError(
                "LocalRunner cannot prove a GPU allocation because "
                "CUDA_VISIBLE_DEVICES is unset; use an allocated environment "
                "or an external platform runner"
            )
        if (
            requested_gpus and
            requested_visible is not None and
            requested_visible != allocated_visible
        ):
            raise RuntimeError(
                "LocalRunner config cannot alter the platform-provided "
                "CUDA_VISIBLE_DEVICES allocation"
            )
        if requested_gpus:
            visible_count = len(
                [
                    device
                    for device in str(allocated_visible).split(",")
                    if device.strip()
                ]
            )
            if visible_count != requested_gpus:
                raise RuntimeError(
                    "LocalRunner GPU visibility does not match the request: "
                    f"requested {requested_gpus}, visible {visible_count}"
                )
        requested_cpus = int(request.resources.get("cpus", 0))
        available_cpus = len(os.sched_getaffinity(0))
        if requested_cpus > available_cpus:
            raise RuntimeError(
                "LocalRunner CPU affinity does not satisfy the request: "
                f"requested {requested_cpus}, available {available_cpus}"
            )
        if "memory" in request.resources:
            raise RuntimeError(
                "LocalRunner cannot reserve memory; use an external platform "
                "runner for memory-constrained actions"
            )
        if "time_limit" in request.resources:
            raise RuntimeError(
                "LocalRunner cannot enforce time_limit; use an external platform runner"
            )
        local_scratch = request.resources.get("local_scratch")
        if local_scratch:
            environment_name = local_scratch["path_environment"]
            configured = request.environment.get(
                environment_name, os.environ.get(environment_name)
            )
            if not configured:
                raise RuntimeError(
                    "LocalRunner has no mapped local scratch path in "
                    f"{environment_name}"
                )
            scratch_path = Path(os.path.expandvars(configured)).expanduser()
            if not scratch_path.is_dir() or not os.access(scratch_path, os.W_OK):
                raise RuntimeError(
                    f"LocalRunner scratch path is not a writable directory: {scratch_path}"
                )
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        record_path = self.jobs_dir / f"{request.client_job_id}.json"
        log_path = self.jobs_dir / f"{request.client_job_id}.log"
        cancel_path = self.jobs_dir.parent / "cancel.requested"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("state") == "COMPLETE":
                return StageResult(**record)
            if record.get("state") == "RUNNING":
                pid, start_time = self._process_reference(
                    str(record.get("backend_ref", ""))
                )
                if (
                    pid is not None and
                    start_time is not None and
                    self._alive(pid, start_time)
                ):
                    while self._alive(pid, start_time):
                        latest = json.loads(
                            record_path.read_text(encoding="utf-8")
                        )
                        if latest.get("state") in TERMINAL:
                            return StageResult(**latest)
                        if cancel_path.exists():
                            self.cancel(request.client_job_id)
                        time.sleep(0.1)
                    latest = json.loads(record_path.read_text(encoding="utf-8"))
                    if latest.get("state") in TERMINAL:
                        return StageResult(**latest)
                orphaned = StageResult(
                    state="ERROR",
                    client_job_id=request.client_job_id,
                    backend_ref=str(record.get("backend_ref", "")),
                    return_code=None,
                    log_path=str(log_path),
                    native_state="ORPHANED",
                    attempt=int(record.get("attempt", 1)),
                )
                self._write_record(record_path, orphaned)
        if cancel_path.exists():
            canceled = StageResult(
                state="CANCELED",
                client_job_id=request.client_job_id,
                backend_ref="local:not-started",
                return_code=None,
                log_path=str(log_path),
                native_state="CANCELED_BEFORE_START",
            )
            self._write_record(record_path, canceled)
            return canceled
        environment = os.environ.copy()
        environment.update(request.environment)
        if request.environment.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
            environment["PYTHONPATH"] = os.pathsep.join(
                [request.environment["PYTHONPATH"], os.environ["PYTHONPATH"]]
            )
        attempt = 1
        if record_path.is_file():
            attempt = int(
                json.loads(record_path.read_text(encoding="utf-8")).get(
                    "attempt", 0
                )
            ) + 1
        with log_path.open("a", encoding="utf-8") as log:
            gate_read, gate_write = os.pipe()
            try:
                # The durable child intentionally survives a controller restart.
                process = subprocess.Popen(  # pylint: disable=consider-using-with
                    [
                        sys.executable,
                        "-c",
                        _GATED_EXEC,
                        str(gate_read),
                        str(record_path),
                        request.client_job_id,
                        str(log_path),
                        str(attempt),
                        *request.command,
                    ],
                    cwd=request.workdir,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    pass_fds=(gate_read,),
                )
                os.close(gate_read)
                gate_read = -1
                process_start_time = self._process_start_time(process.pid)
                if process_start_time is None:
                    raise RuntimeError(
                        "LocalRunner could not bind the child process identity"
                    )
                backend_ref = (
                    f"pid:{process.pid}:start:{process_start_time}"
                )
                running = StageResult(
                    state="RUNNING",
                    client_job_id=request.client_job_id,
                    backend_ref=backend_ref,
                    return_code=None,
                    log_path=str(log_path),
                    native_state="RUNNING",
                    attempt=attempt,
                    attempt_id=f"local-{process.pid}",
                )
                self._write_record(record_path, running)
                if not cancel_path.exists():
                    os.write(gate_write, b"1")
            finally:
                if gate_read >= 0:
                    os.close(gate_read)
                os.close(gate_write)
            if cancel_path.exists() and self._alive(
                process.pid, process_start_time
            ):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return_code = process.wait()
        current = json.loads(record_path.read_text(encoding="utf-8"))
        if current.get("state") == "CANCELING":
            self.cancel(request.client_job_id)
            current = json.loads(record_path.read_text(encoding="utf-8"))
            if current.get("state") == "CANCELING":
                raise RuntimeError("Local process group cancellation remains unacknowledged")
        if current.get("state") in TERMINAL:
            return StageResult(**current)
        state = "COMPLETE" if return_code == 0 else "ERROR"
        result = StageResult(
            state=state,
            client_job_id=request.client_job_id,
            backend_ref=backend_ref,
            return_code=return_code,
            log_path=str(log_path),
            native_state=state,
            attempt=attempt,
            attempt_id=f"local-{process.pid}",
        )
        self._write_record(record_path, result)
        return result

    def cancel(self, client_job_id: str) -> dict[str, Any]:
        """Reconcile cancellation intent with the run-owned process or backend job."""
        # Keep each identity/terminal-state decision explicit before signaling.
        # pylint: disable=too-many-return-statements
        record_path = self.jobs_dir / f"{client_job_id}.json"
        if not record_path.is_file():
            return {"state": "UNKNOWN", "client_job_id": client_job_id}
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("state") in {"RUNNING", "CANCELING"}:
            pid, start_time = self._process_reference(
                str(record.get("backend_ref", ""))
            )
            if start_time is None:
                return {
                    "state": "UNKNOWN",
                    "client_job_id": client_job_id,
                    "reason": "process reference has no start-time identity",
                }
            canceled = StageResult(
                state="CANCELED",
                client_job_id=client_job_id,
                backend_ref=str(record.get("backend_ref", "")),
                return_code=None,
                log_path=record.get("log_path"),
                native_state="CANCELED",
                attempt=int(record.get("attempt", 1)),
                attempt_id=record.get("attempt_id"),
            )
            canceling = StageResult(
                state="CANCELING",
                client_job_id=client_job_id,
                backend_ref=str(record.get("backend_ref", "")),
                return_code=None,
                log_path=record.get("log_path"),
                native_state="CANCELING",
                attempt=int(record.get("attempt", 1)),
                attempt_id=record.get("attempt_id"),
            )
            if record.get("state") == "RUNNING":
                self._write_record(record_path, canceling)
            if (
                pid is not None and
                self._alive(pid, start_time)
            ):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (PermissionError, ProcessLookupError) as exc:
                    if isinstance(exc, PermissionError):
                        return {
                            "state": "UNKNOWN",
                            "client_job_id": client_job_id,
                            "reason": str(exc),
                        }
                deadline = time.monotonic() + 5.0
                while self._alive(pid, start_time) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if self._alive(pid, start_time):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (PermissionError, ProcessLookupError) as exc:
                        if isinstance(exc, PermissionError):
                            return {
                                "state": "UNKNOWN",
                                "client_job_id": client_job_id,
                                "reason": str(exc),
                            }
                    deadline = time.monotonic() + 5.0
                    while (
                        self._alive(pid, start_time) and
                        time.monotonic() < deadline
                    ):
                        time.sleep(0.05)
            # A dead leader does not establish that all workers have stopped.
            # Do not signal a group without a live, start-time-verified leader.
            deadline = time.monotonic() + 5.0
            while pid is not None and self._group_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if pid is not None and self._group_alive(pid):
                return {
                    "state": "UNKNOWN",
                    "client_job_id": client_job_id,
                    "reason": "owned process group has not terminated",
                }
            self._write_record(record_path, canceled)
            return {"state": "CANCELED", "client_job_id": client_job_id}
        return {"state": record["state"], "client_job_id": client_job_id}

    def logs(self, client_job_id: str, cursor: str | None = None) -> dict[str, Any]:
        """Read run-owned job logs from the requested cursor."""
        log_path = self.jobs_dir / f"{client_job_id}.log"
        offset = int(cursor or 0)
        if not log_path.is_file():
            return {"text": "", "cursor": str(offset)}
        with log_path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            value = stream.read()
            next_cursor = stream.tell()
        return {"text": value, "cursor": str(next_cursor)}


class ExternalRunner:
    """Adapter for an executable native-platform four-verb implementation."""

    def __init__(
        self,
        *,
        command: list[str],
        jobs_dir: str | Path,
        poll_seconds: float,
        call_timeout_seconds: float,
    ):
        """Bind the workflow configuration or durable runtime paths."""
        self.command = command
        self.jobs_dir = Path(jobs_dir)
        self.poll_seconds = poll_seconds
        self.call_timeout_seconds = call_timeout_seconds

    def _call(self, verb: str, *arguments: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [*self.command, verb, *arguments],
                text=True,
                capture_output=True,
                check=False,
                timeout=self.call_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Runner {verb} timed out after {self.call_timeout_seconds:g}s"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"Runner {verb} failed ({result.returncode}): {result.stderr.strip()}"
            )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Runner {verb} returned non-JSON output") from exc

    def run(self, request: StageRequest) -> StageResult:
        """Submit or adopt the stage and return its normalized terminal result."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.jobs_dir / f"{request.client_job_id}.request.json"
        request_path.write_text(
            json.dumps(asdict(request), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # submit is idempotent: the backend must adopt a matching client ID.
        cancel_path = self.jobs_dir.parent / "cancel.requested"
        if cancel_path.exists():
            return StageResult(
                state="CANCELED", client_job_id=request.client_job_id,
                backend_ref="external:not-started", return_code=None,
                log_path=None, native_state="CANCELED_BEFORE_START",
            )
        self._call("submit", "--request", str(request_path))
        while True:
            if cancel_path.exists():
                self.cancel(request.client_job_id)
            status = self._call(
                "status", "--client-job-id", request.client_job_id
            )
            state = str(status["state"]).upper()
            expected_image = request.execution_contract.get("container_image")
            if state == "COMPLETE" and expected_image and status.get("container_image") != expected_image:
                raise RuntimeError("Platform runner did not attest the requested container image")
            if state in TERMINAL:
                attempt_id = status.get("attempt_id")
                if (
                    request.execution_contract["attempt_scope"] == "gang" and
                    not attempt_id
                ):
                    raise RuntimeError(
                        "Gang runner status must report an attempt-scoped attempt_id"
                    )
                return StageResult(
                    state=state,
                    client_job_id=request.client_job_id,
                    backend_ref=str(status.get("backend_ref", "")),
                    return_code=status.get("return_code"),
                    log_path=status.get("log_path"),
                    native_state=str(status.get("native_state", state)),
                    attempt=int(status.get("attempt", 1)),
                    container_image=status.get("container_image"),
                    attempt_id=(None if attempt_id is None else str(attempt_id)),
                )
            if state == "UNKNOWN":
                raise RuntimeError(
                    f"Runner lost ownership of {request.client_job_id}; not resubmitting"
                )
            time.sleep(self.poll_seconds)

    def cancel(self, client_job_id: str) -> dict[str, Any]:
        """Reconcile cancellation intent with the run-owned process or backend job."""
        return self._call("cancel", "--client-job-id", client_job_id)

    def logs(self, client_job_id: str, cursor: str | None = None) -> dict[str, Any]:
        """Read run-owned job logs from the requested cursor."""
        arguments = ["--client-job-id", client_job_id]
        if cursor is not None:
            arguments.extend(["--cursor", cursor])
        return self._call("logs", *arguments)


def build_runner(config: dict[str, Any], run_dir: Path) -> Runner:
    """Construct the configured local or external execution adapter."""
    if config["backend"] == "local":
        return LocalRunner(run_dir / "jobs")
    return ExternalRunner(
        command=list(map(str, config["runner_command"])),
        jobs_dir=run_dir / "jobs",
        poll_seconds=float(config["poll_seconds"]),
        call_timeout_seconds=float(config.get("call_timeout_seconds", 30)),
    )
