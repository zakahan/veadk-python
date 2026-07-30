# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process background task manager for long-running CodeEnv turns.

The first implementation intentionally keeps task state in process memory.  A
dedicated daemon thread owns each remote Codex connection, so ending the ADK
invocation (and its event loop) does not cancel the task.  Production services
can replace this manager with a durable implementation without changing the
LongRunningFunctionTool contract.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_MAX_EVENTS_PER_POLL = 200


class CodeTaskCancelled(RuntimeError):
    """Raised inside a worker after a task cancellation was requested."""


@dataclass(frozen=True)
class CodeTaskCredentials:
    """Credentials captured before the ToolContext invocation is released."""

    access_key: str
    secret_key: str
    session_token: str = ""
    region: str = "cn-beijing"
    host: str = ""
    scheme: str = "https"


@dataclass(frozen=True)
class CodeTaskSpec:
    """Immutable input required to execute one remote Codex task."""

    instruction: str
    tool_id: str
    credentials: CodeTaskCredentials
    user_session_id: str
    cwd: str = "/home/gem"
    model: str | None = None
    timeout_seconds: int = 3600
    ttl_seconds: int = 3600
    approval_policy: str = "accept"
    app_name: str = ""
    user_id: str = ""
    veadk_session_id: str = ""
    function_call_id: str = ""


@dataclass
class _TaskRecord:
    task_id: str
    spec: CodeTaskSpec
    status: str = "accepted"
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    events: list[dict[str, Any]] = field(default_factory=list)
    next_sequence: int = 1
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    agentkit_session_id: str | None = None
    codex_thread_id: str | None = None
    codex_turn_id: str | None = None


Worker = Callable[[CodeTaskSpec, "CodeTaskReporter"], dict[str, Any]]


class CodeTaskReporter:
    """Thread-safe worker view over one task record."""

    def __init__(self, manager: CodeTaskManager, task_id: str) -> None:
        self._manager = manager
        self.task_id = task_id

    def transition(self, status: str, message: str | None = None) -> None:
        self._manager._transition(self.task_id, status, message)

    def emit(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        message: str | None = None,
    ) -> None:
        self._manager._emit(self.task_id, event_type, data, message=message)

    def bind_agentkit_session(self, session_id: str) -> None:
        self._manager._bind_identifier(self.task_id, "agentkit_session_id", session_id)

    def bind_codex_thread(self, thread_id: str) -> None:
        self._manager._bind_identifier(self.task_id, "codex_thread_id", thread_id)

    def bind_codex_turn(self, turn_id: str) -> None:
        self._manager._bind_identifier(self.task_id, "codex_turn_id", turn_id)

    @property
    def cancellation_requested(self) -> bool:
        return self._manager._is_cancel_requested(self.task_id)

    def check_cancelled(self) -> None:
        if self.cancellation_requested:
            raise CodeTaskCancelled("code task was cancelled")


class CodeTaskManager:
    """Start, observe, wait for, and cancel CodeEnv tasks."""

    def __init__(self, worker: Worker | None = None) -> None:
        self._worker = worker or _default_worker
        self._condition = threading.Condition(threading.RLock())
        self._tasks: dict[str, _TaskRecord] = {}
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, spec: CodeTaskSpec) -> dict[str, Any]:
        """Persist a task record and start its independent worker thread."""
        task_id = f"ct_{uuid.uuid4().hex}"
        record = _TaskRecord(task_id=task_id, spec=spec)
        with self._condition:
            self._tasks[task_id] = record
            self._append_event_locked(
                record,
                "status_changed",
                {"status": "accepted"},
                message="Code task accepted",
            )
            thread = threading.Thread(
                target=self._run,
                args=(task_id,),
                name=f"veadk-code-task-{task_id[-8:]}",
                daemon=True,
            )
            self._threads[task_id] = thread
            thread.start()
            return self._snapshot_locked(record, include_events=False)

    def get(self, task_id: str) -> dict[str, Any]:
        """Return the latest public snapshot without exposing credentials."""
        with self._condition:
            return self._snapshot_locked(self._record_locked(task_id))

    def poll(
        self,
        task_id: str,
        *,
        cursor: int = 0,
        wait_seconds: float = 0,
        max_events: int = 100,
    ) -> dict[str, Any]:
        """Return events after ``cursor``, optionally waiting for the next one."""
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        max_events = max(1, min(int(max_events), _MAX_EVENTS_PER_POLL))
        deadline = time.monotonic() + wait_seconds

        with self._condition:
            record = self._record_locked(task_id)
            while (
                not any(event["sequence"] > cursor for event in record.events)
                and record.status not in TERMINAL_STATUSES
                and wait_seconds > 0
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
                record = self._record_locked(task_id)

            available = [
                _json_copy(event)
                for event in record.events
                if event["sequence"] > cursor
            ]
            events = available[:max_events]
            next_cursor = events[-1]["sequence"] if events else cursor
            snapshot = self._snapshot_locked(record, include_events=False)
            snapshot.update(
                {
                    "events": events,
                    "next_cursor": next_cursor,
                    "has_more": len(available) > len(events),
                }
            )
            return snapshot

    def wait(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Block until a task reaches a terminal state."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            record = self._record_locked(task_id)
            while record.status not in TERMINAL_STATUSES:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out waiting for code task {task_id}")
                self._condition.wait(remaining)
                record = self._record_locked(task_id)
            return self._snapshot_locked(record)

    def cancel(self, task_id: str) -> dict[str, Any]:
        """Request cancellation; the app-server worker interrupts its turn."""
        with self._condition:
            record = self._record_locked(task_id)
            if record.status in TERMINAL_STATUSES:
                return self._snapshot_locked(record)
            record.cancel_requested = True
            record.status = "cancelling"
            record.updated_at = _utc_now()
            self._append_event_locked(
                record,
                "status_changed",
                {"status": "cancelling"},
                message="Cancellation requested",
            )
            self._condition.notify_all()
            return self._snapshot_locked(record, include_events=False)

    def resume_context(self, task_id: str) -> tuple[str, str, str, dict[str, Any]]:
        """Return ADK routing identifiers and the terminal FunctionResponse body."""
        with self._condition:
            record = self._record_locked(task_id)
            if record.status not in TERMINAL_STATUSES:
                raise RuntimeError(f"code task {task_id} is still {record.status}")
            if not record.spec.function_call_id:
                raise RuntimeError(f"code task {task_id} has no ADK function-call id")
            response: dict[str, Any] = {
                "task_id": task_id,
                "status": record.status,
            }
            if record.result is not None:
                response["result"] = _json_copy(record.result)
            if record.error:
                response["error"] = record.error
            return (
                record.spec.user_id,
                record.spec.veadk_session_id,
                record.spec.function_call_id,
                response,
            )

    def _run(self, task_id: str) -> None:
        reporter = CodeTaskReporter(self, task_id)
        with self._condition:
            spec = self._record_locked(task_id).spec
        try:
            reporter.check_cancelled()
            result = self._worker(spec, reporter)
            reporter.check_cancelled()
        except CodeTaskCancelled as error:
            self._finish(task_id, "cancelled", error=str(error))
        except Exception as error:  # noqa: BLE001 - task errors are persisted
            self._finish(task_id, "failed", error=_safe_error(error))
        else:
            self._finish(task_id, "completed", result=result)
        finally:
            with self._condition:
                self._threads.pop(task_id, None)

    def _finish(
        self,
        task_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._condition:
            record = self._record_locked(task_id)
            if record.cancel_requested and status == "completed":
                status = "cancelled"
                result = None
                error = "code task was cancelled"
            record.status = status
            record.result = _json_copy(result) if result is not None else None
            record.error = error
            record.spec = replace(
                record.spec,
                credentials=CodeTaskCredentials(
                    access_key="",
                    secret_key="",
                    region=record.spec.credentials.region,
                    host=record.spec.credentials.host,
                    scheme=record.spec.credentials.scheme,
                ),
            )
            record.updated_at = _utc_now()
            data: dict[str, Any] = {"status": status}
            if result is not None:
                data["result"] = result
            if error:
                data["error"] = error
            self._append_event_locked(
                record,
                "task_completed" if status == "completed" else "task_stopped",
                data,
                message=error,
            )
            self._condition.notify_all()

    def _transition(
        self, task_id: str, status: str, message: str | None = None
    ) -> None:
        with self._condition:
            record = self._record_locked(task_id)
            if record.status in TERMINAL_STATUSES:
                return
            if record.cancel_requested and status != "cancelling":
                return
            record.status = status
            record.updated_at = _utc_now()
            self._append_event_locked(
                record,
                "status_changed",
                {"status": status},
                message=message,
            )
            self._condition.notify_all()

    def _emit(
        self,
        task_id: str,
        event_type: str,
        data: dict[str, Any] | None,
        *,
        message: str | None,
    ) -> None:
        with self._condition:
            record = self._record_locked(task_id)
            if record.status in TERMINAL_STATUSES:
                return
            self._append_event_locked(record, event_type, data, message=message)
            record.updated_at = _utc_now()
            self._condition.notify_all()

    def _bind_identifier(self, task_id: str, field_name: str, value: str) -> None:
        with self._condition:
            record = self._record_locked(task_id)
            setattr(record, field_name, value)
            record.updated_at = _utc_now()
            self._condition.notify_all()

    def _is_cancel_requested(self, task_id: str) -> bool:
        with self._condition:
            return self._record_locked(task_id).cancel_requested

    def _record_locked(self, task_id: str) -> _TaskRecord:
        record = self._tasks.get(task_id)
        if record is None:
            raise KeyError(f"unknown code task: {task_id}")
        return record

    @staticmethod
    def _append_event_locked(
        record: _TaskRecord,
        event_type: str,
        data: dict[str, Any] | None,
        *,
        message: str | None,
    ) -> None:
        event: dict[str, Any] = {
            "sequence": record.next_sequence,
            "type": event_type,
            "created_at": _utc_now(),
        }
        record.next_sequence += 1
        if message:
            event["message"] = str(message)[:4000]
        if data:
            event["data"] = _json_copy(data)
        record.events.append(event)

    @staticmethod
    def _snapshot_locked(
        record: _TaskRecord, *, include_events: bool = True
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "task_id": record.task_id,
            "status": record.status,
            "is_terminal": record.status in TERMINAL_STATUSES,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "next_cursor": (record.events[-1]["sequence"] if record.events else 0),
        }
        if record.agentkit_session_id:
            snapshot["agentkit_session_id"] = record.agentkit_session_id
        if record.codex_thread_id:
            snapshot["thread_id"] = record.codex_thread_id
        if record.codex_turn_id:
            snapshot["turn_id"] = record.codex_turn_id
        if record.result is not None:
            snapshot["result"] = _json_copy(record.result)
        if record.error:
            snapshot["error"] = record.error
        if include_events:
            snapshot["events"] = _json_copy(record.events)
        return snapshot


def _default_worker(spec: CodeTaskSpec, reporter: CodeTaskReporter) -> dict[str, Any]:
    from veadk.tools.code_task.worker import run_agentkit_code_task

    return run_agentkit_code_task(spec, reporter)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _safe_error(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    return text[:8000]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_CODE_TASK_MANAGER = CodeTaskManager()


def get_code_task_manager() -> CodeTaskManager:
    """Return the process-local manager used by the built-in tool."""
    return _CODE_TASK_MANAGER
