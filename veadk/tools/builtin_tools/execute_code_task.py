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

"""Long-running VEADK tool that delegates coding work to AgentKit CodeEnv."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google.adk.tools import ToolContext
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types

from veadk.tools.builtin_tools._agentkit import (
    get_agentkit_credentials,
    get_agentkit_endpoint_config,
    resolve_agentkit_tool_id,
)
from veadk.tools.code_task import (
    CodeTaskCredentials,
    CodeTaskSpec,
    get_code_task_manager,
)

_DEFAULT_CWD = "/home/gem"
_DEFAULT_TIMEOUT_SECONDS = 3600
_MAX_TASK_SECONDS = 86400


def _start_code_task(
    instruction: str,
    cwd: str = _DEFAULT_CWD,
    model: str | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    tool_context: ToolContext = None,
) -> dict[str, Any]:
    """Delegate a coding task to Codex in an AgentKit CodeEnv sandbox.

    Use this for autonomous repository analysis, code changes, command
    execution, and test runs that may take a long time. The tool returns a
    pending task ID immediately. Do not call it again for the same task; the
    application resumes the agent with the final result when the task finishes.

    Args:
        instruction: Complete coding objective for Codex, including expected
            verification and output.
        cwd: Working directory inside CodeEnv. Defaults to /home/gem.
        model: Optional Codex model override. Usually leave unset so the
            AgentKit Tool configuration chooses the model.
        timeout_seconds: Maximum Codex turn duration, from 1 to 86400 seconds.

    Returns:
        A pending task descriptor containing task_id and status.
    """
    if tool_context is None:
        raise ValueError("tool_context is required for execute_code_task")
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    cwd = cwd.strip()
    if not cwd.startswith("/"):
        raise ValueError("cwd must be an absolute path inside CodeEnv")
    if not isinstance(timeout_seconds, int) or not (
        1 <= timeout_seconds <= _MAX_TASK_SECONDS
    ):
        raise ValueError("timeout_seconds must be between 1 and 86400")

    tool_id = resolve_agentkit_tool_id()
    access_key, secret_key, headers = get_agentkit_credentials(tool_context.state)
    if not access_key or not secret_key:
        raise ValueError("AgentKit credentials are unavailable")
    _, region, host, scheme = get_agentkit_endpoint_config()
    approval_policy = (
        os.getenv("VEADK_CODE_TASK_APPROVAL_POLICY", "accept").strip().lower()
    )
    if approval_policy not in {"accept", "decline"}:
        raise ValueError("VEADK_CODE_TASK_APPROVAL_POLICY must be accept or decline")
    ttl_seconds = min(
        _MAX_TASK_SECONDS,
        max(3600, timeout_seconds + 600),
    )
    task_uuid = uuid.uuid4().hex
    spec = CodeTaskSpec(
        instruction=instruction,
        tool_id=tool_id,
        credentials=CodeTaskCredentials(
            access_key=access_key,
            secret_key=secret_key,
            session_token=headers.get("X-Security-Token", ""),
            region=region,
            host=host,
            scheme=scheme,
        ),
        user_session_id=f"veadk-code-{task_uuid}",
        cwd=cwd,
        model=model.strip() if model and model.strip() else None,
        timeout_seconds=timeout_seconds,
        ttl_seconds=ttl_seconds,
        approval_policy=approval_policy,
        app_name=tool_context.session.app_name,
        user_id=tool_context.user_id,
        veadk_session_id=tool_context.session.id,
        function_call_id=tool_context.function_call_id or "",
    )
    task = get_code_task_manager().submit(spec)
    return {
        "status": "pending",
        "task_id": task["task_id"],
        "message": (
            "The CodeEnv task is running in the background. The application "
            "will resume this function call after the task reaches a terminal "
            "state."
        ),
    }


_start_code_task.__name__ = "execute_code_task"
execute_code_task = LongRunningFunctionTool(func=_start_code_task)
"""LongRunningFunctionTool mounted on a VEADK Agent."""


def get_code_task(task_id: str) -> dict[str, Any]:
    """Return the latest in-process CodeEnv task snapshot."""
    return get_code_task_manager().get(task_id)


def poll_code_task(
    task_id: str,
    cursor: int = 0,
    wait_seconds: float = 0,
    max_events: int = 100,
) -> dict[str, Any]:
    """Read incremental task events for a UI, API handler, or diagnostic."""
    return get_code_task_manager().poll(
        task_id,
        cursor=cursor,
        wait_seconds=wait_seconds,
        max_events=max_events,
    )


def cancel_code_task(task_id: str) -> dict[str, Any]:
    """Request cancellation of an in-process CodeEnv task."""
    return get_code_task_manager().cancel(task_id)


async def wait_code_task(task_id: str, timeout: float | None = None) -> dict[str, Any]:
    """Wait asynchronously until a CodeEnv task is terminal."""
    return await asyncio.to_thread(
        get_code_task_manager().wait,
        task_id,
        timeout,
    )


async def resume_code_task(
    runner: Any,
    task_id: str,
    *,
    timeout: float | None = None,
    run_config: Any = None,
) -> AsyncIterator[Any]:
    """Wait for a task and resume its long-running ADK function call.

    The runner must use the same session service and app name as the runner
    that produced the original long-running function call.
    """
    await wait_code_task(task_id, timeout)
    user_id, session_id, function_call_id, response = (
        get_code_task_manager().resume_context(task_id)
    )
    function_response = types.FunctionResponse(
        id=function_call_id,
        name="execute_code_task",
        response=response,
    )
    kwargs: dict[str, Any] = {
        "user_id": user_id,
        "session_id": session_id,
        "new_message": types.Content(
            role="user",
            parts=[types.Part(function_response=function_response)],
        ),
    }
    if run_config is not None:
        kwargs["run_config"] = run_config
    async for event in runner.run_async(**kwargs):
        yield event


__all__ = [
    "cancel_code_task",
    "execute_code_task",
    "get_code_task",
    "poll_code_task",
    "resume_code_task",
    "wait_code_task",
]
