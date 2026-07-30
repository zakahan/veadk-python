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

"""AgentKit Session provisioning and Codex execution for one code task."""

from __future__ import annotations

import time
from typing import Any

from veadk.tools.code_task.app_server import CodexAppServerClient
from veadk.tools.code_task.manager import (
    CodeTaskReporter,
    CodeTaskSpec,
)

_CREATE_SESSION_START_FAIL_CODE = "ErrCreateSessionFail"


def run_agentkit_code_task(
    spec: CodeTaskSpec, reporter: CodeTaskReporter
) -> dict[str, Any]:
    """Create a CodeEnv session, run Codex, and return its final result."""
    reporter.transition("provisioning_sandbox", "Creating AgentKit CodeEnv session")
    reporter.check_cancelled()
    client = _agentkit_client(spec)
    session_id, endpoint = _create_or_reconcile_session(client, spec, reporter)
    reporter.bind_agentkit_session(session_id)
    reporter.emit(
        "sandbox_ready",
        {
            "agentkit_session_id": session_id,
            "user_session_id": spec.user_session_id,
        },
    )
    reporter.check_cancelled()
    reporter.transition("connecting_codex", "Connecting to Codex app-server")

    def on_event(event_type: str, data: dict[str, Any]) -> None:
        if event_type == "thread_started":
            thread_id = data.get("thread_id")
            if isinstance(thread_id, str):
                reporter.bind_codex_thread(thread_id)
        elif event_type == "turn_started":
            turn_id = data.get("turn_id")
            if isinstance(turn_id, str):
                reporter.bind_codex_turn(turn_id)
            reporter.transition("running", "Codex turn is running")
        reporter.emit(event_type, data)

    with CodexAppServerClient(
        endpoint,
        approval_policy=spec.approval_policy,
    ) as app_server:
        result = app_server.run(
            spec.instruction,
            cwd=spec.cwd,
            model=spec.model,
            timeout_seconds=spec.timeout_seconds,
            event_callback=on_event,
            cancellation_check=lambda: reporter.cancellation_requested,
        )

    result.update(
        {
            "agentkit_session_id": session_id,
            "user_session_id": spec.user_session_id,
        }
    )
    return result


def _agentkit_client(spec: CodeTaskSpec) -> Any:
    from agentkit.sdk.tools.client import AgentkitToolsClient

    client = AgentkitToolsClient(
        access_key=spec.credentials.access_key,
        secret_key=spec.credentials.secret_key,
        region=spec.credentials.region,
        session_token=spec.credentials.session_token,
    )
    if spec.credentials.host:
        client.set_host(spec.credentials.host)
    if spec.credentials.scheme:
        client.set_scheme(spec.credentials.scheme)
    return client


def _create_or_reconcile_session(
    client: Any,
    spec: CodeTaskSpec,
    reporter: CodeTaskReporter,
) -> tuple[str, str]:
    from agentkit.sdk.tools import types as tools_types

    try:
        response = client.create_session(
            tools_types.CreateSessionRequest(
                ToolId=spec.tool_id,
                Ttl=spec.ttl_seconds,
                TtlUnit="second",
                UserSessionId=spec.user_session_id,
            )
        )
    except Exception as error:
        if _CREATE_SESSION_START_FAIL_CODE not in str(error):
            raise
        reconciled = _find_ready_session(client, spec, reporter)
        if reconciled is None:
            raise
        return reconciled

    session_id = str(response.session_id or "").strip()
    endpoint = str(response.endpoint or "").strip()
    if not session_id:
        raise RuntimeError("AgentKit CreateSession response has no SessionId")
    if endpoint:
        return session_id, endpoint
    return _wait_until_ready(client, spec, reporter, session_id)


def _find_ready_session(
    client: Any,
    spec: CodeTaskSpec,
    reporter: CodeTaskReporter,
) -> tuple[str, str] | None:
    from agentkit.sdk.tools import types as tools_types

    for attempt in range(6):
        reporter.check_cancelled()
        response = client.list_sessions(
            tools_types.ListSessionsRequest(
                ToolId=spec.tool_id,
                MaxResults=10,
                Filters=[
                    tools_types.FiltersItemForListSessions(
                        Name="UserSessionId",
                        Values=[spec.user_session_id],
                    )
                ],
            )
        )
        for session in response.session_infos or []:
            if session.user_session_id != spec.user_session_id:
                continue
            if (session.status or "").lower() == "failed":
                raise RuntimeError("AgentKit CodeEnv session failed to start")
            if session.session_id and session.endpoint:
                return session.session_id, session.endpoint
            if session.session_id:
                return _wait_until_ready(
                    client,
                    spec,
                    reporter,
                    session.session_id,
                )
        if attempt < 5:
            time.sleep(5)
    return None


def _wait_until_ready(
    client: Any,
    spec: CodeTaskSpec,
    reporter: CodeTaskReporter,
    session_id: str,
) -> tuple[str, str]:
    from agentkit.sdk.tools import types as tools_types

    deadline = time.monotonic() + min(spec.timeout_seconds, 300)
    while time.monotonic() < deadline:
        reporter.check_cancelled()
        response = client.get_session(
            tools_types.GetSessionRequest(
                ToolId=spec.tool_id,
                SessionId=session_id,
            )
        )
        status = str(response.status or "").lower()
        endpoint = str(response.endpoint or "").strip()
        if status == "ready" and endpoint:
            return session_id, endpoint
        if status == "failed":
            raise RuntimeError("AgentKit CodeEnv session failed to start")
        time.sleep(2)
    raise TimeoutError("AgentKit CodeEnv session did not become ready in time")
