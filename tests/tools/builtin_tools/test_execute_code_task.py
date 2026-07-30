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

from __future__ import annotations

import json
import threading
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.genai import types

from veadk import Agent, Runner
from veadk.memory.short_term_memory import ShortTermMemory
from veadk.tools.builtin_tools import execute_code_task as tool_module
from veadk.tools.code_task import app_server
from veadk.tools.code_task.manager import (
    CodeTaskCredentials,
    CodeTaskManager,
    CodeTaskSpec,
)
from veadk.tools.code_task.worker import (
    _agentkit_client,
    _create_or_reconcile_session,
)


def _spec(**updates: Any) -> CodeTaskSpec:
    values: dict[str, Any] = {
        "instruction": "fix the tests",
        "tool_id": "tool-1",
        "credentials": CodeTaskCredentials("ak", "sk"),
        "user_session_id": "veadk-code-test",
        "app_name": "app",
        "user_id": "user",
        "veadk_session_id": "session",
        "function_call_id": "call-1",
    }
    values.update(updates)
    return CodeTaskSpec(**values)


def test_manager_runs_task_outside_caller_and_polls_incremental_events():
    worker_started = threading.Event()
    allow_completion = threading.Event()

    def worker(spec, reporter):
        assert spec.instruction == "fix the tests"
        reporter.transition("running", "Codex is running")
        reporter.bind_agentkit_session("sandbox-session")
        reporter.emit("assistant_delta", {"text": "working"})
        worker_started.set()
        assert allow_completion.wait(2)
        return {"output": "done", "thread_id": "thread-1", "turn_id": "turn-1"}

    manager = CodeTaskManager(worker=worker)
    pending = manager.submit(_spec())
    assert pending["status"] in {"accepted", "running"}
    assert worker_started.wait(2)

    first = manager.poll(pending["task_id"], cursor=0)
    assert [event["sequence"] for event in first["events"]] == list(
        range(1, len(first["events"]) + 1)
    )
    assert any(event["type"] == "assistant_delta" for event in first["events"])

    allow_completion.set()
    completed = manager.wait(pending["task_id"], timeout=2)
    assert completed["status"] == "completed"
    assert completed["result"]["output"] == "done"
    assert manager._tasks[pending["task_id"]].spec.credentials.access_key == ""
    assert manager._tasks[pending["task_id"]].spec.credentials.secret_key == ""

    incremental = manager.poll(pending["task_id"], cursor=first["next_cursor"])
    assert incremental["is_terminal"] is True
    assert incremental["events"][-1]["type"] == "task_completed"


def test_manager_cancels_a_running_task():
    worker_started = threading.Event()

    def worker(_spec, reporter):
        worker_started.set()
        while True:
            reporter.check_cancelled()
            threading.Event().wait(0.01)

    manager = CodeTaskManager(worker=worker)
    pending = manager.submit(_spec())
    assert worker_started.wait(2)
    manager.cancel(pending["task_id"])

    cancelled = manager.wait(pending["task_id"], timeout=2)
    assert cancelled["status"] == "cancelled"
    assert cancelled["is_terminal"] is True


@pytest.mark.asyncio
async def test_builtin_is_long_running_and_uses_context_credentials(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeManager:
        def submit(self, spec):
            captured["spec"] = spec
            return {"task_id": "ct_test", "status": "accepted"}

    monkeypatch.setattr(tool_module, "get_code_task_manager", lambda: FakeManager())
    monkeypatch.setattr(tool_module, "resolve_agentkit_tool_id", lambda: "code-tool")
    monkeypatch.setattr(
        tool_module,
        "get_agentkit_credentials",
        lambda _state: ("context-ak", "context-sk", {"X-Security-Token": "sts"}),
    )
    monkeypatch.setattr(
        tool_module,
        "get_agentkit_endpoint_config",
        lambda: ("agentkit", "cn-test", "host", "https"),
    )
    context = SimpleNamespace(
        state={
            "VOLCENGINE_ACCESS_KEY": "context-ak",
            "VOLCENGINE_SECRET_KEY": "context-sk",
        },
        session=SimpleNamespace(app_name="app", id="veadk-session"),
        user_id="user",
        function_call_id="function-call",
    )

    assert isinstance(tool_module.execute_code_task, LongRunningFunctionTool)
    assert tool_module.execute_code_task.is_long_running is True
    result = await tool_module.execute_code_task.run_async(
        args={"instruction": "repair this repository"},
        tool_context=context,
    )

    assert result["status"] == "pending"
    assert result["task_id"] == "ct_test"
    spec = captured["spec"]
    assert spec.tool_id == "code-tool"
    assert spec.credentials.access_key == "context-ak"
    assert spec.credentials.secret_key == "context-sk"
    assert spec.credentials.session_token == "sts"
    assert spec.credentials.host == "host"
    assert spec.credentials.scheme == "https"
    assert spec.function_call_id == "function-call"
    declaration = tool_module.execute_code_task._get_declaration()
    assert declaration is not None
    parameters = declaration.parameters_json_schema
    assert parameters is not None
    assert set(parameters["properties"]) == {
        "instruction",
        "cwd",
        "model",
        "timeout_seconds",
    }


@pytest.mark.asyncio
async def test_adk_runner_marks_pending_call_and_resumes_with_same_id(monkeypatch):
    model_requests: list[Any] = []
    final_responses: list[types.FunctionResponse] = []

    class FakeLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request, stream=False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            model_requests.append(llm_request)
            responses = [
                part.function_response
                for content in llm_request.contents
                for part in content.parts or []
                if part.function_response is not None
            ]
            if len(model_requests) == 1:
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id="call-long-running",
                                    name="execute_code_task",
                                    args={"instruction": "fix the tests"},
                                )
                            )
                        ],
                    )
                )
            elif len(model_requests) == 2:
                assert responses[-1].response["status"] == "pending"
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="The task is running.")],
                    )
                )
            else:
                final_responses.extend(responses)
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="The task completed.")],
                    )
                )

    manager = CodeTaskManager(
        worker=lambda _spec, _reporter: {"output": "all tests pass"}
    )
    monkeypatch.setenv("MODEL_AGENT_API_KEY", "test-model-key")
    monkeypatch.setattr(tool_module, "get_code_task_manager", lambda: manager)
    monkeypatch.setattr(tool_module, "resolve_agentkit_tool_id", lambda: "code-tool")
    monkeypatch.setattr(
        tool_module,
        "get_agentkit_credentials",
        lambda _state: ("ak", "sk", {}),
    )
    monkeypatch.setattr(
        tool_module,
        "get_agentkit_endpoint_config",
        lambda: ("agentkit", "cn-test", "host", "https"),
    )
    agent = Agent(
        name="long_running_test_agent",
        instruction="Call execute_code_task exactly once.",
        model=FakeLlm(model="fake"),
        tools=[tool_module.execute_code_task],
    )
    runner = Runner(
        agent=agent,
        app_name="long-running-test",
        short_term_memory=ShortTermMemory(backend="local"),
    )
    await runner.session_service.create_session(
        app_name="long-running-test",
        user_id="user",
        session_id="session",
    )

    task_id: str | None = None
    marked_ids: set[str] = set()
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="repair the repository")],
        ),
    ):
        marked_ids.update(event.long_running_tool_ids or set())
        for response in event.get_function_responses():
            if response.name == "execute_code_task":
                task_id = response.response["task_id"]

    assert marked_ids == {"call-long-running"}
    assert task_id is not None
    assert len(model_requests) == 2

    resumed_text: list[str] = []
    async for event in tool_module.resume_code_task(runner, task_id):
        if event.content:
            resumed_text.extend(
                part.text for part in event.content.parts or [] if part.text is not None
            )

    assert resumed_text == ["The task completed."]
    assert final_responses[-1].id == "call-long-running"
    assert final_responses[-1].response == {
        "task_id": task_id,
        "status": "completed",
        "result": {"output": "all tests pass"},
    }


def test_create_session_uses_seconds_and_returns_endpoint():
    class FakeClient:
        request = None

        def create_session(self, request):
            self.request = request
            return SimpleNamespace(
                session_id="sandbox-session",
                endpoint="https://sandbox.example/?Authorization=secret",
            )

    client = FakeClient()
    reporter = SimpleNamespace(check_cancelled=lambda: None)

    result = _create_or_reconcile_session(client, _spec(), reporter)

    assert result == (
        "sandbox-session",
        "https://sandbox.example/?Authorization=secret",
    )
    assert client.request.tool_id == "tool-1"
    assert client.request.user_session_id == "veadk-code-test"
    assert client.request.ttl_unit == "second"


def test_agentkit_client_honors_shared_host_and_scheme_configuration():
    client = _agentkit_client(
        _spec(
            credentials=CodeTaskCredentials(
                "ak",
                "sk",
                region="cn-test",
                host="agentkit.internal.example",
                scheme="http",
            )
        )
    )

    assert client.service_info.host == "agentkit.internal.example"
    assert client.service_info.scheme == "http"


def test_create_session_reconciles_existing_session_until_ready():
    class FakeClient:
        def create_session(self, _request):
            raise RuntimeError("ErrCreateSessionFail: UserSessionId exists")

        def list_sessions(self, _request):
            return SimpleNamespace(
                session_infos=[
                    SimpleNamespace(
                        user_session_id="veadk-code-test",
                        status="creating",
                        session_id="sandbox-session",
                        endpoint="",
                    )
                ]
            )

        def get_session(self, _request):
            return SimpleNamespace(
                status="ready",
                endpoint="https://sandbox.example/?Authorization=secret",
            )

    reporter = SimpleNamespace(check_cancelled=lambda: None)

    assert _create_or_reconcile_session(FakeClient(), _spec(), reporter) == (
        "sandbox-session",
        "https://sandbox.example/?Authorization=secret",
    )


def test_app_server_url_preserves_endpoint_auth_query():
    assert (
        app_server.app_server_websocket_url(
            "https://sandbox.example/?Authorization=secret"
        )
        == "wss://sandbox.example/v1/codex/app-server/?Authorization=secret"
    )


def test_app_server_connection_error_does_not_expose_endpoint_query(monkeypatch):
    def fail_connect(url, **_kwargs):
        raise RuntimeError(f"could not connect to {url}")

    monkeypatch.setattr(app_server, "connect", fail_connect)

    with (
        pytest.raises(app_server.CodexAppServerError) as raised,
        app_server.CodexAppServerClient(
            "https://sandbox.example/?Authorization=secret"
        ),
    ):
        pass

    assert "secret" not in str(raised.value)
    assert "RuntimeError" in str(raised.value)


def test_app_server_streams_progress_and_returns_authoritative_text(monkeypatch):
    callbacks: list[tuple[str, dict[str, Any]]] = []

    class FakeWebSocket:
        def __init__(self):
            self.sent: list[dict[str, Any]] = []
            self.responded_to: set[str] = set()
            self.sent_early_delta = False
            self.notifications = [
                {
                    "id": "approval-1",
                    "method": "item/commandExecution/requestApproval",
                    "params": {},
                },
                {
                    "method": "item/started",
                    "params": {
                        "turnId": "turn-1",
                        "item": {
                            "id": "item-1",
                            "type": "commandExecution",
                            "command": "pytest",
                        },
                    },
                },
                {
                    "method": "item/agentMessage/delta",
                    "params": {"turnId": "turn-1", "delta": "partial"},
                },
                {
                    "method": "item/completed",
                    "params": {
                        "turnId": "turn-1",
                        "item": {
                            "id": "message-1",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "all tests pass",
                        },
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-1", "status": "completed"}},
                },
            ]

        def send(self, raw):
            self.sent.append(json.loads(raw))

        def recv(self, timeout):
            del timeout
            last = self.sent[-1]
            method = last.get("method")
            request_id = last.get("id")
            if method == "initialize" and request_id not in self.responded_to:
                self.responded_to.add(request_id)
                return json.dumps({"id": last["id"], "result": {}})
            if method == "thread/start" and request_id not in self.responded_to:
                self.responded_to.add(request_id)
                return json.dumps(
                    {
                        "id": last["id"],
                        "result": {"thread": {"id": "thread-1"}},
                    }
                )
            if method == "turn/start" and request_id not in self.responded_to:
                if not self.sent_early_delta:
                    self.sent_early_delta = True
                    return json.dumps(
                        {
                            "method": "item/agentMessage/delta",
                            "params": {
                                "turnId": "turn-1",
                                "delta": "early ",
                            },
                        }
                    )
                self.responded_to.add(request_id)
                return json.dumps(
                    {
                        "id": last["id"],
                        "result": {"turn": {"id": "turn-1"}},
                    }
                )
            return json.dumps(self.notifications.pop(0))

        def close(self):
            return None

    websocket = FakeWebSocket()
    monkeypatch.setattr(app_server, "connect", lambda *_args, **_kwargs: websocket)

    with app_server.CodexAppServerClient(
        "https://sandbox.example/?Authorization=secret"
    ) as client:
        result = client.run(
            "fix it",
            cwd="/home/gem",
            model=None,
            timeout_seconds=30,
            event_callback=lambda event_type, data: callbacks.append(
                (event_type, data)
            ),
            cancellation_check=lambda: False,
        )

    assert result["output"] == "all tests pass"
    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"
    assert ("assistant_delta", {"text": "early "}) in callbacks
    assert ("assistant_delta", {"text": "partial"}) in callbacks
    assert any(event_type == "execution_update" for event_type, _ in callbacks)
    approval_response = next(
        message for message in websocket.sent if message.get("id") == "approval-1"
    )
    assert approval_response["result"]["decision"] == "accept"
