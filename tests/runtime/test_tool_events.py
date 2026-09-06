"""Streaming delegation must work with the Google and VeADK runners."""

import asyncio
from contextlib import aclosing

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner as AdkRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from veadk import Agent, Runner
from veadk.tools.sandbox.codex_sandbox_tool import CodexSandboxTool


class ParentModel(BaseLlm):
    model: str = "offline-parent"

    async def generate_content_async(self, llm_request, stream=False):
        responses = [
            p.function_response
            for c in llm_request.contents
            for p in c.parts or []
            if p.function_response
        ]
        if responses:
            assert responses[-1].response["status"] == "completed"
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="Parent received: " + responses[-1].response["result"]
                        )
                    ],
                )
            )
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="sandbox_task", args={"task": "hello"}, id="call-1"
                            )
                        )
                    ],
                )
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("runner_type", [AdkRunner, Runner])
async def test_progress_precedes_function_response_and_is_not_history(
    monkeypatch, runner_type
):
    release = asyncio.Event()
    stopped = asyncio.Event()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            stopped.set()

        async def create_session(self, key):
            return "session-1"

        async def start_turn(self, sid, task, key):
            return {"turnId": "turn-1"}

        async def events(self, sid, tid):
            for i, text in enumerate(("hello ", "world"), 1):
                yield {
                    "schemaVersion": 1,
                    "eventId": i,
                    "sessionId": sid,
                    "turnId": tid,
                    "type": "message.delta",
                    "payload": {"itemId": "msg-1", "delta": text},
                }
            await release.wait()
            yield {
                "schemaVersion": 1,
                "eventId": 3,
                "sessionId": sid,
                "turnId": tid,
                "type": "turn.completed",
                "payload": {"status": "completed", "finalText": "hello world"},
            }

    monkeypatch.setattr(
        "veadk.tools.sandbox.codex_sandbox_tool.CodexWorkerClient", FakeClient
    )
    agent = Agent(
        name="parent",
        model=ParentModel(),
        model_api_key="offline-test",
        tools=[CodexSandboxTool("http://localhost:8197")],
    )
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="test", user_id="user", session_id="session")
    runner = runner_type(agent=agent, app_name="test", session_service=sessions)
    events = []
    deltas = []
    async with aclosing(
        runner.run_async(
            user_id="user",
            session_id="session",
            new_message=types.Content(
                role="user", parts=[types.Part(text="Delegate this")]
            ),
        )
    ) as stream:
        async with asyncio.timeout(10):
            async for event in stream:
                events.append(event)
                worker = (event.custom_metadata or {}).get("codex_worker")
                if worker and worker["type"] == "message.delta":
                    assert not stopped.is_set()
                    assert not any(e.get_function_responses() for e in events)
                    deltas.append(worker["payload"]["delta"])
                    if len(deltas) == 2:
                        release.set()
    assert "".join(deltas) == "hello world"
    assert any(e.get_function_responses() for e in events)
    assert events[-1].content.parts[0].text == "Parent received: hello world"
    session = await sessions.get_session(
        app_name="test", user_id="user", session_id="session"
    )
    assert not any(
        (e.custom_metadata or {}).get("codex_worker") for e in session.events
    )
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_bridge_cancellation_closes_tool_task():
    from veadk.runtime.tool_events import emit_tool_event, stream_tool_events
    from google.adk.events import Event

    closed = asyncio.Event()

    async def source():
        try:
            await emit_tool_event(Event(author="parent", partial=True))
            await asyncio.Event().wait()
            yield Event(author="parent")
        finally:
            closed.set()

    async with aclosing(stream_tool_events(source())) as stream:
        await anext(stream)
    assert closed.is_set()
