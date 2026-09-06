"""Delegate a task to CodeEnv while exposing progress through VeADK events."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress

from google.adk.events import Event
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from veadk.runtime.tool_events import emit_tool_event
from veadk.tools.sandbox.codex_worker_client import CodexWorkerClient, CodexWorkerError


def _key(*parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class CodexSandboxTool(BaseTool):
    """A returning delegation tool; the parent agent continues after completion.

    endpoint must address an already provisioned, trusted CodeEnv sandbox.
    Credentials remain server-side and are never included in model arguments.
    A sandbox is a trust boundary; use separate endpoints for untrusted tenants.
    """

    emits_progress = True

    def __init__(
        self, endpoint: str, *, api_key: str | None = None, name="sandbox_task"
    ):
        super().__init__(
            name=name,
            description="Delegate a coding task to the remote CodeEnv sandbox. Returns its result; progress streams while it runs.",
        )
        self._endpoint = endpoint
        self._api_key = api_key

    def _get_declaration(self):
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"task": types.Schema(type=types.Type.STRING)},
                required=["task"],
            ),
        )

    async def run_async(self, *, args, tool_context):
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return {"status": "failed", "error": "task must be nonempty text"}
        ctx = tool_context._invocation_context
        call_id = tool_context.function_call_id
        if not call_id:
            raise CodexWorkerError("A stable ADK function call ID is required")
        binding_key = _key(
            ctx.app_name, ctx.user_id, ctx.session.id, self.name, self._endpoint
        )
        state_key = "codex_worker_session_" + binding_key
        turn_key = _key(binding_key, ctx.invocation_id, call_id)
        sid = tid = None
        async with CodexWorkerClient(self._endpoint, api_key=self._api_key) as client:
            try:
                sid = tool_context.state.get(state_key)
                if not sid:
                    sid = await client.create_session(binding_key)
                    tool_context.state[state_key] = sid
                started = await client.start_turn(sid, task, turn_key)
                tid = started["turnId"]
                async for event in client.events(sid, tid):
                    payload = event["payload"]
                    metadata = {"codex_worker": {**event, "functionCallId": call_id}}
                    text = (
                        payload.get("delta")
                        if event["type"] == "message.delta"
                        else None
                    )
                    # Full item text stays in metadata as a replacement snapshot.
                    # Only the normal tool result enters durable parent history.
                    await emit_tool_event(
                        Event(
                            author=ctx.agent.name,
                            invocation_id=ctx.invocation_id,
                            branch=ctx.branch,
                            partial=True,
                            custom_metadata=metadata,
                            content=types.Content(
                                role="model", parts=[types.Part(text=text)]
                            )
                            if text
                            else None,
                        )
                    )
                    if event["type"] == "turn.completed":
                        return {
                            "status": payload["status"],
                            "result": payload.get("finalText", ""),
                            "sessionId": sid,
                            "turnId": tid,
                        }
            except asyncio.CancelledError:
                if sid:
                    # Recover an accepted start whose HTTP response was lost by
                    # looking it up with the SAME idempotency key, never resubmit.
                    async def stop():
                        nonlocal tid
                        if tid is None:
                            tid = (
                                await client.request(
                                    "GET", client.turn_path(sid) + "/by-key/" + turn_key
                                )
                            )["turnId"]
                        await client.cancel(sid, tid)

                    with suppress(CodexWorkerError, asyncio.TimeoutError):
                        await asyncio.wait_for(stop(), timeout=25)
                raise
            except CodexWorkerError as exc:
                return {
                    "status": "unknown",
                    "error": str(exc),
                    "sessionId": sid,
                    "turnId": tid,
                }
