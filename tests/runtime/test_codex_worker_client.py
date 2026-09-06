import json

import httpx
import pytest

from veadk.tools.sandbox.codex_worker_client import CodexWorkerClient, CodexWorkerError


@pytest.mark.asyncio
async def test_endpoint_query_and_key_survive_retry():
    seen = []

    async def handle(request):
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ReadTimeout("do not expose URL", request=request)
        return httpx.Response(200, json={"turnId": "turn-1"})

    client = CodexWorkerClient(
        "https://sandbox.example/v1/codex-worker?api_key=placeholder",
        api_key="header-placeholder",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), headers=client._headers
    ) as http:
        client._http = http
        assert (await client.start_turn("session-1", "test", "stable-key"))[
            "turnId"
        ] == "turn-1"
    assert len(seen) == 2
    assert all(r.url.params["api_key"] == "placeholder" for r in seen)
    assert all(r.headers["Idempotency-Key"] == "stable-key" for r in seen)
    assert all(r.headers["X-API-Key"] == "header-placeholder" for r in seen)


@pytest.mark.asyncio
async def test_reconnect_deduplicates_and_preserves_cursor():
    calls = []

    def event(seq, kind):
        return (
            "data: "
            + json.dumps(
                {
                    "schemaVersion": 1,
                    "eventId": seq,
                    "sessionId": "sid",
                    "turnId": "tid",
                    "type": kind,
                    "payload": {},
                }
            )
            + "\n\n"
        )

    async def handle(request):
        calls.append(request)
        content = event(1, "message.delta")
        if len(calls) > 1:
            content += event(2, "turn.completed")
        return httpx.Response(
            200, text=content, headers={"Content-Type": "text/event-stream"}
        )

    client = CodexWorkerClient("https://sandbox.example")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client._http = http
        events = [e async for e in client.events("sid", "tid")]
    assert [e["eventId"] for e in events] == [1, 2]
    assert [r.headers["Last-Event-ID"] for r in calls] == ["0", "1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 410, 503])
async def test_event_errors_are_sanitized(status):
    client = CodexWorkerClient("https://sandbox.example?api_key=never-print-me")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, text="private provider error")
        )
    ) as http:
        client._http = http
        with pytest.raises(CodexWorkerError) as error:
            await anext(client.events("sid", "tid"))
    assert "never-print-me" not in str(error.value)
    assert "private" not in str(error.value)


@pytest.mark.asyncio
async def test_cancel_does_not_start_a_turn(monkeypatch):
    client = CodexWorkerClient("https://sandbox.example")
    methods = []

    async def request(method, path, **kwargs):
        methods.append((method, path))
        if method == "GET":
            return {"status": "running", "codexTurnId": "codex-id"}
        return {"status": "running"}

    monkeypatch.setattr(client, "request", request)
    await client.cancel("sid", "tid")
    assert methods == [
        ("GET", "/sessions/sid/turns/tid"),
        ("POST", "/sessions/sid/turns/tid/cancel"),
    ]
