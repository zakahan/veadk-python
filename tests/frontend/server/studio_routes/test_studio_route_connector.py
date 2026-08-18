# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the persistent Studio route connector."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from websockets.exceptions import InvalidStatus

import frontend.server.studio_routes.connector as connector
from frontend.server.studio_routes.registry import build_studio_route_registry
from veadk.integrations.agentkit.studio_routes import mount_studio_route_host


@pytest.mark.asyncio
async def test_http_fallback_executes_print_hello_in_studio_bff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VEADK_STUDIO_ROUTE_CHANNEL", "demo")
    app = FastAPI()
    mount_studio_route_host(app=app, enabled=True)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:
        await asyncio.sleep(0.01)

    async def reject_websocket(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise InvalidStatus(SimpleNamespace(status_code=200))  # type: ignore[arg-type]

    monkeypatch.setattr(connector, "connect", reject_websocket)
    ready = asyncio.Event()
    channel_task = asyncio.create_task(
        connector.serve_studio_route_channel(
            endpoint=f"http://127.0.0.1:{port}",
            authorization="",
            registry=build_studio_route_registry(),
            on_ready=ready.set,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{port}/print_hello")
    finally:
        channel_task.cancel()
        await asyncio.gather(channel_task, return_exceptions=True)
        server.should_exit = True
        await server_task

    assert response.status_code == 200
    assert response.json()["message"] == "hello from Studio BFF"
    assert response.json()["executed_by"] == "studio-bff"
    assert isinstance(response.json()["bff_process_id"], int)


@pytest.mark.asyncio
async def test_missing_route_capability_is_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 404

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args

        async def get(self, url: str, *, headers: dict[str, str]) -> _Response:
            del url, headers
            return _Response()

    monkeypatch.setattr(connector.httpx, "AsyncClient", _Client)

    assert not await connector.runtime_supports_bff_routes(
        endpoint="https://runtime.example",
        authorization="",
    )
