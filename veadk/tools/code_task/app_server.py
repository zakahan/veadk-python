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

"""Minimal synchronous Codex app-server client for a background worker."""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from typing_extensions import Self
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from veadk.tools.code_task.manager import CodeTaskCancelled

_APP_SERVER_PATH = "/v1/codex/app-server/"
_CLIENT_VERSION = "0.1.0"
_MAX_PUBLIC_TEXT = 20_000


class CodexAppServerError(RuntimeError):
    """Raised for transport or JSON-RPC failures."""


EventCallback = Callable[[str, dict[str, Any]], None]
CancellationCheck = Callable[[], bool]


def app_server_websocket_url(endpoint: str) -> str:
    """Convert an authenticated sandbox Endpoint into its app-server URL."""
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError("sandbox Endpoint must use http, https, ws, or wss")
    if not parsed.hostname:
        raise ValueError("sandbox Endpoint must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("sandbox Endpoint must not use URL userinfo")
    if parsed.fragment:
        raise ValueError("sandbox Endpoint must not include a fragment")
    normalized_path = parsed.path.rstrip("/")
    if normalized_path not in {"", _APP_SERVER_PATH.rstrip("/")}:
        raise ValueError(f"sandbox Endpoint path must be / or {_APP_SERVER_PATH}")
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    return urlunsplit((scheme, parsed.netloc, _APP_SERVER_PATH, parsed.query, ""))


class CodexAppServerClient:
    """Drive one Codex thread/turn and expose normalized progress events."""

    def __init__(
        self,
        endpoint: str,
        *,
        approval_policy: str = "accept",
        connect_timeout: float = 30,
    ) -> None:
        if approval_policy not in {"accept", "decline"}:
            raise ValueError("approval_policy must be accept or decline")
        self.websocket_url = app_server_websocket_url(endpoint)
        self.approval_policy = approval_policy
        self.connect_timeout = connect_timeout
        self._websocket: Any = None
        self._event_callback: EventCallback | None = None
        self._deferred_messages: deque[dict[str, Any]] = deque()

    def __enter__(self) -> Self:
        try:
            self._websocket = connect(
                self.websocket_url,
                open_timeout=self.connect_timeout,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            )
        except Exception as error:
            raise CodexAppServerError(
                f"failed to connect to CodeEnv app-server ({type(error).__name__})"
            ) from error
        try:
            self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "veadk_code_task",
                        "title": "VeADK Code Task",
                        "version": _CLIENT_VERSION,
                    },
                    "capabilities": {"experimentalApi": False},
                },
                timeout=self.connect_timeout,
            )
            self._send({"method": "initialized"})
        except Exception:
            with suppress(Exception):
                self._websocket.close()
            self._websocket = None
            raise
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._websocket is not None:
            self._websocket.close()
            self._websocket = None

    def run(
        self,
        instruction: str,
        *,
        cwd: str,
        model: str | None,
        timeout_seconds: int,
        event_callback: EventCallback,
        cancellation_check: CancellationCheck,
    ) -> dict[str, Any]:
        """Start a thread and run one turn until completion or cancellation."""
        self._event_callback = event_callback
        thread_params: dict[str, Any] = {"cwd": cwd}
        if model:
            thread_params["model"] = model
        thread_result = self._request(
            "thread/start", thread_params, timeout=self.connect_timeout
        )
        thread = thread_result.get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise CodexAppServerError("thread/start response did not contain thread.id")
        thread_id = thread["id"]
        event_callback("thread_started", {"thread_id": thread_id})

        turn_result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": instruction}],
            },
            timeout=self.connect_timeout,
        )
        turn = turn_result.get("turn")
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
            raise CodexAppServerError("turn/start response did not contain turn.id")
        turn_id = turn["id"]
        event_callback("turn_started", {"turn_id": turn_id})

        deadline = time.monotonic() + timeout_seconds
        streamed_chunks: list[str] = []
        fallback_text: str | None = None
        while True:
            if cancellation_check():
                self._interrupt(thread_id, turn_id)
                raise CodeTaskCancelled("code task was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._interrupt(thread_id, turn_id)
                raise TimeoutError(
                    f"Codex turn timed out after {timeout_seconds} seconds"
                )
            if self._deferred_messages:
                message = self._deferred_messages.popleft()
            else:
                try:
                    message = self._receive(timeout=min(1.0, remaining))
                except TimeoutError:
                    continue
            if self._handle_server_request(message):
                continue
            method = message.get("method")
            params = message.get("params")
            if not isinstance(method, str) or not isinstance(params, dict):
                continue
            message_turn_id = params.get("turnId")

            if method == "item/agentMessage/delta" and message_turn_id == turn_id:
                delta = params.get("delta")
                if isinstance(delta, str) and delta:
                    streamed_chunks.append(delta)
                    event_callback("assistant_delta", {"text": _public_text(delta)})
                continue

            if (
                method in {"item/started", "item/completed"}
                and message_turn_id == turn_id
            ):
                item = params.get("item")
                if isinstance(item, dict):
                    normalized = _normalize_item(
                        item, completed=method == "item/completed"
                    )
                    if normalized is not None:
                        event_callback("execution_update", normalized)
                    if (
                        method == "item/completed"
                        and item.get("type") in {"agentMessage", "agent_message"}
                        and isinstance(item.get("text"), str)
                        and item.get("phase") in {None, "final_answer"}
                    ):
                        fallback_text = item["text"]
                continue

            if method == "turn/completed":
                completed_turn = params.get("turn")
                if (
                    not isinstance(completed_turn, dict)
                    or completed_turn.get("id") != turn_id
                ):
                    continue
                status = str(completed_turn.get("status") or "completed")
                if status == "failed":
                    error = completed_turn.get("error")
                    detail = (
                        error.get("message")
                        if isinstance(error, dict)
                        else "Codex turn failed"
                    )
                    raise CodexAppServerError(str(detail))
                if status != "completed":
                    raise CodexAppServerError(f"Codex turn ended with status {status}")
                final_text = fallback_text or "".join(streamed_chunks)
                return {
                    "output": _public_text(final_text, limit=100_000),
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "turn_status": status,
                }

            if not method.lower().endswith("/delta"):
                event_callback(
                    "codex_notification",
                    {"method": method},
                )

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexAppServerError(
                    f"{method} timed out after {timeout:g} seconds"
                )
            incoming = self._receive(timeout=remaining)
            if self._handle_server_request(incoming):
                continue
            if incoming.get("id") != request_id:
                self._deferred_messages.append(incoming)
                continue
            if "error" in incoming:
                error = incoming["error"]
                if isinstance(error, dict):
                    code = error.get("code", "unknown")
                    detail = error.get("message", "unknown app-server error")
                    raise CodexAppServerError(f"{method} failed ({code}): {detail}")
                raise CodexAppServerError(f"{method} failed: {error}")
            result = incoming.get("result")
            if not isinstance(result, dict):
                raise CodexAppServerError(f"{method} returned a non-object result")
            return result

    def _interrupt(self, thread_id: str, turn_id: str) -> None:
        try:
            self._send(
                {
                    "id": str(uuid.uuid4()),
                    "method": "turn/interrupt",
                    "params": {"threadId": thread_id, "turnId": turn_id},
                }
            )
        except (CodexAppServerError, OSError, WebSocketException):
            return

    def _handle_server_request(self, message: dict[str, Any]) -> bool:
        method = message.get("method")
        if not isinstance(method, str) or "id" not in message:
            return False
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            self._send(
                {
                    "id": message["id"],
                    "result": {"decision": self.approval_policy},
                }
            )
            if self._event_callback is not None:
                self._event_callback(
                    "approval_resolved",
                    {
                        "kind": (
                            "command" if "commandExecution" in method else "file_change"
                        ),
                        "decision": self.approval_policy,
                    },
                )
            return True
        if method == "item/permissions/requestApproval":
            self._send(
                {
                    "id": message["id"],
                    "result": {"permissions": {}, "scope": "turn"},
                }
            )
            return True
        self._send(
            {
                "id": message["id"],
                "error": {
                    "code": -32601,
                    "message": f"unsupported server request: {method}",
                },
            }
        )
        return True

    def _send(self, message: dict[str, Any]) -> None:
        if self._websocket is None:
            raise CodexAppServerError("app-server WebSocket is not connected")
        try:
            self._websocket.send(
                json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            )
        except (OSError, WebSocketException) as error:
            raise CodexAppServerError(
                f"app-server send failed ({type(error).__name__})"
            ) from error

    def _receive(self, *, timeout: float) -> dict[str, Any]:
        if self._websocket is None:
            raise CodexAppServerError("app-server WebSocket is not connected")
        try:
            raw = self._websocket.recv(timeout=timeout)
        except TimeoutError:
            raise
        except (OSError, WebSocketException) as error:
            raise CodexAppServerError(
                f"app-server receive failed ({type(error).__name__})"
            ) from error
        if isinstance(raw, bytes):
            raise CodexAppServerError("app-server sent an unsupported binary frame")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CodexAppServerError("app-server sent invalid JSON") from error
        if not isinstance(message, dict):
            raise CodexAppServerError("app-server sent a non-object JSON message")
        return message


def _normalize_item(item: dict[str, Any], *, completed: bool) -> dict[str, Any] | None:
    item_type = str(item.get("type") or "other")
    item_id = str(item.get("id") or "")
    status = "completed" if completed else "running"
    normalized: dict[str, Any] = {
        "item_id": item_id,
        "kind": _item_kind(item_type),
        "status": status,
    }

    if item_type in {"reasoning"}:
        normalized["title"] = "Codex is reasoning"
        return normalized
    if item_type in {"commandExecution", "command_execution"}:
        normalized["title"] = "Run command"
        normalized["command"] = _public_value(item.get("command"))
        if completed:
            normalized["exit_code"] = _public_value(
                item.get("exitCode", item.get("exit_code"))
            )
            normalized["output"] = _public_value(
                item.get("aggregatedOutput", item.get("aggregated_output"))
            )
        return normalized
    if item_type in {"fileChange", "file_change", "file_changes"}:
        normalized["title"] = "Change files"
        normalized["changes"] = _public_value(item.get("changes", item.get("path")))
        return normalized
    if item_type in {"mcpToolCall", "mcp_tool_call"}:
        normalized["title"] = "Call MCP tool"
        normalized["server"] = _public_value(item.get("server"))
        normalized["tool"] = _public_value(item.get("tool", item.get("name")))
        if completed:
            normalized["result"] = _public_value(item.get("result", item.get("error")))
        return normalized
    if item_type in {"webSearch", "web_search", "web_search_call"}:
        normalized["title"] = "Search the web"
        normalized["query"] = _public_value(item.get("query", item.get("arguments")))
        return normalized
    if item_type in {"agentMessage", "agent_message"}:
        return None
    normalized["title"] = item_type
    return normalized


def _item_kind(item_type: str) -> str:
    normalized = item_type.replace("_", "").lower()
    if normalized == "reasoning":
        return "reasoning"
    if normalized == "commandexecution":
        return "command"
    if normalized in {"filechange", "filechanges"}:
        return "file"
    if normalized == "mcptoolcall":
        return "mcp"
    if normalized in {"websearch", "websearchcall"}:
        return "web"
    return "other"


def _public_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _public_text(value)
    if isinstance(value, list):
        return [_public_value(item) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key)[:200]: _public_value(item)
            for key, item in list(value.items())[:100]
        }
    return _public_text(str(value))


def _public_text(value: str, *, limit: int = _MAX_PUBLIC_TEXT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"
