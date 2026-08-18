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

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from frontend.server.studio_tools import bytedcli_tools
from frontend.server.studio_tools.registry import build_studio_tool_registry


def _mock_bytedcli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout: str = '{"items": [{"id": "result-1"}]}',
    stderr: str = "",
) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(bytedcli_tools.shutil, "which", lambda name: "/bin/bytedcli")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {
            "capture_output": True,
            "check": False,
            "text": True,
            "timeout": 60,
        }
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(bytedcli_tools.subprocess, "run", fake_run)
    return calls


def test_bytedcli_mode_registers_only_the_three_product_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VEADK_STUDIO_TOOL_CHANNEL", "bytedcli")
    monkeypatch.delenv("VEADK_STUDIO_TOOL_MODULE", raising=False)

    registry = build_studio_tool_registry()

    assert [item["id"] for item in registry.public_items()] == [
        "studio_get_codebase_mr_status",
        "studio_query_logs_by_logid",
        "studio_read_lark_doc",
    ]


def test_log_query_builds_a_shell_free_bytedcli_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_bytedcli(monkeypatch)

    result = bytedcli_tools.query_logs_by_logid(
        logid="log-123",
        psm="example.service",
        site="cn",
        level="Error,Warn",
        scan_span_minutes=20,
    )

    assert result["ok"] is True
    assert result["data"] == {"items": [{"id": "result-1"}]}
    assert result["executed_by"] == "studio-bff-bytedcli"
    assert calls == [
        [
            "/bin/bytedcli",
            "--json",
            "--site",
            "cn",
            "log",
            "get-logid-log",
            "log-123",
            "--output",
            "console",
            "--scan-span",
            "20",
            "--psm",
            "example.service",
            "--level",
            "Error,Warn",
        ]
    ]


def test_lark_reader_fetches_the_full_document_by_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_bytedcli(monkeypatch)
    document_url = "https://bytedance.larkoffice.com/wiki/example"

    result = bytedcli_tools.read_lark_doc(document_url=document_url)

    assert result["ok"] is True
    assert calls == [
        [
            "/bin/bytedcli",
            "--json",
            "lark",
            "docs",
            "fetch",
            "--as",
            "user",
            "--doc",
            document_url,
            "--scope",
            "full",
            "--doc-format",
            "markdown",
            "--detail",
            "simple",
        ]
    ]


def test_lark_reader_rejects_non_lark_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_bytedcli(monkeypatch)

    result = bytedcli_tools.read_lark_doc(
        document_url="https://example.com/wiki/not-a-lark-document",
    )

    assert result["ok"] is False
    assert calls == []


def test_bytedcli_failure_does_not_return_raw_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_bytedcli(
        monkeypatch,
        returncode=1,
        stdout="",
        stderr="internal diagnostic that must stay on the BFF",
    )

    result = bytedcli_tools.get_codebase_mr_status(
        selector="123",
        repo="org/repo",
    )

    assert result["ok"] is False
    assert result["error"] == "bytedcli Codebase MR query failed"
    assert result["exit_code"] == 1
    assert "detail" not in result
