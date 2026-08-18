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

"""Studio-owned BFF tools backed by the locally authenticated bytedcli."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any
from urllib.parse import urlsplit

from frontend.server.studio_tools.registry import StudioTool, StudioToolRegistry
from veadk.utils.logger import get_logger

logger = get_logger(__name__)

_BYTEDCLI_TIMEOUT_SECONDS = 60


def _run_bytedcli_json(
    arguments: list[str],
    *,
    failure_message: str,
    timeout_seconds: int = _BYTEDCLI_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run bytedcli without a shell and keep raw diagnostics on the BFF."""

    executable = shutil.which("bytedcli")
    if executable is None:
        return {
            "ok": False,
            "error": "bytedcli was not found in PATH; install it before using this tool",
        }
    try:
        completed = subprocess.run(
            [executable, "--json", *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"bytedcli timed out after {timeout_seconds} seconds",
        }
    except OSError as error:
        logger.warning("Failed to start bytedcli: %s", error)
        return {"ok": False, "error": "failed to start bytedcli"}

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        logger.warning(
            "%s exit_code=%s detail=%s",
            failure_message,
            completed.returncode,
            detail[:4000],
        )
        return {
            "ok": False,
            "error": failure_message,
            "exit_code": completed.returncode,
        }

    output = completed.stdout.strip()
    if not output:
        return {"ok": False, "error": "bytedcli returned an empty response"}
    try:
        data = json.loads(output)
    except json.JSONDecodeError as error:
        logger.warning("bytedcli returned invalid JSON: %s", error)
        return {"ok": False, "error": "bytedcli returned invalid JSON"}
    return {"ok": True, "data": data}


def _execution_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "executed_by": "studio-bff-bytedcli",
        "bff_process_id": os.getpid(),
    }


def query_logs_by_logid(
    *,
    logid: str,
    psm: str = "",
    site: str = "cn",
    vregion: str = "",
    level: str = "",
    scan_span_minutes: int = 10,
) -> dict[str, Any]:
    arguments = ["--site", site]
    if vregion:
        arguments.extend(["--vregion", vregion])
    arguments.extend(
        [
            "log",
            "get-logid-log",
            logid,
            "--output",
            "console",
            "--scan-span",
            str(scan_span_minutes),
        ]
    )
    if psm:
        arguments.extend(["--psm", psm])
    if level:
        arguments.extend(["--level", level])
    return _execution_result(
        _run_bytedcli_json(
            arguments,
            failure_message="bytedcli log query failed",
        )
    )


def get_codebase_mr_status(
    *,
    selector: str,
    repo: str = "",
    site: str = "cn",
    page_size: int = 20,
) -> dict[str, Any]:
    arguments = [
        "--site",
        site,
        "codebase",
        "mr",
        "status",
        selector,
        "--page-size",
        str(page_size),
    ]
    if repo:
        arguments.extend(["--repo", repo])
    return _execution_result(
        _run_bytedcli_json(
            arguments,
            failure_message="bytedcli Codebase MR query failed",
        )
    )


def read_lark_doc(*, document_url: str) -> dict[str, Any]:
    parsed_url = urlsplit(document_url)
    hostname = (parsed_url.hostname or "").lower()
    supported_host = any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in ("larkoffice.com", "feishu.cn", "larksuite.com")
    )
    if (
        parsed_url.scheme != "https"
        or not supported_host
        or not parsed_url.path.strip("/")
    ):
        return _execution_result(
            {
                "ok": False,
                "error": "document_url must be a valid HTTPS Feishu/Lark document URL",
            }
        )
    arguments = [
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
    return _execution_result(
        _run_bytedcli_json(
            arguments,
            failure_message="bytedcli Lark document fetch failed",
        )
    )


def register_bytedcli_tools(registry: StudioToolRegistry) -> None:
    """Register the supported read-only local tools in the Studio registry."""

    registry.register(
        StudioTool(
            name="studio_query_logs_by_logid",
            display_name="按 LogID 查询日志",
            description=(
                "Use the locally authenticated bytedcli to query internal Argos or "
                "LogService logs for a LogID. Use it when the user provides a LogID "
                "and asks to diagnose an online request. This is a read-only query."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "logid": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                        "description": "The exact request LogID to query.",
                    },
                    "psm": {
                        "type": "string",
                        "maxLength": 256,
                        "description": "Optional service PSM used to narrow results.",
                    },
                    "site": {
                        "type": "string",
                        "enum": ["cn", "boe", "i18n-tt", "us-ttp", "eu-ttp"],
                        "default": "cn",
                    },
                    "vregion": {"type": "string", "maxLength": 128},
                    "level": {
                        "type": "string",
                        "maxLength": 64,
                        "description": "Optional level such as Error or Error,Warn.",
                    },
                    "scan_span_minutes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 120,
                        "default": 10,
                    },
                },
                "required": ["logid"],
                "additionalProperties": False,
            },
            executor=lambda arguments: query_logs_by_logid(
                logid=str(arguments["logid"]),
                psm=str(arguments.get("psm") or ""),
                site=str(arguments.get("site") or "cn"),
                vregion=str(arguments.get("vregion") or ""),
                level=str(arguments.get("level") or ""),
                scan_span_minutes=int(arguments.get("scan_span_minutes") or 10),
            ),
            executor_revision="bytedcli-logid-v1",
            timeout_ms=75_000,
            idempotent=True,
            risk_level="medium",
        )
    )
    registry.register(
        StudioTool(
            name="studio_get_codebase_mr_status",
            display_name="查询 Codebase MR 状态",
            description=(
                "Use the locally authenticated bytedcli to read a Codebase merge "
                "request's mergeability, review state, and CI checks. Use it when "
                "the user asks about a specific MR. This tool never changes the MR."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2048,
                        "description": "MR number, URL, or source branch.",
                    },
                    "repo": {
                        "type": "string",
                        "maxLength": 512,
                        "description": "Optional Codebase repository path.",
                    },
                    "site": {
                        "type": "string",
                        "enum": ["cn", "boe", "i18n-tt", "us-ttp", "eu-ttp"],
                        "default": "cn",
                    },
                    "page_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                    },
                },
                "required": ["selector"],
                "additionalProperties": False,
            },
            executor=lambda arguments: get_codebase_mr_status(
                selector=str(arguments["selector"]),
                repo=str(arguments.get("repo") or ""),
                site=str(arguments.get("site") or "cn"),
                page_size=int(arguments.get("page_size") or 20),
            ),
            executor_revision="bytedcli-codebase-mr-v1",
            timeout_ms=75_000,
            idempotent=True,
            risk_level="medium",
        )
    )
    registry.register(
        StudioTool(
            name="studio_read_lark_doc",
            display_name="读取飞书文档",
            description=(
                "Read the full Markdown content of a Feishu or Lark document URL "
                "through the locally authenticated bytedcli user. Use it only when "
                "the user provides a document or Wiki URL. This is a read-only fetch."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "document_url": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2048,
                        "description": "HTTPS URL of the Feishu/Lark document or Wiki page.",
                    },
                },
                "required": ["document_url"],
                "additionalProperties": False,
            },
            executor=lambda arguments: read_lark_doc(
                document_url=str(arguments["document_url"]).strip(),
            ),
            executor_revision="bytedcli-lark-fetch-v1",
            timeout_ms=75_000,
            idempotent=True,
            risk_level="medium",
        )
    )
