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

"""BFF-side tool definitions, revisioning, validation, and execution."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from veadk.integrations.agentkit.studio_channel import (
    StudioToolManifest,
    catalog_revision,
)

ToolExecutor = Callable[[dict[str, Any]], Any]


class StudioToolExecutionError(RuntimeError):
    """A safe error that can be returned across the Studio channel."""


@dataclass(frozen=True)
class StudioTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    executor: ToolExecutor
    executor_revision: str = "v1"
    timeout_ms: int = 30_000
    idempotent: bool = False
    risk_level: str = "low"

    def manifest(self) -> StudioToolManifest:
        return StudioToolManifest(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            executor_revision=self.executor_revision,
            timeout_ms=self.timeout_ms,
            idempotent=self.idempotent,
            risk_level=self.risk_level,
        )


class StudioToolRegistry:
    """Owns local executors; only manifests cross the WebSocket boundary."""

    def __init__(self) -> None:
        self._tools: dict[tuple[str, str], StudioTool] = {}
        self._latest: dict[str, str] = {}

    def register(self, tool: StudioTool) -> None:
        manifest = tool.manifest()
        try:
            Draft202012Validator.check_schema(manifest.input_schema)
        except SchemaError as error:
            raise ValueError(
                f"Invalid JSON Schema for {tool.name}: {error.message}"
            ) from error
        key = (manifest.name, manifest.executor_revision)
        if key in self._tools:
            raise ValueError(
                f"Studio tool already registered: {manifest.name}@{manifest.executor_revision}"
            )
        self._tools[key] = tool
        self._latest[manifest.name] = manifest.executor_revision

    def manifests(self) -> list[dict[str, Any]]:
        manifests = [
            self._tools[(name, revision)].manifest().model_dump(mode="json")
            for name, revision in sorted(self._latest.items())
        ]
        return manifests

    @property
    def revision(self) -> str:
        return catalog_revision(self.manifests())

    @property
    def enabled(self) -> bool:
        return bool(self._latest)

    async def execute(
        self,
        *,
        name: str,
        executor_revision: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = self._tools.get((name, executor_revision))
        if tool is None:
            raise StudioToolExecutionError(
                f"Studio tool revision is unavailable: {name}@{executor_revision}"
            )
        try:
            Draft202012Validator(tool.input_schema).validate(arguments)
        except ValidationError as error:
            raise StudioToolExecutionError(
                f"Invalid arguments for Studio tool {name}: {error.message}"
            ) from error

        if inspect.iscoroutinefunction(tool.executor):
            return await tool.executor(arguments)
        return await asyncio.to_thread(tool.executor, arguments)


def _register_demo_tools(registry: StudioToolRegistry) -> None:
    def current_time(arguments: dict[str, Any]) -> dict[str, Any]:
        timezone_name = str(arguments.get("timezone") or "Asia/Shanghai")
        now = datetime.now(ZoneInfo(timezone_name))
        return {
            "timezone": timezone_name,
            "iso_time": now.isoformat(),
            "executed_by": "studio-bff",
            "bff_process_id": os.getpid(),
        }

    registry.register(
        StudioTool(
            name="studio_current_time",
            description=(
                "Return the current time from the local VeADK Studio BFF. Use this "
                "when the user asks for the current time and mention that execution "
                "was confirmed on the Studio BFF."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "enum": ["Asia/Shanghai", "UTC"],
                        "description": "Timezone used to format the current time.",
                    }
                },
                "additionalProperties": False,
            },
            executor=current_time,
            executor_revision="demo-time-v1",
            idempotent=True,
        )
    )

    def multiply(arguments: dict[str, Any]) -> dict[str, Any]:
        left = int(arguments["left"])
        right = int(arguments["right"])
        return {
            "left": left,
            "right": right,
            "product": left * right,
            "executed_by": "studio-bff",
            "bff_process_id": os.getpid(),
        }

    registry.register(
        StudioTool(
            name="studio_multiply",
            description=(
                "Multiply two integers in the local VeADK Studio BFF. Always use "
                "this tool for multiplication so the reverse tool channel can be verified."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            executor=multiply,
            executor_revision="demo-multiply-v1",
            idempotent=True,
        )
    )


def build_studio_tool_registry() -> StudioToolRegistry:
    """Build the registry selected by server-owned Studio configuration."""

    registry = StudioToolRegistry()
    mode = os.getenv("VEADK_STUDIO_TOOL_CHANNEL", "").strip().lower()
    if mode in {"1", "true", "yes", "demo"}:
        _register_demo_tools(registry)

    module_name = os.getenv("VEADK_STUDIO_TOOL_MODULE", "").strip()
    if module_name:
        module = importlib.import_module(module_name)
        register_tools = getattr(module, "register_tools", None)
        if not callable(register_tools):
            raise RuntimeError(f"{module_name} must export register_tools(registry)")
        register_tools(registry)
    return registry
