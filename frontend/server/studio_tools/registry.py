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
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
    display_name: str = ""
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
        return self.snapshot().manifests()

    @property
    def revision(self) -> str:
        return self.snapshot().revision

    @property
    def enabled(self) -> bool:
        return bool(self._latest)

    def public_items(self) -> list[dict[str, Any]]:
        return self.snapshot().public_items()

    def snapshot(
        self, selected_names: Sequence[str] | None = None
    ) -> StudioToolCatalogSnapshot:
        """Freeze selected latest tool revisions for one run.

        ``None`` preserves the legacy full-catalog behavior. An explicit empty
        sequence selects no BFF tools.
        """

        names = (
            sorted(self._latest)
            if selected_names is None
            else list(dict.fromkeys(selected_names))
        )
        unknown = sorted(set(names) - self._latest.keys())
        if unknown:
            raise ValueError("Unknown Studio tools: " + ", ".join(unknown))
        tools = {
            (name, self._latest[name]): self._tools[(name, self._latest[name])]
            for name in names
        }
        return StudioToolCatalogSnapshot(tools)

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


class StudioToolCatalogSnapshot:
    """Immutable per-run view of BFF manifests and executors."""

    def __init__(self, tools: dict[tuple[str, str], StudioTool]) -> None:
        self._tools = MappingProxyType(dict(tools))
        self._latest = MappingProxyType(
            {name: revision for name, revision in self._tools}
        )
        self._manifests = tuple(
            self._tools[(name, revision)].manifest().model_dump(mode="json")
            for name, revision in sorted(self._latest.items())
        )
        self._revision = catalog_revision(list(self._manifests))

    @property
    def enabled(self) -> bool:
        return bool(self._tools)

    @property
    def revision(self) -> str:
        return self._revision

    def manifests(self) -> list[dict[str, Any]]:
        return [dict(manifest) for manifest in self._manifests]

    def public_items(self) -> list[dict[str, Any]]:
        return [
            {
                "id": name,
                "name": tool.display_name or name,
                "description": tool.description,
                "riskLevel": tool.risk_level,
            }
            for (name, revision), tool in sorted(self._tools.items())
            if self._latest[name] == revision
        ]

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
                f"Studio tool is unavailable in this run: {name}@{executor_revision}"
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


def build_studio_tool_registry() -> StudioToolRegistry:
    """Build the registry selected by server-owned Studio configuration."""

    registry = StudioToolRegistry()
    mode = os.getenv("VEADK_STUDIO_TOOL_CHANNEL", "").strip().lower()
    if mode in {"1", "true", "yes", "demo", "bytedcli"}:
        from frontend.server.studio_tools.bytedcli_tools import (
            register_bytedcli_tools,
        )

        register_bytedcli_tools(registry)

    module_name = os.getenv("VEADK_STUDIO_TOOL_MODULE", "").strip()
    if module_name:
        module = importlib.import_module(module_name)
        register_tools = getattr(module, "register_tools", None)
        if not callable(register_tools):
            raise RuntimeError(f"{module_name} must export register_tools(registry)")
        register_tools(registry)
    return registry
