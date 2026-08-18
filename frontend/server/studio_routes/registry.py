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

"""Studio BFF-owned route declarations and local handler execution."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from veadk.integrations.agentkit.studio_routes import (
    StudioRouteManifest,
    route_catalog_revision,
)

RouteExecutor = Callable[[dict[str, Any]], Any]


class StudioRouteExecutionError(RuntimeError):
    """A safe BFF handler error that may cross the reverse-route channel."""


@dataclass(frozen=True)
class StudioRouteResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None


@dataclass(frozen=True)
class StudioRoute:
    id: str
    method: str
    path: str
    executor: RouteExecutor
    handler_revision: str = "v1"
    timeout_ms: int = 30_000
    response_mode: str = "json"

    def manifest(self) -> StudioRouteManifest:
        return StudioRouteManifest(
            id=self.id,
            method=self.method.upper(),
            path=self.path,
            handler_revision=self.handler_revision,
            timeout_ms=self.timeout_ms,
            response_mode=self.response_mode,
        )


class StudioRouteRegistry:
    """Own local route handlers; only declarative manifests leave the BFF."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], StudioRoute] = {}
        self._routes_by_id: dict[tuple[str, str], StudioRoute] = {}

    def register(self, route: StudioRoute) -> None:
        manifest = route.manifest()
        route_key = (manifest.method, manifest.path)
        id_key = (manifest.id, manifest.handler_revision)
        if route_key in self._routes:
            raise ValueError(
                f"Studio route already registered: {manifest.method} {manifest.path}"
            )
        if id_key in self._routes_by_id:
            raise ValueError(
                f"Studio route id already registered: "
                f"{manifest.id}@{manifest.handler_revision}"
            )
        self._routes[route_key] = route
        self._routes_by_id[id_key] = route

    def manifests(self) -> list[dict[str, Any]]:
        return [
            route.manifest().model_dump(mode="json")
            for _, route in sorted(self._routes.items())
        ]

    @property
    def revision(self) -> str:
        return route_catalog_revision(self.manifests())

    @property
    def enabled(self) -> bool:
        return bool(self._routes)

    async def execute(
        self,
        *,
        route_id: str,
        handler_revision: str,
        request: dict[str, Any],
    ) -> StudioRouteResponse:
        route = self._routes_by_id.get((route_id, handler_revision))
        if route is None:
            raise StudioRouteExecutionError(
                f"Studio route handler is unavailable: {route_id}@{handler_revision}"
            )
        if inspect.iscoroutinefunction(route.executor):
            result = await route.executor(request)
        else:
            result = await asyncio.to_thread(route.executor, request)
        if isinstance(result, StudioRouteResponse):
            return result
        return StudioRouteResponse(body=result)


def _register_demo_routes(registry: StudioRouteRegistry) -> None:
    def print_hello(request: dict[str, Any]) -> StudioRouteResponse:
        del request
        return StudioRouteResponse(
            headers={"content-type": "application/json"},
            body={
                "message": "hello from Studio BFF",
                "executed_by": "studio-bff",
                "bff_process_id": os.getpid(),
            },
        )

    registry.register(
        StudioRoute(
            id="print_hello",
            method="GET",
            path="/print_hello",
            executor=print_hello,
            handler_revision="demo-print-hello-v1",
        )
    )


def build_studio_route_registry() -> StudioRouteRegistry:
    """Build the BFF route registry selected by server-owned configuration."""

    registry = StudioRouteRegistry()
    mode = os.getenv("VEADK_STUDIO_ROUTE_CHANNEL", "").strip().lower()
    if mode in {"1", "true", "yes", "demo"}:
        _register_demo_routes(registry)
    return registry
