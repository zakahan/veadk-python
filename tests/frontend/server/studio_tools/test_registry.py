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

import pytest

from frontend.server.studio_tools.registry import (
    StudioTool,
    StudioToolExecutionError,
    StudioToolRegistry,
)


def _registry() -> StudioToolRegistry:
    registry = StudioToolRegistry()
    registry.register(
        StudioTool(
            name="studio_multiply",
            description="Multiply two integers in Studio.",
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            executor=lambda args: {"product": args["left"] * args["right"]},
            executor_revision="v1",
        )
    )
    return registry


@pytest.mark.asyncio
async def test_registry_validates_arguments_and_executes_revision() -> None:
    registry = _registry()

    result = await registry.execute(
        name="studio_multiply",
        executor_revision="v1",
        arguments={"left": 6, "right": 7},
    )

    assert result == {"product": 42}
    assert registry.manifests()[0]["executor_revision"] == "v1"
    assert registry.revision.startswith("sha256:")


@pytest.mark.asyncio
async def test_registry_rejects_arguments_before_executor() -> None:
    registry = _registry()

    with pytest.raises(StudioToolExecutionError, match="Invalid arguments"):
        await registry.execute(
            name="studio_multiply",
            executor_revision="v1",
            arguments={"left": "six", "right": 7},
        )


@pytest.mark.asyncio
async def test_new_executor_revision_becomes_the_next_catalog_snapshot() -> None:
    registry = _registry()
    first_revision = registry.revision
    registry.register(
        StudioTool(
            name="studio_multiply",
            description="Multiply two integers with the updated Studio executor.",
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            executor=lambda args: {
                "product": args["left"] * args["right"],
                "revision": "v2",
            },
            executor_revision="v2",
        )
    )

    assert registry.revision != first_revision
    assert registry.manifests()[0]["executor_revision"] == "v2"
    assert await registry.execute(
        name="studio_multiply",
        executor_revision="v2",
        arguments={"left": 3, "right": 5},
    ) == {"product": 15, "revision": "v2"}
