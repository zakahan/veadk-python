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

"""Semantic observability helpers shared by VeADK and ADK-native skills."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span

from veadk.utils.adk_compat import get_event_function_responses


@dataclass(frozen=True)
class ActiveSkill:
    """Low-cardinality metadata for the skill active in one invocation context."""

    name: str
    skill_id: str = ""
    space_id: str = ""
    version: str = ""
    invocation_id: str = ""


_active_skill: ContextVar[ActiveSkill | None] = ContextVar(
    "veadk_active_skill", default=None
)

_SKILL_OPERATIONS = {
    "skills_tool": "load",
    "load_skill": "load",
    "load_skill_resource": "load_resource",
    "run_skill_script": "run_script",
    "list_skills": "list",
    "search_skills": "search",
}


def set_active_skill(skill: ActiveSkill) -> None:
    """Set the skill used by subsequent model and tool calls in this context."""

    _active_skill.set(skill)


def get_active_skill() -> ActiveSkill | None:
    return _active_skill.get()


def active_skill_span_attributes(invocation_id: str = "") -> dict[str, str]:
    skill = get_active_skill()
    if skill is None or (
        invocation_id and skill.invocation_id and invocation_id != skill.invocation_id
    ):
        return {}
    attributes = {"skill.name": skill.name}
    if skill.skill_id:
        attributes["skill.id"] = skill.skill_id
    if skill.space_id:
        attributes["skill.space.id"] = skill.space_id
    if skill.version:
        attributes["skill.version"] = skill.version
    return attributes


def active_skill_metric_attributes(invocation_id: str = "") -> dict[str, str]:
    skill = get_active_skill()
    if skill is None or (
        invocation_id and skill.invocation_id and invocation_id != skill.invocation_id
    ):
        return {}
    return {
        "skill_name": skill.name,
        "skill_id": skill.skill_id,
        "skill_space_id": skill.space_id,
        "skill_version": skill.version,
    }


def set_active_skill_attributes(span: Span) -> None:
    invocation_id = str(
        (getattr(span, "attributes", None) or {}).get("invocation.id", "")
    )
    for name, value in active_skill_span_attributes(invocation_id).items():
        span.set_attribute(name, value)


def observe_skill_tool_call(
    span: Span,
    tool: Any,
    args: dict[str, Any],
    function_response_event: Any,
) -> None:
    """Annotate ADK tool spans with stable Skill semantics.

    Google ADK exposes SkillToolset operations as ordinary tools, while VeADK's
    legacy implementation exposes ``skills_tool``. Recognizing both here keeps
    the observability contract independent of the selected Skill runtime.
    """

    tool_name = getattr(tool, "name", "")
    operation = _SKILL_OPERATIONS.get(tool_name)
    if operation is None:
        set_active_skill_attributes(span)
        return

    skill = _skill_from_tool_call(tool, args)
    failed, error_type = _tool_call_failed(function_response_event)
    invocation_id = str(
        (getattr(span, "attributes", None) or {}).get("invocation.id", "")
    )
    if skill and not failed and operation in {"load", "load_resource", "run_script"}:
        skill = ActiveSkill(
            name=skill.name,
            skill_id=skill.skill_id,
            space_id=skill.space_id,
            version=skill.version,
            invocation_id=invocation_id,
        )
        set_active_skill(skill)

    span.set_attribute("skill.operation", operation)
    span.set_attribute("skill.phase", "completed" if not failed else "failed")
    span.set_attribute("gen_ai.operation.name", f"skill.{operation}")
    set_active_skill_attributes(span)
    if error_type:
        span.set_attribute("error.type", error_type)

    event_attributes = active_skill_span_attributes(invocation_id)
    event_attributes["skill.operation"] = operation
    event_name = {
        "load": "skill.selected" if not failed else "skill.load_failed",
        "run_script": "skill.completed" if not failed else "skill.failed",
        "load_resource": "skill.resource_loaded"
        if not failed
        else "skill.resource_load_failed",
    }.get(operation, f"skill.{operation}")
    span.add_event(event_name, attributes=event_attributes)

    # The legacy SkillsTool records its metric inside run_async so direct uses
    # that bypass ADK telemetry remain observable. Avoid double counting here.
    if tool_name != "skills_tool":
        _record_skill_metrics(span, operation, not failed, error_type)


def _skill_from_tool_call(tool: Any, args: dict[str, Any]) -> ActiveSkill | None:
    name = str(args.get("skill_name") or args.get("command") or "").strip()
    current = get_active_skill()
    if not name:
        return current

    skill_id = ""
    space_id = ""
    version = ""
    skills = getattr(tool, "skills", None)
    if isinstance(skills, dict):
        skill = skills.get(name)
        if skill is not None:
            skill_id = str(getattr(skill, "id", "") or "")
            space_id = str(getattr(skill, "skill_space_id", "") or "")
            version = str(getattr(skill, "version", "") or "")
    return ActiveSkill(name, skill_id=skill_id, space_id=space_id, version=version)


def _tool_call_failed(function_response_event: Any) -> tuple[bool, str]:
    responses = get_event_function_responses(function_response_event)
    if not responses:
        return False, ""
    response = getattr(responses[0], "response", None)
    if response is None and isinstance(responses[0], dict):
        response = responses[0].get("response")

    if isinstance(response, dict):
        error = response.get("error")
        status = str(response.get("status", "")).lower()
        if error:
            return True, "skill_execution_error"
        if status in {"error", "failed", "failure"}:
            return True, status
    if isinstance(response, str) and response.lstrip().lower().startswith(
        ("error:", "execution failed:")
    ):
        return True, "skill_execution_error"
    return False, ""


def _record_skill_metrics(
    span: Span, operation: str, success: bool, error_type: str = ""
) -> None:
    from veadk.tracing.telemetry.telemetry import meter_uploader

    if meter_uploader and hasattr(meter_uploader, "record_skill_operation"):
        meter_uploader.record_skill_operation(
            span=span,
            operation=operation,
            attributes=active_skill_metric_attributes(
                str((getattr(span, "attributes", None) or {}).get("invocation.id", ""))
            ),
            success=success,
            error_type=error_type,
        )
