from types import SimpleNamespace
from unittest.mock import patch

from veadk.tracing.telemetry import skill_observability


class FakeSpan:
    def __init__(self, invocation_id: str = "invocation-1") -> None:
        self.attributes = {"invocation.id": invocation_id}
        self.events = []

    def set_attribute(self, name, value) -> None:
        self.attributes[name] = value

    def add_event(self, name, attributes=None) -> None:
        self.events.append((name, attributes or {}))


def test_google_skill_load_activates_skill_and_records_semantics():
    span = FakeSpan()
    tool = SimpleNamespace(name="load_skill")
    response = SimpleNamespace(response={"status": "success"})

    with (
        patch.object(
            skill_observability,
            "get_event_function_responses",
            return_value=[response],
        ),
        patch.object(skill_observability, "_record_skill_metrics") as record,
    ):
        skill_observability.observe_skill_tool_call(
            span, tool, {"skill_name": "pdf"}, object()
        )

    assert span.attributes["skill.name"] == "pdf"
    assert span.attributes["skill.operation"] == "load"
    assert span.attributes["skill.phase"] == "completed"
    assert span.events[0][0] == "skill.selected"
    record.assert_called_once()


def test_active_skill_does_not_leak_into_another_invocation():
    skill_observability.set_active_skill(
        skill_observability.ActiveSkill(name="pdf", invocation_id="invocation-previous")
    )

    assert skill_observability.active_skill_span_attributes("invocation-next") == {}
    assert skill_observability.active_skill_metric_attributes("invocation-next") == {}


def test_skill_script_failure_is_annotated_without_replacing_active_skill():
    skill_observability.set_active_skill(
        skill_observability.ActiveSkill(name="pdf", invocation_id="invocation-1")
    )
    span = FakeSpan()
    tool = SimpleNamespace(name="run_skill_script")
    response = SimpleNamespace(response={"error": "script failed"})

    with (
        patch.object(
            skill_observability,
            "get_event_function_responses",
            return_value=[response],
        ),
        patch.object(skill_observability, "_record_skill_metrics") as record,
    ):
        skill_observability.observe_skill_tool_call(
            span, tool, {"skill_name": "pdf"}, object()
        )

    assert span.attributes["skill.phase"] == "failed"
    assert span.attributes["error.type"] == "skill_execution_error"
    assert skill_observability.get_active_skill().name == "pdf"
    assert record.call_args.args[2] is False
