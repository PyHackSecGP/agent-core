"""Test cases for agent_core.models module."""
from agent_core.models import (
    AgentMessage, ToolCall, ToolResult, ToolSchema,
    LLMResponse, AgentResult, Artifact,
    StopReason, ToolRisk, ExecutionPolicy,
)


def test_tool_call_construction():
    tc = ToolCall(name="my_tool", arguments={"x": 1}, call_id="abc")
    assert tc.name == "my_tool"
    assert tc.arguments == {"x": 1}
    assert tc.call_id == "abc"


def test_llm_response_has_tool_calls_true():
    tc = ToolCall(name="t", arguments={}, call_id="1")
    r = LLMResponse(tool_calls=[tc])
    assert r.has_tool_calls is True


def test_llm_response_has_tool_calls_false():
    r = LLMResponse(content="hello")
    assert r.has_tool_calls is False


def test_agent_result_full_fields():
    tc = ToolCall(name="t", arguments={}, call_id="1")
    art = Artifact(name="report.html", path="/tmp/report.html", content_type="report")
    r = AgentResult(
        output="done",
        stop_reason=StopReason.DONE,
        iterations=3,
        tool_calls=[tc],
        artifacts=[art],
    )
    assert r.stop_reason == StopReason.DONE
    assert r.error is None
    assert len(r.tool_calls) == 1
    assert r.artifacts[0].content_type == "report"


def test_agent_result_max_iterations_has_no_output():
    r = AgentResult(
        output=None,
        stop_reason=StopReason.MAX_ITERATIONS,
        iterations=20,
        tool_calls=[],
        error="exceeded iterations",
    )
    assert r.output is None
    assert r.error == "exceeded iterations"


def test_agent_message_defaults():
    m = AgentMessage(role="user", content="hi")
    assert m.tool_calls == []
    assert m.tool_results == []


def test_tool_risk_enum():
    assert ToolRisk.NONE.value == "none"
    assert ToolRisk.EXECUTION.value == "execution"
    assert ToolRisk.EXTERNAL_SIDE_EFFECT.value == "external"


def test_execution_policy_enum():
    assert ExecutionPolicy.READ_ONLY.value == "read_only"
    assert ExecutionPolicy.AUTO.value == "auto"
