# tests/test_tool.py
import json
import pytest
from agent_core.tool import tool, ToolRegistry
from agent_core.models import ToolCall, ToolRisk, ExecutionPolicy


def test_tool_decorator_registers_schema():
    @tool(description="Add two integers")
    def add(a: int, b: int) -> int:
        return a + b

    registry = ToolRegistry([add])
    schemas = registry.schemas()
    assert any(s.name == "add" for s in schemas)
    s = next(x for x in schemas if x.name == "add")
    assert s.description == "Add two integers"
    assert "a" in s.parameters["properties"]
    assert s.parameters["required"] == ["a", "b"]


def test_registry_execute_success():
    @tool(description="Multiply")
    def multiply(x: int, y: int) -> int:
        return x * y

    registry = ToolRegistry([multiply])
    result = registry.execute(
        ToolCall(name="multiply", arguments={"x": 3, "y": 4}, call_id="1"),
        ExecutionPolicy.AUTO,
    )
    assert not result.is_error
    assert json.loads(result.content) == 12


def test_registry_preserves_call_id():
    @tool(description="Fixed")
    def fixed() -> str:
        return "ok"

    registry = ToolRegistry([fixed])
    result = registry.execute(ToolCall(name="fixed", arguments={}, call_id="xyz"), ExecutionPolicy.AUTO)
    assert result.call_id == "xyz"


def test_registry_unknown_tool_is_error():
    registry = ToolRegistry([])
    result = registry.execute(ToolCall(name="ghost", arguments={}, call_id="1"), ExecutionPolicy.AUTO)
    assert result.is_error
    assert "Unknown tool" in result.content


def test_registry_captures_exception():
    @tool(description="Broken")
    def broken() -> str:
        raise ValueError("intentional")

    registry = ToolRegistry([broken])
    result = registry.execute(ToolCall(name="broken", arguments={}, call_id="1"), ExecutionPolicy.AUTO)
    assert result.is_error
    assert "intentional" in result.content


def test_policy_read_only_blocks_execution_risk():
    @tool(description="Risky", risk=ToolRisk.EXECUTION)
    def risky_op() -> str:
        return "ran"

    registry = ToolRegistry([risky_op])
    result = registry.execute(
        ToolCall(name="risky_op", arguments={}, call_id="1"),
        ExecutionPolicy.READ_ONLY,
    )
    assert result.is_error
    assert "not permitted" in result.content


def test_policy_read_only_allows_none_risk():
    @tool(description="Safe read", risk=ToolRisk.NONE)
    def safe_read() -> str:
        return "data"

    registry = ToolRegistry([safe_read])
    result = registry.execute(
        ToolCall(name="safe_read", arguments={}, call_id="1"),
        ExecutionPolicy.READ_ONLY,
    )
    assert not result.is_error


def test_policy_safe_blocks_external_side_effect():
    @tool(description="Send alert", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)
    def send_alert(msg: str) -> str:
        return "sent"

    registry = ToolRegistry([send_alert])
    result = registry.execute(
        ToolCall(name="send_alert", arguments={"msg": "hi"}, call_id="1"),
        ExecutionPolicy.SAFE,
    )
    assert result.is_error
    assert "not permitted" in result.content


def test_policy_auto_allows_external_side_effect():
    @tool(description="Send alert", risk=ToolRisk.EXTERNAL_SIDE_EFFECT)
    def send_alert2(msg: str) -> str:
        return "sent"

    registry = ToolRegistry([send_alert2])
    result = registry.execute(
        ToolCall(name="send_alert2", arguments={"msg": "hi"}, call_id="1"),
        ExecutionPolicy.AUTO,
    )
    assert not result.is_error


def test_policy_auto_blocks_confirmation_required():
    @tool(description="Dangerous", risk=ToolRisk.FILESYSTEM, requires_confirmation=True)
    def dangerous_write() -> str:
        return "written"

    registry = ToolRegistry([dangerous_write])
    result = registry.execute(
        ToolCall(name="dangerous_write", arguments={}, call_id="1"),
        ExecutionPolicy.AUTO,
    )
    assert result.is_error
    assert "requires confirmation" in result.content


def test_policy_human_confirm_allows_confirmation_required():
    @tool(description="Dangerous allowed", risk=ToolRisk.FILESYSTEM, requires_confirmation=True)
    def gated_write() -> str:
        return "written"

    registry = ToolRegistry([gated_write])
    result = registry.execute(
        ToolCall(name="gated_write", arguments={}, call_id="1"),
        ExecutionPolicy.HUMAN_CONFIRM,
    )
    assert not result.is_error
