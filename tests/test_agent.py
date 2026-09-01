# tests/test_agent.py
import pytest
from agent_core.agent import Agent, AgentLoopError
from agent_core.tool import tool, ToolRegistry
from agent_core.memory import ConversationMemory
from agent_core.models import (
    AgentResult, LLMResponse, ToolCall, StopReason,
    ExecutionPolicy, ToolRisk,
)
from agent_core.llm.base import LLMProvider


class MockProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._iter = iter(responses)

    def chat(self, messages, tools) -> LLMResponse:
        return next(self._iter)


# ── Invariant tests ──────────────────────────────────────────────────────────

def test_invariant_never_exceeds_max_iterations():
    @tool(description="Spin")
    def spin() -> str:
        return "spinning"

    registry = ToolRegistry([spin])
    responses = [LLMResponse(tool_calls=[ToolCall(name="spin", arguments={}, call_id=str(i))]) for i in range(25)]
    agent = Agent(provider=MockProvider(responses), registry=registry, max_iterations=3)
    result = agent.run("spin forever")
    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert result.iterations == 3


def test_invariant_tool_error_reaches_model_not_exception():
    @tool(description="Always fails")
    def broken() -> str:
        raise RuntimeError("tool failed")

    registry = ToolRegistry([broken])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="broken", arguments={}, call_id="e1")]),
        LLMResponse(content="I saw the error"),
    ])
    agent = Agent(provider=provider, registry=registry)
    result = agent.run("use broken tool")
    assert result.stop_reason == StopReason.DONE
    assert result.iterations == 2


def test_invariant_unknown_tool_name_produces_error_result():
    registry = ToolRegistry([])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="nonexistent", arguments={}, call_id="x")]),
        LLMResponse(content="ok"),
    ])
    agent = Agent(provider=provider, registry=registry)
    result = agent.run("call nonexistent")
    assert result.stop_reason == StopReason.DONE


def test_invariant_policy_violation_produces_error_result_not_crash():
    @tool(description="Risky op", risk=ToolRisk.EXECUTION)
    def risky() -> str:
        return "ran"

    registry = ToolRegistry([risky])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="risky", arguments={}, call_id="r1")]),
        LLMResponse(content="blocked"),
    ])
    agent = Agent(provider=provider, registry=registry, policy=ExecutionPolicy.READ_ONLY)
    result = agent.run("run risky op")
    assert result.stop_reason == StopReason.DONE


def test_invariant_tool_calls_audit_trail_complete():
    @tool(description="Echo")
    def echo(msg: str) -> str:
        return msg

    registry = ToolRegistry([echo])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="echo", arguments={"msg": "a"}, call_id="1")]),
        LLMResponse(tool_calls=[ToolCall(name="echo", arguments={"msg": "b"}, call_id="2")]),
        LLMResponse(content="done"),
    ])
    agent = Agent(provider=provider, registry=registry)
    result = agent.run("echo twice")
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].call_id == "1"
    assert result.tool_calls[1].call_id == "2"


# ── Sequence tests ───────────────────────────────────────────────────────────

def test_sequence_direct_text_response():
    registry = ToolRegistry([])
    agent = Agent(provider=MockProvider([LLMResponse(content="immediate")]), registry=registry)
    result = agent.run("What is 2+2?")
    assert result.stop_reason == StopReason.DONE
    assert result.output == "immediate"
    assert result.iterations == 1


def test_sequence_tool_then_text():
    @tool(description="Greet")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry = ToolRegistry([greet])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="greet", arguments={"name": "World"}, call_id="c1")]),
        LLMResponse(content="Greeted"),
    ])
    agent = Agent(provider=provider, registry=registry)
    result = agent.run("Greet the world")
    assert result.stop_reason == StopReason.DONE
    assert result.output == "Greeted"
    assert result.iterations == 2


def test_sequence_memory_contains_full_history():
    @tool(description="Echo")
    def echo2(msg: str) -> str:
        return msg

    registry = ToolRegistry([echo2])
    provider = MockProvider([
        LLMResponse(tool_calls=[ToolCall(name="echo2", arguments={"msg": "hi"}, call_id="m1")]),
        LLMResponse(content="done"),
    ])
    memory = ConversationMemory()
    agent = Agent(provider=provider, registry=registry, memory=memory)
    agent.run("Echo hi")
    roles = [m.role for m in memory.messages()]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_sequence_system_prompt_prepended():
    registry = ToolRegistry([])
    provider = MockProvider([LLMResponse(content="ok")])
    memory = ConversationMemory()
    agent = Agent(
        provider=provider, registry=registry, memory=memory,
        system_prompt="You are a triage bot.",
    )
    agent.run("triage these findings")
    msgs = memory.messages()
    assert msgs[0].role == "system"
    assert "triage bot" in msgs[0].content
