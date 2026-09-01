# tests/test_memory.py
import tempfile
from agent_core.memory import ConversationMemory
from agent_core.models import AgentMessage, ToolCall, ToolResult


def test_add_and_retrieve():
    mem = ConversationMemory()
    mem.add(AgentMessage(role="user", content="hello"))
    msgs = mem.messages()
    assert len(msgs) == 1
    assert msgs[0].content == "hello"


def test_messages_returns_copy():
    mem = ConversationMemory()
    mem.add(AgentMessage(role="user", content="hi"))
    msgs = mem.messages()
    msgs.clear()
    assert len(mem.messages()) == 1


def test_clear():
    mem = ConversationMemory()
    mem.add(AgentMessage(role="user", content="hi"))
    mem.clear()
    assert mem.messages() == []


def test_persist_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ConversationMemory(session_id="sess1", persist_dir=tmpdir)
        mem.add(AgentMessage(role="user", content="persisted"))
        mem.add(AgentMessage(
            role="assistant",
            tool_calls=[ToolCall(name="t", arguments={"x": 1}, call_id="c1")],
        ))
        mem.add(AgentMessage(
            role="tool",
            tool_results=[ToolResult(call_id="c1", content="42")],
        ))

        mem2 = ConversationMemory(session_id="sess1", persist_dir=tmpdir)
        assert mem2.load() is True
        msgs = mem2.messages()
        assert len(msgs) == 3
        assert msgs[0].content == "persisted"
        assert msgs[1].tool_calls[0].name == "t"
        assert msgs[2].tool_results[0].content == "42"


def test_load_returns_false_when_no_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ConversationMemory(session_id="nonexistent", persist_dir=tmpdir)
        assert mem.load() is False


def test_no_persist_dir_works_in_memory_only():
    mem = ConversationMemory()
    mem.add(AgentMessage(role="user", content="hi"))
    assert len(mem.messages()) == 1
