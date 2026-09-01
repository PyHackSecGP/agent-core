# tests/test_ollama.py
import json
import unittest.mock as mock
from agent_core.llm.ollama import OllamaProvider
from agent_core.models import AgentMessage, ToolCall, ToolResult, ToolSchema


def _fake_urlopen(body: dict):
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = mock.MagicMock(return_value=False)
    return resp


def test_ollama_text_response():
    provider = OllamaProvider(model="hermes3:70b")
    body = {"message": {"role": "assistant", "content": "hello", "tool_calls": []}}
    with mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(body)):
        result = provider.chat([AgentMessage(role="user", content="hi")], [])
    assert result.content == "hello"
    assert not result.has_tool_calls


def test_ollama_tool_call_response():
    provider = OllamaProvider(model="hermes3:70b")
    body = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}],
        }
    }
    with mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(body)):
        result = provider.chat([AgentMessage(role="user", content="add")], [])
    assert result.has_tool_calls
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].arguments == {"a": 1, "b": 2}
    assert result.tool_calls[0].call_id != ""


def test_ollama_tool_results_expand_to_separate_messages():
    """One AgentMessage(role=tool) with N results → N Ollama messages."""
    provider = OllamaProvider(model="hermes3:70b")
    body = {"message": {"role": "assistant", "content": "done", "tool_calls": []}}
    captured: list[dict] = []

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data))
        return _fake_urlopen(body)

    messages = [
        AgentMessage(role="user", content="go"),
        AgentMessage(role="assistant", tool_calls=[
            ToolCall(name="t1", arguments={}, call_id="id1"),
            ToolCall(name="t2", arguments={}, call_id="id2"),
        ]),
        AgentMessage(role="tool", tool_results=[
            ToolResult(call_id="id1", content="r1"),
            ToolResult(call_id="id2", content="r2"),
        ]),
    ]
    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        provider.chat(messages, [])

    sent_msgs = captured[0]["messages"]
    tool_msgs = [m for m in sent_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "id1"
    assert tool_msgs[1]["tool_call_id"] == "id2"


def test_ollama_sends_tool_schemas():
    provider = OllamaProvider(model="hermes3:70b")
    body = {"message": {"role": "assistant", "content": "ok", "tool_calls": []}}
    captured: list[dict] = []

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data))
        return _fake_urlopen(body)

    schema = ToolSchema(
        name="greet",
        description="Greet someone",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    )
    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        provider.chat([AgentMessage(role="user", content="hi")], [schema])

    tools_sent = captured[0]["tools"]
    assert len(tools_sent) == 1
    assert tools_sent[0]["function"]["name"] == "greet"
