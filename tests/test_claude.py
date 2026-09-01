# tests/test_claude.py
import unittest.mock as mock
from agent_core.llm.claude import ClaudeProvider
from agent_core.models import AgentMessage, ToolCall, ToolResult, ToolSchema


def _sdk_response(blocks: list[dict]):
    content = []
    for b in blocks:
        blk = mock.MagicMock()
        blk.type = b["type"]
        if b["type"] == "text":
            blk.text = b["text"]
        else:
            blk.id = b["id"]
            blk.name = b["name"]
            blk.input = b["input"]
        content.append(blk)
    resp = mock.MagicMock()
    resp.content = content
    return resp


def test_claude_text_response():
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    sdk_resp = _sdk_response([{"type": "text", "text": "hello"}])
    with mock.patch("anthropic.Anthropic") as Cls:
        Cls.return_value.messages.create.return_value = sdk_resp
        provider._client = Cls.return_value
        result = provider.chat([AgentMessage(role="user", content="hi")], [])
    assert result.content == "hello"
    assert not result.has_tool_calls


def test_claude_tool_call_response():
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    sdk_resp = _sdk_response([
        {"type": "tool_use", "id": "tu_1", "name": "add", "input": {"a": 2, "b": 3}}
    ])
    with mock.patch("anthropic.Anthropic") as Cls:
        Cls.return_value.messages.create.return_value = sdk_resp
        provider._client = Cls.return_value
        result = provider.chat([AgentMessage(role="user", content="add")], [])
    assert result.has_tool_calls
    assert result.tool_calls[0].name == "add"
    assert result.tool_calls[0].call_id == "tu_1"
    assert result.tool_calls[0].arguments == {"a": 2, "b": 3}


def test_claude_tool_results_become_user_message():
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    sdk_resp = _sdk_response([{"type": "text", "text": "done"}])
    captured: list[dict] = []

    def fake_create(**kw):
        captured.append(kw)
        return sdk_resp

    with mock.patch("anthropic.Anthropic") as Cls:
        Cls.return_value.messages.create.side_effect = fake_create
        provider._client = Cls.return_value
        messages = [
            AgentMessage(role="user", content="go"),
            AgentMessage(role="assistant", tool_calls=[ToolCall(name="t", arguments={}, call_id="id1")]),
            AgentMessage(role="tool", tool_results=[ToolResult(call_id="id1", content="result")]),
        ]
        provider.chat(messages, [])

    sent = captured[0]["messages"]
    assert sent[-1]["role"] == "user"
    block = sent[-1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "id1"
    assert block["content"] == "result"


def test_claude_schemas_sent_as_tools():
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    sdk_resp = _sdk_response([{"type": "text", "text": "ok"}])
    captured: list[dict] = []

    def fake_create(**kw):
        captured.append(kw)
        return sdk_resp

    schema = ToolSchema(
        name="greet",
        description="Greet",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    )
    with mock.patch("anthropic.Anthropic") as Cls:
        Cls.return_value.messages.create.side_effect = fake_create
        provider._client = Cls.return_value
        provider.chat([AgentMessage(role="user", content="hi")], [schema])

    assert captured[0]["tools"][0]["name"] == "greet"
    assert captured[0]["tools"][0]["input_schema"] == schema.parameters
