# agent_core/llm/claude.py
from __future__ import annotations
import anthropic

from ..models import AgentMessage, LLMResponse, ToolCall, ToolResult, ToolSchema
from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider using the official SDK with native tool use."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 4096) -> None:
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[AgentMessage], tools: list[ToolSchema]) -> LLMResponse:
        """Send messages to Claude and return the response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[_to_msg(m) for m in messages],
            tools=[_to_tool(t) for t in tools],
        )
        tool_calls: list[ToolCall] = []
        content = ""
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, arguments=block.input, call_id=block.id))
            elif block.type == "text":
                content = block.text
        return LLMResponse(content=content, tool_calls=tool_calls)


def _to_msg(m: AgentMessage) -> dict:
    if m.role == "tool":
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tr.call_id, "content": tr.content, "is_error": tr.is_error}
                for tr in m.tool_results
            ],
        }
    if m.tool_calls:
        return {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tc.call_id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content}


def _to_tool(t: ToolSchema) -> dict:
    return {"name": t.name, "description": t.description, "input_schema": t.parameters}
