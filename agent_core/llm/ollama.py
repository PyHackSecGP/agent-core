# agent_core/llm/ollama.py
from __future__ import annotations
import json
import urllib.request
import uuid

from ..models import AgentMessage, LLMResponse, ToolCall, ToolResult, ToolSchema
from .base import LLMProvider

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama HTTP API provider — stdlib only, no extra dependencies."""

    def __init__(self, model: str = "hermes3:70b", base_url: str = DEFAULT_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[AgentMessage], tools: list[ToolSchema]) -> LLMResponse:
        """Send messages to Ollama and return the response."""
        payload = {
            "model": self.model,
            "messages": _flatten(messages),
            "tools": [_tool_schema(t) for t in tools],
            "stream": False,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())

        msg = body["message"]
        tool_calls = [
            ToolCall(
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", {}),
                call_id=tc.get("id", str(uuid.uuid4())),
            )
            for tc in msg.get("tool_calls", [])
        ]
        return LLMResponse(content=msg.get("content", ""), tool_calls=tool_calls)


def _flatten(messages: list[AgentMessage]) -> list[dict]:
    result: list[dict] = []
    for m in messages:
        if m.role == "tool":
            for tr in m.tool_results:
                result.append({"role": "tool", "content": tr.content, "tool_call_id": tr.call_id})
        elif m.tool_calls:
            result.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ],
            })
        else:
            result.append({"role": m.role, "content": m.content})
    return result


def _tool_schema(t: ToolSchema) -> dict:
    return {
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
    }
