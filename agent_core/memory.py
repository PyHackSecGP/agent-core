# agent_core/memory.py
from __future__ import annotations
import json
from pathlib import Path

from .models import AgentMessage, ToolCall, ToolResult


class ConversationMemory:
    """In-memory message history with optional JSON file persistence."""

    def __init__(self, session_id: str = "", persist_dir: str | None = None) -> None:
        self._messages: list[AgentMessage] = []
        self._persist_path: Path | None = None
        if persist_dir and session_id:
            self._persist_path = Path(persist_dir) / f"{session_id}.json"
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, message: AgentMessage) -> None:
        """Append a message and persist if a persist path is configured."""
        self._messages.append(message)
        if self._persist_path is not None:
            self._save()

    def messages(self) -> list[AgentMessage]:
        """Return a copy of the message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()

    def load(self) -> bool:
        """Load messages from disk. Returns False if no persisted file exists."""
        if self._persist_path is None or not self._persist_path.exists():
            return False
        data: list[dict] = json.loads(self._persist_path.read_text())
        self._messages = [_from_dict(d) for d in data]
        return True

    def _save(self) -> None:
        assert self._persist_path is not None
        self._persist_path.write_text(json.dumps([_to_dict(m) for m in self._messages], indent=2))


def _to_dict(m: AgentMessage) -> dict:
    """Serialize an AgentMessage to a dictionary."""
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [{"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id} for tc in m.tool_calls],
        "tool_results": [{"call_id": tr.call_id, "content": tr.content, "is_error": tr.is_error} for tr in m.tool_results],
    }


def _from_dict(d: dict) -> AgentMessage:
    """Deserialize a dictionary to an AgentMessage."""
    return AgentMessage(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
        tool_results=[ToolResult(**tr) for tr in d.get("tool_results", [])],
    )
