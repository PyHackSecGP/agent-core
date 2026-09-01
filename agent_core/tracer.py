# agent_core/tracer.py
from __future__ import annotations
import json
import time
from pathlib import Path

from .models import ToolCall, ToolResult


class AgentTracer:
    """JSONL observability logger — one line per tool call."""

    def __init__(self, session_id: str, log_dir: str | None = None) -> None:
        self.session_id = session_id
        self._path: Path | None = None
        if log_dir:
            self._path = Path(log_dir) / f"{session_id}.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_tool_call(
        self,
        agent: str,
        iteration: int,
        provider: str,
        model: str,
        call: ToolCall,
        result: ToolResult,
        duration_ms: float,
    ) -> None:
        """Log a single tool call to the JSONL file."""
        self._write({
            "ts": time.time(),
            "session_id": self.session_id,
            "agent": agent,
            "iteration": iteration,
            "provider": provider,
            "model": model,
            "tool_name": call.name,
            "tool_args": call.arguments,
            "tool_duration_ms": duration_ms,
            "tool_result": (result.content or "")[:500],
            "tool_error": result.is_error,
        })

    def log_error(self, agent: str, error: str) -> None:
        """Log an agent-level error to the JSONL file."""
        self._write({
            "ts": time.time(),
            "session_id": self.session_id,
            "agent": agent,
            "error": error,
        })

    def _write(self, entry: dict) -> None:
        """Write a single entry to the JSONL file."""
        if self._path is not None:
            with self._path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
