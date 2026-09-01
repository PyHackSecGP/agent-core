# tests/test_tracer.py
import json
import tempfile
from pathlib import Path
from agent_core.tracer import AgentTracer
from agent_core.models import ToolCall, ToolResult
import pytest


def test_tracer_writes_jsonl_on_tool_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracer = AgentTracer(session_id="s1", log_dir=tmpdir)
        call = ToolCall(name="parse_log", arguments={"file": "/var/log/auth.log"}, call_id="c1")
        result = ToolResult(call_id="c1", content="42 events", is_error=False)
        tracer.log_tool_call(
            agent="LogMonitorAgent", iteration=1,
            provider="OllamaProvider", model="hermes3:70b",
            call=call, result=result, duration_ms=123.4,
        )
        log_file = Path(tmpdir) / "s1.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool_name"] == "parse_log"
        assert entry["tool_error"] is False
        assert entry["tool_duration_ms"] == pytest.approx(123.4)
        assert entry["agent"] == "LogMonitorAgent"


def test_tracer_writes_error_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracer = AgentTracer(session_id="s2", log_dir=tmpdir)
        tracer.log_error(agent="TriageAgent", error="provider timeout")
        log_file = Path(tmpdir) / "s2.jsonl"
        entry = json.loads(log_file.read_text().strip())
        assert entry["error"] == "provider timeout"


def test_tracer_no_log_dir_does_not_crash():
    tracer = AgentTracer(session_id="s3", log_dir=None)
    call = ToolCall(name="t", arguments={}, call_id="c")
    result = ToolResult(call_id="c", content="ok")
    tracer.log_tool_call("agent", 0, "Ollama", "model", call, result, 10.0)  # must not raise


def test_tracer_truncates_long_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracer = AgentTracer(session_id="s4", log_dir=tmpdir)
        call = ToolCall(name="t", arguments={}, call_id="c")
        result = ToolResult(call_id="c", content="x" * 1000)
        tracer.log_tool_call("agent", 0, "Ollama", "model", call, result, 5.0)
        entry = json.loads(Path(tmpdir, "s4.jsonl").read_text().strip())
        assert len(entry["tool_result"]) <= 500
