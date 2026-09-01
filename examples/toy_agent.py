#!/usr/bin/env python3
"""End-to-end smoke test: agent uses add + multiply to answer a math goal."""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_core import Agent, tool, ToolRegistry, AgentTracer
from agent_core.models import ExecutionPolicy
from agent_core.llm import get_provider


@tool(description="Add two integers and return the sum")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(description="Multiply two integers and return the product")
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def main() -> None:
    """Run the toy agent smoke test."""
    provider = get_provider(
        provider=os.environ.get("AGENT_PROVIDER", "ollama"),
        model=os.environ.get("AGENT_MODEL", "hermes3:70b"),
    )
    registry = ToolRegistry([add, multiply])

    with tempfile.TemporaryDirectory() as log_dir:
        tracer = AgentTracer(session_id="toy-demo", log_dir=log_dir)
        agent = Agent(
            provider=provider,
            registry=registry,
            max_iterations=10,
            policy=ExecutionPolicy.AUTO,
            tracer=tracer,
            name="ToyAgent",
            system_prompt="You are a math assistant. Use the available tools to compute answers.",
        )

        goal = "Calculate (3 + 4) * 2 using the available tools, then give me the final numeric answer."
        print(f"Goal: {goal}\n")

        result = agent.run(goal)

        print(f"Output:     {result.output}")
        print(f"Iterations: {result.iterations}")
        print(f"Stop:       {result.stop_reason.value}")
        print(f"Tools used: {[tc.name for tc in result.tool_calls]}")


if __name__ == "__main__":
    main()
