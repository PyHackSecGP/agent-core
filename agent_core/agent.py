# agent_core/agent.py
from __future__ import annotations
import time

from .memory import ConversationMemory
from .models import AgentMessage, AgentResult, ExecutionPolicy, StopReason, ToolCall
from .tool import ToolRegistry
from .tracer import AgentTracer
from .llm.base import LLMProvider


class AgentLoopError(Exception):
    """Raised when the agent loop terminates abnormally."""
    pass


class Agent:
    """Tool-calling agent loop with policy enforcement, memory, and observability."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_iterations: int = 20,
        memory: ConversationMemory | None = None,
        system_prompt: str = "",
        policy: ExecutionPolicy = ExecutionPolicy.AUTO,
        tracer: AgentTracer | None = None,
        name: str = "agent",
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self.memory = memory or ConversationMemory()
        self.system_prompt = system_prompt
        self.policy = policy
        self.tracer = tracer
        self.name = name

    def run(self, goal: str) -> AgentResult:
        """Run the agent loop until the model returns final text or max_iterations is hit."""
        all_tool_calls: list[ToolCall] = []

        if self.system_prompt:
            self.memory.add(AgentMessage(role="system", content=self.system_prompt))
        self.memory.add(AgentMessage(role="user", content=goal))

        for i in range(self.max_iterations):
            response = self.provider.chat(self.memory.messages(), self.registry.schemas())

            if response.has_tool_calls:
                self.memory.add(AgentMessage(role="assistant", tool_calls=response.tool_calls))
                results = []
                for tc in response.tool_calls:
                    t0 = time.monotonic()
                    result = self.registry.execute(tc, self.policy)
                    duration_ms = (time.monotonic() - t0) * 1000
                    results.append(result)
                    all_tool_calls.append(tc)
                    if self.tracer:
                        self.tracer.log_tool_call(
                            agent=self.name,
                            iteration=i,
                            provider=type(self.provider).__name__,
                            model=getattr(self.provider, "model", "unknown"),
                            call=tc,
                            result=result,
                            duration_ms=duration_ms,
                        )
                self.memory.add(AgentMessage(role="tool", tool_results=results))
            else:
                self.memory.add(AgentMessage(role="assistant", content=response.content))
                return AgentResult(
                    output=response.content,
                    stop_reason=StopReason.DONE,
                    iterations=i + 1,
                    tool_calls=all_tool_calls,
                )

        return AgentResult(
            output=None,
            stop_reason=StopReason.MAX_ITERATIONS,
            iterations=self.max_iterations,
            tool_calls=all_tool_calls,
            error=f"Agent did not finish within {self.max_iterations} iterations",
        )
