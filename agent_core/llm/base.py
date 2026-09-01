# agent_core/llm/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from ..models import AgentMessage, LLMResponse, ToolSchema


class LLMProvider(ABC):
    """Abstract base class for LLM provider integrations."""

    @abstractmethod
    def chat(self, messages: list[AgentMessage], tools: list[ToolSchema]) -> LLMResponse:
        """Send messages and tool schemas to the LLM. Return the response."""
        ...
