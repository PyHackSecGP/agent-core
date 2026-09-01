from __future__ import annotations
import os

from .base import LLMProvider
from .claude import ClaudeProvider
from .ollama import OllamaProvider

__all__ = ["LLMProvider", "ClaudeProvider", "OllamaProvider", "get_provider"]


def get_provider(provider: str = "", model: str = "") -> LLMProvider:
    """Return LLMProvider from name or AGENT_PROVIDER env var.

    provider: "claude" | "ollama" (default: "ollama")
    model: override default model name
    """
    p = provider or os.environ.get("AGENT_PROVIDER", "ollama")
    if p == "claude":
        return ClaudeProvider(model=model or "claude-sonnet-4-6")
    return OllamaProvider(model=model or "hermes3:70b")
