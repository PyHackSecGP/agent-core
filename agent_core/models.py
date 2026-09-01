"""Data models for agent-core package."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(Enum):
    """Enumeration of reasons agent loop terminates."""

    DONE = "done"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


class ToolRisk(Enum):
    """Enumeration of risk levels for tool execution."""

    NONE = "none"
    FILESYSTEM = "filesystem"
    EXECUTION = "execution"
    EXTERNAL_SIDE_EFFECT = "external"


class ExecutionPolicy(Enum):
    """Enumeration of tool execution policies."""

    READ_ONLY = "read_only"  # NONE risk only
    SAFE = "safe"  # NONE + FILESYSTEM
    AUTO = "auto"  # All except requires_confirmation tools
    HUMAN_CONFIRM = "human"  # All tools including confirmation-required


@dataclass
class ToolSchema:
    """Schema definition for a tool that an agent can call.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing the tool's input parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A request by the LLM to execute a tool.

    Attributes:
        name: Name of the tool to call.
        arguments: Dictionary of arguments to pass to the tool.
        call_id: Unique identifier for this tool call (empty by default).
    """

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    """Result returned after executing a tool.

    Attributes:
        call_id: ID of the ToolCall this result corresponds to.
        content: Output or result of the tool execution.
        is_error: Whether the tool execution resulted in an error.
    """

    call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentMessage:
    """A message in the agent conversation history.

    Attributes:
        role: Role of the sender (e.g., "user", "assistant", "system").
        content: Text content of the message.
        tool_calls: List of tool calls made in this message.
        tool_results: List of tool results received.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: Text content of the response.
        tool_calls: List of tool calls requested by the LLM.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        """Check if this response contains any tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class Artifact:
    """A file or data artifact produced by the agent.

    Attributes:
        name: Filename or identifier for the artifact.
        path: File system path or URI where the artifact is stored.
        content_type: Classification of artifact content (e.g., "report", "sarif", "patch", "writeup").
    """

    name: str
    path: str
    content_type: str


@dataclass
class AgentResult:
    """Final result of an agent execution run.

    Attributes:
        output: Final output text from the agent (None if max iterations reached).
        stop_reason: Reason the agent loop terminated.
        iterations: Number of iterations the agent ran.
        tool_calls: All tool calls made during execution.
        error: Error message if stop_reason is ERROR, else None.
        artifacts: List of artifacts produced by the agent.
    """

    output: str | None
    stop_reason: StopReason
    iterations: int
    tool_calls: list[ToolCall]
    error: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
