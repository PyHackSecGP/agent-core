# agent_core/tool.py
from __future__ import annotations
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .models import ExecutionPolicy, ToolCall, ToolResult, ToolRisk, ToolSchema

_REGISTRY: dict[str, "ToolDefinition"] = {}


@dataclass
class ToolDefinition:
    """Internal representation of a registered tool."""

    fn: Callable
    name: str
    schema: ToolSchema
    risk: ToolRisk
    requires_confirmation: bool

    def call(self, arguments: dict[str, Any]) -> str:
        """Execute the tool function and return JSON-serialised result."""
        result = self.fn(**arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)


def tool(
    description: str,
    risk: ToolRisk = ToolRisk.NONE,
    requires_confirmation: bool = False,
) -> Callable:
    """Decorator that registers a function as an agent tool."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[fn.__name__] = ToolDefinition(
            fn=fn,
            name=fn.__name__,
            schema=_build_schema(fn, description),
            risk=risk,
            requires_confirmation=requires_confirmation,
        )
        return fn
    return decorator


def _build_schema(fn: Callable, description: str) -> ToolSchema:
    """Build a ToolSchema from a function signature and description."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        hint = hints.get(name, str)
        properties[name] = {"type": _py_to_json(hint), "description": name}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return ToolSchema(
        name=fn.__name__,
        description=description,
        parameters={"type": "object", "properties": properties, "required": required},
    )


def _py_to_json(hint: Any) -> str:
    """Convert a Python type hint to a JSON Schema type."""
    return {
        str: "string", int: "integer", float: "number",
        bool: "boolean", list: "array", dict: "object",
    }.get(hint, "string")


def _is_permitted(defn: ToolDefinition, policy: ExecutionPolicy) -> bool:
    """Check if a tool is permitted under the given execution policy."""
    if policy == ExecutionPolicy.READ_ONLY:
        return defn.risk == ToolRisk.NONE
    if policy == ExecutionPolicy.SAFE:
        return defn.risk in (ToolRisk.NONE, ToolRisk.FILESYSTEM)
    return True


class ToolRegistry:
    """Registry of available tools with policy-enforced dispatch."""

    def __init__(self, tools: list[Callable] | None = None) -> None:
        if tools is None:
            self._tools: dict[str, ToolDefinition] = dict(_REGISTRY)
        else:
            self._tools = {
                fn.__name__: _REGISTRY[fn.__name__]
                for fn in tools
                if fn.__name__ in _REGISTRY
            }

    def schemas(self) -> list[ToolSchema]:
        """Return JSON schemas for all registered tools."""
        return [d.schema for d in self._tools.values()]

    def execute(self, call: ToolCall, policy: ExecutionPolicy) -> ToolResult:
        """Execute a tool call under the given policy. Always returns ToolResult — never raises."""
        defn = self._tools.get(call.name)
        if defn is None:
            return ToolResult(call_id=call.call_id, content=f"Unknown tool: {call.name}", is_error=True)

        if not _is_permitted(defn, policy):
            return ToolResult(
                call_id=call.call_id,
                content=f"Tool '{call.name}' (risk={defn.risk.value}) not permitted under policy {policy.value}",
                is_error=True,
            )

        if defn.requires_confirmation and policy != ExecutionPolicy.HUMAN_CONFIRM:
            return ToolResult(
                call_id=call.call_id,
                content=f"Tool '{call.name}' requires confirmation; current policy is {policy.value}",
                is_error=True,
            )

        try:
            content = defn.call(call.arguments)
            return ToolResult(call_id=call.call_id, content=content)
        except Exception as exc:
            return ToolResult(call_id=call.call_id, content=str(exc), is_error=True)
