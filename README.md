# agent-core — Shared agentic tool-calling loop for P1-P5 security projects

![tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![llm](https://img.shields.io/badge/llm-agnostic%20(Ollama%20%7C%20Claude)-purple)
![status](https://img.shields.io/badge/status-internal%20library-informational)

---

## 60-Second Overview

**What this is.** A tiny, reusable `Agent` class that wraps the ReAct tool-calling loop (think + act + observe, repeat until done) with four things every real-world security agent needs: a tool registry you populate with a `@tool` decorator, an LLM-provider abstraction so Ollama and Claude are swappable, a step-budget cap so a confused model cannot spin forever, and a JSONL tracer so you can audit every call after the fact.

**Who uses it.**

- **P1 — SAST+DAST Triage Tool** — `TriageAgent` consumes raw scanner findings and classifies them.
- **P2 — Threat Model Generator** — `ThreatModelAgent` turns architecture descriptions into STRIDE threats.
- **P3 — AI Log Anomaly Detector** — `LogAnalysisAgent` scores log events and maps them to MITRE ATT&CK.
- **P4 — AI Pentest Report Assistant** — the analyzer agent inside `p4-mini-crs` writes the executive report.
- **P5 — ATLAS** — the planner agent inside the autonomous HTB pipeline chooses the next recon action.

**Why.** All five projects were about to grow their own copy of the same loop. Writing it five times means fixing the same bug (max-iterations off-by-one, unknown-tool crash, policy-bypass risk) five times. One library, one audit trail.

```
pip install -e /home/tony/projects/agent-core
```

---

## For Interviewers

### Problem Statement

Every project in the P1-P5 portfolio needed the same primitive: a loop that lets an LLM call Python functions, inspect the results, and decide what to do next. The first two projects each hand-rolled it. By the third it was clear the loop was going to be rewritten five times with five subtly different bug surfaces — the dangerous kind where a tool exception on P1 crashes the agent, but on P2 silently returns empty text, and nobody notices which behaviour is actually correct. `agent-core` is the shared abstraction: one loop, one policy model, one tracer, five consumers.

### Skills Demonstrated

- **ReAct loop pattern** — the classic *reason → act → observe → reason* cycle, implemented in ~45 lines in `agent_core/agent.py`.
- **Tool registration via decorator** — `@tool(description=..., risk=...)` auto-generates a JSON Schema from the Python signature using `inspect` + `typing.get_type_hints`. No manual schema maintenance.
- **LLM provider abstraction** — `LLMProvider` ABC with two concrete implementations: `OllamaProvider` (stdlib-only HTTP, zero extra deps) and `ClaudeProvider` (official Anthropic SDK with native tool use).
- **Policy-enforced execution** — four-tier `ExecutionPolicy` (`READ_ONLY`, `SAFE`, `AUTO`, `HUMAN_CONFIRM`) crossed with per-tool `ToolRisk` (`NONE`, `FILESYSTEM`, `EXECUTION`, `EXTERNAL_SIDE_EFFECT`). Policy violations become `ToolResult(is_error=True)` the model can read — never a crash.
- **Step-cap + forced-conclude** — `max_iterations` guarantees the loop terminates; when hit, the result carries `StopReason.MAX_ITERATIONS` instead of hanging or raising.
- **Audit trail** — `AgentTracer` writes one JSONL line per tool call with provider, model, args, duration, truncated output, and error flag. Grep-friendly, SIEM-friendly.
- **Dataclass models over Pydantic** — zero runtime dependency for the core; dataclasses are enough.
- **Pytest, no fixtures theatre** — 46 tests across 7 files split into *invariant* tests (never crashes, never exceeds budget) and *sequence* tests (loop produces the expected transcript).

### Architecture

```
                              ┌─────────────────────────────────┐
                              │           Agent                 │
   goal: str  ──────────────▶ │  (ReAct loop, max_iterations)   │
                              └──────────────┬──────────────────┘
                                             │
                   ┌─────────────────────────┼──────────────────────────┐
                   │                         │                          │
                   ▼                         ▼                          ▼
         ┌──────────────────┐      ┌──────────────────┐       ┌──────────────────┐
         │ ConversationMem  │      │  LLMProvider     │       │  ToolRegistry    │
         │  (role history)  │      │   (ABC)          │       │  @tool decorator │
         └──────────────────┘      │                  │       │                  │
                                   │  ├─ Ollama       │       │  ├─ schema()     │
                                   │  └─ Claude       │       │  └─ execute()    │
                                   └──────────────────┘       │     ↓            │
                                                              │  policy check    │
                                                              │     ↓            │
                                                              │  user Python fn  │
                                                              └──────────────────┘

   Loop (up to max_iterations):

       memory ──▶ provider.chat(messages, schemas) ──▶ LLMResponse
                                                            │
                     ┌──────────────────────────────────────┤
                     │                                      │
         has_tool_calls? YES                      has_tool_calls? NO
                     │                                      │
                     ▼                                      ▼
         registry.execute(each call)                return AgentResult(
                     │                                 output=content,
                     ▼                                 stop_reason=DONE,
          append ToolResult to memory                  iterations=i+1,
                     │                                 tool_calls=all)
                     ▼
          tracer.log_tool_call(...)
                     │
                     └──▶ next iteration

   Fell off the end?  ──▶  AgentResult(output=None, stop_reason=MAX_ITERATIONS)
```

### Design Decisions

- **Why ReAct, not plan-then-execute.** Plan-execute looks tidier on a slide but hard-fails when step 3 depends on something only visible from step 2's output. Security findings and recon are exactly that shape: you cannot plan the whole thing up front because the triage path depends on what the scanner actually returned. ReAct re-reasons every step, so it survives surprises for free.
- **Why callable tools, not JSON-schemas-only.** The user writes a normal Python function. The decorator reads its signature with `inspect.signature` + `typing.get_type_hints` and generates the JSON schema. One source of truth. No drift between "what the schema says" and "what the function does".
- **Why step cap, not time cap.** A step cap is deterministic, trivially testable, and model-agnostic. A wall-clock cap depends on network latency and model speed and would make the invariant tests flaky.
- **Why `MAX_ITERATIONS` returns a result, not raises.** Callers (P1-P5) always need to log and return something, never crash out of the loop. Raising means every caller writes the same `try/except` five times. Returning a result with `stop_reason=MAX_ITERATIONS` keeps the loop a value-returning function and leaves the error-handling decision to the caller.
- **Why provider-agnostic.** P1-P4 run on Ollama for cost (hermes3:70b, zero API bill). P5 ATLAS runs on Claude for reasoning quality on CTFs. Same agent code, different provider in the constructor.
- **Why dataclasses, not Pydantic.** Core has one runtime dep (`anthropic`) and that's for the Claude provider only. Ollama provider is stdlib. The models never need validation at construction — they're produced by code we control, not parsed from untrusted input.
- **Why tool errors become `ToolResult(is_error=True)`.** The model has to see the failure to recover. If the loop raised, the model would never learn its tool call was wrong. Policy violations, unknown tool names, and raised exceptions all funnel into the same error-shaped result so the model's recovery path is uniform.

### Metrics

| Metric | Value |
| --- | --- |
| Core LOC | 553 across 10 files |
| Tests | 46 across 7 files (`pytest` only, no fixtures) |
| Runtime deps | 1 (`anthropic`, only for Claude provider; Ollama provider is stdlib) |
| Consumers | 5 (P1 triage, P2 threat model, P3 log anomaly, P4 report, P5 ATLAS planner) |
| Python | 3.11+ |

---

## Quick Start

```bash
# From any consumer project:
pip install -e /home/tony/projects/agent-core
```

Hello-world with one tool:

```python
from agent_core import Agent, tool, ToolRegistry
from agent_core.llm.ollama import OllamaProvider

@tool(description="Return the length of a string.")
def strlen(text: str) -> int:
    return len(text)

agent = Agent(
    provider=OllamaProvider(model="hermes3:70b"),
    registry=ToolRegistry([strlen]),
    system_prompt="You are a helper. Use the strlen tool when asked about lengths.",
)

result = agent.run("How long is the word 'elephant'?")
print(result.output)          # -> "The word 'elephant' is 8 characters long."
print(result.stop_reason)     # -> StopReason.DONE
print(result.iterations)      # -> 2
print(result.tool_calls)      # -> [ToolCall(name='strlen', arguments={'text': 'elephant'}, ...)]
```

Swap to Claude by changing one line:

```python
from agent_core.llm.claude import ClaudeProvider
agent = Agent(provider=ClaudeProvider(model="claude-sonnet-4-6"), registry=ToolRegistry([strlen]))
```

---

## API Reference

### `Agent`

```python
Agent(
    provider: LLMProvider,
    registry: ToolRegistry,
    max_iterations: int = 20,
    memory: ConversationMemory | None = None,
    system_prompt: str = "",
    policy: ExecutionPolicy = ExecutionPolicy.AUTO,
    tracer: AgentTracer | None = None,
    name: str = "agent",
)
```

- `run(goal: str) -> AgentResult` — executes the loop. Always returns a result; never raises for tool failures.

### `@tool` decorator

```python
@tool(
    description: str,                     # shown to the model
    risk: ToolRisk = ToolRisk.NONE,       # NONE | FILESYSTEM | EXECUTION | EXTERNAL_SIDE_EFFECT
    requires_confirmation: bool = False,  # true -> only runnable under HUMAN_CONFIRM policy
)
def my_tool(arg1: str, arg2: int = 0) -> str | dict: ...
```

The JSON schema is built from the signature. Non-string return values are JSON-serialised before being handed back to the model.

### `ToolRegistry`

```python
ToolRegistry(tools: list[Callable] | None = None)
    .schemas() -> list[ToolSchema]
    .execute(call: ToolCall, policy: ExecutionPolicy) -> ToolResult
```

Passing `None` picks up every `@tool`-decorated function in the process. Passing an explicit list scopes an agent to only the tools you want it to see.

### `LLMProvider` protocol

```python
class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[AgentMessage], tools: list[ToolSchema]) -> LLMResponse: ...
```

Two shipped implementations:

- `agent_core.llm.ollama.OllamaProvider(model="hermes3:70b", base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))`
- `agent_core.llm.claude.ClaudeProvider(model="claude-sonnet-4-6", max_tokens=4096)`

### Result schema

```python
@dataclass
class AgentResult:
    output: str | None            # final text, None if max iterations hit
    stop_reason: StopReason       # DONE | MAX_ITERATIONS | ERROR
    iterations: int               # number of loop steps actually executed
    tool_calls: list[ToolCall]    # complete audit trail
    error: str | None = None
    artifacts: list[Artifact] = []
```

---

## Examples

### 1. Triage a security finding (P1 pattern)

```python
from agent_core import Agent, tool, ToolRegistry, ExecutionPolicy
from agent_core.llm.ollama import OllamaProvider

@tool(description="Look up CVSS score for a CVE id.")
def cvss_lookup(cve_id: str) -> dict:
    # ... call NVD or local cache ...
    return {"cve": cve_id, "score": 7.5, "vector": "AV:N/AC:L/PR:N/UI:N/..."}

@tool(description="Check if a path is reachable from any entrypoint.")
def is_reachable(path: str) -> bool:
    return path.startswith("src/api/")

agent = Agent(
    provider=OllamaProvider(),
    registry=ToolRegistry([cvss_lookup, is_reachable]),
    system_prompt="You triage SAST findings. Mark as DISMISS when CVSS<4 or path unreachable.",
    policy=ExecutionPolicy.READ_ONLY,
)
result = agent.run("Finding: SQL injection in src/legacy/old_db.py referencing CVE-2021-44228.")
```

### 2. Analyze a single log event (P3 pattern)

```python
@tool(description="Fetch previous 10 events from the same source IP.")
def recent_events(ip: str) -> list[dict]:
    return [...]

@tool(description="Lookup MITRE ATT&CK technique by name.")
def attack_lookup(name: str) -> dict:
    return {"id": "T1078", "tactic": "Initial Access", ...}

agent = Agent(
    provider=OllamaProvider(),
    registry=ToolRegistry([recent_events, attack_lookup]),
    system_prompt="You score anomalies 0-100 and map to MITRE ATT&CK.",
    max_iterations=10,
)
result = agent.run('Event: {"ts": "...", "src_ip": "203.0.113.5", "action": "auth_fail", "count": 47}')
```

### 3. Write a short pentest report section (P4 pattern)

```python
from agent_core.llm.claude import ClaudeProvider

@tool(description="Read the finding record by id.")
def read_finding(finding_id: str) -> dict:
    return {"id": finding_id, "title": "RCE via unsafe deserialization", ...}

agent = Agent(
    provider=ClaudeProvider(model="claude-sonnet-4-6"),
    registry=ToolRegistry([read_finding]),
    system_prompt="Write a 10-line executive summary suitable for a non-technical reader.",
    max_iterations=5,
)
result = agent.run("Summarize finding F-00123.")
print(result.output)
```

---

## Technical Reference

### Install

```bash
# Editable install (recommended for consumer projects in the same checkout):
cd /path/to/consumer-project
pip install -e /home/tony/projects/agent-core

# Or depend by path in the consumer's pyproject.toml:
# dependencies = ["agent-core @ file:///home/tony/projects/agent-core"]
```

Python 3.11+ required (for `X | Y` type syntax and `StrEnum`-adjacent patterns in `models.py`).

### LLM provider implementations

**Ollama (`agent_core/llm/ollama.py`).** Pure `urllib.request`. Reads `OLLAMA_HOST` env var (defaults to `http://localhost:11434`). Converts internal messages into Ollama's `/api/chat` tool-use format. 120-second request timeout. Zero extra deps.

**Claude (`agent_core/llm/claude.py`).** Uses the official `anthropic` SDK. Reads `ANTHROPIC_API_KEY` the standard way (SDK handles it). Converts internal messages into Anthropic's native `tool_use` / `tool_result` content blocks. System prompt is pulled out of the message list and passed as the top-level `system=` kwarg.

### Config

| What | Where | Default |
| --- | --- | --- |
| Ollama endpoint | `OLLAMA_HOST` env var | `http://localhost:11434` |
| Anthropic key | `ANTHROPIC_API_KEY` env var | *(none; required for Claude)* |
| Step budget | `Agent(max_iterations=...)` | `20` |
| Execution policy | `Agent(policy=...)` | `ExecutionPolicy.AUTO` |
| Memory persistence | `ConversationMemory(session_id=..., persist_dir=...)` | in-memory only |
| Tracing | `Agent(tracer=AgentTracer(session_id, log_dir=...))` | disabled |

### Step budget semantics

`max_iterations` counts *provider calls*, not tool executions. One iteration = one `provider.chat()` + (if the response had tool calls) the full fan-out of those tool executions. Hitting the cap returns `StopReason.MAX_ITERATIONS` with the full `tool_calls` audit trail intact.

### Extending — adding a new provider

Subclass `LLMProvider` and implement `chat()`:

```python
from agent_core.llm.base import LLMProvider
from agent_core.models import AgentMessage, LLMResponse, ToolCall, ToolSchema

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o") -> None:
        import openai
        self._client = openai.OpenAI()
        self.model = model

    def chat(self, messages: list[AgentMessage], tools: list[ToolSchema]) -> LLMResponse:
        # Translate AgentMessage -> OpenAI messages, ToolSchema -> OpenAI tool spec.
        # Translate response.choices[0].message back into LLMResponse(content=..., tool_calls=[...]).
        ...
```

Mirror the two shipped providers for message/tool translation shape.

### Extending — tool patterns

- **Pure function tool.** Decorate; the schema is auto-built. Return a string or anything JSON-serialisable.
- **Side-effecting tool.** Mark with `risk=ToolRisk.EXECUTION` (and optionally `requires_confirmation=True`). Callers can then enforce `ExecutionPolicy.READ_ONLY` / `SAFE` to block it.
- **Filesystem tool.** `risk=ToolRisk.FILESYSTEM` — permitted under `SAFE` but not under `READ_ONLY`.
- **External side-effect tool.** `risk=ToolRisk.EXTERNAL_SIDE_EFFECT` for anything that touches a third-party service (email, webhook, paid API). Blocked unless policy is `AUTO` or `HUMAN_CONFIRM`.

Tool functions should **not** raise for predictable error cases — return an error-shaped string instead so the model can read it and recover. The registry *does* catch raised exceptions and convert them to error results, but intentional error-paths are clearer than ones that look like bugs.

---

## Where It's Used

| Project | Role of agent-core | Key tools |
| --- | --- | --- |
| **P1 — SAST+DAST Triage Tool** | `TriageAgent` classifies scanner findings as EXPLOITABLE / NEEDS-REVIEW / DISMISS and emits SARIF annotations. | `cvss_lookup`, `is_reachable`, `check_waf`, `write_sarif` |
| **P2 — Threat Model Generator** | `ThreatModelAgent` turns architecture text into STRIDE threats, data flows, trust boundaries. | `list_components`, `enumerate_flows`, `score_stride` |
| **P3 — AI Log Anomaly Detector** | `LogAnalysisAgent` scores single events, maps to MITRE ATT&CK, enforces allowlist. | `recent_events`, `attack_lookup`, `check_allowlist` |
| **P4 — AI Pentest Report Assistant** | Analyzer agent inside `p4-mini-crs` composes the executive summary from a finding store. | `read_finding`, `cvss_lookup`, `write_markdown` |
| **P5 — ATLAS** | Planner agent inside the autonomous HTB pipeline picks the next recon action; recon/exploit/privesc agents each use their own tool set over the same loop. | `nmap_scan`, `nse_run`, `hydra_try`, `ssh_run` |

---

## Why This Was Built

Three motivations, in order:

1. **Reuse.** Five projects, five copies of a loop was five chances to drift. One library, one bugfix.
2. **Policy enforceability.** A tool marked `EXECUTION` must be blockable without the caller writing a per-tool `if`. Centralising the policy check in the registry made the one-line guard possible.
3. **Audit trail.** Security projects need a defensible record of what the model did. The tracer writes one JSONL line per tool call — grep-friendly, SIEM-ingestible, no custom log parser needed.

The library is intentionally *not* a framework. It has no CLI, no YAML config, no server mode. Consumers import the `Agent` class and compose. That is the entire surface area.

---

## FAQ

**Q: Why not use LangChain / LlamaIndex / Autogen / crewai?**
Each of those pulls in a tree of dependencies larger than the entire P1-P5 codebase combined, and each imposes its own tool-definition DSL. The whole loop here is ~45 lines. Importing a megaframework to replace 45 lines is the opposite of lazy.

**Q: Why not use Anthropic's tool_use SDK directly in each project?**
Because then every project is Claude-locked. The Ollama provider was the whole point — Ollama is free, runs locally on the homelab, and handles P1-P4's workload fine. The abstraction is the escape hatch for cost.

**Q: Can tools be async?**
Not currently. All five consumers are batch jobs where blocking is fine. If a consumer ever needs streaming or concurrent tool fan-out, add it to the loop then, not now.

**Q: What happens if the model hallucinates a tool name?**
`ToolRegistry.execute` returns `ToolResult(is_error=True, content="Unknown tool: X")`. The model sees the error on the next iteration and typically self-corrects. Covered by `test_invariant_unknown_tool_name_produces_error_result`.

**Q: What happens if a tool function raises an exception?**
The registry catches it, converts the exception message to a `ToolResult(is_error=True)`, and the loop continues. The model sees the error text. Covered by `test_invariant_tool_error_reaches_model_not_exception`.

**Q: Why `max_iterations=20` as the default?**
Empirically, every P1-P5 real task finishes in under 10 steps. 20 is "double the headroom and still cheap". Override per agent — ATLAS planner uses more, triage agents use less.

**Q: Can two agents share tools?**
Yes. The `@tool` decorator registers into a process-wide `_REGISTRY`. Instantiating `ToolRegistry()` with no arguments picks up everything; instantiating with an explicit list scopes an agent to a subset. Keeps the common case one-line and the scoped case explicit.

**Q: Is there a plan-execute version?**
No, and probably never. The portfolio's workloads are discovery-shaped, not planning-shaped. If a future project genuinely needs plan-execute, add it as a sibling `PlanAgent` class, don't retrofit the ReAct one.

---

## License

MIT. Internal portfolio code. Not currently published to PyPI — install from path.
