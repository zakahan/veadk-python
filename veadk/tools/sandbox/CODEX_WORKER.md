# Streaming CodeEnv delegation

```python
from veadk import Agent
from veadk.tools.sandbox.codex_sandbox_tool import CodexSandboxTool

agent = Agent(
    name="assistant",
    instruction="Delegate coding work to sandbox_task, then summarize its result.",
    tools=[CodexSandboxTool(endpoint="https://your-codeenv-session", api_key=None)],
)
```

Requires an existing CodeEnv session exposing the Rust worker protocol v1. The
endpoint may include platform authentication query parameters, which are retained.
The parent agent's model credentials are separate from Codex's sandbox model.
Automatic AgentKit provisioning is not performed by this tool.

`veadk.Agent` opts into an invocation-local event bridge for tools declaring
`emits_progress`. This works under both VeADK Runner and the Google ADK Runner used
by the AgentKit app. Ordinary tools and other agent runtimes keep their existing
behavior. The bridge waits for consumer acknowledgement before advancing ordinary
agent events, preserving Runner's session-append ordering.

Progress uses partial ADK Events with `custom_metadata["codex_worker"]`, including
worker session/turn/item identifiers and the originating functionCallId. Text
deltas also populate `content.parts[].text`. Complete item snapshots and worker
finalText are metadata, not duplicate text chunks. UI consumers should replace
messages by itemId and show tool progress from the metadata. Only the tool's final
function response is durable history; the parent then continues its own answer.

Bindings live in ADK session state and keys include app/user/session/tool/endpoint.
Start retries preserve the original idempotency key. SSE reconnects use the last
consumed cursor and never start another turn. Unknown, failed or interrupted
results remain explicit. Different trust domains need separate sandbox endpoints.
The tool does not transfer host Python tools, skills or credentials into CodeEnv.

Closing/cancelling the agent iterator cancels its producer task, closes HTTP
resources and attempts turn/interrupt. If the start response was lost, cancellation
queries the turn by its original key; it never starts a new task just to cancel it.
UI transport disconnect policy belongs to the application: preserve the invocation
if browser disconnect should not cancel execution.

Focused tests:

```bash
pytest tests/runtime/test_tool_events.py tests/runtime/test_codex_worker_client.py
```

Tests use fake models/transport and require no real model credentials. The full
Rust + real Codex integration case lives in actb-mono under
`agentkit/client/code_env/execute_code_agent`.
