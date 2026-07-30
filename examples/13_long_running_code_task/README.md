# 13 · Long-running AgentKit CodeEnv task

This example demonstrates the complete `LongRunningFunctionTool`
pause/resume flow:

1. A VEADK Agent calls `execute_code_task` once, and ADK marks the call as
   long-running.
2. The tool immediately returns `pending + task_id`, so the initial Agent
   invocation doesn't wait for the remote job.
3. A process-local worker thread creates an AgentKit CodeEnv Session and drives
   Codex through app-server.
4. The application displays incremental progress with `poll_code_task`.
5. Once terminal, `resume_code_task` injects the final response under the same
   function-call ID in a new invocation, so the Agent can summarize the result.

Only `execute_code_task` is registered on the Agent. `poll_code_task` and
`resume_code_task` are called by the host application.

> 中文版见 [README.zh.md](./README.zh.md)

## Configuration

The caller only supplies AgentKit credentials and the CodeEnv Tool ID:

```bash
export VOLCENGINE_ACCESS_KEY=...
export VOLCENGINE_SECRET_KEY=...
export VOLCENGINE_SESSION_TOKEN=...   # optional for temporary credentials
export AGENTKIT_TOOL_ID=...
```

Configure the outer VEADK Agent model normally. The CodeEnv Tool itself must
already contain the model URL, key, and model name used by Codex in the sandbox.

## Run

```bash
python examples/13_long_running_code_task/main.py \
  "Inspect the repository in /home/gem, fix its failing tests, and rerun them"
```

## First-version limitations

- Task state, events, and in-flight credentials live only in the current VEADK
  process; credentials are cleared when a task becomes terminal.
- The task is independent from the original invocation event loop, but a VEADK
  process restart still loses it.
- Multi-process deployments require sticky routing; a durable task queue and
  cross-process lookup are deferred to a later version.
- The AgentKit Session remains available until its TTL expires.
- Command and file approvals inside isolated CodeEnv are accepted by default.
  Set `VEADK_CODE_TASK_APPROVAL_POLICY=decline` to reject them.
- `resume_code_task` must receive a Runner that shares the original session
  service.
