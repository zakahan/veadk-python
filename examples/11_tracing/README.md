# 11 · Tracing & observability

See exactly what the agent did — every LLM call and tool call — by attaching a
tracer. Each run gets a `trace_id` (32 hex chars) you can search in an
observability platform.

> 中文版见 [README.zh.md](./README.zh.md)

## Core idea

```python
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

tracer = OpentelemetryTracer(exporters=[...])   # exporters optional
agent = Agent(tracers=[tracer], tools=[...])

answer = await runner.run(messages="...", session_id="demo-session")
runner.get_trace_id()    # the trace id to correlate in your backend
```

- **`tracers=[...]`** — attach one or more tracers to the agent.
- Each LLM call and tool call becomes a span with timing, inputs, and outputs.
- **`runner.get_trace_id()`** — the 32-char id that ties those spans together;
  search it in your observability platform's UI.
- With **no exporter** the spans are kept in-memory (no creds needed); you still
  get a `trace_id`. Add exporters to also ship them to a platform.

## Agent + custom tool trace

`agent.py` is a complete example: the agent registers a `get_city_weather` tool,
and the tool creates an additional `weather.lookup` business span. A run produces
a hierarchy like this:

```text
Agent invocation
└── Tool: get_city_weather       # created automatically by VeADK
    └── weather.lookup          # created by the business code
```

The essential code is:

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from veadk import Agent
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

weather_tracer = trace.get_tracer("veadk.examples.tracing.weather")


def get_city_weather(city: str) -> dict[str, str]:
    """Get the current weather for a city."""
    with weather_tracer.start_as_current_span(
        "weather.lookup",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        span.set_attribute("weather.city", city.lower().strip())
        span.set_attribute("weather.provider", "demo-fixed-data")
        span.add_event("weather.lookup.started")

        try:
            weather = "Sunny, 25°C"
            span.set_attribute("weather.found", True)
            span.add_event("weather.lookup.completed")
            return {"result": weather}
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


agent_tracer = OpentelemetryTracer()
root_agent = Agent(
    name="traced_agent",
    instruction="Always use get_city_weather for weather questions.",
    tools=[get_city_weather],
    tracers=[agent_tracer],
)
```

`OpentelemetryTracer` initializes the global OpenTelemetry provider. Because the
tool runs inside the agent's active context, `start_as_current_span` makes the
custom span a child of the tool span; no manual `trace_id` propagation is needed.

- `set_attribute` adds searchable, aggregatable business dimensions.
- `add_event` records significant events during the span.
- `record_exception` plus `StatusCode.ERROR` records failures.
- Do not put API keys, tokens, full user input, or other sensitive data in span
  attributes or events.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # set MODEL_AGENT_API_KEY (+ AK/SK and ENABLE_* to export)
python main.py
```

The script prints the active exporters, the agent's answer, and the trace id. The
default question makes the agent call the custom-traced `get_city_weather` tool.

## Deploy to AgentKit

The deployment entry point in `app.py` uses `AgentkitAgentServerApp` directly:

```python
from agentkit.apps import AgentkitAgentServerApp
from veadk.memory.short_term_memory import ShortTermMemory

from agent import create_agent

# AgentKit Runtime already manages the platform APMPlus processor. Do not add a
# second exporter here.
root_agent = create_agent()

agent_server = AgentkitAgentServerApp(
    agent=root_agent,
    short_term_memory=ShortTermMemory(backend="local"),
)
app = agent_server.app
```

The local `main.py` explicitly calls `build_exporters()` instead. This keeps
environment-controlled export available locally while preventing AgentKit from
registering both its platform exporter and a manual exporter for the same spans.

The included `.dockerignore` excludes `.env`, preventing local secrets from
being copied into the image. Configure the Volcengine AK/SK used by the AgentKit
CLI on the deployment machine, then create the deployment configuration:

```bash
export VOLCENGINE_ACCESS_KEY=<deployment-access-key>
export VOLCENGINE_SECRET_KEY=<deployment-secret-key>

veadk agentkit config \
  --agent_name custom-trace-agent \
  --entry_point app.py \
  --dependencies_file requirements.txt \
  --language Python \
  --language_version 3.12 \
  --launch_type cloud \
  --cloud_provider volcengine \
  --region cn-beijing \
  --runtime_envs MODEL_AGENT_API_KEY=<ark-api-key> \
  --runtime_envs OTEL_SDK_DISABLED=false \
  --runtime_envs ENABLE_APMPLUS=true \
  --runtime_envs OBSERVABILITY_OPENTELEMETRY_APMPLUS_SERVICE_NAME=custom-trace-agent \
  --runtime_envs OBSERVABILITY_OPENTELEMETRY_APMPLUS_API_KEY=<apmplus-api-key>

veadk agentkit launch
veadk agentkit status
veadk agentkit invoke "What is the weather in Beijing?"
```

The runtime needs at least `MODEL_AGENT_API_KEY`. To see custom spans in a
platform, it also needs:

- `OTEL_SDK_DISABLED=false`;
- at least one of `ENABLE_APMPLUS`, `ENABLE_COZELOOP`, or `ENABLE_TLS` enabled;
- the selected exporter's credentials and service/workspace/topic id;
- network access from the Runtime to the model and exporter endpoints.

`MODEL_AGENT_NAME`, `MODEL_AGENT_PROVIDER`, `MODEL_AGENT_API_BASE`, `HOST`, and
`PORT` have defaults and can be overridden as needed. Without an exporter,
traces remain only in process memory and are not suitable for AgentKit
production observability.

## Exporters

This example builds exporters based on `ENABLE_*` env flags (config comes from
`.env`):

- **APMPlus** (`ENABLE_APMPLUS=true`) — auth via `VOLCENGINE_ACCESS_KEY` /
  `SECRET_KEY` (auto token) or `..._APMPLUS_API_KEY`; service name via
  `..._APMPLUS_SERVICE_NAME`.
- **CozeLoop** (`ENABLE_COZELOOP=true`) — `..._COZELOOP_API_KEY`;
  `..._COZELOOP_SERVICE_NAME` is the space id.
- **Volcengine TLS** (`ENABLE_TLS=true`) — Volcengine AK/SK;
  `..._TLS_SERVICE_NAME` is the topic id, `..._TLS_REGION` the region.

(Full env var names are `OBSERVABILITY_OPENTELEMETRY_<PLATFORM>_*`; see
`.env.example`.) Endpoints have defaults, so you usually only set the key / id.

```python
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.exporters.cozeloop_exporter import CozeloopExporter
from veadk.tracing.telemetry.exporters.tls_exporter import TLSExporter

tracer = OpentelemetryTracer(exporters=[APMPlusExporter(), CozeloopExporter(), TLSExporter()])
```

You can enable several at once — the same spans are sent to all of them, and
also kept in-memory.

> ℹ️ The local `runner.save_tracing_file(...)` dump is intentionally not used
> here: on the current VeADK + Google ADK combination its session→trace filter
> can return an empty file. The platform exporters above are the reliable way to
> inspect traces. See the
> [configuration docs](https://volcengine.github.io/veadk-python/) for more.
