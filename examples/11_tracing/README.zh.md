# 11 · 链路追踪与可观测性

通过挂载 tracer，清楚地看到智能体做了什么 —— 每一次大模型调用、每一次工具调用。
每次运行都会得到一个 `trace_id`（32 位十六进制），可在可观测平台中检索。

> English version: [README.md](./README.md)

## 核心思想

```python
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

tracer = OpentelemetryTracer(exporters=[...])   # exporters 可选
agent = Agent(tracers=[tracer], tools=[...])

answer = await runner.run(messages="...", session_id="demo-session")
runner.get_trace_id()    # 用于在后端平台关联检索的 trace id
```

- **`tracers=[...]`** —— 为智能体挂载一个或多个 tracer。
- 每一次大模型调用与工具调用都会成为一个 span，包含耗时、输入与输出。
- **`runner.get_trace_id()`** —— 串联这些 span 的 32 位 id；在可观测平台 UI 中检索它。
- **不配 exporter** 时 span 仅在内存中收集（无需凭证），你依然能拿到 `trace_id`；
  加上 exporter 就能同时上报到平台。

## Agent + 自定义 Tool Trace

本目录的 `agent.py` 是一个完整示例：Agent 注册了 `get_city_weather` tool，
tool 内部额外创建 `weather.lookup` 业务 span。运行一次后，链路结构类似：

```text
Agent invocation
└── Tool: get_city_weather       # VeADK 自动创建
    └── weather.lookup          # 业务代码自定义
```

核心代码如下：

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

`OpentelemetryTracer` 初始化全局 OpenTelemetry provider。tool 在 Agent 调用上下文中
执行时，`start_as_current_span` 会读取当前上下文，因此自定义 span 会成为 tool span
的子节点，不需要手动传递 `trace_id`。

- `set_attribute`：添加可检索、可聚合的业务维度。
- `add_event`：记录 span 生命周期中的关键事件。
- `record_exception` + `StatusCode.ERROR`：记录异常及失败状态。
- 不要把 API Key、Token、完整用户输入或其他敏感信息写入 attribute/event。

## 运行步骤

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 MODEL_AGENT_API_KEY（要上报再填 AK/SK 和 ENABLE_*）
python main.py
```

脚本会打印启用了哪些 exporter、Agent 回答，以及 trace id。默认问题会让 Agent
调用带自定义 trace 的 `get_city_weather` tool。

## 部署到 AgentKit

部署入口 `app.py` 直接使用 `AgentkitAgentServerApp`：

```python
from agentkit.apps import AgentkitAgentServerApp
from veadk.memory.short_term_memory import ShortTermMemory

from agent import create_agent

# AgentKit Runtime 已经管理平台 APMPlus processor，这里不再手动添加 exporter。
root_agent = create_agent()

agent_server = AgentkitAgentServerApp(
    agent=root_agent,
    short_term_memory=ShortTermMemory(backend="local"),
)
app = agent_server.app
```

本地 `main.py` 则显式调用 `build_exporters()`。这样本地仍可按 `ENABLE_*` 上报，
AgentKit 部署时不会同时注册“平台 exporter + 手动 exporter”，避免相同
`trace_id` / `span_id` 被重复上传。

目录中的 `.dockerignore` 会排除 `.env`，避免将本地密钥打进镜像。先在部署机器上
配置 AgentKit 使用的火山引擎账号 AK/SK，再生成部署配置：

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
veadk agentkit invoke "北京天气怎么样？"
```

运行时至少需要 `MODEL_AGENT_API_KEY`。为了在平台看到自定义 span，还需要：

- `OTEL_SDK_DISABLED=false`；
- 开启至少一个 `ENABLE_APMPLUS` / `ENABLE_COZELOOP` / `ENABLE_TLS`；
- 配置对应 exporter 的凭证和 service/workspace/topic id；
- Runtime 网络能够访问模型与 exporter endpoint。

`MODEL_AGENT_NAME`、`MODEL_AGENT_PROVIDER`、`MODEL_AGENT_API_BASE`、`HOST` 和
`PORT` 都有默认值，可按需覆盖。如果没有开启 exporter，trace 只保存在当前进程
内存中，不适合作为 AgentKit 生产观测方案。

## Exporter

本示例根据 `ENABLE_*` 环境变量来构建 exporter（具体配置从 `.env` 读取）：

- **APMPlus**（`ENABLE_APMPLUS=true`）—— 用 `VOLCENGINE_ACCESS_KEY` /
  `SECRET_KEY` 自动取 token，或填 `..._APMPLUS_API_KEY`；服务名用
  `..._APMPLUS_SERVICE_NAME`。
- **CozeLoop**（`ENABLE_COZELOOP=true`）—— `..._COZELOOP_API_KEY`；
  `..._COZELOOP_SERVICE_NAME` 是 space id。
- **火山 TLS**（`ENABLE_TLS=true`）—— 火山 AK/SK；`..._TLS_SERVICE_NAME` 是
  topic id，`..._TLS_REGION` 是地域。

（完整环境变量名为 `OBSERVABILITY_OPENTELEMETRY_<平台>_*`，见 `.env.example`。）
端点都有默认值，通常只需填 key / id。

```python
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.exporters.cozeloop_exporter import CozeloopExporter
from veadk.tracing.telemetry.exporters.tls_exporter import TLSExporter

tracer = OpentelemetryTracer(exporters=[APMPlusExporter(), CozeloopExporter(), TLSExporter()])
```

可以同时开启多个 —— 同一批 span 会发往所有平台，并同时保留在内存中。

> ℹ️ 这里有意没有用本地的 `runner.save_tracing_file(...)` 导出：在当前
> VeADK + Google ADK 的组合下，其“会话→trace”过滤可能返回空文件。
> 上面的平台 exporter 才是查看 trace 的可靠方式。更多见
> [配置文档](https://volcengine.github.io/veadk-python/)。
