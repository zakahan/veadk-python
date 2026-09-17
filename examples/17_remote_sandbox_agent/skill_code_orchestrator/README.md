# Skill + CodeEnv 调度案例

参考相邻的 `multi_agents`：一个普通 `veadk.Agent` 作为 root，
通过 `sub_agents=[skill_agent, codeenv_agent]` 注册两个远端子 agent。
root 选择执行者，子 agent 通过原生 ADK 事件流直接展示过程和结果。

## 安装与配置

PyPI `veadk-python==1.1.10` 的 wheel 不包含 `AgentkitRemoteSandboxAgent`。
因此 requirements.txt 固定官方 main 的 commit
`c0cf4fb0f22eacaee33e99afb941fa2f38a3900a`，并固定本案例验证的 ADK/AgentKit SDK 版本。

从本目录执行（Python 3.12）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

填写 `.env`：

| 变量 | 用途 |
| --- | --- |
| `AGENTKIT_SKILL_TOOL_ID` | 显式传给 Skill agent 的 `tool_id` |
| `AGENTKIT_CODEENV_TOOL_ID` | 显式传给 CodeEnv agent 的 `tool_id` |
| `REGION` | 沙箱和部署地域，默认 `cn-beijing` |
| `MODEL_AGENT_NAME` / `MODEL_AGENT_API_KEY` | root 调度模型 |
| `VOLCENGINE_ACCESS_KEY` / `VOLCENGINE_SECRET_KEY` | 本地访问沙箱的云凭证；Runtime 可用获授权的服务角色 |

本案例不使用共用的 `AGENTKIT_TOOL_ID`。两个独立变量分别传入两个 agent，
`tool_type` 分别显式设为 `Skill` 和 `CodeEnv`。
沿用 SDK 的地域解析，配置 `REGION` 即可；不要额外设置会覆盖它的 `AGENTKIT_TOOL_REGION`。
Skill Tool 需要提供 A2A 服务且安装所需技能；CodeEnv Tool 需要支持 Codex Worker v1 和 `tool_events`。
这里连接已有 Tool，不创建 Tool 或安装沙箱内技能。

## 本地运行

```bash
python app.py
```

另一个终端：

```bash
python client.py '让你的 subAgent 列出可用技能，说明哪些可以处理 CSV'
python client.py '让你的 subAgent 用 Python 计算 2 + 3，并展示执行结果'
python client.py --session-id 上次输出的ID '把刚才的结果乘以 10'
```

请提供实际技能名称和数据。客户端标出事件所属 agent，持续读取整个 SSE 流。

## 调度实现

`agent.py` 是完整的 agent 定义，直接使用 `transfer_to_agent`，
没有结构化 planner、自定义循环或事件重包装。
prompt 要求 root 先说明再转交，实际文本与调用顺序由模型和 ADK 决定。
当前远端类仍读取用户原话；本案例不改写远端输入，也不保证同轮自动串联两个沙箱。
两个沙箱的文件系统不共享，需要双方可访问的地址或共享存储来交换文件。
服务使用内存 ADK 会话，重启或多副本间切换不保证续接；此案例用于单实例演示。

## AgentKit CLI 部署

使用 `ak 0.52.7` 的 `hybrid` 模式：本机 Docker 构建 Linux amd64 镜像，
推送到火山镜像仓库，再部署到云端 Runtime。需要本机 Docker 服务可用，
并配置 CLI 登录/云凭证以及上述环境变量或 `.env`。
从本仓库 checkout 内运行：脚本将 requirements 中固定的 commit 从本地 Git
打包到被 Git 忽略的 `vendor/veadk.zip`，Docker 直接安装该源码包。

```bash
python deploy.py --prepare-only
# 生成本地 agentkit.yaml；可按实际资源修改服务角色、网络等配置。
ak launch --config-file agentkit.yaml --platform linux/amd64
ak status --config-file agentkit.yaml
ak invoke run --config-file agentkit.yaml --raw '让你的 subAgent 用 Python 计算 2 + 3'
```

也可以 `python deploy.py` 一次完成配置、本地构建、推送和部署。
构建日志直接显示在终端；后续构建可复用本机 Docker 缓存。
原有 cloud 配置首次运行脚本时会迁移到 hybrid，保留仓库及 Runtime 身份。
切换不会取消已经提交的云端流水线。
部署地域从 `REGION` 读取；项目和名称可以通过 `--project`、`--name` 指定。
`deploy.py` 把模型和两个 ToolId 等配置写入 Runtime 环境变量，并保留已有配置中的 Runtime ID。
生成的 `agentkit.yaml` 可能含凭证，权限为 0600 且被 Git 忽略；不要提交或分享。
`.dockerignore` 排除本地 `.env` 和部署配置。

## 离线验证

仓库根目录已有开发环境时：

```bash
.venv/bin/python -m pytest tests/agents/test_remote_sandbox_agent.py -q
```

运行远端 agent 的现有回归测试，覆盖原生转交、流式事件和沙箱会话复用等行为。
