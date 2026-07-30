# 13 · AgentKit CodeEnv 长任务

这个示例展示完整的 `LongRunningFunctionTool` 暂停/恢复流程：

1. VEADK Agent 调用一次 `execute_code_task`，ADK 将该调用标记为 long-running。
2. 工具立即返回 `pending + task_id`，首次 Agent invocation 无需等待远端任务。
3. 进程内后台线程创建 AgentKit CodeEnv Session，并通过 Codex app-server
   执行任务。
4. 应用通过 `poll_code_task` 展示增量进度。
5. 任务结束后，`resume_code_task` 在新的 invocation 中把同一个
   function-call ID 的最终结果回填，Agent 继续生成最终总结。

只有 `execute_code_task` 注册给 Agent；`poll_code_task` 和
`resume_code_task` 由宿主应用调用。

> English version: [README.md](./README.md)

## 配置

调用方只需要提供 AgentKit 鉴权信息和 CodeEnv Tool ID：

```bash
export VOLCENGINE_ACCESS_KEY=...
export VOLCENGINE_SECRET_KEY=...
export VOLCENGINE_SESSION_TOKEN=...   # 临时凭证时可选
export AGENTKIT_TOOL_ID=...
```

此外仍需按常规方式配置外层 VEADK Agent 的模型。CodeEnv Tool 本身需要预先配置
沙箱内 Codex 使用的模型地址、Key 和模型名。

## 运行

```bash
python examples/13_long_running_code_task/main.py \
  "检查 /home/gem 中的仓库，修复测试失败并重新运行测试"
```

## 第一版限制

- 任务、事件和运行期间的鉴权信息只保存在当前 VEADK 进程内；任务结束后会清空凭证。
- 任务不依赖原 Agent invocation 的 event loop，客户端断开后后台线程可以继续；
  但 VEADK 进程重启会丢失任务。
- 多进程部署需要粘性路由；持久化任务队列与跨进程查询留给下一版。
- 沙箱 Session 完成后保留到 AgentKit TTL 到期，便于检查修改结果。
- 默认自动同意隔离 CodeEnv 内的命令和文件修改审批。设置
  `VEADK_CODE_TASK_APPROVAL_POLICY=decline` 可改为拒绝。
- `resume_code_task` 必须使用与首次调用共享同一个 session service 的 Runner。
