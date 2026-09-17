"""Native VeADK multi-agent example with Skill and CodeEnv sandboxes."""

import os

from veadk import Agent
from veadk.agents.agentkit_remote_sandbox_agent import AgentkitRemoteSandboxAgent

skill_agent = AgentkitRemoteSandboxAgent(
    name="skill_agent",
    description="使用远端已安装的技能处理文档、表格等任务，直接流式展示执行过程和结果。",
    tool_id=os.environ["AGENTKIT_SKILL_TOOL_ID"],
    tool_type="Skill",
    request_timeout=900,
    expiry_buffer=90,
)

codeenv_agent = AgentkitRemoteSandboxAgent(
    name="codeenv_agent",
    description="编写和运行代码、处理数据及文件，直接流式展示执行过程和真实结果。",
    tool_id=os.environ["AGENTKIT_CODEENV_TOOL_ID"],
    tool_type="CodeEnv",
    request_timeout=900,
    expiry_buffer=90,
)

root_agent = Agent(
    name="skill_code_orchestrator",
    description="根据用户需求，将任务交给 Skill 或 CodeEnv 远端子 agent。",
    instruction=(
        "你是调度器。需要使用技能时交给 skill_agent；"
        "需要编写或运行代码、操作文件时交给 codeenv_agent。"
        "用户说‘让你的 subAgent 执行’时，是要求你选择并调用这里的子 agent。"
        "调用前先用一句简短自然中文说明要做什么，"
        "然后在同一轮使用 transfer_to_agent 转交，不要只说明而不调用。"
        "不要自己执行任务，不要编造执行结果或已安装的技能。"
        "子 agent 会直接流式展示执行过程并回答用户，无需你重复总结。"
    ),
    sub_agents=[skill_agent, codeenv_agent],
)
