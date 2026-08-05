# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentKit deployment entry point for the custom tracing agent."""

import os

from agent import create_agent
from agentkit.apps import AgentkitAgentServerApp

from veadk.memory.short_term_memory import ShortTermMemory

# AgentKit Runtime manages the platform APMPlus processor. Do not manually add
# another APMPlusExporter here, otherwise every span is uploaded twice.
root_agent = create_agent()

agent_server = AgentkitAgentServerApp(
    agent=root_agent,
    short_term_memory=ShortTermMemory(backend="local"),
)

# AgentKit imports this ASGI application in the deployed runtime.
app = agent_server.app


if __name__ == "__main__":
    agent_server.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
