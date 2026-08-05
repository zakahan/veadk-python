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

"""Run the same custom-traced agent locally from the command line."""

import asyncio

from agent import build_exporters, create_agent

from veadk import Runner

SESSION_ID = "demo-session"

exporters = build_exporters()
root_agent = create_agent(exporters=exporters)


async def main() -> None:
    print(
        "Exporters:",
        [type(exporter).__name__ for exporter in exporters]
        or "in-memory only (no cloud export)",
    )

    runner = Runner(agent=root_agent, app_name="tracing_demo")
    answer = await runner.run(messages="北京今天天气怎么样？", session_id=SESSION_ID)
    print("Answer:", answer)
    print("Trace id:", runner.get_trace_id())


if __name__ == "__main__":
    asyncio.run(main())
