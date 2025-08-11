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


from veadk.a2a.remote_ve_agent import RemoteVeAgent
from veadk.memory.short_term_memory import ShortTermMemory
from veadk.runner import Runner

remote = RemoteVeAgent(name="weatherreporter", url="http://0.0.0.0:8000")


async def remote_call():
    resp = await Runner(
        agent=remote,
        short_term_memory=ShortTermMemory(),
        app_name="test",
        user_id="user_id",
    ).run(
        messages="Query the weather in Beijing, then check if there is any content about weather in the knowledge base, and then check if there is any content about weather in the memory.",
        # messages="Query the weather in Beijing, ",
        session_id="session_id",
    )
    print(resp)


if __name__ == "__main__":
    import asyncio

    asyncio.run(remote_call())
