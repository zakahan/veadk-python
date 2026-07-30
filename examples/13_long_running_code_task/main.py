# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a background CodeEnv task, stream progress, then resume a VEADK agent."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from google.genai import types

from veadk import Agent, Runner
from veadk.memory.short_term_memory import ShortTermMemory
from veadk.tools.builtin_tools.execute_code_task import (
    execute_code_task,
    poll_code_task,
    resume_code_task,
)

APP_NAME = "long_running_code_task"
USER_ID = "demo-user"
SESSION_ID = "demo-session"


def _text(event: Any) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text or "" for part in event.content.parts)


def _pending_task_id(event: Any) -> str | None:
    for response in event.get_function_responses():
        if response.name != "execute_code_task":
            continue
        body = response.response
        if isinstance(body, dict) and body.get("status") == "pending":
            task_id = body.get("task_id")
            if isinstance(task_id, str):
                return task_id
    return None


async def main() -> None:
    instruction = (
        " ".join(sys.argv[1:]).strip()
        or "Inspect the repository in /home/gem, run its tests, fix one concrete "
        "failure if present, rerun the relevant tests, and summarize the result."
    )
    agent = Agent(
        name="code_task_coordinator",
        instruction=(
            "Delegate repository coding work to execute_code_task. Call it once, "
            "then wait for the application to resume you with the final result."
        ),
        tools=[execute_code_task],
    )
    memory = ShortTermMemory()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        short_term_memory=memory,
    )
    await memory.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    task_id: str | None = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=instruction)],
        ),
    ):
        if text := _text(event):
            print(f"[{event.author}] {text}")
        task_id = task_id or _pending_task_id(event)

    if task_id is None:
        raise RuntimeError("the agent did not create a CodeEnv task")
    print(f"[application] task accepted: {task_id}")

    cursor = 0
    while True:
        snapshot = await asyncio.to_thread(
            poll_code_task,
            task_id,
            cursor,
            10,
            100,
        )
        for task_event in snapshot["events"]:
            cursor = task_event["sequence"]
            print(
                f"[code-task:{cursor}] {task_event['type']}: "
                f"{task_event.get('message') or task_event.get('data') or ''}"
            )
        if snapshot["is_terminal"]:
            break

    async for event in resume_code_task(runner, task_id):
        if text := _text(event):
            print(f"[{event.author}] {text}")


if __name__ == "__main__":
    asyncio.run(main())
