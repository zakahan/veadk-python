"""Invocation-local progress events emitted while an ADK tool is running."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
from contextvars import ContextVar
from typing import Any

from google.adk.events import Event

_sink: ContextVar[asyncio.Queue | None] = ContextVar("veadk_tool_events", default=None)


async def emit_tool_event(event: Event) -> None:
    """Publish progress through the current VeADK Agent's event iterator.

    Progress is transient. The normal function response is the durable result.
    Calling a streaming tool without a VeADK Agent bridge is an explicit error.
    """
    sink = _sink.get()
    if sink is None:
        raise RuntimeError("Streaming tools require veadk.Agent")
    await sink.put(event)


async def stream_tool_events(source):
    """Merge the agent iterator and tool progress without waiting for a tool.

    The context variable is set inside the producer task, so concurrent agents,
    nested invocations, and early generator closure cannot share a mutable sink.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
    finished = object()

    async def produce():
        token = _sink.set(queue)
        try:
            async with aclosing(source):
                async for event in source:
                    consumed = asyncio.Event()
                    await queue.put((event, consumed))
                    # ADK Runner appends durable events after receiving them.
                    # Do not advance the LLM flow until that append completes.
                    await consumed.wait()
        except Exception as exc:
            await queue.put(exc)
        finally:
            _sink.reset(token)
        await queue.put(finished)

    task = asyncio.create_task(produce())
    try:
        while True:
            item = await queue.get()
            if item is finished:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple):
                event, consumed = item
                try:
                    yield event
                finally:
                    consumed.set()
            else:
                yield item
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
