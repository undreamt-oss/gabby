# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded daemon workers for synchronous extension callbacks."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any


class SyncCallbackOverloaded(RuntimeError):
    """The bounded queue for synchronous extensions is full."""


@dataclass
class _Job:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[Any]
    callback: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _started: bool = False
    _cancelled: bool = False

    def claim(self) -> bool:
        """Atomically mark this job started unless its waiter abandoned it first."""
        with self._lock:
            if self._cancelled:
                return False
            self._started = True
            return True

    def cancel(self) -> None:
        """Mark a still-queued job so a worker skips its callback."""
        with self._lock:
            if not self._started:
                self._cancelled = True


class SyncCallbackPool:
    """Run synchronous extensions without using asyncio's default executor.

    Workers are daemon threads so a timed-out, uncooperative extension cannot hold
    process shutdown open. Worker and queue counts are bounded to apply backpressure.
    Cancellation abandons the result; it cannot stop a callback already running.
    """

    def __init__(self, *, workers: int = 8, queue_size: int = 256) -> None:
        if workers < 1 or queue_size < 1:
            raise ValueError("workers and queue_size must be positive")
        self._workers = workers
        self._jobs: queue.Queue[_Job] = queue.Queue(maxsize=queue_size)
        self._start_lock = threading.Lock()
        self._started = False

    async def run(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _Job(loop, future, callback, args, kwargs)
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise SyncCallbackOverloaded("Synchronous extension capacity is exhausted") from exc
        # Some embedded loops do not promptly wake for callbacks scheduled from worker
        # threads. Periodic waits provide a bounded wakeup path while preserving async
        # cancellation and without blocking the event loop.
        try:
            while not future.done():
                await asyncio.wait({future}, timeout=0.05)
            return future.result()
        except asyncio.CancelledError:
            job.cancel()
            if not future.done():
                future.cancel()
            raise

    def _start(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            for index in range(self._workers):
                thread = threading.Thread(
                    target=self._worker,
                    name=f"gabby-sync-{index}",
                    daemon=True,
                )
                thread.start()
            self._started = True

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if not job.claim():
                continue
            try:
                result = job.callback(*job.args, **job.kwargs)
            except Exception as exc:
                self._schedule_result(job, error=exc)
            else:
                self._schedule_result(job, result=result)

    @staticmethod
    def _schedule_result(job: _Job, *, result: Any = None, error: Exception | None = None) -> None:
        def complete() -> None:
            if job.future.done():
                return
            if error is not None:
                job.future.set_exception(error)
            else:
                job.future.set_result(result)

        with suppress(RuntimeError):
            job.loop.call_soon_threadsafe(complete)


_sync_callback_pool = SyncCallbackPool()


async def run_sync_callback(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Dispatch a sync extension and await its result without blocking the loop."""
    return await _sync_callback_pool.run(callback, *args, **kwargs)
