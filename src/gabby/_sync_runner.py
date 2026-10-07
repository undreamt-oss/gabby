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
"""Persistent event-loop bridge for repeatable synchronous Agent.run calls."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class SyncLoopBridge:
    """Reuse one asyncio loop on the calling thread for synchronous calls."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_id: int | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the persistent loop has been initialized."""
        return self._loop is not None and not self._loop.is_closed()

    @property
    def is_loop_thread(self) -> bool:
        """Return whether the current thread owns the persistent loop."""
        return self._thread_id == threading.get_ident()

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run a coroutine on the persistent loop, rejecting cross-thread use."""
        try:
            loop = self._get_loop()
            return loop.run_until_complete(coroutine)
        except BaseException:
            coroutine.close()
            raise

    def stop(self) -> None:
        """Close the persistent loop after its asynchronous work has finished."""
        loop = self._loop
        if loop is None:
            return
        if not self.is_loop_thread:
            raise RuntimeError("The synchronous bridge must be closed on its owning thread")
        if loop.is_running():
            raise RuntimeError("Cannot close the synchronous bridge while its loop is running")
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
            self._loop = None
            self._thread_id = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            if not self.is_loop_thread:
                raise RuntimeError(
                    "A synchronous Agent can only be used from the thread that first called run()"
                )
            if self._loop.is_running():
                raise RuntimeError("Cannot call the synchronous bridge from its running event loop")
            return self._loop

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Cannot start the synchronous bridge inside a running event loop")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._thread_id = threading.get_ident()
        return loop
