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
"""Lifecycle and thread-affinity checks for the sync-to-async bridge."""

from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from gabby._sync_runner import SyncLoopBridge


def test_bridge_reuses_its_loop_and_can_be_restarted_after_close() -> None:
    bridge = SyncLoopBridge()
    assert bridge.is_running is False

    assert bridge.run(asyncio.sleep(0, result="first")) == "first"
    loop = bridge._loop
    assert bridge.is_running is True
    assert bridge.run(asyncio.sleep(0, result="second")) == "second"
    assert bridge._loop is loop

    bridge.stop()
    assert bridge.is_running is False
    bridge.stop()
    assert bridge.run(asyncio.sleep(0, result="restarted")) == "restarted"
    bridge.stop()


def test_bridge_rejects_running_loop_reentry_and_closes_the_coroutine() -> None:
    bridge = SyncLoopBridge()
    uninitialized = SyncLoopBridge()

    async def reenter() -> None:
        nested = asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="running event loop"):
            bridge.run(nested)
        assert inspect.getcoroutinestate(nested) == inspect.CORO_CLOSED

        initial = asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="inside a running event loop"):
            uninitialized.run(initial)
        assert inspect.getcoroutinestate(initial) == inspect.CORO_CLOSED

    bridge.run(reenter())
    bridge.stop()


def test_bridge_rejects_calls_and_close_from_non_owner_thread() -> None:
    bridge = SyncLoopBridge()
    bridge.run(asyncio.sleep(0))
    errors: list[str] = []

    def use_from_other_thread() -> None:
        with pytest.raises(RuntimeError, match="only be used from the thread") as error:
            bridge.run(asyncio.sleep(0))
        errors.append(str(error.value))
        with pytest.raises(RuntimeError, match="owning thread") as close_error:
            bridge.stop()
        errors.append(str(close_error.value))

    worker = threading.Thread(target=use_from_other_thread)
    worker.start()
    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert len(errors) == 2
    bridge.stop()


def test_bridge_rejects_closing_while_its_loop_is_running() -> None:
    bridge = SyncLoopBridge()

    async def stop_from_loop() -> None:
        with pytest.raises(RuntimeError, match="while its loop is running"):
            bridge.stop()

    bridge.run(stop_from_loop())
    bridge.stop()
