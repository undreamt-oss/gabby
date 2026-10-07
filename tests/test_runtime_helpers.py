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
"""Bounds and callback scheduling contracts used by the async runtime."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from gabby._sync import SyncCallbackPool
from gabby.runtime import (
    _add_usage,
    _bounded_json_dumps,
    _invoke,
    _invoke_with_timeout,
    _reject_json_constant,
)
from gabby.tools import ToolError


def test_bounded_json_serialization_counts_utf8_and_supports_fallback_values() -> None:
    assert _bounded_json_dumps("é", max_bytes=4, tool_name="echo") == '"é"'
    fallback_value = object()
    assert json.loads(_bounded_json_dumps(fallback_value, max_bytes=100, tool_name="echo")) == str(
        fallback_value
    )

    with pytest.raises(ToolError, match="max_result_bytes=3"):
        _bounded_json_dumps("é", max_bytes=3, tool_name="echo")
    with pytest.raises(ValueError, match="Out of range float values"):
        _bounded_json_dumps(float("nan"), max_bytes=100, tool_name="echo")


def test_usage_aggregation_ignores_invalid_values_and_overflow() -> None:
    total: dict[str, int | float] = {"requests": 1}
    _add_usage(
        total,
        {
            "requests": 2,
            "fractional": 0.5,
            "negative": -1,
            "boolean": True,
            "infinite": float("inf"),
            "nan": float("nan"),
            "": 1,
            "x" * 65: 1,
            12: 1,
        },
    )
    _add_usage(total, {"huge": 10**10000})

    assert total == {"requests": 3, "fractional": 0.5}


def test_json_constant_parser_rejects_nonstandard_numbers() -> None:
    with pytest.raises(ValueError, match="Invalid JSON numeric constant: NaN"):
        _reject_json_constant("NaN")


@pytest.mark.asyncio
async def test_invoke_supports_async_callbacks_and_sync_callbacks_returning_awaitables() -> None:
    async def async_callback(value: int) -> int:
        return value + 1

    async def resolve() -> str:
        return "resolved"

    def sync_callback() -> Any:
        return resolve()

    assert await _invoke(async_callback, 2) == 3
    assert await _invoke(sync_callback) == "resolved"


@pytest.mark.asyncio
async def test_sync_callback_timeout_does_not_block_and_late_result_is_consumed() -> None:
    release = threading.Event()

    def slow_callback() -> str:
        release.wait(timeout=1)
        return "late result"

    with pytest.raises(TimeoutError):
        await _invoke_with_timeout(slow_callback, timeout=0.01)

    release.set()
    await asyncio.sleep(0.02)


def test_sync_callback_pool_rejects_unbounded_configuration() -> None:
    with pytest.raises(ValueError, match="workers and queue_size"):
        SyncCallbackPool(workers=0)
    with pytest.raises(ValueError, match="workers and queue_size"):
        SyncCallbackPool(queue_size=0)


@pytest.mark.asyncio
async def test_sync_callback_cancellation_after_result_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SyncCallbackPool(workers=1)
    callback_finished = threading.Event()

    def callback() -> str:
        callback_finished.set()
        return "completed"

    async def complete_then_cancel(futures: set[asyncio.Future[Any]], *, timeout: float) -> None:
        del timeout
        assert callback_finished.wait(timeout=1)
        next(iter(futures)).set_result("completed")
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "wait", complete_then_cancel)
        with pytest.raises(asyncio.CancelledError):
            await pool.run(callback)
    assert callback_finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["timeout", "caller"])
async def test_timed_out_sync_callback_is_skipped_if_still_queued(
    monkeypatch: pytest.MonkeyPatch, cancellation: str
) -> None:
    pool = SyncCallbackPool(workers=1, queue_size=2)
    monkeypatch.setattr("gabby.runtime.run_sync_callback", pool.run)
    started = threading.Event()
    release = threading.Event()
    late_callback_ran = threading.Event()

    def blocking_callback() -> str:
        started.set()
        release.wait(timeout=1)
        return "released"

    def queued_callback() -> str:
        late_callback_ran.set()
        return "should not run"

    async def wait_until_started() -> None:
        for _ in range(1000):
            if started.is_set():
                return
            await asyncio.sleep(0.001)
        pytest.fail("blocking callback did not start")

    active = asyncio.create_task(_invoke(blocking_callback))
    await wait_until_started()
    if cancellation == "timeout":
        with pytest.raises(TimeoutError):
            await _invoke_with_timeout(queued_callback, timeout=0.02)
    else:
        waiter = asyncio.create_task(_invoke_with_timeout(queued_callback, timeout=2))
        for _ in range(1000):
            if pool._jobs.qsize() == 1:
                break
            await asyncio.sleep(0.001)
        assert pool._jobs.qsize() == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    release.set()
    assert await active == "released"
    await asyncio.sleep(0.02)
    assert not late_callback_ran.is_set()
