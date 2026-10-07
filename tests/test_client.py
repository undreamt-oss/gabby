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
"""Contract tests for the public Gabby HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gabby import GabbyAPIError, GabbyClient, RunResult, StreamEvent


def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_async_run_posts_stateless_context_and_parses_result() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"output": "done", "metadata": {"cost": 2}, "trace_id": "trace-1", "trace": None},
        )

    async with GabbyClient(
        "http://localhost:8787/prefix",
        "agent / one",
        bearer_token="secret",
        transport=_transport(handle),
    ) as client:
        result = await client.arun("task", context={"project": "alpha"}, include_trace=False)

    assert result == RunResult("done", {"cost": 2}, "trace-1", None)
    assert requests[0].url.raw_path == b"/prefix/v1/agents/agent%20%2F%20one/run"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content) == {
        "input": "task",
        "context": {"project": "alpha"},
        "memory": {},
        "metadata": {},
        "include_trace": False,
    }


@pytest.mark.asyncio
async def test_async_stream_decodes_events_and_rejects_mismatched_names() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            content=(
                b'id: 7\nevent: text_delta\ndata: {"type":"text_delta","data":{"text":"hi"}}\n\n'
            ),
        )

    async with GabbyClient(
        "http://127.0.0.1:8787", "agent", transport=_transport(handle)
    ) as client:
        events = [
            event
            async for event in client.astream(
                "task", idempotency_key="retry-1234567890", last_event_id="6"
            )
        ]
    assert events == [StreamEvent("text_delta", {"text": "hi"}, "7")]
    assert requests[0].headers["idempotency-key"] == "retry-1234567890"
    assert requests[0].headers["last-event-id"] == "6"

    def mismatched(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'event: completed\ndata: {"type":"error","data":{}}\n\n',
        )

    async with GabbyClient("http://localhost", "agent", transport=_transport(mismatched)) as client:
        with pytest.raises(ValueError, match="does not match"):
            _ = [event async for event in client.astream("task")]


@pytest.mark.asyncio
async def test_http_errors_are_typed_and_responses_are_bounded() -> None:
    def error(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            429, json={"detail": {"error": "busy", "error_type": "CapacityLimitError"}}
        )

    async with GabbyClient("http://localhost", "agent", transport=_transport(error)) as client:
        with pytest.raises(GabbyAPIError, match="CapacityLimitError: busy") as caught:
            await client.arun("task")
    assert caught.value.status_code == 429
    assert caught.value.error_type == "CapacityLimitError"

    def large(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"x" * 32)

    async with GabbyClient(
        "http://localhost", "agent", max_response_bytes=16, transport=_transport(large)
    ) as client:
        with pytest.raises(ValueError, match="max_response_bytes"):
            await client.arun("task")


def test_sync_wrappers_reuse_client_and_close_cleanly() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"type":"completed","data":{"result":{"output":"ok"}}}\n\n',
            )
        return httpx.Response(
            200, json={"output": "ok", "metadata": {}, "trace_id": "t", "trace": None}
        )

    with GabbyClient("http://localhost", "agent", transport=_transport(handle)) as client:
        assert client.run("one").output == "ok"
        assert [event.type for event in client.stream("two")] == ["completed"]
    assert calls == 2


@pytest.mark.parametrize(
    "base_url",
    ["http://example.test", "https://user:pass@example.test", "https://example.test/?token=x"],
)
def test_client_rejects_unsafe_or_credential_bearing_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        GabbyClient(base_url, "agent")
