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
"""Edge-contract tests for the public HTTP client."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from gabby import GabbyAPIError, GabbyClient


def _transport(
    body: bytes,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body, headers=headers)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("base_url", "agent_name", "kwargs", "error", "message"),
    [
        ("", "agent", {}, ValueError, "base_url"),
        ("file://localhost/agent", "agent", {}, ValueError, "base_url"),
        ("https:///missing-host", "agent", {}, ValueError, "base_url"),
        ("http://example.test", "agent", {}, ValueError, "HTTPS"),
        ("http://[::1]", "agent", {}, None, ""),
        ("http://localhost.", "agent", {}, None, ""),
        ("http://LOCALHOST", "agent", {}, None, ""),
        ("http://127.0.0.2", "agent", {}, None, ""),
        ("https://example.test", "", {}, ValueError, "agent_name"),
        ("https://example.test", "agent", {"bearer_token": ""}, ValueError, "bearer_token"),
        (
            "https://example.test",
            "agent",
            {"bearer_token": 7},
            ValueError,
            "bearer_token",
        ),
        (
            "https://example.test",
            "agent",
            {"max_response_bytes": True},
            ValueError,
            "max_response_bytes",
        ),
        (
            "https://example.test",
            "agent",
            {"max_response_bytes": 0},
            ValueError,
            "max_response_bytes",
        ),
        (
            "https://example.test",
            "agent",
            {"headers": [("x-test", "value")]},
            TypeError,
            "headers",
        ),
        (
            "https://example.test",
            "agent",
            {"headers": {"": "value"}},
            ValueError,
            "non-empty string",
        ),
        (
            "https://example.test",
            "agent",
            {"headers": {"x-test": 1}},
            ValueError,
            "string values",
        ),
        (
            "https://example.test",
            "agent",
            {"headers": {"Host": "other.test"}},
            ValueError,
            "transport-controlled",
        ),
        (
            "https://example.test",
            "agent",
            {"headers": {"X-Test": "one", "x-test": "two"}},
            ValueError,
            "duplicate",
        ),
        (
            "https://example.test",
            "agent",
            {"bearer_token": "token", "headers": {"authorization": "Basic token"}},
            ValueError,
            "not both",
        ),
    ],
)
def test_client_construction_validates_transport_and_credentials(
    base_url: str,
    agent_name: str,
    kwargs: dict[str, Any],
    error: type[Exception] | None,
    message: str,
) -> None:
    if error is None:
        client = GabbyClient(base_url, agent_name, **kwargs)
        client.close()
        return
    with pytest.raises(error, match=message):
        GabbyClient(base_url, agent_name, **kwargs)


@pytest.mark.parametrize(
    ("input_value", "kwargs", "error", "message"),
    [
        ("", {}, ValueError, "input"),
        (1, {}, ValueError, "input"),
        ("task", {"context": []}, TypeError, "context"),
        ("task", {"memory": []}, TypeError, "memory"),
        ("task", {"metadata": []}, TypeError, "metadata"),
        ("task", {"include_trace": 1}, TypeError, "include_trace"),
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_invalid_request_values_before_http_call(
    input_value: Any,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    async with GabbyClient("http://localhost", "agent") as client:
        with pytest.raises(error, match=message):
            await client.arun(input_value, **kwargs)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"{}",
        json.dumps({"output": 1, "trace_id": "t"}).encode(),
        json.dumps({"output": "ok", "trace_id": "t", "metadata": []}).encode(),
        json.dumps({"output": "ok", "trace_id": "t", "trace": []}).encode(),
    ],
)
@pytest.mark.asyncio
async def test_run_rejects_malformed_public_response_schema(body: bytes) -> None:
    async with GabbyClient("http://localhost", "agent", transport=_transport(body)) as client:
        with pytest.raises(ValueError, match="public response schema"):
            await client.arun("task")


@pytest.mark.parametrize(
    ("body", "expected_type", "expected_message"),
    [
        (b"not-json", "HTTPError", "Gabby request failed"),
        (b'{"detail":"denied"}', "HTTPError", "denied"),
        (
            b'{"detail":{"error_type":"BusyError","error":"try later"}}',
            "BusyError",
            "try later",
        ),
        (
            b'{"detail":{"error_type":1,"error":false}}',
            "HTTPError",
            "Gabby request failed",
        ),
        (b'{"detail":"\xff"}', "HTTPError", "Gabby request failed"),
    ],
)
@pytest.mark.asyncio
async def test_http_error_bodies_are_typed_with_safe_fallbacks(
    body: bytes, expected_type: str, expected_message: str
) -> None:
    async with GabbyClient(
        "http://localhost", "agent", transport=_transport(body, status_code=503)
    ) as client:
        with pytest.raises(GabbyAPIError) as caught:
            await client.arun("task")

    assert caught.value.status_code == 503
    assert caught.value.error_type == expected_type
    assert expected_message in str(caught.value)


@pytest.mark.parametrize(
    ("body", "headers", "limit", "message"),
    [
        (
            b'data: {"type":"done","data":{}}\n\n',
            {"content-type": "application/json"},
            1024,
            "text/event-stream",
        ),
        (
            b"data: {bad}\n\n",
            {"content-type": "text/event-stream"},
            1024,
            "invalid JSON",
        ),
        (
            b"data: []\n\n",
            {"content-type": "text/event-stream"},
            1024,
            "JSON object",
        ),
        (
            b'data: {"type":"","data":{}}\n\n',
            {"content-type": "text/event-stream"},
            1024,
            "invalid type or data",
        ),
        (
            b'data: {"type":"done","data":[]}\n\n',
            {"content-type": "text/event-stream"},
            1024,
            "invalid type or data",
        ),
        (
            b'event: other\ndata: {"type":"done","data":{}}\n\n',
            {"content-type": "text/event-stream"},
            1024,
            "does not match",
        ),
        (
            b"data: \xff\n\n",
            {"content-type": "text/event-stream"},
            1024,
            "valid UTF-8",
        ),
        (
            b'data: {"type":"done","data":{}}',
            {"content-type": "text/event-stream"},
            1024,
            "final event delimiter",
        ),
        (
            b'data: {"type":"done","data":{}}\n\n',
            {"content-type": "text/event-stream"},
            8,
            "max_response_bytes",
        ),
    ],
)
@pytest.mark.asyncio
async def test_stream_rejects_invalid_content_and_sse_frames(
    body: bytes, headers: dict[str, str], limit: int, message: str
) -> None:
    async with GabbyClient(
        "http://localhost",
        "agent",
        max_response_bytes=limit,
        transport=_transport(body, headers=headers),
    ) as client:
        with pytest.raises(ValueError, match=message):
            _ = [event async for event in client.astream("task")]


@pytest.mark.asyncio
async def test_stream_accepts_comments_multiline_data_and_crlf_frames() -> None:
    body = b': ping\r\ndata\r\ndata: {"type":\r\ndata: "progress","data":{}}\r\n\r\n'
    async with GabbyClient(
        "http://localhost",
        "agent",
        transport=_transport(body, headers={"content-type": "text/event-stream"}),
    ) as client:
        events = [event async for event in client.astream("task")]

    assert [event.type for event in events] == ["progress"]


@pytest.mark.asyncio
async def test_stream_http_error_and_async_loop_affinity() -> None:
    error_client = GabbyClient(
        "http://localhost",
        "agent",
        transport=_transport(b'{"detail":"unavailable"}', status_code=503),
    )
    async with error_client:
        with pytest.raises(GabbyAPIError, match="unavailable"):
            _ = [event async for event in error_client.astream("task")]

    client = GabbyClient("http://localhost", "agent")
    with pytest.raises(RuntimeError, match="Synchronous client methods"):
        client.run("task")
    client.close()

    client = GabbyClient("http://localhost", "agent")
    async with client:
        await client.aclose()
        await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.arun("after close")


def test_sync_client_rejects_switching_to_async_api() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": "ok", "metadata": {}, "trace_id": "t", "trace": None},
        )

    client = GabbyClient("http://localhost", "agent", transport=httpx.MockTransport(handle))
    assert client.run("sync").output == "ok"

    async def call_async() -> None:
        with pytest.raises(RuntimeError, match="same event loop|consistently"):
            await client.arun("async")

    asyncio.run(call_async())
    client.close()
