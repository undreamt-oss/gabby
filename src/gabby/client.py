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
"""Async-first HTTP client for Gabby's stateless execution API."""

from __future__ import annotations

import asyncio
import ipaddress
import json
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from ._sync_runner import SyncLoopBridge
from .server import DEFAULT_MAX_HTTP_RESPONSE_BYTES


@dataclass(frozen=True)
class RunResult:
    """One stateless remote execution result."""

    output: str
    metadata: dict[str, Any]
    trace_id: str
    trace: dict[str, Any] | None


@dataclass(frozen=True)
class StreamEvent:
    """One typed event received from Gabby's SSE endpoint."""

    type: str
    data: dict[str, Any]
    event_id: str | None = None


class GabbyAPIError(RuntimeError):
    """A sanitized HTTP error returned by a Gabby service."""

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.status_code = status_code
        self.error_type = error_type


class GabbyClient:
    """Call one configured Gabby service without retaining conversation state.

    ``arun`` and ``astream`` are the primary async interface. ``run`` and ``stream``
    are synchronous wrappers for scripts and synchronous applications. A client may
    use either interface, but must not switch between them after its first call.
    The caller owns context and memory and decides what to send on each request.
    """

    def __init__(
        self,
        base_url: str,
        agent_name: str,
        *,
        bearer_token: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout | float = 180.0,
        max_response_bytes: int = DEFAULT_MAX_HTTP_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty HTTP(S) URL")
        parsed = urlsplit(base_url)
        host = parsed.hostname
        if (
            parsed.scheme not in {"http", "https"}
            or host is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host.lower() in {"localhost", "localhost."}
        if parsed.scheme == "http" and not is_loopback:
            raise ValueError("base_url must use HTTPS outside loopback")
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be a non-empty string")
        if bearer_token is not None and (not isinstance(bearer_token, str) or not bearer_token):
            raise ValueError("bearer_token must be a non-empty string when provided")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if headers is not None and not isinstance(headers, Mapping):
            raise TypeError("headers must be a mapping of HTTP header names to values")

        request_headers: dict[str, str] = {}
        seen_headers: set[str] = set()
        for name, value in (headers or {}).items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise ValueError("headers must contain non-empty string names and string values")
            normalized = name.lower()
            if normalized in seen_headers or normalized in {
                "host",
                "content-length",
                "idempotency-key",
                "last-event-id",
            }:
                raise ValueError("headers must not contain duplicate or transport-controlled names")
            seen_headers.add(normalized)
            request_headers[name] = value
        if bearer_token is not None:
            if "authorization" in seen_headers:
                raise ValueError("set bearer_token or Authorization in headers, not both")
            request_headers["Authorization"] = f"Bearer {bearer_token}"

        self._base_url = base_url.rstrip("/")
        self._agent_name = quote(agent_name, safe="")
        self._headers = request_headers
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mode: str | None = None
        self._bridge: SyncLoopBridge | None = None
        self._closed = False

    async def arun(
        self,
        input: str,
        *,
        context: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_trace: bool = True,
    ) -> RunResult:
        """Submit one task and return its output and optional execution trace."""
        self._bind("async")
        url = self._url("run")
        payload = self._payload(input, context, memory, metadata, include_trace)
        client = self._get_client()
        async with client.stream("POST", url, json=payload, headers=self._headers) as response:
            body = await self._read_bounded(response)
            self._raise_for_response(response, body)
        try:
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError
            output = value["output"]
            trace_id = value["trace_id"]
            result_metadata = value.get("metadata", {})
            trace = value.get("trace")
            if (
                not isinstance(output, str)
                or not isinstance(trace_id, str)
                or not isinstance(result_metadata, dict)
                or (trace is not None and not isinstance(trace, dict))
            ):
                raise ValueError
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Gabby run response did not match the public response schema") from exc
        return RunResult(output, result_metadata, trace_id, trace)

    async def astream(
        self,
        input: str,
        *,
        context: Mapping[str, Any] | None = None,
        memory: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        include_trace: bool = True,
        idempotency_key: str | None = None,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield progress events, optionally allowing reconnect to the same transient run.

        Reuse ``idempotency_key`` and pass the last yielded ``event_id`` as
        ``last_event_id`` after a transport interruption. Resumption works only while the
        server's in-process journal retains the execution.
        """
        self._bind("async")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 128
            or any(not 0x21 <= ord(char) <= 0x7E for char in idempotency_key)
        ):
            raise ValueError("idempotency_key must contain 1 to 128 visible ASCII characters")
        if last_event_id is not None and (
            idempotency_key is None
            or not isinstance(last_event_id, str)
            or len(last_event_id) > 20
            or not last_event_id.isascii()
            or not last_event_id.isdecimal()
        ):
            raise ValueError("last_event_id requires a valid idempotency_key and numeric event ID")
        url = self._url("stream")
        payload = self._payload(input, context, memory, metadata, include_trace)
        headers = dict(self._headers)
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        client = self._get_client()
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            await self._check_response(response)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type != "text/event-stream":
                raise ValueError("Gabby stream endpoint did not return text/event-stream")
            async for event in _decode_sse(response, max_bytes=self._max_response_bytes):
                yield event

    def run(self, input: str, **kwargs: Any) -> RunResult:
        """Synchronous convenience wrapper around :meth:`arun`."""
        self._bind("sync")
        if self._bridge is None:
            self._bridge = SyncLoopBridge()
        return self._bridge.run(self.arun(input, **kwargs))

    def stream(self, input: str, **kwargs: Any) -> Iterator[StreamEvent]:
        """Synchronous iterator wrapper around :meth:`astream`."""
        self._bind("sync")
        if self._bridge is None:
            self._bridge = SyncLoopBridge()
        bridge = self._bridge
        events = self.astream(input, **kwargs)

        def iterate() -> Iterator[StreamEvent]:
            try:
                while True:
                    try:

                        async def next_event() -> StreamEvent:
                            return await events.__anext__()

                        yield bridge.run(next_event())
                    except StopAsyncIteration:
                        return
            finally:
                bridge.run(events.aclose())

        return iterate()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._closed:
            return
        self._bind("async")
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._closed = True

    def close(self) -> None:
        """Close a synchronous client and its event-loop bridge."""
        if self._mode == "async":
            raise RuntimeError("This client uses the asynchronous API; await aclose()")
        if self._bridge is not None:
            self._bridge.run(self.aclose())
            self._bridge.stop()
            self._bridge = None
        self._closed = True

    async def __aenter__(self) -> GabbyClient:
        self._bind("async")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    def __enter__(self) -> GabbyClient:
        self._bind("sync")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _bind(self, mode: str) -> None:
        if self._closed:
            raise RuntimeError("GabbyClient is closed")
        bridge_call = (
            self._mode == "sync" and self._bridge is not None and self._bridge.is_loop_thread
        )
        if self._mode is not None and self._mode != mode and not bridge_call:
            raise RuntimeError("Use either the async or synchronous client API consistently")
        if mode == "sync":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError("Synchronous client methods cannot run inside an event loop")
        else:
            loop = asyncio.get_running_loop()
            if self._loop is not None and self._loop is not loop:
                raise RuntimeError("An async GabbyClient must be used on the same event loop")
            self._loop = loop
        if not bridge_call:
            self._mode = mode

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
        return self._client

    def _url(self, action: str) -> str:
        return f"{self._base_url}/v1/agents/{self._agent_name}/{action}"

    @staticmethod
    def _payload(
        input: str,
        context: Mapping[str, Any] | None,
        memory: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        include_trace: bool,
    ) -> dict[str, Any]:
        if not isinstance(input, str) or not input:
            raise ValueError("input must be a non-empty string")
        for name, value in (("context", context), ("memory", memory), ("metadata", metadata)):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
        if not isinstance(include_trace, bool):
            raise TypeError("include_trace must be a boolean")
        return {
            "input": input,
            "context": dict(context or {}),
            "memory": dict(memory or {}),
            "metadata": dict(metadata or {}),
            "include_trace": include_trace,
        }

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > self._max_response_bytes:
                raise ValueError("Gabby HTTP response exceeded max_response_bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    async def _check_response(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = await self._read_bounded(response)
        self._raise_for_response(response, body)

    @staticmethod
    def _raise_for_response(response: httpx.Response, body: bytes) -> None:
        if response.is_success:
            return
        error_type = "HTTPError"
        message = "Gabby request failed"
        try:
            document = json.loads(body)
            detail = document.get("detail") if isinstance(document, dict) else None
            if isinstance(detail, dict):
                if isinstance(detail.get("error_type"), str):
                    error_type = detail["error_type"]
                if isinstance(detail.get("error"), str):
                    message = detail["error"]
            elif isinstance(detail, str):
                message = detail
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        raise GabbyAPIError(response.status_code, error_type, message)


async def _decode_sse(response: httpx.Response, *, max_bytes: int) -> AsyncIterator[StreamEvent]:
    """Decode bounded SSE frames and validate the Gabby JSON event envelope."""
    total = 0
    line_buffer = bytearray()
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []

    def dispatch() -> StreamEvent | None:
        nonlocal event_name, event_id, data_lines
        if not data_lines:
            event_name = None
            event_id = None
            data_lines = []
            return None
        try:
            value = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise ValueError("Gabby SSE event contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Gabby SSE event must contain a JSON object")
        kind, data = value.get("type"), value.get("data")
        if not isinstance(kind, str) or not kind or not isinstance(data, dict):
            raise ValueError("Gabby SSE event has an invalid type or data object")
        if event_name is not None and event_name != kind:
            raise ValueError("Gabby SSE event name does not match its JSON envelope")
        event_name = None
        result_event_id = event_id
        event_id = None
        data_lines = []
        return StreamEvent(kind, data, result_event_id)

    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("Gabby SSE response exceeded max_response_bytes")
        line_buffer.extend(chunk)
        offset = 0
        while True:
            newline = line_buffer.find(b"\n", offset)
            if newline < 0:
                if len(line_buffer) - offset > max_bytes:
                    raise ValueError("Gabby SSE line exceeded max_response_bytes")
                break
            raw_line = bytes(line_buffer[offset:newline])
            offset = newline + 1
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if not raw_line:
                event = dispatch()
                if event is not None:
                    yield event
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Gabby SSE stream is not valid UTF-8") from exc
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value
            elif field == "id":
                if "\x00" not in value:
                    event_id = value
            elif field == "data":
                data_lines.append(value)
        if offset:
            del line_buffer[:offset]

    if line_buffer or data_lines:
        raise ValueError("Gabby SSE stream ended before the final event delimiter")
