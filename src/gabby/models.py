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
"""Async model provider contracts and built-in chat completion adapters."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from inspect import signature
from itertools import chain
from typing import Any, ClassVar, Protocol, cast
from urllib.parse import quote

import httpx

from ._sync import run_sync_callback
from .config import (
    DEFAULT_MAX_MODEL_REQUEST_BYTES,
    DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    validate_model_credentials,
    validate_model_endpoint,
)

_MAX_TOOL_RESPONSE_TEMPLATE_BYTES = 64 * 1024


@dataclass
class ModelResponse:
    """Normalized completion text, tool calls, provider usage, and raw response data."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStreamDelta:
    """Provider-neutral piece of a streamed model response."""

    content_delta: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name_delta: str | None = None
    tool_arguments_delta: str | None = None
    provider_metadata: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


class StreamingModelProvider(Protocol):
    """Optional model-provider interface for incremental response deltas."""

    name: str

    def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
    ) -> AsyncIterator[ModelStreamDelta]:
        """Yield incremental text, partial tool calls, usage, and finish information."""
        ...


class ModelProvider(Protocol):
    """Asynchronous model interface shared by every reasoning backend."""

    name: str

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
    ) -> ModelResponse:
        """Return a completion or structured tool calls within the supplied timeout."""
        ...


class ModelError(RuntimeError):
    """The provider failed to return a usable completion."""


class RetryableModelError(ModelError):
    """A transient provider failure that a configured runtime policy may retry."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, (int, float))
            or not math.isfinite(retry_after_seconds)
            or retry_after_seconds < 0
        ):
            raise ValueError("retry_after_seconds must be a finite non-negative number")
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ModelResponseSizeError(ModelError):
    """A provider response exceeded the configured UTF-8 byte limit."""


class ModelRequestSizeError(ValueError):
    """A serialized model request exceeded the agent's configured byte limit."""


def ensure_model_request_size(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    temperature: float | None,
    max_bytes: int,
    streaming: bool = False,
    supports_tool_choice: bool = True,
) -> None:
    """Reject an oversized canonical provider request without building its full JSON body."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if streaming:
        payload["stream"] = True
    if tools:
        payload["tools"] = tools
        if supports_tool_choice:
            payload["tool_choice"] = "auto"
    if temperature is not None:
        payload["temperature"] = temperature

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    byte_count = 0
    try:
        for chunk in encoder.iterencode(payload):
            byte_count += len(chunk.encode("utf-8"))
            if byte_count > max_bytes:
                raise ModelRequestSizeError(
                    f"Serialized model request exceeded max_model_request_bytes={max_bytes}"
                )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ModelRequestSizeError):
            raise
        raise ModelRequestSizeError("Model request is not valid JSON") from exc


def _method_accepts_keyword(method: Any, name: str) -> bool:
    """Check whether a provider method accepts an optional limit keyword."""
    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == name for parameter in parameters)


def ensure_model_response_size(response: ModelResponse, *, max_bytes: int) -> None:
    """Bound normalized response text and structured tool-call strings without copying them."""
    _count_utf8_payload((response.content, response.tool_calls, response.usage), max_bytes)


def _count_utf8_payload(values: Any, max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    stack: list[tuple[Iterator[Any], int | None]] = [(iter(values), None)]
    active: set[int] = set()
    byte_count = 0
    node_count = 0
    while stack:
        iterator, container_id = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            if container_id is not None:
                active.remove(container_id)
            continue
        node_count += 1
        if node_count > max_bytes or len(stack) > 128:
            raise ModelResponseSizeError("Model response exceeded its structural size limit")
        if isinstance(item, str):
            remaining = max_bytes - byte_count
            encoded_length = 0
            for character in item:
                codepoint = ord(character)
                if codepoint <= 0x7F:
                    encoded_length += 1
                elif codepoint <= 0x7FF:
                    encoded_length += 2
                elif 0xD800 <= codepoint <= 0xDFFF:
                    raise ModelResponseSizeError("Model response contains invalid Unicode")
                elif codepoint <= 0xFFFF:
                    encoded_length += 3
                else:
                    encoded_length += 4
                if encoded_length > remaining:
                    raise ModelResponseSizeError(
                        f"Model response exceeded max_model_response_bytes={max_bytes}"
                    )
            byte_count += encoded_length
        elif isinstance(item, (list, tuple, dict)):
            identity = id(item)
            if identity in active:
                raise ModelResponseSizeError("Model response contains recursive data")
            active.add(identity)
            children = chain.from_iterable(item.items()) if isinstance(item, dict) else iter(item)
            stack.append((iter(children), identity))
    return byte_count


async def complete_with_response_limit(
    provider: ModelProvider,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    temperature: float | None,
    timeout_seconds: float,
    max_response_bytes: int,
) -> ModelResponse:
    """Call a provider with its native response cap when supported, then validate its result."""
    complete = provider.complete
    kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "model": model,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
    }
    if _method_accepts_keyword(complete, "max_response_bytes"):
        kwargs["max_response_bytes"] = max_response_bytes
    response = await complete(**kwargs)
    if not isinstance(response, ModelResponse):
        raise ModelError("Model provider returned an invalid response object")
    ensure_model_response_size(response, max_bytes=max_response_bytes)
    return response


async def stream_with_response_limit(
    provider: StreamingModelProvider,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    temperature: float | None,
    timeout_seconds: float,
    max_response_bytes: int,
) -> AsyncIterator[ModelStreamDelta]:
    """Forward provider deltas while bounding accumulated text and tool-call data."""
    _validate_response_limit(max_response_bytes)
    stream = provider.stream
    kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "model": model,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
    }
    if _method_accepts_keyword(stream, "max_response_bytes"):
        kwargs["max_response_bytes"] = max_response_bytes
    used_bytes = 0
    async for delta in stream(**kwargs):
        if not isinstance(delta, ModelStreamDelta):
            raise ModelError("Model provider returned a malformed stream delta")
        if any(
            value is not None and not isinstance(value, str)
            for value in (
                delta.content_delta,
                delta.tool_call_id,
                delta.tool_name_delta,
                delta.tool_arguments_delta,
                delta.finish_reason,
            )
        ):
            raise ModelError("Model provider returned a malformed stream delta")
        if delta.tool_call_index is not None and (
            isinstance(delta.tool_call_index, bool)
            or not isinstance(delta.tool_call_index, int)
            or delta.tool_call_index < 0
        ):
            raise ModelError("Model provider returned a malformed stream delta")
        if delta.usage is not None and not isinstance(delta.usage, dict):
            raise ModelError("Model provider returned a malformed stream delta")
        if delta.provider_metadata is not None and not isinstance(delta.provider_metadata, dict):
            raise ModelError("Model provider returned malformed call metadata")
        delta_bytes = _count_utf8_payload(
            (
                delta.content_delta,
                delta.tool_call_id,
                delta.tool_name_delta,
                delta.tool_arguments_delta,
                delta.usage,
                delta.provider_metadata,
            ),
            max_response_bytes,
        )
        if used_bytes + delta_bytes > max_response_bytes:
            raise ModelResponseSizeError(
                f"Model response exceeded max_model_response_bytes={max_response_bytes}"
            )
        used_bytes += delta_bytes
        yield delta


def _validate_response_limit(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_response_bytes must be a positive integer")


async def _read_response_limited(response: httpx.Response, *, max_bytes: int) -> bytearray:
    """Read a response body incrementally and stop before buffering beyond its cap."""
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            raise ModelResponseSizeError(
                f"Model response exceeded max_model_response_bytes={max_bytes}"
            )
        body.extend(chunk)
    return body


def _extract_sse_lines(buffer: bytearray, *, final: bool = False) -> Iterator[bytes]:
    """Yield complete LF, CRLF, or CR terminated lines from a bounded byte buffer."""
    start = 0
    while True:
        carriage_return = buffer.find(b"\r", start)
        line_feed = buffer.find(b"\n", start)
        candidates = [position for position in (carriage_return, line_feed) if position >= 0]
        if not candidates:
            break
        boundary = min(candidates)
        if buffer[boundary] == 13 and boundary + 1 == len(buffer) and not final:
            break
        ending = 2 if buffer[boundary : boundary + 2] == b"\r\n" else 1
        yield bytes(buffer[start:boundary])
        start = boundary + ending
    if start:
        del buffer[:start]
    if final and buffer:
        yield bytes(buffer)
        buffer.clear()


async def _iter_response_lines_limited(
    response: httpx.Response, *, max_bytes: int
) -> AsyncIterator[str]:
    """Yield UTF-8 SSE lines while limiting the complete decoded response body."""
    total_bytes = 0
    buffer = bytearray()

    def decode_line(value: bytes) -> str:
        if value.endswith(b"\r"):
            value = value[:-1]
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelError("Model stream returned invalid UTF-8") from exc

    async for chunk in response.aiter_bytes():
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise ModelResponseSizeError(
                f"Model response exceeded max_model_response_bytes={max_bytes}"
            )
        buffer.extend(chunk)
        for raw_line in _extract_sse_lines(buffer):
            yield decode_line(raw_line)
    for raw_line in _extract_sse_lines(buffer, final=True):
        yield decode_line(raw_line)


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse and bound a delta-seconds Retry-After header without exposing it."""
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, 5.0)


@dataclass
class OpenAICompatibleProvider:
    """Chat completions adapter for APIs implementing the OpenAI-compatible contract."""

    supports_tool_choice: ClassVar[bool] = True

    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    extra_headers: dict[str, str] = field(default_factory=dict)
    name: str = "openai_compatible"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Prevent built-in adapters from sending prompts over remote cleartext HTTP."""
        validate_model_endpoint(self.base_url, provider=self.name)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OpenAICompatibleProvider:
        """Construct a provider from model configuration without embedding secret values."""
        validate_model_credentials(config, provider="openai_compatible")
        return cls(
            base_url=str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            extra_headers=dict(config.get("headers", {})),
        )

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> ModelResponse:
        """Request and validate a non-streamed chat completion."""
        _validate_response_limit(max_response_bytes)
        key = self.api_key or os.environ.get(self.api_key_env)
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            if self.supports_tool_choice:
                payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                timeout=httpx.Timeout(timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    # Provider bodies can echo prompts, headers, or other private data.
                    # Keep the status useful for diagnosis without exposing that body.
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=retry_after,
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                response_body = await _read_response_limited(response, max_bytes=max_response_bytes)
                data = json.loads(response_body)
        except ModelError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            # These failures may be transient. The retry policy remains caller-owned.
            raise RetryableModelError("Model request failed") from None
        except (httpx.HTTPError, ValueError):
            # Exception strings may contain the request URL or provider supplied data.
            raise ModelError("Model request failed") from None
        try:
            if not isinstance(data, dict):
                raise TypeError("Response must be an object")
            choices = data["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError("Response choices must be a non-empty list")
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise TypeError("Response choice must be an object")
            choice = first_choice["message"]
            if not isinstance(choice, dict):
                raise TypeError("Response message must be an object")
            content = choice.get("content")
            tool_calls = choice.get("tool_calls", []) or []
            usage = data.get("usage", {})
            if content is not None and not isinstance(content, str):
                raise TypeError("Message content must be text or null")
            if not isinstance(tool_calls, list) or not isinstance(usage, dict):
                raise TypeError("Invalid tool_calls or usage shape")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelError(
                "Provider response did not include a valid choices[0].message"
            ) from exc
        return ModelResponse(content=content, tool_calls=tool_calls, usage=usage, raw=data)

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> AsyncIterator[ModelStreamDelta]:
        """Stream OpenAI-compatible chat completion deltas, including partial tool calls."""
        _validate_response_limit(max_response_bytes)
        key = self.api_key or os.environ.get(self.api_key_env)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            if self.supports_tool_choice:
                payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self.extra_headers,
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                timeout=httpx.Timeout(timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=retry_after,
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                async for line in _iter_response_lines_limited(
                    response, max_bytes=max_response_bytes
                ):
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError as exc:
                        raise ModelError("Model stream returned invalid JSON") from exc
                    if not isinstance(chunk, dict):
                        raise ModelError("Model stream returned an invalid event")
                    if "error" in chunk:
                        raise ModelError("Model stream returned an error event")
                    usage = chunk.get("usage")
                    if usage is not None and not isinstance(usage, dict):
                        raise ModelError("Model stream returned invalid token usage")
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list):
                        raise ModelError("Model stream returned invalid choices")
                    if not choices:
                        if usage:
                            yield ModelStreamDelta(usage=usage)
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict) or not isinstance(
                        choice.get("delta", {}), dict
                    ):
                        raise ModelError("Model stream returned an invalid choice")
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise ModelError("Model stream returned non-text content")
                    emitted = False
                    if content:
                        yield ModelStreamDelta(content_delta=content)
                        emitted = True
                    calls = delta.get("tool_calls", [])
                    if not isinstance(calls, list):
                        raise ModelError("Model stream returned invalid tool calls")
                    for item in calls:
                        if not isinstance(item, dict):
                            raise ModelError("Model stream returned an invalid tool call")
                        index = item.get("index")
                        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                            raise ModelError("Model stream returned an invalid tool call index")
                        call_id = item.get("id")
                        function = item.get("function", {})
                        if call_id is not None and not isinstance(call_id, str):
                            raise ModelError("Model stream returned an invalid tool call ID")
                        if not isinstance(function, dict):
                            raise ModelError("Model stream returned an invalid function delta")
                        name = function.get("name")
                        arguments = function.get("arguments")
                        if name is not None and not isinstance(name, str):
                            raise ModelError("Model stream returned an invalid tool name")
                        if arguments is not None and not isinstance(arguments, str):
                            raise ModelError("Model stream returned invalid tool arguments")
                        yield ModelStreamDelta(
                            tool_call_index=index,
                            tool_call_id=call_id,
                            tool_name_delta=name,
                            tool_arguments_delta=arguments,
                        )
                        emitted = True
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None and not isinstance(finish_reason, str):
                        raise ModelError("Model stream returned an invalid finish reason")
                    if usage or finish_reason is not None or not emitted:
                        yield ModelStreamDelta(usage=usage, finish_reason=finish_reason)
        except ModelError:
            raise
        except httpx.TimeoutException:
            raise RetryableModelError("Model stream exceeded its timeout") from None
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            raise RetryableModelError("Model stream request failed") from None
        except httpx.HTTPError:
            raise ModelError("Model stream request failed") from None

    async def aclose(self) -> None:
        """Close the lazily created HTTP client and release its connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Translate Gabby's OpenAI-shaped internal messages to Anthropic Messages input."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ModelError("Model request contains an invalid message")
        role = message["role"]
        content = message.get("content")
        if role == "system":
            if not isinstance(content, str):
                raise ModelError("System message content must be text")
            system_parts.append(content)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ModelError("Tool result is missing its call ID")
            if not isinstance(content, str):
                raise ModelError("Tool result content must be text")
            pending_results.append(
                {"type": "tool_result", "tool_use_id": call_id, "content": content}
            )
            continue
        if role not in {"user", "assistant"}:
            raise ModelError("Model request contains an unsupported message role")
        if pending_results:
            converted.append({"role": "user", "content": pending_results})
            pending_results = []
        blocks: list[dict[str, Any]] = []
        if content is not None:
            if not isinstance(content, str):
                raise ModelError("Message content must be text or null")
            if content:
                blocks.append({"type": "text", "text": content})
        calls = message.get("tool_calls", [])
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ModelError("Only assistant messages may contain tool calls")
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise ModelError("Model request contains a malformed tool call")
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                except ValueError:
                    raise ModelError("Model request contains malformed tool arguments") from None
                if not isinstance(arguments, dict):
                    raise ModelError("Tool arguments must be a JSON object")
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                )
        converted.append({"role": role, "content": blocks or ""})
    if pending_results:
        converted.append({"role": "user", "content": pending_results})
    return "\n\n".join(system_parts), converted


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate normalized function definitions to Anthropic tool schemas."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("parameters"), dict)
        ):
            raise ModelError("Model request contains an invalid tool schema")
        item: dict[str, Any] = {
            "name": function["name"],
            "input_schema": function["parameters"],
        }
        if isinstance(function.get("description"), str):
            item["description"] = function["description"]
        converted.append(item)
    return converted


@dataclass
class AnthropicProvider:
    """Native Anthropic Messages API adapter with normalized Gabby tool calls."""

    api_key: str | None = None
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 1024
    extra_headers: dict[str, str] = field(default_factory=dict)
    name: str = "anthropic"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_model_endpoint(self.base_url, provider=self.name)
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 200_000
        ):
            raise ValueError("max_tokens must be an integer from 1 through 200000")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AnthropicProvider:
        """Build the provider from a model mapping without embedding credentials."""
        validate_model_credentials(config, provider="anthropic")
        max_tokens = config.get("max_tokens", 1024)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("model.max_tokens must be an integer")
        return cls(
            base_url=str(config.get("base_url", "https://api.anthropic.com/v1")).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "ANTHROPIC_API_KEY")),
            max_tokens=max_tokens,
            extra_headers=dict(config.get("headers", {})),
        )

    def _request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        streaming: bool,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        system, converted_messages = _anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": converted_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _anthropic_tools(tools)
        if temperature is not None:
            payload["temperature"] = temperature
        if streaming:
            payload["stream"] = True
        try:
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            size = 0
            for chunk in encoder.iterencode(payload):
                size += len(chunk.encode("utf-8"))
                if size > DEFAULT_MAX_MODEL_REQUEST_BYTES:
                    raise ModelRequestSizeError(
                        "Serialized model request exceeded "
                        f"max_model_request_bytes={DEFAULT_MAX_MODEL_REQUEST_BYTES}"
                    )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ModelRequestSizeError):
                raise
            raise ModelRequestSizeError("Model request is not valid JSON") from None
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Accept": "text/event-stream" if streaming else "application/json",
            **self.extra_headers,
        }
        key = self.api_key or os.environ.get(self.api_key_env)
        if key:
            headers["x-api-key"] = key
        return payload, headers

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> ModelResponse:
        """Request one Anthropic Messages completion and normalize content/tool use."""
        _validate_response_limit(max_response_bytes)
        payload, headers = self._request(
            messages=messages, tools=tools, model=model, temperature=temperature, streaming=False
        )
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        try:
            async with client.stream(
                "POST", "/messages", json=payload, timeout=httpx.Timeout(timeout_seconds)
            ) as response:
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=_retry_after_seconds(
                                response.headers.get("Retry-After")
                            ),
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                data = json.loads(
                    await _read_response_limited(response, max_bytes=max_response_bytes)
                )
        except ModelError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            raise RetryableModelError("Model request failed") from None
        except (httpx.HTTPError, ValueError):
            raise ModelError("Model request failed") from None
        try:
            if not isinstance(data, dict) or not isinstance(data.get("content"), list):
                raise TypeError
            text: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in data["content"]:
                if not isinstance(block, dict):
                    raise TypeError
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text.append(block["text"])
                elif block.get("type") == "tool_use":
                    if (
                        not isinstance(block.get("id"), str)
                        or not isinstance(block.get("name"), str)
                        or not isinstance(block.get("input"), dict)
                    ):
                        raise TypeError
                    calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(
                                    block["input"], ensure_ascii=False, separators=(",", ":")
                                ),
                            },
                        }
                    )
                else:
                    raise TypeError
            usage_data = data.get("usage", {})
            if not isinstance(usage_data, dict):
                raise TypeError
            usage = {
                "prompt_tokens": usage_data.get("input_tokens", 0),
                "completion_tokens": usage_data.get("output_tokens", 0),
            }
        except (KeyError, TypeError, ValueError):
            raise ModelError("Provider response did not include valid message content") from None
        return ModelResponse(content="".join(text) or None, tool_calls=calls, usage=usage, raw=data)

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> AsyncIterator[ModelStreamDelta]:
        """Stream Anthropic text and tool-use events as provider-neutral deltas."""
        _validate_response_limit(max_response_bytes)
        payload, headers = self._request(
            messages=messages, tools=tools, model=model, temperature=temperature, streaming=True
        )
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)
        tool_blocks: set[int] = set()
        try:
            async with client.stream(
                "POST", "/messages", json=payload, timeout=httpx.Timeout(timeout_seconds)
            ) as response:
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=_retry_after_seconds(
                                response.headers.get("Retry-After")
                            ),
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                async for line in _iter_response_lines_limited(
                    response, max_bytes=max_response_bytes
                ):
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except ValueError:
                        raise ModelError("Model stream returned invalid JSON") from None
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        raise ModelError("Model stream returned an invalid event")
                    kind = event["type"]
                    if kind == "error":
                        raise ModelError("Model stream returned an error event")
                    if kind == "content_block_start":
                        index = event.get("index")
                        block = event.get("content_block")
                        if (
                            isinstance(index, bool)
                            or not isinstance(index, int)
                            or index < 0
                            or not isinstance(block, dict)
                        ):
                            raise ModelError("Model stream returned an invalid content block")
                        if block.get("type") == "tool_use":
                            if not isinstance(block.get("id"), str) or not isinstance(
                                block.get("name"), str
                            ):
                                raise ModelError("Model stream returned an invalid tool call")
                            tool_blocks.add(index)
                            yield ModelStreamDelta(
                                tool_call_index=index,
                                tool_call_id=block["id"],
                                tool_name_delta=block["name"],
                            )
                            initial = block.get("input", {})
                            if not isinstance(initial, dict):
                                raise ModelError("Model stream returned invalid tool input")
                            if initial:
                                yield ModelStreamDelta(
                                    tool_call_index=index,
                                    tool_arguments_delta=json.dumps(
                                        initial, ensure_ascii=False, separators=(",", ":")
                                    ),
                                )
                    elif kind == "content_block_delta":
                        index = event.get("index")
                        delta = event.get("delta")
                        if (
                            isinstance(index, bool)
                            or not isinstance(index, int)
                            or not isinstance(delta, dict)
                        ):
                            raise ModelError("Model stream returned an invalid content delta")
                        if delta.get("type") == "text_delta":
                            value = delta.get("text")
                            if not isinstance(value, str):
                                raise ModelError("Model stream returned non-text content")
                            if value:
                                yield ModelStreamDelta(content_delta=value)
                        elif delta.get("type") == "input_json_delta" and index in tool_blocks:
                            value = delta.get("partial_json")
                            if not isinstance(value, str):
                                raise ModelError("Model stream returned invalid tool arguments")
                            yield ModelStreamDelta(
                                tool_call_index=index, tool_arguments_delta=value
                            )
                    elif kind == "message_start":
                        message = event.get("message")
                        usage_data = message.get("usage") if isinstance(message, dict) else None
                        if isinstance(usage_data, dict):
                            yield ModelStreamDelta(
                                usage={"prompt_tokens": usage_data.get("input_tokens", 0)}
                            )
                    elif kind == "message_delta":
                        delta = event.get("delta", {})
                        usage_data = event.get("usage", {})
                        if not isinstance(delta, dict) or not isinstance(usage_data, dict):
                            raise ModelError("Model stream returned invalid message metadata")
                        finish = delta.get("stop_reason")
                        usage = (
                            {"completion_tokens": usage_data.get("output_tokens", 0)}
                            if usage_data
                            else None
                        )
                        if finish is not None and not isinstance(finish, str):
                            raise ModelError("Model stream returned an invalid finish reason")
                        yield ModelStreamDelta(usage=usage, finish_reason=finish)
        except ModelError:
            raise
        except httpx.TimeoutException:
            raise RetryableModelError("Model stream exceeded its timeout") from None
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            raise RetryableModelError("Model stream request failed") from None
        except httpx.HTTPError:
            raise ModelError("Model stream request failed") from None

    async def aclose(self) -> None:
        """Close the provider's connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


def _gemini_request_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Convert Gabby's OpenAI-shaped messages into Gemini GenerateContent contents."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    call_metadata: dict[str, dict[str, Any]] = {}

    def flush_tool_results() -> None:
        if pending_tool_results:
            contents.append({"role": "user", "parts": pending_tool_results.copy()})
            pending_tool_results.clear()

    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ModelError("Model request contains an invalid message")
        role = message["role"]
        content = message.get("content")
        if role == "system":
            if not isinstance(content, str):
                raise ModelError("System message content must be text")
            system_parts.append(content)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in tool_names:
                raise ModelError("Tool result is missing a known call ID")
            if not isinstance(content, str):
                raise ModelError("Tool result content must be text")
            metadata = call_metadata.pop(call_id, {})
            try:
                result = json.loads(content)
            except ValueError:
                result = {"result": content}
            if not isinstance(result, dict):
                result = {"result": result}
            function_response: dict[str, Any] = {
                "name": tool_names.pop(call_id),
                "response": result,
            }
            provider_call_id = metadata.get("gemini_function_id")
            if provider_call_id is not None:
                function_response["id"] = provider_call_id
            pending_tool_results.append({"functionResponse": function_response})
            continue
        if role not in {"user", "assistant"}:
            raise ModelError("Model request contains an unsupported message role")
        flush_tool_results()
        parts: list[dict[str, Any]] = []
        original_response_parts: list[dict[str, Any]] | None = None
        if content is not None:
            if not isinstance(content, str):
                raise ModelError("Message content must be text or null")
            if content:
                parts.append({"text": content})
        calls = message.get("tool_calls", [])
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ModelError("Only assistant messages may contain tool calls")
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise ModelError("Model request contains a malformed tool call")
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                except ValueError:
                    raise ModelError("Model request contains malformed tool arguments") from None
                if not isinstance(arguments, dict):
                    raise ModelError("Tool arguments must be a JSON object")
                call_id = call["id"]
                if call_id in tool_names:
                    raise ModelError("Model request contains duplicate tool call IDs")
                tool_names[call_id] = function["name"]
                metadata = call.get("provider_metadata", {})
                if not isinstance(metadata, dict):
                    raise ModelError("Model request contains invalid provider call metadata")
                preserved_parts = metadata.get("gemini_response_parts")
                if preserved_parts is not None:
                    if not isinstance(preserved_parts, list) or any(
                        not isinstance(item, dict) for item in preserved_parts
                    ):
                        raise ModelError("Model request contains invalid preserved model parts")
                    if (
                        original_response_parts is not None
                        and original_response_parts != preserved_parts
                    ):
                        raise ModelError("Model request contains conflicting preserved model parts")
                    original_response_parts = preserved_parts
                function_call: dict[str, Any] = {
                    "name": function["name"],
                    "args": arguments,
                }
                for source, target in (
                    ("gemini_function_id", "id"),
                    ("gemini_thought_signature", "thoughtSignature"),
                ):
                    value = metadata.get(source)
                    if value is not None:
                        if not isinstance(value, str):
                            raise ModelError(
                                "Model request contains invalid provider call metadata"
                            )
                        function_call[target] = value
                call_metadata[call_id] = metadata
                parts.append({"functionCall": function_call})
        if original_response_parts is not None:
            contents.append({"role": "model", "parts": original_response_parts})
        elif parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    flush_tool_results()
    system_instruction = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    return system_instruction, contents


def _gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized function definitions into Gemini function declarations."""
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("parameters"), dict)
        ):
            raise ModelError("Model request contains an invalid tool schema")
        declaration: dict[str, Any] = {
            "name": function["name"],
            "parametersJsonSchema": function["parameters"],
        }
        description = function.get("description")
        if isinstance(description, str):
            declaration["description"] = description
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _gemini_usage(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ModelError("Provider response contained invalid token usage")
    usage: dict[str, Any] = {}
    for source, target in (
        ("promptTokenCount", "prompt_tokens"),
        ("candidatesTokenCount", "completion_tokens"),
        ("totalTokenCount", "total_tokens"),
    ):
        count = value.get(source)
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ModelError("Provider response contained invalid token usage")
            usage[target] = count
    return usage


def _gemini_candidate_parts(
    data: object,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    if not isinstance(data, dict):
        raise ModelError("Provider response was not a JSON object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ModelError("Provider response did not contain a candidate")
    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise ModelError("Provider response contained invalid content parts")
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ModelError("Provider response contained an invalid content part")
        function_call = part.get("functionCall")
        if part.get("thought") is True and function_call is None:
            continue
        text = part.get("text")
        if text is not None:
            if not isinstance(text, str):
                raise ModelError("Provider response contained non-text content")
            if part.get("thought") is not True:
                text_parts.append(text)
        if function_call is not None:
            if (
                not isinstance(function_call, dict)
                or not isinstance(function_call.get("name"), str)
                or not function_call["name"]
                or not isinstance(function_call.get("args", {}), dict)
            ):
                raise ModelError("Provider response contained an invalid function call")
            provider_metadata: dict[str, str] = {}
            provider_call_id = function_call.get("id")
            if provider_call_id is not None:
                if not isinstance(provider_call_id, str) or not provider_call_id:
                    raise ModelError("Provider response contained an invalid function call ID")
                provider_metadata["gemini_function_id"] = provider_call_id
            thought_signature = part.get("thoughtSignature")
            if thought_signature is not None:
                if not isinstance(thought_signature, str) or not thought_signature:
                    raise ModelError("Provider response contained an invalid thought signature")
                provider_metadata["gemini_thought_signature"] = thought_signature
            normalized_call: dict[str, Any] = {
                "id": uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": function_call["name"],
                    "arguments": json.dumps(
                        function_call.get("args", {}),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                },
            }
            if provider_metadata:
                normalized_call["provider_metadata"] = provider_metadata
            calls.append(normalized_call)
        if text is None and function_call is None and part.get("thought") is not True:
            raise ModelError("Provider response contained an unsupported content part")
    finish_reason = candidate.get("finishReason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ModelError("Provider response contained an invalid finish reason")
    if calls:
        metadata = calls[0].setdefault("provider_metadata", {})
        metadata["gemini_response_parts"] = parts
    return calls, "".join(text_parts) or None, _gemini_usage(data.get("usageMetadata"))


@dataclass
class GeminiProvider:
    """Native Google Gemini GenerateContent adapter with normalized tool calling."""

    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    max_output_tokens: int = 1024
    extra_headers: dict[str, str] = field(default_factory=dict)
    name: str = "gemini"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_model_endpoint(self.base_url, provider=self.name)
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or not 1 <= self.max_output_tokens <= 200_000
        ):
            raise ValueError("max_output_tokens must be an integer from 1 through 200000")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GeminiProvider:
        """Build the provider from model configuration without embedding credentials."""
        validate_model_credentials(config, provider="gemini")
        max_output_tokens = config.get("max_output_tokens", 1024)
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
            raise ValueError("model.max_output_tokens must be an integer")
        return cls(
            base_url=str(
                config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
            ).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
            max_output_tokens=max_output_tokens,
            extra_headers=dict(config.get("headers", {})),
        )

    def _request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        streaming: bool,
    ) -> tuple[dict[str, Any], dict[str, str], str]:
        if not isinstance(model, str) or not model or len(model) > 512:
            raise ModelError("Model identifier is invalid")
        system_instruction, contents = _gemini_request_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": self.max_output_tokens},
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = _gemini_tools(tools)
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        try:
            size = 0
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for chunk in encoder.iterencode(payload):
                size += len(chunk.encode("utf-8"))
                if size > DEFAULT_MAX_MODEL_REQUEST_BYTES:
                    raise ModelRequestSizeError(
                        "Serialized model request exceeded "
                        f"max_model_request_bytes={DEFAULT_MAX_MODEL_REQUEST_BYTES}"
                    )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ModelRequestSizeError):
                raise
            raise ModelRequestSizeError("Model request is not valid JSON") from None
        api_key = self.api_key or os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if api_key:
            headers["x-goog-api-key"] = api_key
        action = "streamGenerateContent?alt=sse" if streaming else "generateContent"
        endpoint = f"/models/{quote(model, safe='')}:" + action
        return payload, headers, endpoint

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> ModelResponse:
        """Request one Gemini completion and normalize text, calls, and usage."""
        _validate_response_limit(max_response_bytes)
        payload, headers, endpoint = self._request(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            streaming=False,
        )
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url, headers={"Content-Type": "application/json"}
            )
        try:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=_retry_after_seconds(
                                response.headers.get("Retry-After")
                            ),
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                raw = json.loads(
                    await _read_response_limited(response, max_bytes=max_response_bytes)
                )
        except ModelError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            raise RetryableModelError("Model request failed") from None
        except (httpx.HTTPError, ValueError):
            raise ModelError("Model request failed") from None
        try:
            calls, content, usage = _gemini_candidate_parts(raw)
        except ModelError:
            raise
        except (TypeError, ValueError):
            raise ModelError("Provider response did not include valid candidate content") from None
        if not isinstance(raw, dict):
            raise ModelError("Provider response was not a JSON object")
        return ModelResponse(content=content, tool_calls=calls, usage=usage, raw=raw)

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> AsyncIterator[ModelStreamDelta]:
        """Stream Gemini text and complete function calls as normalized deltas."""
        _validate_response_limit(max_response_bytes)
        payload, headers, endpoint = self._request(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            streaming=True,
        )
        headers["Accept"] = "text/event-stream"
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url, headers={"Content-Type": "application/json"}
            )
        call_index = 0
        response_parts: list[dict[str, Any]] = []
        try:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    if response.status_code in {408, 425, 429} or response.status_code >= 500:
                        raise RetryableModelError(
                            f"Model returned HTTP {response.status_code}",
                            retry_after_seconds=_retry_after_seconds(
                                response.headers.get("Retry-After")
                            ),
                        )
                    raise ModelError(f"Model returned HTTP {response.status_code}")
                async for line in _iter_response_lines_limited(
                    response, max_bytes=max_response_bytes
                ):
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                    except ValueError:
                        raise ModelError("Model stream returned invalid JSON") from None
                    if not isinstance(data, dict):
                        raise ModelError("Model stream returned an invalid event")
                    candidates = data.get("candidates")
                    if not isinstance(candidates, list) or not candidates:
                        usage = _gemini_usage(data.get("usageMetadata"))
                        if usage:
                            yield ModelStreamDelta(usage=usage)
                        continue
                    if not isinstance(candidates[0], dict):
                        raise ModelError("Model stream returned an invalid candidate")
                    candidate_content = candidates[0].get("content", {})
                    event_parts = (
                        candidate_content.get("parts", [])
                        if isinstance(candidate_content, dict)
                        else None
                    )
                    if not isinstance(event_parts, list) or any(
                        not isinstance(part, dict) for part in event_parts
                    ):
                        raise ModelError("Model stream returned invalid content parts")
                    response_parts.extend(event_parts)
                    calls, content, usage = _gemini_candidate_parts(data)
                    if content:
                        yield ModelStreamDelta(content_delta=content)
                    for call_offset, call in enumerate(calls):
                        function = call["function"]
                        call_metadata = dict(call.get("provider_metadata", {}))
                        if call_offset == 0:
                            call_metadata["gemini_response_parts"] = response_parts.copy()
                        yield ModelStreamDelta(
                            tool_call_index=call_index,
                            tool_call_id=call["id"],
                            tool_name_delta=function["name"],
                            tool_arguments_delta=function["arguments"],
                            provider_metadata=call_metadata or None,
                        )
                        call_index += 1
                    candidate = data["candidates"][0]
                    finish_reason = candidate.get("finishReason")
                    if usage or finish_reason is not None:
                        yield ModelStreamDelta(
                            usage=usage or None,
                            finish_reason=finish_reason,
                        )
        except ModelError:
            raise
        except httpx.TimeoutException:
            raise RetryableModelError("Model stream exceeded its timeout") from None
        except (httpx.NetworkError, httpx.RemoteProtocolError):
            raise RetryableModelError("Model stream request failed") from None
        except httpx.HTTPError:
            raise ModelError("Model stream request failed") from None

    async def aclose(self) -> None:
        """Close the provider's connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


@dataclass
class OllamaProvider(OpenAICompatibleProvider):
    """Ollama's OpenAI-compatible `/v1` chat completions interface."""

    name: str = "ollama"
    supports_tool_choice: ClassVar[bool] = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OllamaProvider:
        """Construct an Ollama adapter, defaulting to its local compatible endpoint."""
        validate_model_credentials(config, provider="ollama")
        return cls(
            base_url=str(config.get("base_url", "http://localhost:11434/v1")).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "GABBY_OLLAMA_API_KEY")),
            extra_headers=dict(config.get("headers", {})),
        )


@dataclass
class HuggingFaceInferenceProvider(OpenAICompatibleProvider):
    """Hugging Face Inference Providers chat-completions router adapter."""

    base_url: str = "https://router.huggingface.co/v1"
    api_key_env: str = "HF_TOKEN"
    name: str = "huggingface"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HuggingFaceInferenceProvider:
        """Construct the hosted Hugging Face chat adapter from agent model settings."""
        validate_model_credentials(config, provider="huggingface")
        return cls(
            base_url=str(config.get("base_url", "https://router.huggingface.co/v1")).rstrip("/"),
            api_key_env=str(config.get("api_key_env", "HF_TOKEN")),
            extra_headers=dict(config.get("headers", {})),
        )


@dataclass
class TransformersProvider:
    """Optional local Hugging Face Transformers chat-completion adapter.

    The model is loaded lazily and generation runs outside the asyncio event loop. Model-specific
    chat templates control prompt formatting and structured tool-call parsing.
    """

    model_id: str
    token_env: str = "HF_TOKEN"
    revision: str | None = None
    adapter_id: str | None = None
    adapter_revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    device: str = "cpu"
    max_new_tokens: int = 1024
    max_input_tokens: int = 32768
    tool_response_template: dict[str, Any] | None = None
    name: str = "transformers"
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _generation_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _call_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate resource bounds before the optional backend or model is loaded."""
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty Hugging Face model ID or local path")
        self.model_id = self.model_id.strip()
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be a non-empty string or None")
        if self.adapter_id is not None and (
            not isinstance(self.adapter_id, str) or not self.adapter_id.strip()
        ):
            raise ValueError("adapter_id must be a non-empty PEFT adapter ID or path")
        if self.adapter_revision is not None and (
            not isinstance(self.adapter_revision, str) or not self.adapter_revision.strip()
        ):
            raise ValueError("adapter_revision must be a non-empty string or None")
        if self.adapter_revision is not None and self.adapter_id is None:
            raise ValueError("adapter_revision requires adapter_id")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty PyTorch device")
        self.device = self.device.strip()
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or not 1 <= self.max_new_tokens <= 16384
        ):
            raise ValueError("max_new_tokens must be an integer from 1 through 16384")
        if (
            isinstance(self.max_input_tokens, bool)
            or not isinstance(self.max_input_tokens, int)
            or not 1 <= self.max_input_tokens <= 1_000_000
        ):
            raise ValueError("max_input_tokens must be an integer from 1 through 1000000")
        if self.tool_response_template is not None:
            if not isinstance(self.tool_response_template, Mapping):
                raise ValueError("tool_response_template must be a JSON object")
            try:
                encoded_template = json.dumps(
                    _thaw_json_value(self.tool_response_template),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                if len(encoded_template.encode("utf-8")) > _MAX_TOOL_RESPONSE_TEMPLATE_BYTES:
                    raise ValueError(
                        "tool_response_template exceeds "
                        f"{_MAX_TOOL_RESPONSE_TEMPLATE_BYTES} UTF-8 bytes"
                    )
                self.tool_response_template = json.loads(encoded_template)
            except (TypeError, UnicodeEncodeError) as exc:
                raise ValueError("tool_response_template must contain bounded JSON values") from exc

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TransformersProvider:
        """Construct local model settings without importing or loading heavyweight libraries."""
        from .config import ConfigError, validate_model_credentials

        validate_model_credentials(config, provider="transformers")
        model_id = config.get("model")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigError("model.model must be a non-empty Hugging Face model ID or path")
        token_env = config.get("api_key_env", "HF_TOKEN")
        revision = config.get("revision")
        adapter_id = config.get("adapter_id")
        adapter_revision = config.get("adapter_revision")
        cache_dir = config.get("cache_dir")
        device = config.get("device", "cpu")
        local_files_only = config.get("local_files_only", False)
        max_new_tokens = config.get("max_new_tokens", 1024)
        max_input_tokens = config.get("max_input_tokens", 32768)
        tool_response_template = config.get("tool_response_template")
        if not isinstance(token_env, str) or not token_env:
            raise ConfigError("model.api_key_env must be a non-empty environment variable name")
        if revision is not None and not isinstance(revision, str):
            raise ConfigError("model.revision must be a string")
        if adapter_id is not None and not isinstance(adapter_id, str):
            raise ConfigError("model.adapter_id must be a string")
        if adapter_revision is not None and not isinstance(adapter_revision, str):
            raise ConfigError("model.adapter_revision must be a string")
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise ConfigError("model.cache_dir must be a string")
        if not isinstance(device, str):
            raise ConfigError("model.device must be a string")
        if not isinstance(local_files_only, bool):
            raise ConfigError("model.local_files_only must be a boolean")
        try:
            return cls(
                model_id=model_id,
                token_env=token_env,
                revision=revision,
                adapter_id=adapter_id,
                adapter_revision=adapter_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                device=device,
                max_new_tokens=max_new_tokens,
                max_input_tokens=max_input_tokens,
                tool_response_template=tool_response_template,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    ) -> ModelResponse:
        """Generate one bounded completion on a worker thread, preserving loop responsiveness."""
        _validate_response_limit(max_response_bytes)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        stop_event = threading.Event()
        async with self._call_lock:
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await run_sync_callback(
                        self._complete_sync,
                        messages,
                        tools,
                        model,
                        temperature,
                        max_response_bytes,
                        stop_event,
                    )
                    if not isinstance(response, ModelResponse):
                        raise ModelError("Local model worker returned an invalid response")
                    return response
            except TimeoutError:
                stop_event.set()
                raise ModelError("Local Transformers generation exceeded its timeout") from None
            except asyncio.CancelledError:
                stop_event.set()
                raise

    def _complete_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_name: str,
        temperature: float | None,
        max_response_bytes: int,
        stop_event: threading.Event,
    ) -> ModelResponse:
        """Load and run the model synchronously while serializing access to its weights."""
        with self._generation_lock:
            try:
                tokenizer, model, torch, stopping_criteria = self._load_model()
                prepared_messages = _transformers_messages(messages)
                prompt_options = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                }
                if tools:
                    without_tools = tokenizer.apply_chat_template(
                        prepared_messages,
                        **prompt_options,
                    )
                inputs = tokenizer.apply_chat_template(
                    prepared_messages,
                    tools=tools or None,
                    **prompt_options,
                )
                input_count = int(inputs["input_ids"].shape[-1])
                if input_count > self.max_input_tokens:
                    raise ModelError(
                        f"Local model input exceeded max_input_tokens={self.max_input_tokens}"
                    )
                prompt_ids = inputs["input_ids"][0]
                if tools and prompt_ids.tolist() == without_tools["input_ids"][0].tolist():
                    raise ModelError(
                        "This local model chat template did not include the configured tool schemas"
                    )
                inputs = inputs.to(model.device)

                class StopOnCancellation(stopping_criteria):  # type: ignore[valid-type,misc]
                    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                        del scores, kwargs
                        return torch.full(
                            (input_ids.shape[0],),
                            stop_event.is_set(),
                            dtype=torch.bool,
                            device=input_ids.device,
                        )

                generation_options: dict[str, Any] = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": temperature is not None and temperature > 0,
                    "stopping_criteria": [StopOnCancellation()],
                }
                if temperature is not None and temperature > 0:
                    generation_options["temperature"] = temperature
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_options)
                generated_ids = generated[0][input_count:]
                content = tokenizer.decode(generated_ids, skip_special_tokens=not bool(tools))
                if model_name != self.model_id:
                    raise ModelError(
                        "The requested model does not match the configured local model"
                    )
                response = _transformers_response(
                    tokenizer,
                    content,
                    tools,
                    prefix=prompt_ids,
                    response_template=self.tool_response_template,
                )
                ensure_model_response_size(response, max_bytes=max_response_bytes)
                return response
            except ModelError:
                raise
            except Exception:
                raise ModelError("Local Transformers model execution failed") from None

    def _load_model(self) -> tuple[Any, Any, Any, Any]:
        """Load the optional Transformers and PyTorch backend on first use."""
        if self._model is not None and self._tokenizer is not None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
            except ImportError:
                raise ModelError(
                    "Install Gabby's optional 'transformers' dependencies to use this provider"
                ) from None
            return self._tokenizer, self._model, torch, StoppingCriteria
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

            token = os.environ.get(self.token_env)
            load_options: dict[str, Any] = {
                "local_files_only": self.local_files_only,
                "trust_remote_code": False,
            }
            if token:
                load_options["token"] = token
            if self.revision:
                load_options["revision"] = self.revision
            if self.cache_dir:
                load_options["cache_dir"] = self.cache_dir
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_options)
            model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype="auto",
                    use_safetensors=True,
                    **load_options,
                ),
            )
            if self.adapter_id is not None:
                try:
                    from peft import PeftModel
                except ImportError:
                    raise ModelError(
                        "Install Gabby's optional 'transformers-adapters' dependencies "
                        "to load a PEFT adapter"
                    ) from None
                adapter_options: dict[str, Any] = {
                    "is_trainable": False,
                    "local_files_only": self.local_files_only,
                    "use_safetensors": True,
                }
                if token:
                    adapter_options["token"] = token
                if self.adapter_revision:
                    adapter_options["revision"] = self.adapter_revision
                if self.cache_dir:
                    adapter_options["cache_dir"] = self.cache_dir
                model = cast(
                    Any,
                    PeftModel.from_pretrained(model, self.adapter_id, **adapter_options),
                )
            model.to(self.device)
            model.eval()
        except ModelError:
            raise
        except ImportError:
            raise ModelError(
                "Install Gabby's optional 'transformers' dependencies to use this provider"
            ) from None
        except Exception:
            raise ModelError("Local Transformers model could not be loaded") from None
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model, torch, StoppingCriteria

    async def aclose(self) -> None:
        """Release model and tokenizer references after the owning agent drains its runs."""
        async with self._call_lock:
            await run_sync_callback(self._clear_model)

    def _clear_model(self) -> None:
        with self._generation_lock:
            self._model = None
            self._tokenizer = None


def _transformers_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gabby's OpenAI-compatible tool-call history to Transformers chat format."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            converted: list[dict[str, Any]] = []
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ModelError(
                        "Tool history is not compatible with local model chat templates"
                    )
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        raise ModelError("Tool history contains invalid JSON arguments") from None
                converted.append({"name": function["name"], "arguments": arguments})
            copied["tool_calls"] = converted
        normalized.append(copied)
    return normalized


def _transformers_response(
    tokenizer: Any,
    content: str,
    tools: list[dict[str, Any]],
    *,
    prefix: Any = None,
    response_template: dict[str, Any] | None = None,
) -> ModelResponse:
    """Parse model-specific response templates into Gabby's normalized response contract."""
    if not tools:
        return ModelResponse(content=content)
    parser = getattr(tokenizer, "parse_response", None)
    if not callable(parser):
        raise ModelError("This tokenizer does not provide structured local tool-call parsing")
    if prefix is None:
        raise ModelError("Local model tool response parsing requires the rendered prompt prefix")
    try:
        parser_options: dict[str, Any] = {"prefix": prefix, "tools": tools}
        if response_template is not None:
            parser_options["schema"] = response_template
            content = _unwrap_transformers_json_fence(content)
        parsed = parser(content, **parser_options)
    except Exception:
        raise ModelError("Local model tool response could not be parsed") from None
    if not isinstance(parsed, dict):
        raise ModelError("Local model returned an invalid structured response")
    parsed_content = parsed.get("content")
    if parsed_content is not None and not isinstance(parsed_content, str):
        raise ModelError("Local model returned invalid response content")
    valid_names = {
        function.get("name")
        for tool in tools
        if isinstance(tool, dict)
        and isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }
    raw_calls = parsed.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ModelError("Local model returned invalid tool calls")
    normalized_calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        function = call.get("function", call) if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if name not in valid_names or not isinstance(arguments, dict):
            raise ModelError("Local model returned an invalid tool call")
        normalized_calls.append(
            {
                "id": f"transformers-{uuid.uuid4().hex}-{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
                },
            }
        )
    return ModelResponse(content=parsed_content or None, tool_calls=normalized_calls)


def _thaw_json_value(value: Any) -> Any:
    """Copy frozen agent configuration containers into JSON-serializable values."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json_value(child) for child in value]
    return value


def _unwrap_transformers_json_fence(content: str) -> str:
    """Remove one complete Markdown JSON fence while preserving trailing control tokens."""
    match = re.fullmatch(
        r"[ \t]*```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```"
        r"(?P<suffix>(?:<\|[^>]+\|>)*[ \t\r\n]*)",
        content,
        flags=re.IGNORECASE,
    )
    if match is None:
        return content
    return match.group("body") + match.group("suffix")
