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
#
# Copyright 2026-present Gabby Contributors.

"""Model-provider decorator that filters complete and streaming payloads."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any, cast

from .config import DEFAULT_MAX_MODEL_REQUEST_BYTES, DEFAULT_MAX_MODEL_RESPONSE_BYTES
from .models import (
    ModelRequestSizeError,
    ModelResponse,
    ModelResponseSizeError,
    ModelStreamDelta,
    _count_utf8_payload,
    ensure_model_request_size,
    ensure_model_response_size,
)
from .payload_filters import DefaultPayloadFilter, PayloadFilter


async def _apply_filters_request(
    filters: Sequence[PayloadFilter],
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    purpose: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for payload_filter in filters:
        result = payload_filter.filter_request(
            messages=messages, tools=tools, model=model, purpose=purpose
        )
        if inspect.isawaitable(result):
            result = await result
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], list)
            or not isinstance(result[1], list)
        ):
            raise TypeError("PayloadFilter.filter_request() must return (messages, tools) lists")
        messages, tools = result
    return messages, tools


async def _apply_filters_response(
    filters: Sequence[PayloadFilter], *, response: ModelResponse, purpose: str
) -> ModelResponse:
    for payload_filter in filters:
        result = payload_filter.filter_response(response=response, purpose=purpose)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ModelResponse):
            raise TypeError("PayloadFilter.filter_response() must return ModelResponse")
        response = result
    return response


class FilteredModelProvider:
    """Transparent provider wrapper that filters every request and response payload.

    The built-in default filter redacts each text delta as it arrives. Custom response filters are
    buffered until the complete response is available so no unapproved text or partial tool
    arguments escape before validation.
    """

    def __init__(
        self,
        provider: Any,
        filters: Sequence[PayloadFilter],
        *,
        purpose: str,
        max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES,
    ) -> None:
        self._provider = provider
        self._filters = tuple(filters)
        self._purpose = purpose
        self._max_request_bytes = max_request_bytes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int | None = None,
    ) -> ModelResponse:
        filtered_messages, filtered_tools = await _apply_filters_request(
            self._filters,
            messages=messages,
            tools=tools,
            model=model,
            purpose=self._purpose,
        )
        self._check_request_size(
            messages=filtered_messages,
            tools=filtered_tools,
            model=model,
            temperature=temperature,
            streaming=False,
        )
        method = self._provider.complete
        kwargs: dict[str, Any] = {
            "messages": filtered_messages,
            "tools": filtered_tools,
            "model": model,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
        }
        if max_response_bytes is not None and _accepts_keyword(method, "max_response_bytes"):
            kwargs["max_response_bytes"] = max_response_bytes
        response = await method(**kwargs)
        if not isinstance(response, ModelResponse):
            raise TypeError("model provider returned an invalid response object")
        filtered_response = await _apply_filters_response(
            self._filters, response=response, purpose=self._purpose
        )
        ensure_model_response_size(
            filtered_response,
            max_bytes=(
                max_response_bytes
                if max_response_bytes is not None
                else DEFAULT_MAX_MODEL_RESPONSE_BYTES
            ),
        )
        return filtered_response

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
        max_response_bytes: int | None = None,
    ) -> AsyncIterator[ModelStreamDelta]:
        if not callable(getattr(self._provider, "stream", None)):
            response = await self.complete(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
            if response.content:
                yield ModelStreamDelta(content_delta=response.content)
            for index, tool_call in enumerate(response.tool_calls):
                function = tool_call.get("function", {})
                yield ModelStreamDelta(
                    tool_call_index=index,
                    tool_call_id=tool_call.get("id"),
                    tool_name_delta=function.get("name"),
                    tool_arguments_delta=function.get("arguments"),
                    provider_metadata=tool_call.get("provider_metadata"),
                )
            yield ModelStreamDelta(finish_reason="stop", usage=response.usage)
            return
        filtered_messages, filtered_tools = await _apply_filters_request(
            self._filters,
            messages=messages,
            tools=tools,
            model=model,
            purpose=self._purpose,
        )
        self._check_request_size(
            messages=filtered_messages,
            tools=filtered_tools,
            model=model,
            temperature=temperature,
            streaming=True,
        )
        method = self._provider.stream
        kwargs: dict[str, Any] = {
            "messages": filtered_messages,
            "tools": filtered_tools,
            "model": model,
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
        }
        if max_response_bytes is not None and _accepts_keyword(method, "max_response_bytes"):
            kwargs["max_response_bytes"] = max_response_bytes
        response_limit = (
            max_response_bytes
            if max_response_bytes is not None
            else DEFAULT_MAX_MODEL_RESPONSE_BYTES
        )
        stream_default_filter = (
            len(self._filters) == 1 and type(self._filters[0]) is DefaultPayloadFilter
        )
        deltas: list[ModelStreamDelta] = []
        default_stream_mode = "probing"
        default_stream_probe = ""
        used_bytes = 0
        emitted_bytes = 0
        async for delta in method(**kwargs):
            if not isinstance(delta, ModelStreamDelta):
                raise TypeError("model provider returned a malformed stream delta")
            try:
                delta_bytes = _stream_delta_bytes(delta, response_limit)
            except (TypeError, ValueError, UnicodeError):
                raise TypeError("model provider returned malformed stream metadata") from None
            used_bytes += delta_bytes
            if used_bytes > response_limit:
                raise ModelResponseSizeError(
                    f"Model response exceeded max_model_response_bytes={response_limit}"
                )
            if stream_default_filter:
                if default_stream_mode == "json":
                    deltas.append(delta)
                    continue
                if default_stream_mode == "probing":
                    deltas.append(delta)
                    if delta.content_delta is None:
                        continue
                    default_stream_probe += delta.content_delta
                    first_content = default_stream_probe.lstrip()
                    if not first_content:
                        continue
                    if first_content[0] in '{["-0123456789':
                        # Structured responses, including scalar numbers and strings, need a
                        # complete parse so JSON types and values survive filtering. Plain text
                        # remains streamed below.
                        default_stream_mode = "json"
                        continue
                    default_stream_mode = "text"
                    buffered, deltas = deltas, []
                    for buffered_delta in buffered:
                        filtered_delta = self._filter_default_stream_delta(buffered_delta)
                        emitted_bytes += _stream_delta_bytes(filtered_delta, response_limit)
                        if emitted_bytes > response_limit:
                            raise ModelResponseSizeError(
                                f"Model response exceeded max_model_response_bytes={response_limit}"
                            )
                        yield filtered_delta
                    continue
                filtered_delta = self._filter_default_stream_delta(delta)
                emitted_bytes += _stream_delta_bytes(filtered_delta, response_limit)
                if emitted_bytes > response_limit:
                    raise ModelResponseSizeError(
                        f"Model response exceeded max_model_response_bytes={response_limit}"
                    )
                yield filtered_delta
            elif self._filters:
                deltas.append(delta)
            else:
                # With no filters there is nothing to approve at end-of-stream. Forward each
                # bounded delta so retries and client cancellation retain their stream semantics.
                yield delta
        if stream_default_filter:
            if default_stream_mode == "probing":
                # Empty or whitespace-only content has no JSON document to parse.
                default_stream_mode = "text"
                buffered, deltas = deltas, []
                for buffered_delta in buffered:
                    filtered_delta = self._filter_default_stream_delta(buffered_delta)
                    emitted_bytes += _stream_delta_bytes(filtered_delta, response_limit)
                    if emitted_bytes > response_limit:
                        raise ModelResponseSizeError(
                            f"Model response exceeded max_model_response_bytes={response_limit}"
                        )
                    yield filtered_delta
                return
            if default_stream_mode == "text":
                return
            response = await _apply_filters_response(
                self._filters,
                response=_response_from_deltas(deltas),
                purpose=self._purpose,
            )
            ensure_model_response_size(response, max_bytes=response_limit)
            if response.content:
                yield ModelStreamDelta(content_delta=response.content)
            for index, tool_call in enumerate(response.tool_calls):
                function = tool_call.get("function", {})
                yield ModelStreamDelta(
                    tool_call_index=index,
                    tool_call_id=tool_call.get("id"),
                    tool_name_delta=function.get("name"),
                    tool_arguments_delta=function.get("arguments"),
                    provider_metadata=tool_call.get("provider_metadata"),
                )
            last = deltas[-1] if deltas else None
            yield ModelStreamDelta(
                finish_reason=(last.finish_reason if last is not None else "stop"),
                usage=(response.usage if response.usage else None),
                provider_metadata=(last.provider_metadata if last is not None else None),
            )
            return
        if not self._filters:
            return
        response = await _apply_filters_response(
            self._filters, response=_response_from_deltas(deltas), purpose=self._purpose
        )
        ensure_model_response_size(response, max_bytes=response_limit)
        if response.content:
            yield ModelStreamDelta(content_delta=response.content)
        for index, tool_call in enumerate(response.tool_calls):
            function = tool_call.get("function", {})
            yield ModelStreamDelta(
                tool_call_index=index,
                tool_call_id=tool_call.get("id"),
                tool_name_delta=function.get("name"),
                tool_arguments_delta=function.get("arguments"),
                provider_metadata=tool_call.get("provider_metadata"),
            )
        last = deltas[-1] if deltas else None
        yield ModelStreamDelta(
            finish_reason=(last.finish_reason if last is not None else "stop"),
            usage=(response.usage if response.usage else None),
            provider_metadata=(last.provider_metadata if last is not None else None),
        )

    async def aclose(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def close(self) -> None:
        close = getattr(self._provider, "close", None)
        if callable(close):
            close()

    def _check_request_size(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        streaming: bool,
    ) -> None:
        try:
            ensure_model_request_size(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_bytes=self._max_request_bytes,
                streaming=streaming,
                supports_tool_choice=bool(getattr(self._provider, "supports_tool_choice", True)),
            )
        except ModelRequestSizeError as exc:
            raise ValueError(
                f"Filtered model request exceeds configured size limit: {exc}"
            ) from exc

    def _filter_default_stream_delta(self, delta: ModelStreamDelta) -> ModelStreamDelta:
        if delta.content_delta is None:
            return delta
        default_filter = cast(DefaultPayloadFilter, self._filters[0])
        filtered_content = default_filter.filter_stream_delta(
            content_delta=delta.content_delta,
            purpose=self._purpose,
        )
        if not isinstance(filtered_content, str):
            raise TypeError("DefaultPayloadFilter.filter_stream_delta() must return str")
        return replace(delta, content_delta=filtered_content)


def _accepts_keyword(method: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == name for parameter in parameters)


def _stream_delta_bytes(delta: ModelStreamDelta, max_bytes: int) -> int:
    """Count response data, excluding repeated normalized-delta envelope overhead."""
    return _count_utf8_payload(
        (
            delta.content_delta,
            delta.tool_call_id,
            delta.tool_name_delta,
            delta.tool_arguments_delta,
            delta.usage,
            delta.provider_metadata,
        ),
        max_bytes=max_bytes,
    )


def _response_from_deltas(deltas: list[ModelStreamDelta]) -> ModelResponse:
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    for delta in deltas:
        if delta.content_delta:
            content.append(delta.content_delta)
        if delta.tool_call_index is not None:
            call = calls.setdefault(
                delta.tool_call_index,
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if delta.tool_call_id:
                call["id"] = delta.tool_call_id
            if delta.tool_name_delta:
                call["function"]["name"] += delta.tool_name_delta
            if delta.tool_arguments_delta:
                call["function"]["arguments"] += delta.tool_arguments_delta
            if delta.provider_metadata:
                call.setdefault("provider_metadata", {}).update(delta.provider_metadata)
        if delta.finish_reason is not None:
            finish_reason = delta.finish_reason
        if delta.usage is not None:
            usage.update(delta.usage)
    return ModelResponse(
        content="".join(content) or None,
        tool_calls=[calls[index] for index in sorted(calls)],
        usage=usage,
        raw={"finish_reason": finish_reason} if finish_reason is not None else {},
    )
