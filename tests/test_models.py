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
"""Contract checks for the OpenAI-compatible asynchronous provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gabby.models import (
    HuggingFaceInferenceProvider,
    ModelError,
    ModelResponse,
    ModelResponseSizeError,
    ModelStreamDelta,
    OllamaProvider,
    OpenAICompatibleProvider,
    RetryableModelError,
    _anthropic_messages,
    _anthropic_tools,
    _extract_sse_lines,
    _iter_response_lines_limited,
    _method_accepts_keyword,
    _validate_response_limit,
    complete_with_response_limit,
    ensure_model_request_size,
    ensure_model_response_size,
    stream_with_response_limit,
)

_ASYNC_CLIENT = httpx.AsyncClient


def _provider(handler: Any, monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://model.test/v1",
        extra_headers={"X-Request-Source": "gabby-test"},
    )
    return provider


@pytest.mark.asyncio
async def test_openai_compatible_provider_sends_tools_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    provider = _provider(respond, monkeypatch)
    try:
        result = await provider.complete(
            messages=[{"role": "user", "content": "look this up"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            model="fake-model",
            temperature=0,
            timeout_seconds=4,
        )
    finally:
        await provider.aclose()

    assert requests[0].url == "https://model.test/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert requests[0].headers["x-request-source"] == "gabby-test"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "fake-model"
    assert payload["messages"] == [{"role": "user", "content": "look this up"}]
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["temperature"] == 0
    assert result.content is None
    assert result.tool_calls[0]["function"]["name"] == "lookup"
    assert result.usage["prompt_tokens"] == 11


@pytest.mark.asyncio
async def test_openai_compatible_provider_reports_http_and_response_shape_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(lambda _: httpx.Response(503, text="upstream unavailable"), monkeypatch)
    try:
        with pytest.raises(RetryableModelError, match="HTTP 503"):
            await provider.complete(messages=[], tools=[], model="fake")
    finally:
        await provider.aclose()

    payloads: tuple[dict[str, Any], ...] = (
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": 4}}]},
        {"choices": [{"message": {"tool_calls": {"bad": "shape"}}}]},
        {"choices": [{"message": {"content": "ok"}}], "usage": []},
        {},
    )
    for payload in payloads:
        provider = _provider(lambda _, value=payload: httpx.Response(200, json=value), monkeypatch)
        try:
            with pytest.raises(ModelError, match="valid choices"):
                await provider.complete(messages=[], tools=[], model="fake")
        finally:
            await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "retry_after", "expected_delay"),
    [
        (429, "2", 2.0),
        (503, "not-a-number", None),
        (503, "nan", None),
        (503, "-1", None),
        (503, "100", 5.0),
        (400, "1", None),
    ],
)
async def test_http_provider_classifies_transient_status_and_bounds_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    retry_after: str,
    expected_delay: float | None,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(status, headers={"Retry-After": retry_after}, text="private"),
        monkeypatch,
    )
    try:
        expected_error = RetryableModelError if status == 429 or status == 503 else ModelError
        with pytest.raises(expected_error) as error:
            await provider.complete(messages=[], tools=[], model="fake")
        if isinstance(error.value, RetryableModelError):
            assert error.value.retry_after_seconds == expected_delay
        assert "private" not in str(error.value)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_http_errors_do_not_expose_response_body(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    secret = "private-prompt-and-token"
    provider = _provider(lambda _: httpx.Response(503, text=secret), monkeypatch)
    try:
        with pytest.raises(RetryableModelError, match="HTTP 503") as error:
            if streaming:
                async for _ in provider.stream(messages=[], tools=[], model="fake"):
                    pass
            else:
                await provider.complete(messages=[], tools=[], model="fake")
        assert secret not in str(error.value)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_transport_errors_use_sanitized_messages(
    monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    secret = "https://model.test/private-query-token"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(secret, request=request)

    provider = _provider(fail, monkeypatch)
    try:
        with pytest.raises(ModelError) as error:
            if streaming:
                async for _ in provider.stream(messages=[], tools=[], model="fake"):
                    pass
            else:
                await provider.complete(messages=[], tools=[], model="fake")
        assert secret not in str(error.value)
        assert error.value.__suppress_context__
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_provider_reads_api_key_from_configured_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "environment-key")
    request_headers: list[httpx.Headers] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_headers.append(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        api_key_env="MODEL_API_KEY", base_url="https://model.test/v1"
    )
    transport = httpx.MockTransport(respond)

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    try:
        result = await provider.complete(messages=[], tools=[], model="fake")
    finally:
        await provider.aclose()

    assert request_headers[0]["authorization"] == "Bearer environment-key"
    assert result.content == "ok"


def test_provider_config_mapping_and_credential_boundary() -> None:
    from gabby.config import ConfigError

    provider = OpenAICompatibleProvider.from_config(
        {
            "base_url": "https://models.example/v1/",
            "api_key_env": "TEAM_MODEL_KEY",
            "headers": {"X-Project": "gabby"},
        }
    )
    assert provider.base_url == "https://models.example/v1"
    assert provider.api_key_env == "TEAM_MODEL_KEY"
    assert provider.extra_headers == {"X-Project": "gabby"}
    with pytest.raises(ConfigError, match="cannot be stored"):
        OpenAICompatibleProvider.from_config({"api_key": "inline-secret"})


@pytest.mark.asyncio
async def test_openai_compatible_provider_streams_text_tool_calls_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    events = [
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "look", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "up", "arguments": "}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 7},
        },
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    provider = _provider(respond, monkeypatch)
    try:
        deltas = [
            delta
            async for delta in provider.stream(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function"}],
                model="fake-model",
            )
        ]
    finally:
        await provider.aclose()

    assert json.loads(requests[0].content)["stream"] is True
    assert "text/event-stream" in requests[0].headers["accept"]
    assert [delta.content_delta for delta in deltas if delta.content_delta] == ["Hel", "lo"]
    tool_deltas = [delta for delta in deltas if delta.tool_call_index is not None]
    assert len(tool_deltas) == 2
    assert tool_deltas[0] == ModelStreamDelta(
        tool_call_index=0,
        tool_call_id="call-1",
        tool_name_delta="look",
        tool_arguments_delta="{",
    )
    assert deltas[-1].usage == {"total_tokens": 7}


@pytest.mark.asyncio
async def test_openai_compatible_provider_emits_usage_only_stream_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = {"choices": [], "usage": {"total_tokens": 3}}
    provider = _provider(
        lambda _: httpx.Response(
            200,
            text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        ),
        monkeypatch,
    )
    try:
        deltas = [delta async for delta in provider.stream(messages=[], tools=[], model="small")]
    finally:
        await provider.aclose()

    assert deltas == [ModelStreamDelta(usage={"total_tokens": 3})]


@pytest.mark.asyncio
async def test_provider_recreates_client_after_close_and_sanitizes_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            httpx.Response(200, json={"choices": [{"message": {"content": "first"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": "second"}}]}),
        )
    )
    provider = _provider(lambda _: next(responses), monkeypatch)
    first = await provider.complete(messages=[], tools=[], model="small")
    await provider.aclose()
    second = await provider.complete(messages=[], tools=[], model="small")
    await provider.aclose()
    assert (first.content, second.content) == ("first", "second")

    invalid = _provider(lambda _: httpx.Response(200, content=b"not-json"), monkeypatch)
    try:
        with pytest.raises(ModelError, match="Model request failed"):
            await invalid.complete(messages=[], tools=[], model="small")
    finally:
        await invalid.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (429, "busy", "HTTP 429"),
        (200, "data: not-json\n\n", "invalid JSON"),
        (200, "data: []\n\n", "invalid event"),
        (200, 'data: {"error":"upstream"}\n\n', "error event"),
        (200, 'data: {"usage":[]}\n\n', "invalid token usage"),
        (200, 'data: {"choices":{}}\n\n', "invalid choices"),
        (200, 'data: {"choices":[null]}\n\n', "invalid choice"),
        (200, 'data: {"choices":[{"delta":{"content":2}}]}\n\n', "non-text content"),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":[{"index":true}]}}]}\n\n',
            "tool call index",
        ),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":1}]}}]}\n\n',
            "tool call ID",
        ),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":[]}]}}]}\n\n',
            "function delta",
        ),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":1}}]}}]}\n\n',
            "tool name",
        ),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":1}}]}}]}\n\n',
            "tool arguments",
        ),
        (
            200,
            'data: {"choices":[{"delta":{},"finish_reason":1}]}\n\n',
            "finish reason",
        ),
        (
            200,
            'data: {"choices":[{"delta":{"tool_calls":{}}}]}\n\n',
            "invalid tool calls",
        ),
    ],
)
async def test_openai_compatible_provider_reports_stream_errors(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: str,
    message: str,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(
            status,
            text=body,
            headers={"content-type": "text/event-stream"},
        ),
        monkeypatch,
    )
    try:
        with pytest.raises(ModelError, match=message):
            async for _ in provider.stream(
                messages=[], tools=[], model="fake-model", timeout_seconds=1
            ):
                pass
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_ollama_provider_uses_compatible_completion_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"local"}}]}\n\ndata: [DONE]\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "local"}}]},
        )

    transport = httpx.MockTransport(respond)
    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_factory(transport=transport, **kwargs),
    )
    provider = OllamaProvider.from_config({"model": "qwen3:8b"})
    try:
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        completion = await provider.complete(messages=[], tools=tools, model="qwen3:8b")
        deltas = [
            delta async for delta in provider.stream(messages=[], tools=tools, model="qwen3:8b")
        ]
    finally:
        await provider.aclose()

    assert provider.name == "ollama"
    assert completion.content == "local"
    assert [delta.content_delta for delta in deltas if delta.content_delta] == ["local"]
    assert {str(request.url) for request in requests} == {
        "http://localhost:11434/v1/chat/completions"
    }
    assert all("authorization" not in request.headers for request in requests)
    assert all("tools" in json.loads(request.content) for request in requests)
    assert all("tool_choice" not in json.loads(request.content) for request in requests)


@pytest.mark.asyncio
async def test_huggingface_inference_provider_uses_router_token_tools_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            event = {"choices": [{"delta": {"content": "streamed"}, "finish_reason": None}]}
            return httpx.Response(
                200,
                text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hosted"}}], "usage": {"total_tokens": 4}},
        )

    transport = httpx.MockTransport(respond)

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    provider = HuggingFaceInferenceProvider.from_config({})
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    try:
        completion = await provider.complete(
            messages=[{"role": "user", "content": "query"}], tools=tools, model="org/model"
        )
        deltas = [
            delta async for delta in provider.stream(messages=[], tools=tools, model="org/model")
        ]
    finally:
        await provider.aclose()

    assert provider.name == "huggingface"
    assert [str(request.url) for request in requests] == [
        "https://router.huggingface.co/v1/chat/completions",
        "https://router.huggingface.co/v1/chat/completions",
    ]
    assert all(request.headers["authorization"] == "Bearer hf-test-token" for request in requests)
    assert all(json.loads(request.content)["tools"] == tools for request in requests)
    assert completion.content == "hosted"
    assert completion.usage["total_tokens"] == 4
    assert [delta.content_delta for delta in deltas if delta.content_delta] == ["streamed"]


@pytest.mark.asyncio
async def test_response_limit_wrapper_supports_legacy_providers_and_rejects_bad_results() -> None:
    class LegacyProvider:
        name = "legacy"

        async def complete(
            self,
            *,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            model: str,
            temperature: float | None = None,
            timeout_seconds: float = 120,
        ) -> ModelResponse:
            return ModelResponse(content="ok")

    result = await complete_with_response_limit(
        LegacyProvider(),
        messages=[],
        tools=[],
        model="small",
        temperature=None,
        timeout_seconds=1,
        max_response_bytes=10,
    )
    assert result.content == "ok"

    class OversizedProvider(LegacyProvider):
        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="ééé")

    with pytest.raises(ModelResponseSizeError):
        await complete_with_response_limit(
            OversizedProvider(),
            messages=[],
            tools=[],
            model="small",
            temperature=None,
            timeout_seconds=1,
            max_response_bytes=5,
        )

    class InvalidProvider(LegacyProvider):
        async def complete(self, **_: Any) -> Any:
            return {"content": "not normalized"}

    with pytest.raises(ModelError, match="invalid response object"):
        await complete_with_response_limit(
            InvalidProvider(),
            messages=[],
            tools=[],
            model="small",
            temperature=None,
            timeout_seconds=1,
            max_response_bytes=10,
        )


@pytest.mark.asyncio
async def test_stream_response_limit_wrapper_checks_provider_deltas() -> None:
    class LegacyStreamProvider:
        name = "legacy-stream"

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="é")
            yield ModelStreamDelta(usage={"tokens": 1})

    events = [
        delta
        async for delta in stream_with_response_limit(
            LegacyStreamProvider(),
            messages=[],
            tools=[],
            model="small",
            temperature=None,
            timeout_seconds=1,
            max_response_bytes=100,
        )
    ]
    assert len(events) == 2

    with pytest.raises(ModelResponseSizeError):
        async for _ in stream_with_response_limit(
            LegacyStreamProvider(),
            messages=[],
            tools=[],
            model="small",
            temperature=None,
            timeout_seconds=1,
            max_response_bytes=7,
        ):
            pass

    class MalformedStreamProvider:
        name = "malformed"

        async def stream(self, **_: Any) -> Any:
            yield {"content_delta": "bad"}

    with pytest.raises(ModelError, match="malformed stream delta"):
        async for _ in stream_with_response_limit(
            MalformedStreamProvider(),
            messages=[],
            tools=[],
            model="small",
            temperature=None,
            timeout_seconds=1,
            max_response_bytes=10,
        ):
            pass


@pytest.mark.parametrize(
    "delta",
    [
        ModelStreamDelta(content_delta=1),  # type: ignore[arg-type]
        ModelStreamDelta(tool_call_index=True),
        ModelStreamDelta(tool_call_index=-1),
        ModelStreamDelta(tool_call_id=1),  # type: ignore[arg-type]
        ModelStreamDelta(finish_reason=1),  # type: ignore[arg-type]
        ModelStreamDelta(usage=[]),  # type: ignore[arg-type]
        ModelStreamDelta(provider_metadata=[]),  # type: ignore[arg-type]
    ],
)
@pytest.mark.asyncio
async def test_stream_response_limit_wrapper_rejects_malformed_delta_fields(
    delta: ModelStreamDelta,
) -> None:
    class MalformedProvider:
        name = "malformed"

        async def stream(self, **_kwargs: Any) -> Any:
            yield delta

    with pytest.raises(ModelError):
        _ = [
            item
            async for item in stream_with_response_limit(
                MalformedProvider(),
                messages=[],
                tools=[],
                model="small",
                temperature=None,
                timeout_seconds=1,
                max_response_bytes=128,
            )
        ]


def test_model_request_and_response_bounds_reject_invalid_or_recursive_payloads() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ensure_model_request_size(
            messages=[], tools=[], model="small", temperature=None, max_bytes=True
        )
    with pytest.raises(ValueError, match="not valid JSON"):
        ensure_model_request_size(
            messages=[{"content": float("nan")}],
            tools=[],
            model="small",
            temperature=None,
            max_bytes=100,
        )
    with pytest.raises(ModelResponseSizeError, match="recursive data"):
        recursive: list[Any] = []
        recursive.append(recursive)
        ensure_model_response_size(ModelResponse(tool_calls=recursive), max_bytes=100)
    with pytest.raises(ModelResponseSizeError, match="invalid Unicode"):
        ensure_model_response_size(ModelResponse(content="\ud800"), max_bytes=100)


@pytest.mark.asyncio
async def test_provider_rejects_oversized_completion_body_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(
            200,
            content=b'{"choices":[{"message":{"content":"this is too large"}}]}',
        ),
        monkeypatch,
    )
    try:
        with pytest.raises(ModelResponseSizeError, match="max_model_response_bytes=8"):
            await provider.complete(messages=[], tools=[], model="small", max_response_bytes=8)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
async def test_provider_stream_accepts_sse_line_endings_and_unterminated_final_line(
    monkeypatch: pytest.MonkeyPatch, separator: str
) -> None:
    event = json.dumps({"choices": [{"delta": {"content": "ok"}}]})
    body = f"data: {event}{separator}{separator}data: [DONE]"
    provider = _provider(
        lambda _: httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        ),
        monkeypatch,
    )
    try:
        deltas = [delta async for delta in provider.stream(messages=[], tools=[], model="small")]
    finally:
        await provider.aclose()

    assert [delta.content_delta for delta in deltas] == ["ok"]


@pytest.mark.asyncio
async def test_provider_stream_rejects_invalid_utf8_and_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(
            200,
            content=b"data: \xff\n",
            headers={"content-type": "text/event-stream"},
        ),
        monkeypatch,
    )
    try:
        with pytest.raises(ModelError, match="invalid UTF-8"):
            async for _ in provider.stream(messages=[], tools=[], model="small"):
                pass
    finally:
        await provider.aclose()

    provider = _provider(
        lambda _: httpx.Response(
            200,
            content=b'data: {"choices":[]}\n\n',
            headers={"content-type": "text/event-stream"},
        ),
        monkeypatch,
    )
    try:
        with pytest.raises(ModelResponseSizeError, match="max_model_response_bytes=8"):
            async for _ in provider.stream(
                messages=[], tools=[], model="small", max_response_bytes=8
            ):
                pass
    finally:
        await provider.aclose()


def test_model_response_bound_rejects_invalid_limit_and_deep_structure() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ensure_model_response_size(ModelResponse(content="ok"), max_bytes=False)

    nested: Any = "leaf"
    for _ in range(130):
        nested = [nested]
    with pytest.raises(ModelResponseSizeError, match="structural size limit"):
        ensure_model_response_size(ModelResponse(tool_calls=[nested]), max_bytes=10_000)

    ensure_model_response_size(ModelResponse(content="€😀"), max_bytes=7)
    with pytest.raises(ModelResponseSizeError, match="max_model_response_bytes=6"):
        ensure_model_response_size(ModelResponse(content="€😀"), max_bytes=6)
    assert not _method_accepts_keyword([].append, "max_response_bytes")


@pytest.mark.parametrize("retry_after", [True, -1, float("inf"), "1"])
def test_retryable_model_error_rejects_invalid_retry_after(retry_after: Any) -> None:
    with pytest.raises(ValueError, match="retry_after_seconds"):
        RetryableModelError("transient", retry_after_seconds=retry_after)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_model_response_limit_validator_rejects_invalid_values(limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _validate_response_limit(limit)


def test_sse_line_parser_handles_split_crlf_and_final_unterminated_line() -> None:
    buffer = bytearray(b"first\r")
    assert list(_extract_sse_lines(buffer)) == []
    buffer.extend(b"\nsecond\rthird")
    assert list(_extract_sse_lines(buffer, final=True)) == [b"first", b"second", b"third"]
    assert buffer == bytearray()


@pytest.mark.parametrize(
    "messages",
    [
        [None],
        [{"role": 1}],
        [{"role": "system", "content": 1}],
        [{"role": "tool", "content": "result"}],
        [{"role": "tool", "tool_call_id": "call-1", "content": None}],
        [{"role": "developer", "content": "unsupported"}],
        [{"role": "user", "content": True}],
        [{"role": "user", "tool_calls": "not a list"}],
        [{"role": "assistant", "tool_calls": [None]}],
        [{"role": "assistant", "tool_calls": [{"id": "call-1", "function": {}}]}],
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "lookup", "arguments": "not json"},
                    }
                ],
            }
        ],
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup", "arguments": "[]"}}],
            }
        ],
    ],
)
def test_anthropic_message_adapter_rejects_malformed_history(messages: Any) -> None:
    with pytest.raises(ModelError):
        _anthropic_messages(messages)


@pytest.mark.parametrize(
    "tools",
    [
        [None],
        [{"function": "invalid"}],
        [{"function": {"name": "lookup"}}],
    ],
)
def test_anthropic_tool_adapter_rejects_invalid_schemas(tools: Any) -> None:
    with pytest.raises(ModelError, match="invalid tool schema"):
        _anthropic_tools(tools)


@pytest.mark.asyncio
async def test_limited_sse_response_lines_validate_utf8_and_cumulative_bytes() -> None:
    response = httpx.Response(200, content=b"data: ok\r\n\r\n")
    assert [line async for line in _iter_response_lines_limited(response, max_bytes=32)] == [
        "data: ok",
        "",
    ]

    invalid = httpx.Response(200, content=b"data: \xff\n")
    with pytest.raises(ModelError, match="invalid UTF-8"):
        _ = [line async for line in _iter_response_lines_limited(invalid, max_bytes=32)]

    oversized = httpx.Response(200, content=b"data: 123456789\n")
    with pytest.raises(ModelResponseSizeError, match="max_model_response_bytes=8"):
        _ = [line async for line in _iter_response_lines_limited(oversized, max_bytes=8)]


@pytest.mark.asyncio
async def test_response_limit_wrapper_forwards_native_limit_keywords() -> None:
    class NativeLimitProvider:
        name = "native-limit"
        received_limit: int | None = None

        async def complete(
            self,
            *,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            model: str,
            temperature: float | None = None,
            timeout_seconds: float = 120,
            max_response_bytes: int = 1024,
        ) -> ModelResponse:
            del messages, tools, model, temperature, timeout_seconds
            self.received_limit = max_response_bytes
            return ModelResponse(content="ok")

    provider = NativeLimitProvider()
    await complete_with_response_limit(
        provider,
        messages=[],
        tools=[],
        model="small",
        temperature=None,
        timeout_seconds=1,
        max_response_bytes=17,
    )
    assert provider.received_limit == 17

    class NativeLimitStreamProvider:
        name = "native-limit-stream"
        received_limit: int | None = None

        async def stream(
            self,
            *,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            model: str,
            temperature: float | None = None,
            timeout_seconds: float = 120,
            max_response_bytes: int = 1024,
        ) -> Any:
            del messages, tools, model, temperature, timeout_seconds
            self.received_limit = max_response_bytes
            yield ModelStreamDelta(content_delta="ok")

    stream_provider = NativeLimitStreamProvider()
    async for _ in stream_with_response_limit(
        stream_provider,
        messages=[],
        tools=[],
        model="small",
        temperature=None,
        timeout_seconds=1,
        max_response_bytes=19,
    ):
        pass
    assert stream_provider.received_limit == 19
