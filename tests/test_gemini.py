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
"""Mocked REST contract tests for the native Google Gemini provider."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gabby import Agent, AgentDefinition, GeminiProvider
from gabby.models import (
    ModelError,
    ModelRequestSizeError,
    ModelResponseSizeError,
    RetryableModelError,
    _gemini_candidate_parts,
)
from gabby.tools import Tool, ToolRegistry

_ASYNC_CLIENT = httpx.AsyncClient


def _provider(
    handler: Any, monkeypatch: pytest.MonkeyPatch, *, max_output_tokens: int = 321
) -> GeminiProvider:
    transport = httpx.MockTransport(handler)

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return GeminiProvider(
        api_key="test-key",
        base_url="https://gemini.test/v1beta",
        max_output_tokens=max_output_tokens,
        extra_headers={"X-Request-Source": "gabby-test"},
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"candidates": "invalid"},
        {"candidates": []},
        {"candidates": [None]},
        {"candidates": [{"content": {"parts": "invalid"}}]},
        {"candidates": [{"content": {"parts": [None]}}]},
        {"candidates": [{"content": {"parts": [{"text": 1}]}}]},
        {"candidates": [{"content": {"parts": [{"functionCall": []}]}}]},
        {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "lookup", "args": []}}]}}
            ]
        },
        {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "lookup", "args": {}, "id": 3}}]}}
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"name": "lookup", "args": {}},
                                "thoughtSignature": 4,
                            }
                        ]
                    }
                }
            ]
        },
        {"candidates": [{"content": {"parts": [{"unknown": True}]}}]},
        {"candidates": [{"content": {"parts": []}, "finishReason": 4}]},
        {
            "candidates": [{"content": {"parts": []}}],
            "usageMetadata": {"promptTokenCount": True},
        },
    ],
)
def test_gemini_candidate_parser_rejects_malformed_provider_shapes(payload: Any) -> None:
    with pytest.raises(ModelError):
        _gemini_candidate_parts(payload)


@pytest.mark.asyncio
async def test_gemini_completion_maps_tools_usage_and_thought_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    original_parts = [
        {"text": "I will inspect the data.", "thoughtSignature": "text-signature"},
        {
            "functionCall": {
                "name": "lookup",
                "args": {"term": "schedule"},
                "id": "google-call-7",
            },
            "thoughtSignature": "call-signature",
        },
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": original_parts},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 13,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 18,
                },
            },
        )

    provider = _provider(respond, monkeypatch)
    try:
        response = await provider.complete(
            messages=[
                {"role": "system", "content": "Follow the task policy."},
                {"role": "user", "content": "Look up the schedule."},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Search a knowledge source.",
                        "parameters": {
                            "type": "object",
                            "properties": {"term": {"type": "string"}},
                            "required": ["term"],
                        },
                    },
                }
            ],
            model="gemini-test-model",
            temperature=0.2,
        )
    finally:
        await provider.aclose()

    assert requests[0].url.path == "/v1beta/models/gemini-test-model:generateContent"
    assert requests[0].headers["x-goog-api-key"] == "test-key"
    assert requests[0].headers["x-request-source"] == "gabby-test"
    payload = json.loads(requests[0].content)
    assert payload["systemInstruction"] == {"parts": [{"text": "Follow the task policy."}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "Look up the schedule."}]}]
    assert payload["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "lookup",
                    "description": "Search a knowledge source.",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {"term": {"type": "string"}},
                        "required": ["term"],
                    },
                }
            ]
        }
    ]
    assert payload["generationConfig"] == {"maxOutputTokens": 321, "temperature": 0.2}
    assert response.content == "I will inspect the data."
    call = response.tool_calls[0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"term": "schedule"}
    assert call["provider_metadata"] == {
        "gemini_function_id": "google-call-7",
        "gemini_thought_signature": "call-signature",
        "gemini_response_parts": original_parts,
    }
    assert response.usage == {
        "prompt_tokens": 13,
        "completion_tokens": 5,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_gemini_follow_up_returns_original_parts_and_matching_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    model_parts = [
        {"text": "Checking now.", "thoughtSignature": "text-signature"},
        {
            "functionCall": {"name": "lookup", "args": {"term": "schedule"}, "id": "fc-1"},
            "thoughtSignature": "call-signature",
        },
    ]
    provider_call_id = "fc-1"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": model_parts}}]},
            )
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "The schedule is ready."}]}}]},
        )

    provider = _provider(respond, monkeypatch)
    try:
        first = await provider.complete(
            messages=[{"role": "user", "content": "Find schedule."}],
            tools=[],
            model="gemini-test-model",
        )
        call = first.tool_calls[0]
        await provider.complete(
            messages=[
                {"role": "user", "content": "Find schedule."},
                {"role": "assistant", "content": first.content, "tool_calls": first.tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": '{"matches": ["meeting"]}',
                },
            ],
            tools=[],
            model="gemini-test-model",
        )
    finally:
        await provider.aclose()

    follow_up = json.loads(requests[1].content)
    assert follow_up["contents"][1] == {"role": "model", "parts": model_parts}
    assert follow_up["contents"][2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "lookup",
                    "response": {"matches": ["meeting"]},
                    "id": provider_call_id,
                }
            }
        ],
    }


@pytest.mark.asyncio
async def test_gemini_stream_maps_text_calls_usage_and_preserved_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    events = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Thinking", "thought": True, "thoughtSignature": "sig-text"}
                        ]
                    }
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Done."},
                            {
                                "functionCall": {
                                    "name": "lookup",
                                    "args": {"term": "x"},
                                    "id": "stream-call-1",
                                },
                                "thoughtSignature": "sig-call",
                            },
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = _provider(respond, monkeypatch)
    try:
        deltas = [
            delta
            async for delta in provider.stream(
                messages=[{"role": "user", "content": "Find x."}],
                tools=[],
                model="gemini-test-model",
            )
        ]
    finally:
        await provider.aclose()

    assert requests[0].url.path == "/v1beta/models/gemini-test-model:streamGenerateContent"
    assert requests[0].url.query == b"alt=sse"
    assert requests[0].headers["accept"] == "text/event-stream"
    assert [delta.content_delta for delta in deltas if delta.content_delta] == ["Done."]
    tool_delta = next(delta for delta in deltas if delta.tool_call_id)
    assert tool_delta.tool_name_delta == "lookup"
    assert json.loads(tool_delta.tool_arguments_delta or "") == {"term": "x"}
    assert tool_delta.provider_metadata == {
        "gemini_function_id": "stream-call-1",
        "gemini_thought_signature": "sig-call",
        "gemini_response_parts": [
            {"text": "Thinking", "thought": True, "thoughtSignature": "sig-text"},
            {"text": "Done."},
            {
                "functionCall": {
                    "name": "lookup",
                    "args": {"term": "x"},
                    "id": "stream-call-1",
                },
                "thoughtSignature": "sig-call",
            },
        ],
    }
    assert deltas[-1].usage == {"prompt_tokens": 4, "completion_tokens": 2}


@pytest.mark.asyncio
async def test_gemini_agent_stream_round_trips_tool_call_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    original_parts = [
        {
            "functionCall": {
                "name": "lookup",
                "args": {"term": "Gemini"},
                "id": "runtime-call-1",
            },
            "thoughtSignature": "runtime-signature",
        }
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            event = {"candidates": [{"content": {"parts": original_parts}}]}
        else:
            event = {"candidates": [{"content": {"parts": [{"text": "Gemini is available."}]}}]}
        body = f"data: {json.dumps(event)}\n\n".encode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider = _provider(respond, monkeypatch)
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Look up a product.",
            parameters={
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            handler=lambda term: {"found": term},
        )
    )
    agent = Agent(
        AgentDefinition(
            name="gemini-agent",
            model={"provider": "gemini", "model": "gemini-test-model"},
            tools=["lookup"],
            policies={
                "allowed_tools": ["lookup"],
                "max_steps": 2,
                "timeout_seconds": 5,
            },
        ),
        model=provider,
        tools=registry,
    )
    try:
        events = [event async for event in agent.astream("Find a product.")]
    finally:
        await agent.aclose()

    assert events[-1].type == "completed"
    assert events[-1].data["result"]["output"] == "Gemini is available."
    assert "runtime-signature" not in json.dumps(
        [{"type": event.type, "data": event.data} for event in events]
    )
    follow_up = json.loads(requests[1].content)
    model_turn = next(content for content in follow_up["contents"] if content["role"] == "model")
    assert model_turn["parts"] == original_parts
    function_response = next(
        part["functionResponse"]
        for content in follow_up["contents"]
        for part in content["parts"]
        if "functionResponse" in part
    )
    assert function_response == {
        "name": "lookup",
        "response": {"found": "Gemini"},
        "id": "runtime-call-1",
    }


@pytest.mark.asyncio
async def test_gemini_sanitizes_http_errors_bounds_responses_and_retries_transient_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def oversized(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"candidates": [{"content": {"parts": [{"text": "' + b"x" * 128 + b'"}]}}]}',
        )

    provider = _provider(oversized, monkeypatch)
    try:
        with pytest.raises(ModelResponseSizeError):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                model="gemini-test-model",
                max_response_bytes=64,
            )
    finally:
        await provider.aclose()

    def unavailable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="private quota details", headers={"retry-after": "2"})

    provider = _provider(unavailable, monkeypatch)
    try:
        with pytest.raises(RetryableModelError, match="HTTP 429") as failure:
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                model="gemini-test-model",
            )
        assert failure.value.retry_after_seconds == 2
        assert "private quota" not in str(failure.value)
    finally:
        await provider.aclose()


def test_gemini_provider_configuration_and_remote_transport_validation() -> None:
    provider = GeminiProvider.from_config(
        {"api_key_env": "CUSTOM_GEMINI_KEY", "max_output_tokens": 4000}
    )
    assert provider.api_key_env == "CUSTOM_GEMINI_KEY"
    assert provider.max_output_tokens == 4000
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"

    definition = AgentDefinition(
        name="gemini-agent",
        model={"provider": "gemini", "model": "gemini-test", "max_output_tokens": 99},
    )
    agent = Agent(definition)
    assert isinstance(agent.model, GeminiProvider)
    assert agent.model.max_output_tokens == 99

    with pytest.raises(ValueError, match="HTTPS"):
        GeminiProvider(base_url="http://models.example.test/v1beta")
    with pytest.raises(ValueError, match="max_output_tokens"):
        GeminiProvider(max_output_tokens=0)


@pytest.mark.asyncio
async def test_gemini_rejects_oversized_serialized_requests_before_network_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    provider = _provider(respond, monkeypatch)
    try:
        with pytest.raises(ModelRequestSizeError):
            await provider.complete(
                messages=[{"role": "user", "content": "x" * (4 * 1024 * 1024)}],
                tools=[],
                model="gemini-test-model",
            )
    finally:
        await provider.aclose()
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "message"),
    [
        ("data: not-json\n\n", "invalid JSON"),
        ("data: []\n\n", "invalid event"),
        ('data: {"candidates":[null]}\n\n', "invalid candidate"),
        (
            'data: {"candidates":[{"content":{"parts":["bad"]}}]}\n\n',
            "invalid content parts",
        ),
    ],
)
async def test_gemini_stream_rejects_invalid_sse_event_shapes(
    event: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(200, text=f"event: message\n{event}"),
        monkeypatch,
    )
    try:
        with pytest.raises(ModelError, match=message):
            async for _ in provider.stream(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model="gemini-test-model",
            ):
                pass
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_stream_ignores_comments_and_emits_usage_only_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(
        lambda _: httpx.Response(
            200,
            text=': heartbeat\nevent: message\ndata: {"usageMetadata":{"promptTokenCount":3}}\n\n',
        ),
        monkeypatch,
    )
    try:
        deltas = [
            delta
            async for delta in provider.stream(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model="gemini-test-model",
            )
        ]
    finally:
        await provider.aclose()

    assert len(deltas) == 1
    assert deltas[0].usage == {"prompt_tokens": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected", "message"),
    [
        (httpx.Response(400, text="private provider details"), ModelError, "HTTP 400"),
        (
            httpx.Response(200, json={"candidates": [{"content": {"parts": "bad"}}]}),
            ModelError,
            "invalid content parts",
        ),
    ],
)
async def test_gemini_completion_sanitizes_provider_and_shape_errors(
    response: httpx.Response,
    expected: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(lambda _: response, monkeypatch)
    try:
        with pytest.raises(expected, match=message) as failure:
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model="gemini-test-model",
            )
        assert "private provider details" not in str(failure.value)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_completion_classifies_transport_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private socket details", request=request)

    provider = _provider(fail, monkeypatch)
    try:
        with pytest.raises(RetryableModelError, match="request failed") as failure:
            await provider.complete(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model="gemini-test-model",
            )
        assert "private socket details" not in str(failure.value)
    finally:
        await provider.aclose()


def test_gemini_request_rejects_invalid_model_identifiers() -> None:
    provider = GeminiProvider()
    for model in ("", "x" * 513):
        with pytest.raises(ModelError, match="identifier is invalid"):
            provider._request(
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                model=model,
                temperature=None,
                streaming=False,
            )
