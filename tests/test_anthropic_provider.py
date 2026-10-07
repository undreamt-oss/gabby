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
"""Contract checks for the native Anthropic Messages adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from gabby.agent import Agent
from gabby.config import AgentDefinition, ConfigError, validate_agent_definition
from gabby.models import AnthropicProvider, ModelError, ModelResponseSizeError
from gabby.tools import Tool, ToolRegistry

_ASYNC_CLIENT = httpx.AsyncClient


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(respond)

    def make_client(**kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return requests


@pytest.mark.asyncio
async def test_completion_maps_tool_messages_and_anthropic_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _install_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Found it. "},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
        ),
    )
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        result = await provider.complete(
            model="claude-test",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Find x."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "old-call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "old-call", "content": "result"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find a value",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        )
    finally:
        await provider.aclose()

    request = requests[0]
    payload = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert payload["system"] == "Be concise."
    assert payload["max_tokens"] == 1024
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "old-call", "content": "result"}],
    }
    assert result.content == "Found it. "
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 4}
    assert result.tool_calls[0]["function"]["arguments"] == '{"q":"x"}'


@pytest.mark.asyncio
async def test_stream_maps_text_tool_arguments_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[dict[str, Any]] = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 6}}},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "lookup", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Checking"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '"x"}'},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n" for frame in frames)
    requests = _install_transport(
        monkeypatch,
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body),
    )
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        deltas = [
            delta
            async for delta in provider.stream(
                model="claude-test", messages=[{"role": "user", "content": "check"}], tools=[]
            )
        ]
    finally:
        await provider.aclose()

    assert json.loads(requests[0].content)["stream"] is True
    assert "".join(delta.content_delta or "" for delta in deltas) == "Checking"
    tool_parts = [delta for delta in deltas if delta.tool_call_index is not None]
    assert tool_parts[0].tool_call_id == "toolu_2"
    assert tool_parts[0].tool_name_delta == "lookup"
    assert "".join(delta.tool_arguments_delta or "" for delta in tool_parts) == '{"q":"x"}'
    assert deltas[-1].finish_reason == "tool_use"
    assert deltas[-1].usage == {"completion_tokens": 5}


def test_config_and_transport_constraints() -> None:
    provider = AnthropicProvider.from_config({"model": "claude-test", "max_tokens": 2048})
    assert provider.max_tokens == 2048
    assert provider.api_key_env == "ANTHROPIC_API_KEY"
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider.from_config({"max_tokens": True})
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider(max_tokens=200_001)
    with pytest.raises(ConfigError, match="HTTPS"):
        AnthropicProvider(base_url="http://example.test/v1")
    for invalid in (True, 0, 200_001):
        definition = AgentDefinition(
            name="bad-anthropic-agent",
            model={"provider": "anthropic", "model": "claude-test", "max_tokens": invalid},
        )
        with pytest.raises(ConfigError, match="model.max_tokens"):
            validate_agent_definition(definition)


@pytest.mark.asyncio
async def test_completion_sanitizes_provider_errors_and_bounds_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "prompt-and-key-should-not-leak"
    _install_transport(
        monkeypatch,
        lambda _: httpx.Response(401, text=secret),
    )
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        with pytest.raises(ModelError) as raised:
            await provider.complete(model="claude-test", messages=[], tools=[])
        assert secret not in str(raised.value)
    finally:
        await provider.aclose()

    _install_transport(
        monkeypatch,
        lambda _: httpx.Response(200, json={"content": [{"type": "text", "text": "too long"}]}),
    )
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        with pytest.raises(ModelResponseSizeError):
            await provider.complete(
                model="claude-test", messages=[], tools=[], max_response_bytes=5
            )
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"content": "not a list"},
        {"content": [None]},
        {"content": [{"type": "image", "source": {}}]},
        {"content": [{"type": "tool_use", "id": 1, "name": "lookup", "input": {}}]},
        {"content": [{"type": "tool_use", "id": "id", "name": "lookup", "input": []}]},
        {"content": [], "usage": []},
    ],
)
async def test_completion_rejects_malformed_anthropic_response_shapes(
    monkeypatch: pytest.MonkeyPatch, body: Any
) -> None:
    _install_transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        with pytest.raises(ModelError, match="valid message content"):
            await provider.complete(model="claude-test", messages=[], tools=[])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "prefix_frames", "message"),
    [
        ("not-json", (), "invalid JSON"),
        ("[]", (), "invalid event"),
        ('{"type":"error"}', (), "error event"),
        (
            '{"type":"content_block_start","index":true,"content_block":{}}',
            (),
            "invalid content block",
        ),
        (
            '{"type":"content_block_start","index":0,"content_block":'
            '{"type":"tool_use","id":1,"name":"lookup","input":{}}}',
            (),
            "invalid tool call",
        ),
        (
            '{"type":"content_block_start","index":0,"content_block":'
            '{"type":"tool_use","id":"id","name":"lookup","input":[]}}',
            (),
            "invalid tool input",
        ),
        (
            '{"type":"content_block_delta","index":true,"delta":{}}',
            (),
            "invalid content delta",
        ),
        (
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":2}}',
            (),
            "non-text content",
        ),
        (
            '{"type":"content_block_delta","index":0,"delta":'
            '{"type":"input_json_delta","partial_json":2}}',
            (
                '{"type":"content_block_start","index":0,"content_block":'
                '{"type":"tool_use","id":"id","name":"lookup","input":{}}}',
            ),
            "invalid tool arguments",
        ),
        (
            '{"type":"message_delta","delta":[],"usage":{}}',
            (),
            "invalid message metadata",
        ),
        (
            '{"type":"message_delta","delta":{"stop_reason":1},"usage":{}}',
            (),
            "invalid finish reason",
        ),
    ],
)
async def test_stream_rejects_malformed_anthropic_events(
    monkeypatch: pytest.MonkeyPatch,
    frame: str,
    prefix_frames: tuple[str, ...],
    message: str,
) -> None:
    body = "".join(f"data: {item}\n\n" for item in (*prefix_frames, frame))
    _install_transport(monkeypatch, lambda _: httpx.Response(200, text=body))
    provider = AnthropicProvider(api_key="test", base_url="https://api.anthropic.test/v1")
    try:
        with pytest.raises(ModelError, match=message):
            async for _ in provider.stream(model="claude-test", messages=[], tools=[]):
                pass
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_agent_executes_anthropic_tool_cycle_statelessly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    seen_payloads: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        seen_payloads.append(payload)
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_lookup",
                            "name": "lookup",
                            "input": {"query": "weather"},
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            )
        assert payload["messages"][-1]["content"][0]["type"] == "tool_result"
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Sunny"}],
                "usage": {"input_tokens": 14, "output_tokens": 1},
            },
        )

    _install_transport(monkeypatch, respond)
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="lookup",
            description="Look up weather",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
            handler=lambda query: f"Weather for {query}",
        )
    )
    definition = AgentDefinition(
        name="anthropic-test",
        model={"provider": "anthropic", "model": "claude-test"},
        tools=["lookup"],
        policies={"max_steps": 3, "allowed_tools": ["lookup"]},
    )
    agent = Agent(definition, tools=tools)
    try:
        result = await agent.arun("What is the weather?")
    finally:
        await agent.aclose()

    assert result.output == "Sunny"
    assert calls == 2
    assert seen_payloads[0]["tools"][0]["name"] == "lookup"
    assert seen_payloads[1]["messages"][-1]["content"][0]["content"] == '"Weather for weather"'
