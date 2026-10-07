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
"""MCP adapter contract tests."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from mcp import Client
except ImportError:
    pytest.skip("MCP optional dependency is not installed", allow_module_level=True)

from gabby.agent import Agent
from gabby.config import AgentDefinition
from gabby.mcp_adapter import create_mcp_server
from gabby.models import ModelResponse


class _SequenceModel:
    name = "mcp-test-model"

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.requests.append(kwargs)
        return ModelResponse(content=self.outputs.pop(0))


def _agent(model: _SequenceModel) -> Agent:
    return Agent(
        AgentDefinition(
            name="mcp-test-agent",
            description="MCP adapter contract fixture",
            model={"provider": "test", "model": "test"},
            instructions="Respond with the configured fixture output.",
            policies={"timeout_seconds": 1, "max_steps": 2},
        ),
        model=model,
    )


@pytest.mark.asyncio
async def test_mcp_tool_exposes_stateless_run_and_injects_request_maps() -> None:
    model = _SequenceModel("first response", "second response")
    agent = _agent(model)
    server = create_mcp_server(agent)

    async with Client(server) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == ["run_agent"]
        assert set(tools.tools[0].input_schema["properties"]) == {
            "input",
            "context",
            "memory",
            "metadata",
        }

        first = await client.call_tool(
            "run_agent",
            {
                "input": "first task",
                "context": {"project": "alpha"},
                "memory": {"caller_note": "temporary"},
                "metadata": {"request": "one"},
            },
        )
        second = await client.call_tool(
            "run_agent",
            {"input": "second task", "context": {"project": "beta"}},
        )

        assert first.is_error is False
        assert first.structured_content is not None
        assert first.structured_content["output"] == "first response"
        assert first.structured_content["trace_id"]
        assert first.structured_content["metadata"]["agent"] == "mcp-test-agent"
        assert second.is_error is False
        assert second.structured_content is not None
        assert second.structured_content["output"] == "second response"
        assert first.structured_content["trace_id"] != second.structured_content["trace_id"]

    assert agent._closed is True
    assert model.outputs == []
    assert len(model.requests) == 2
    first_messages = json.dumps(model.requests[0]["messages"])
    second_messages = json.dumps(model.requests[1]["messages"])
    assert "temporary" in first_messages and "temporary" not in second_messages
    assert "alpha" in first_messages and "beta" in second_messages


@pytest.mark.asyncio
async def test_mcp_tool_rejects_oversized_response_without_exposing_output() -> None:
    agent = _agent(_SequenceModel("x" * 1024))
    server = create_mcp_server(agent, max_response_bytes=256)

    async with Client(server) as client:
        result = await client.call_tool("run_agent", {"input": "large answer"})

    assert result.is_error is True
    error_text = getattr(result.content[0], "text", "")
    assert error_text == "Error executing tool run_agent"
    assert "x" * 256 not in error_text
    assert agent._closed is True


@pytest.mark.asyncio
async def test_mcp_tool_redacts_agent_execution_errors() -> None:
    class BrokenModel:
        name = "mcp-test-model"

        async def complete(self, **_: Any) -> ModelResponse:
            raise RuntimeError("private provider response and token")

    agent = _agent(BrokenModel())  # type: ignore[arg-type]
    server = create_mcp_server(agent)

    async with Client(server) as client:
        result = await client.call_tool("run_agent", {"input": "fail safely"})

    assert result.is_error is True
    error_text = getattr(result.content[0], "text", "")
    assert error_text == "Error executing tool run_agent"
    assert "private provider response" not in error_text
    assert agent._closed is True


@pytest.mark.asyncio
async def test_mcp_tool_rejects_non_json_agent_result(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(_SequenceModel("unused"))

    async def invalid_result(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            output="answer",
            metadata={"unserializable": object()},
            trace=SimpleNamespace(trace_id="trace-id"),
        )

    monkeypatch.setattr(agent, "arun", invalid_result)
    server = create_mcp_server(agent)

    async with Client(server) as client:
        result = await client.call_tool("run_agent", {"input": "invalid response"})

    assert result.is_error is True
    error_text = getattr(result.content[0], "text", "")
    assert error_text == "Error executing tool run_agent"
    assert "unserializable" not in error_text
    assert agent._closed is True


def test_mcp_server_requires_an_agent_and_bounded_response_size() -> None:
    with pytest.raises(TypeError, match="agent must be a Gabby Agent"):
        create_mcp_server(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 256"):
        create_mcp_server(_agent(_SequenceModel("unused")), max_response_bytes=255)


def test_mcp_sdk_is_loaded_only_when_the_adapter_is_constructed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(_SequenceModel("unused"))
    monkeypatch.setitem(sys.modules, "mcp.server", None)

    with pytest.raises(ImportError, match=r"install gabby-agent-runtime\[mcp\]"):
        create_mcp_server(agent)

    agent.close()
