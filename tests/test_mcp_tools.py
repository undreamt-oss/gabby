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
"""Contract checks for importing host-connected MCP tools into Gabby."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from mcp import Client
    from mcp.server import MCPServer
except ImportError:
    pytest.skip("MCP optional dependency is not installed", allow_module_level=True)

from gabby.agent import Agent
from gabby.config import AgentDefinition
from gabby.mcp_tools import register_mcp_tools
from gabby.models import ModelResponse
from gabby.tools import Tool, ToolError, ToolErrorCode, ToolRegistry


class _DispatchModel:
    name = "mcp-tools-model"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "mcp-call-1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": '{"text":"gabby"}',
                        },
                    }
                ]
            )
        return ModelResponse(content="MCP tool completed")


@pytest.mark.asyncio
async def test_agent_uses_host_connected_mcp_tool_and_host_keeps_client_lifecycle() -> None:
    server = MCPServer(name="TextTools", description="Text operations")

    async def reverse(text: str) -> dict[str, str]:
        return {"reversed": text[::-1]}

    server.add_tool(reverse, name="reverse_text", structured_output=True)
    registry = ToolRegistry()
    model = _DispatchModel("mcp_texttools_reverse_text")
    definition = AgentDefinition(
        name="mcp-consumer",
        model={"provider": "test", "model": "test"},
        tools=["mcp_texttools_reverse_text"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 5,
            "allowed_tools": ["mcp_texttools_reverse_text"],
            "allowed_permissions": ["mcp:text:read"],
        },
    )

    async with Client(server) as client:
        mapping = await register_mcp_tools(
            registry,
            client,
            server_name="texttools",
            permissions=["mcp:text:read"],
        )
        assert mapping == {"mcp_texttools_reverse_text": "reverse_text"}
        tool = registry.get("mcp_texttools_reverse_text")
        assert tool.permissions == ("mcp:text:read",)
        agent = Agent(definition, model=model, tools=registry)
        try:
            result = await agent.arun("Reverse gabby")
        finally:
            await agent.aclose()

        assert result.output == "MCP tool completed"
        assert len(model.calls) == 2
        second_request = model.calls[1]["messages"]
        assert any(
            message.get("role") == "tool"
            and json.loads(message["content"]) == {"reversed": "ybbag"}
            for message in second_request
        )


@pytest.mark.asyncio
async def test_mcp_tool_errors_are_sanitized_before_they_reach_agent() -> None:
    server = MCPServer(name="PrivateTools", description="Private test")

    async def fail() -> str:
        raise RuntimeError("private endpoint token")

    server.add_tool(fail, name="fail")
    registry = ToolRegistry()
    async with Client(server) as client:
        mapping = await register_mcp_tools(registry, client, server_name="private")
        tool = registry.get(next(iter(mapping)))
        assert tool.handler is not None
        with pytest.raises(ToolError) as raised:
            await tool.handler()
    assert raised.value.code == ToolErrorCode.EXECUTION_FAILED
    assert str(raised.value) == "MCP tool returned an error"
    assert "private endpoint token" not in str(raised.value)


@pytest.mark.asyncio
async def test_mcp_discovery_is_bounded_and_does_not_partially_register() -> None:
    class ManyTools:
        async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name=f"tool-{index}",
                        description="fixture",
                        input_schema={"type": "object", "properties": {}},
                    )
                    for index in range(2)
                ],
                next_cursor=None,
            )

        async def call_tool(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("discovery-limit check must not call a tool")

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="max_tools=1"):
        await register_mcp_tools(registry, ManyTools(), server_name="limited", max_tools=1)
    assert registry.names() == []


@pytest.mark.asyncio
async def test_mcp_name_collisions_and_unsupported_schemas_fail_closed() -> None:
    class OneTool:
        async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="lookup",
                        description="fixture",
                        input_schema={"type": "array"},
                    )
                ],
                next_cursor=None,
            )

        async def call_tool(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("schema validation must run before tool invocation")

    with pytest.raises(ValueError, match="object input schema"):
        await register_mcp_tools(ToolRegistry(), OneTool(), server_name="search")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="mcp_search_lookup",
            description="already registered",
            parameters={"type": "object", "properties": {}},
            output_schema={"type": "null"},
            handler=lambda: None,
        )
    )
    with pytest.raises(ValueError, match="name collision"):
        await register_mcp_tools(
            registry,
            OneToolWithValidSchema(),
            server_name="search",
        )


@pytest.mark.asyncio
async def test_mcp_text_results_are_mapped_and_unsupported_content_is_rejected() -> None:
    class ContentClient(OneToolWithValidSchema):
        def __init__(self, result: Any) -> None:
            self.result = result

        async def call_tool(self, *_args: Any, **_kwargs: Any) -> Any:
            return self.result

    text_client = ContentClient(
        SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[
                SimpleNamespace(type="text", text="first"),
                SimpleNamespace(type="text", text="second"),
            ],
        )
    )
    text_registry = ToolRegistry()
    text_mapping = await register_mcp_tools(text_registry, text_client, server_name="text")
    text_tool = text_registry.get(next(iter(text_mapping)))
    assert text_tool.handler is not None
    assert await text_tool.handler() == {"text": ["first", "second"]}

    image_client = ContentClient(
        SimpleNamespace(
            is_error=False,
            structured_content=None,
            content=[SimpleNamespace(type="image", data="private image data")],
        )
    )
    image_registry = ToolRegistry()
    image_mapping = await register_mcp_tools(image_registry, image_client, server_name="image")
    image_tool = image_registry.get(next(iter(image_mapping)))
    assert image_tool.handler is not None
    with pytest.raises(ToolError, match="unsupported content"):
        await image_tool.handler()


@pytest.mark.asyncio
async def test_mcp_schema_size_and_description_bounds_fail_before_registration() -> None:
    class OversizedSchema(OneToolWithValidSchema):
        async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="large",
                        description="ok",
                        input_schema={"type": "object", "properties": {"x": "a" * 128}},
                    )
                ],
                next_cursor=None,
            )

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="schema exceeds"):
        await register_mcp_tools(
            registry,
            OversizedSchema(),
            server_name="large",
            max_schema_bytes=64,
        )
    assert registry.names() == []

    class OversizedDescription(OneToolWithValidSchema):
        async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="large",
                        description="x" * 65,
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                next_cursor=None,
            )

    with pytest.raises(ValueError, match="description exceeds"):
        await register_mcp_tools(
            ToolRegistry(),
            OversizedDescription(),
            server_name="large",
            max_schema_bytes=64,
        )


class OneToolWithValidSchema:
    async def list_tools(self, *, cursor: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description="fixture",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            next_cursor=None,
        )

    async def call_tool(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("collision check must run before tool invocation")
