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
"""SDK-independent contracts for importing host-connected MCP tools."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from gabby.mcp_tools import _tool_name, _validate_schema_size, register_mcp_tools
from gabby.tools import ToolError, ToolErrorCode, ToolRegistry


def _remote(name: str = "lookup", *, schema: Any = None, description: Any = "description") -> Any:
    return SimpleNamespace(
        name=name,
        input_schema=schema if schema is not None else {"type": "object", "properties": {}},
        description=description,
    )


class _Client:
    def __init__(self, pages: list[Any] | None = None, result: Any = None) -> None:
        self.pages = list(pages or [SimpleNamespace(tools=[_remote()], next_cursor=None)])
        self.result = result or SimpleNamespace(
            is_error=False,
            structured_content={"ok": True},
            content=[],
        )
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    async def list_tools(self, *, cursor: str | None = None) -> Any:
        del cursor
        value = self.pages.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return await value()
        return value

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, read_timeout_seconds: float
    ) -> Any:
        self.calls.append((name, arguments, read_timeout_seconds))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    ("registry", "client", "options", "error"),
    [
        (object(), _Client(), {}, TypeError),
        (ToolRegistry(), _Client(), {"server_name": " "}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x" * 257}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x", "timeout_seconds": True}, ValueError),
        (
            ToolRegistry(),
            _Client(),
            {"server_name": "x", "timeout_seconds": float("nan")},
            ValueError,
        ),
        (ToolRegistry(), _Client(), {"server_name": "x", "max_result_bytes": 0}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x", "max_tools": 0}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x", "max_tools": 257}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x", "max_schema_bytes": 0}, ValueError),
        (ToolRegistry(), _Client(), {"server_name": "x", "permissions": ("read", 3)}, ValueError),
        (ToolRegistry(), object(), {"server_name": "x"}, TypeError),
    ],
)
async def test_register_validates_host_owned_inputs(
    registry: Any, client: Any, options: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        await register_mcp_tools(registry, client, **options)


def test_tool_names_are_sanitized_and_long_names_have_stable_digest() -> None:
    assert _tool_name("server one", "find/file") == "mcp_server_one_find_file"
    first = _tool_name("s" * 80, "t" * 80)
    assert first == _tool_name("s" * 80, "t" * 80)
    assert len(first) == 64 and first[55] == "_"


@pytest.mark.parametrize(
    "schema",
    [
        {"x": float("inf")},
        {1: "value"},
        {"x": "\ud800"},
        {"x": object()},
    ],
)
def test_schema_validator_rejects_non_finite_json(schema: dict[Any, Any]) -> None:
    with pytest.raises(ValueError):
        _validate_schema_size(schema, max_bytes=1024)


def test_schema_validator_rejects_cycles_depth_and_byte_overflow() -> None:
    _validate_schema_size({"x": [True, None, 3, 2.5]}, max_bytes=1024)
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive
    with pytest.raises(ValueError, match="cycles"):
        _validate_schema_size(recursive, max_bytes=1024)

    deep: Any = "leaf"
    for _ in range(66):
        deep = {"x": deep}
    with pytest.raises(ValueError, match="structural"):
        _validate_schema_size(deep, max_bytes=100_000)
    with pytest.raises(ValueError, match="byte limit"):
        _validate_schema_size({"x": "a" * 100}, max_bytes=16)
    with pytest.raises(ValueError, match="byte limit"):
        _validate_schema_size({"type": "object", "properties": {}}, max_bytes=10)


@pytest.mark.asyncio
async def test_discovery_paginates_and_rejects_bad_pages_cursors_and_server_errors() -> None:
    registry = ToolRegistry()
    client = _Client(
        [
            SimpleNamespace(tools=[_remote("first")], next_cursor="next"),
            SimpleNamespace(tools=[_remote("second")], next_cursor=None),
        ]
    )
    mapping = await register_mcp_tools(registry, client, server_name="research")
    assert set(mapping.values()) == {"first", "second"}

    for page in (SimpleNamespace(tools=()), SimpleNamespace(tools=[], next_cursor=4)):
        with pytest.raises(ValueError, match="tool list|pagination cursor"):
            await register_mcp_tools(ToolRegistry(), _Client([page]), server_name="bad")
    too_many = SimpleNamespace(tools=[_remote("a"), _remote("b")], next_cursor=None)
    with pytest.raises(ValueError, match="max_tools"):
        await register_mcp_tools(
            ToolRegistry(), _Client([too_many]), server_name="limited", max_tools=1
        )
    with pytest.raises(RuntimeError, match="discovery failed") as failure:
        await register_mcp_tools(
            ToolRegistry(), _Client([RuntimeError("private endpoint detail")]), server_name="bad"
        )
    assert failure.value.__cause__ is None and "private endpoint" not in str(failure.value)

    async def slow_page() -> Any:
        await asyncio.sleep(1)
        return SimpleNamespace(tools=[], next_cursor=None)

    with pytest.raises(TimeoutError, match="discovery exceeded"):
        await register_mcp_tools(
            ToolRegistry(), _Client([slow_page]), server_name="slow", timeout_seconds=0.001
        )
    with pytest.raises(asyncio.CancelledError):
        await register_mcp_tools(
            ToolRegistry(), _Client([asyncio.CancelledError()]), server_name="cancelled"
        )


@pytest.mark.asyncio
async def test_discovery_bounds_remote_names_and_never_partially_registers() -> None:
    for remote in (_remote(""), SimpleNamespace()):
        with pytest.raises(ValueError, match="tool"):
            await register_mcp_tools(
                ToolRegistry(),
                _Client([SimpleNamespace(tools=[remote], next_cursor=None)]),
                server_name="bad",
            )
    with pytest.raises(ValueError, match="over 256"):
        await register_mcp_tools(
            ToolRegistry(),
            _Client([SimpleNamespace(tools=[_remote("x" * 257)], next_cursor=None)]),
            server_name="bad",
        )
    schemas: list[Any] = [[], {"type": "array"}]
    for schema in schemas:
        remote = SimpleNamespace(name="bad", input_schema=schema, description="")
        with pytest.raises(ValueError, match="object input schema"):
            await register_mcp_tools(
                ToolRegistry(),
                _Client([SimpleNamespace(tools=[remote], next_cursor=None)]),
                server_name="bad",
            )
    registry = ToolRegistry()
    page = SimpleNamespace(
        tools=[_remote("valid"), _remote("bad", schema={"type": "array"})], next_cursor=None
    )
    with pytest.raises(ValueError):
        await register_mcp_tools(registry, _Client([page]), server_name="atomic")
    assert registry.names() == []


@pytest.mark.asyncio
async def test_remote_name_collisions_and_invalid_description_are_rejected() -> None:
    duplicate = SimpleNamespace(tools=[_remote("a/b"), _remote("a?b")], next_cursor=None)
    with pytest.raises(ValueError, match="collision"):
        await register_mcp_tools(ToolRegistry(), _Client([duplicate]), server_name="collision")
    long_prefix = "a" * 80
    distinct_long_names = SimpleNamespace(
        tools=[_remote(long_prefix + "x"), _remote(long_prefix + "y")], next_cursor=None
    )
    mapping = await register_mcp_tools(
        ToolRegistry(), _Client([distinct_long_names]), server_name="collision"
    )
    assert len(mapping) == 2 and len(set(mapping)) == 2
    invalid_description = SimpleNamespace(
        tools=[_remote("x", description="\ud800")], next_cursor=None
    )
    with pytest.raises(ValueError):
        await register_mcp_tools(ToolRegistry(), _Client([invalid_description]), server_name="bad")
    huge_description = SimpleNamespace(tools=[_remote("x", description="x" * 65)], next_cursor=None)
    with pytest.raises(ValueError, match="description exceeds"):
        await register_mcp_tools(
            ToolRegistry(), _Client([huge_description]), server_name="bad", max_schema_bytes=64
        )
    default_schema = SimpleNamespace(name="default", description=None)
    defaults_registry = ToolRegistry()
    await register_mcp_tools(
        defaults_registry,
        _Client([SimpleNamespace(tools=[default_schema], next_cursor=None)]),
        server_name="defaults",
    )
    default_tool = defaults_registry.get("mcp_defaults_default")
    assert default_tool.description.endswith(": ")
    assert default_tool.parameters == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_imported_handler_covers_structured_text_and_error_contracts() -> None:
    successful_client = _Client()
    successful_registry = ToolRegistry()
    mapping = await register_mcp_tools(
        successful_registry,
        successful_client,
        server_name="remote",
        permissions=["net:read"],
        timeout_seconds=4,
    )
    successful_tool = successful_registry.get(next(iter(mapping)))
    assert successful_tool.permissions == ("net:read",)
    assert successful_tool.handler is not None
    assert await successful_tool.handler(query="q") == {"ok": True}
    assert successful_client.calls == [("lookup", {"query": "q"}, 4.0)]

    cases = [
        (
            SimpleNamespace(is_error=True, structured_content=None, content=[]),
            "returned an error",
            ToolErrorCode.EXECUTION_FAILED,
        ),
        (
            SimpleNamespace(is_error=False, structured_content=[], content=[]),
            "invalid structured",
            ToolErrorCode.INVALID_RESULT,
        ),
        (
            SimpleNamespace(is_error=False, structured_content=None, content=None),
            "invalid content",
            ToolErrorCode.INVALID_RESULT,
        ),
        (
            SimpleNamespace(
                is_error=False,
                structured_content=None,
                content=[SimpleNamespace(type="image", text="x")],
            ),
            "unsupported content",
            ToolErrorCode.INVALID_RESULT,
        ),
        (
            SimpleNamespace(
                is_error=False,
                structured_content=None,
                content=[SimpleNamespace(type="text", text=3)],
            ),
            "unsupported content",
            ToolErrorCode.INVALID_RESULT,
        ),
    ]
    for result, message, code in cases:
        registry = ToolRegistry()
        mapping = await register_mcp_tools(registry, _Client(result=result), server_name="remote")
        handler = registry.get(next(iter(mapping))).handler
        assert handler is not None
        with pytest.raises(ToolError, match=message) as error:
            await handler()
        assert error.value.code == code

    broken_registry = ToolRegistry()
    mapping = await register_mcp_tools(
        broken_registry, _Client(result=RuntimeError("secret")), server_name="remote"
    )
    handler = broken_registry.get(next(iter(mapping))).handler
    assert handler is not None
    with pytest.raises(ToolError, match="call failed") as error:
        await handler()
    assert error.value.code == ToolErrorCode.EXECUTION_FAILED
    assert "secret" not in str(error.value)

    text_registry = ToolRegistry()
    mapping = await register_mcp_tools(
        text_registry,
        _Client(result=SimpleNamespace(is_error=False, structured_content=None, content=[])),
        server_name="remote",
    )
    handler = text_registry.get(next(iter(mapping))).handler
    assert handler is not None and await handler() == {"text": []}

    valid_text_registry = ToolRegistry()
    mapping = await register_mcp_tools(
        valid_text_registry,
        _Client(
            result=SimpleNamespace(
                is_error=False,
                structured_content=None,
                content=[SimpleNamespace(type="text", text="ok")],
            )
        ),
        server_name="remote",
    )
    handler = valid_text_registry.get(next(iter(mapping))).handler
    assert handler is not None and await handler() == {"text": ["ok"]}

    async def cancelled_call(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    cancel_client = _Client()
    cancel_client.call_tool = cancelled_call  # type: ignore[method-assign]
    cancel_registry = ToolRegistry()
    mapping = await register_mcp_tools(cancel_registry, cancel_client, server_name="remote")
    handler = cancel_registry.get(next(iter(mapping))).handler
    assert handler is not None
    with pytest.raises(asyncio.CancelledError):
        await handler()
