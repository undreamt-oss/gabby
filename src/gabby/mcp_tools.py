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
"""Host-injected MCP client tools for Gabby agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from .tools import Tool, ToolError, ToolErrorCode, ToolRegistry

DEFAULT_MAX_MCP_TOOLS = 128
MAX_MCP_TOOLS = 256
DEFAULT_MAX_MCP_SCHEMA_BYTES = 64 * 1024


class _SchemaTooLarge(ValueError):
    pass


def _tool_name(server_name: str, remote_name: str) -> str:
    """Build a unique Gabby-compatible name while retaining a digest for long names."""
    original = f"mcp_{server_name}_{remote_name}"
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", original)
    if len(normalized) <= 64:
        return normalized
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[:55]}_{digest}"


def _validate_schema_size(schema: dict[str, Any], *, max_bytes: int) -> None:
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = 0

    def count_text_bytes(value: str) -> int:
        byte_count = 0
        for character in value:
            codepoint = ord(character)
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError("MCP tool schema must be finite JSON")
            byte_count += (
                1
                if codepoint <= 0x7F
                else 2
                if codepoint <= 0x7FF
                else 3
                if codepoint <= 0xFFFF
                else 4
            )
            if byte_count > max_bytes:
                raise _SchemaTooLarge("MCP tool schema exceeds the configured byte limit")
        return byte_count

    visited_nodes = 0
    ancestors: set[int] = set()

    def validate(value: Any, depth: int = 0) -> None:
        nonlocal visited_nodes
        visited_nodes += 1
        if visited_nodes > 10_000 or depth > 64:
            raise ValueError("MCP tool schema exceeds structural limits")
        if isinstance(value, str):
            count_text_bytes(value)
        elif isinstance(value, bool) or value is None or isinstance(value, int):
            return
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("MCP tool schema must be finite JSON")
        elif isinstance(value, (list, dict)):
            identity = id(value)
            if identity in ancestors:
                raise ValueError("MCP tool schema must not contain cycles")
            ancestors.add(identity)
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                if isinstance(value, dict):
                    if not isinstance(key, str):
                        raise ValueError("MCP tool schema must have string keys")
                    count_text_bytes(key)
                validate(item, depth + 1)
            ancestors.remove(identity)
        else:
            raise ValueError("MCP tool schema must be finite JSON")

    validate(schema)
    try:
        for chunk in encoder.iterencode(schema):
            size += count_text_bytes(chunk)
            if size > max_bytes:
                raise _SchemaTooLarge("MCP tool schema exceeds the configured byte limit")
    except _SchemaTooLarge:
        raise
    except (TypeError, ValueError):
        raise ValueError("MCP tool schema must be finite JSON") from None


async def register_mcp_tools(
    registry: ToolRegistry,
    client: Any,
    *,
    server_name: str,
    permissions: tuple[str, ...] | list[str] = (),
    timeout_seconds: float = 30,
    max_result_bytes: int = 1024 * 1024,
    max_tools: int = DEFAULT_MAX_MCP_TOOLS,
    max_schema_bytes: int = DEFAULT_MAX_MCP_SCHEMA_BYTES,
) -> dict[str, str]:
    """Register tools from an already-connected MCP client without taking ownership of it.

    The host establishes and closes the MCP client connection. Every imported tool is a
    host-trusted handler; Gabby policy and permission checks still run before each call. MCP
    responses with structured JSON are passed through, while unstructured responses accept only
    text blocks. Other MCP content types fail closed instead of entering model context.
    """
    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be a ToolRegistry")
    if not isinstance(server_name, str) or not server_name.strip():
        raise ValueError("server_name must be a non-empty string")
    if len(server_name.encode("utf-8")) > 256:
        raise ValueError("server_name must be at most 256 UTF-8 bytes")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    if (
        isinstance(max_result_bytes, bool)
        or not isinstance(max_result_bytes, int)
        or max_result_bytes < 1
    ):
        raise ValueError("max_result_bytes must be a positive integer")
    if (
        isinstance(max_tools, bool)
        or not isinstance(max_tools, int)
        or not 1 <= max_tools <= MAX_MCP_TOOLS
    ):
        raise ValueError(f"max_tools must be an integer from 1 through {MAX_MCP_TOOLS}")
    if (
        isinstance(max_schema_bytes, bool)
        or not isinstance(max_schema_bytes, int)
        or max_schema_bytes < 1
    ):
        raise ValueError("max_schema_bytes must be a positive integer")
    if not isinstance(permissions, (tuple, list)) or any(
        not isinstance(permission, str) or not permission for permission in permissions
    ):
        raise ValueError("permissions must be non-empty strings")
    list_tools = getattr(client, "list_tools", None)
    call_tool = getattr(client, "call_tool", None)
    if not callable(list_tools) or not callable(call_tool):
        raise TypeError("client must provide async list_tools() and call_tool() methods")

    remote_tools: list[Any] = []
    cursor: str | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            while True:
                page = await list_tools(cursor=cursor)
                page_tools = getattr(page, "tools", None)
                if not isinstance(page_tools, list):
                    raise ValueError("MCP server returned an invalid tool list")
                if len(remote_tools) + len(page_tools) > max_tools:
                    raise ValueError(f"MCP server exposes more than max_tools={max_tools}")
                remote_tools.extend(page_tools)
                cursor = getattr(page, "next_cursor", None)
                if cursor is None:
                    break
                if not isinstance(cursor, str) or not cursor:
                    raise ValueError("MCP server returned an invalid pagination cursor")
    except TimeoutError:
        raise TimeoutError("MCP tool discovery exceeded its timeout") from None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise RuntimeError("MCP tool discovery failed") from None

    names: dict[str, str] = {}
    pending: list[Tool] = []
    for remote in remote_tools:
        remote_name = getattr(remote, "name", None)
        if not isinstance(remote_name, str) or not remote_name:
            raise ValueError("MCP server returned a tool without a valid name")
        if len(remote_name.encode("utf-8")) > 256:
            raise ValueError("MCP server returned a tool name over 256 UTF-8 bytes")
        gabby_name = _tool_name(server_name.strip(), remote_name)
        if gabby_name in names or gabby_name in registry.names():
            raise ValueError(f"MCP tool name collision for {gabby_name!r}")
        names[gabby_name] = remote_name
        schema = getattr(remote, "input_schema", None)
        if schema is None:
            schema = {"type": "object", "properties": {}}
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError(f"MCP tool {remote_name!r} must declare an object input schema")
        _validate_schema_size(schema, max_bytes=max_schema_bytes)
        description = getattr(remote, "description", None)
        if not isinstance(description, str):
            description = ""
        if len(description.encode("utf-8")) > max_schema_bytes:
            raise ValueError("MCP tool description exceeds the configured schema byte limit")

        def create_handler(remote_tool_name: str) -> Callable[..., Any]:
            async def invoke_mcp_tool(**arguments: Any) -> Any:
                try:
                    result = await call_tool(
                        remote_tool_name,
                        arguments,
                        read_timeout_seconds=float(timeout_seconds),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise ToolError(
                        "MCP tool call failed", code=ToolErrorCode.EXECUTION_FAILED
                    ) from None
                if getattr(result, "is_error", False):
                    raise ToolError(
                        "MCP tool returned an error", code=ToolErrorCode.EXECUTION_FAILED
                    )
                structured = getattr(result, "structured_content", None)
                if structured is not None:
                    if not isinstance(structured, Mapping):
                        raise ToolError(
                            "MCP tool returned invalid structured content",
                            code=ToolErrorCode.INVALID_RESULT,
                        )
                    return dict(structured)
                blocks = getattr(result, "content", None)
                if not isinstance(blocks, list):
                    raise ToolError(
                        "MCP tool returned invalid content", code=ToolErrorCode.INVALID_RESULT
                    )
                text_blocks: list[str] = []
                for block in blocks:
                    if getattr(block, "type", None) != "text" or not isinstance(
                        getattr(block, "text", None), str
                    ):
                        raise ToolError(
                            "MCP tool returned unsupported content",
                            code=ToolErrorCode.INVALID_RESULT,
                        )
                    text_blocks.append(block.text)
                return {"text": text_blocks}

            return invoke_mcp_tool

        pending.append(
            Tool(
                name=gabby_name,
                description=(f"MCP server {server_name.strip()} tool {remote_name}: {description}"),
                parameters=schema,
                output_schema={"type": ["object", "array", "string", "number", "boolean", "null"]},
                handler=create_handler(remote_name),
                timeout_seconds=float(timeout_seconds),
                max_result_bytes=max_result_bytes,
                permissions=permissions,
            )
        )

    for tool in pending:
        registry.register(tool)
    return names
