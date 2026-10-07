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
"""Optional MCP adapter for local, stateless agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .agent import Agent
from .server import (
    DEFAULT_MAX_HTTP_RESPONSE_BYTES,
    ResponseSizeLimitError,
    _bounded_json_bytes,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer


@dataclass(frozen=True)
class _AgentContext:
    agent: Agent
    max_response_bytes: int


def create_mcp_server(
    agent: Agent,
    *,
    max_response_bytes: int = DEFAULT_MAX_HTTP_RESPONSE_BYTES,
) -> Any:
    """Expose one agent through a local MCP stdio server.

    The MCP server owns the agent's run lifecycle and closes it when its transport exits.
    Injected model, tool, retriever, and other host-owned handles retain their normal ownership.
    Install the optional ``mcp`` extra to use this adapter.
    """
    if not isinstance(agent, Agent):
        raise TypeError("agent must be a Gabby Agent")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes < 256
    ):
        raise ValueError("max_response_bytes must be an integer of at least 256")
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver import Context
    except ImportError:
        raise ImportError(
            "MCP support requires the optional dependency; install gabby-agent-runtime[mcp]"
        ) from None

    @asynccontextmanager
    async def lifespan(_server: MCPServer[_AgentContext]) -> AsyncIterator[_AgentContext]:
        async with agent:
            yield _AgentContext(agent, max_response_bytes)

    server = MCPServer(name="Gabby", description="Stateless agent execution", lifespan=lifespan)

    async def run_agent(
        input: str,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        """Execute one independent Gabby agent run with caller-supplied context and memory."""
        if ctx is None:
            raise RuntimeError("MCP request context is unavailable")
        state = ctx.request_context.lifespan_context
        try:
            result = await state.agent.arun(
                input,
                context=context,
                memory=memory,
                metadata=metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise RuntimeError("Gabby agent execution failed") from None
        response = {
            "output": result.output,
            "metadata": result.metadata,
            "trace_id": result.trace.trace_id,
        }
        try:
            _bounded_json_bytes(
                response,
                max_bytes=state.max_response_bytes,
                default=None,
            )
        except ResponseSizeLimitError:
            raise RuntimeError("Gabby agent result exceeds the configured response limit") from None
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            raise RuntimeError("Gabby agent returned an invalid JSON result") from None
        return response

    # MCP identifies injected context from the evaluated annotation. Context is imported lazily
    # because this module is part of Gabby's core package but the SDK is an optional extra.
    run_agent.__annotations__["ctx"] = Context[_AgentContext]
    server.add_tool(run_agent, name="run_agent", structured_output=True)
    return server
