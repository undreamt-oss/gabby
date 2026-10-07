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
"""Bounded composition of stateless agents through the tool contract."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass

from .agent import Agent
from .tools import Tool, ToolContext, ToolError, ToolErrorCode
from .tracing import _PARENT_TRACE_ID

_AGENT_DELEGATION_PATH: ContextVar[tuple[int | str, ...]] = ContextVar(
    "gabby_agent_delegation_path", default=()
)
_MAX_AGENT_DELEGATION_DEPTH = 8


@dataclass(frozen=True)
class AgentTool:
    """Expose another stateless ``Agent`` as one explicitly granted parent tool.

    The child receives only the task string produced for this tool call. Parent context, memory,
    metadata, and caller identity are not forwarded. Child policy, tools, retrieval, verification,
    and model configuration remain independent. The host owns both agents and must close them.
    """

    agent: Agent
    name: str
    description: str
    timeout_seconds: float = 60.0
    max_input_bytes: int = 64 * 1024
    max_result_bytes: int = 1024 * 1024
    permission: str = "agent:invoke"
    forward_principal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.agent, Agent):
            raise TypeError("agent must be a gabby.Agent")
        if not isinstance(self.forward_principal, bool):
            raise TypeError("forward_principal must be a boolean")
        if not isinstance(self.permission, str) or not self.permission:
            raise ValueError("permission must be a non-empty string")

    def to_tool(self) -> Tool:
        """Build the tool to register in a parent agent's ``ToolRegistry``."""

        async def invoke(input: str, tool_context: ToolContext) -> dict[str, str]:
            path = _AGENT_DELEGATION_PATH.get()
            child_name = self.agent.definition.name
            parent_identity: int | str = (
                tool_context.agent_instance_id
                if tool_context.agent_instance_id is not None
                else tool_context.agent_name
            )
            child_identity: int | str = (
                id(self.agent) if tool_context.agent_instance_id is not None else child_name
            )
            if not path or path[-1] != parent_identity:
                path = (*path, parent_identity)
            if child_identity in path:
                raise ToolError(
                    "Agent delegation cycle detected", code=ToolErrorCode.TOOL_UNAVAILABLE
                )
            if len(path) >= _MAX_AGENT_DELEGATION_DEPTH:
                raise ToolError(
                    "Agent delegation depth limit reached", code=ToolErrorCode.TOOL_UNAVAILABLE
                )

            token = _AGENT_DELEGATION_PATH.set((*path, child_identity))
            parent_trace_token = _PARENT_TRACE_ID.set(tool_context.run_id)
            try:
                result = await self.agent.arun(
                    input,
                    principal=tool_context.principal if self.forward_principal else None,
                )
            except asyncio.CancelledError:
                raise
            finally:
                _PARENT_TRACE_ID.reset(parent_trace_token)
                _AGENT_DELEGATION_PATH.reset(token)
            return {
                "agent": child_name,
                "output": result.output,
                "trace_id": result.trace.trace_id,
            }

        return Tool(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string", "minLength": 1}},
                "required": ["input"],
                "additionalProperties": False,
            },
            handler=invoke,
            timeout_seconds=self.timeout_seconds,
            max_input_bytes=self.max_input_bytes,
            max_result_bytes=self.max_result_bytes,
            output_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "output": {"type": "string"},
                    "trace_id": {"type": "string"},
                },
                "required": ["agent", "output", "trace_id"],
                "additionalProperties": False,
            },
            permissions=(self.permission,),
            context_parameter="tool_context",
        )
