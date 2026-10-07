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
"""Tests for bounded stateless agent composition."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from gabby import AgentTool, Principal, ToolContext, ToolError
from gabby.agent import Agent
from gabby.config import AgentDefinition
from gabby.models import ModelResponse
from gabby.tools import Tool, ToolRegistry


class ScriptedModel:
    name = "scripted"

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _definition(name: str, *, tools: list[str] | None = None) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        model={"provider": "test", "model": name},
        tools=tools or [],
        policies={"max_steps": 4, "timeout_seconds": 5},
    )


@pytest.mark.asyncio
async def test_agent_tool_rejects_invalid_definition_options() -> None:
    with pytest.raises(TypeError, match="gabby.Agent"):
        AgentTool(object(), name="child", description="delegate")  # type: ignore[arg-type]

    agent = Agent(_definition("validation-child"), model=ScriptedModel())
    try:
        with pytest.raises(TypeError, match="forward_principal"):
            AgentTool(
                agent,
                name="child",
                description="delegate",
                forward_principal=1,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="permission"):
            AgentTool(agent, name="child", description="delegate", permission="")
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_tool_delegates_only_the_explicit_task_and_returns_trace_reference() -> None:
    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[tuple[str, Any]] = []

        async def on_event(self, *, trace_id: str, agent_name: str, event: Any) -> None:
            self.events.append((trace_id, event))

    child_model = ScriptedModel(ModelResponse(content="Focused findings"))
    child_tracer = CapturingTracer()
    child = Agent(_definition("researcher"), model=child_model, tracer=child_tracer)
    parent_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {
                        "name": "research",
                        "arguments": json.dumps({"input": "Find primary sources"}),
                    },
                }
            ]
        ),
        ModelResponse(content="Research complete"),
    )
    registry = ToolRegistry()
    registry.register(
        AgentTool(
            child,
            name="research",
            description="Ask the research agent to investigate one focused task.",
        ).to_tool()
    )
    parent_definition = _definition("coordinator", tools=["research"])
    parent_definition.policies["allowed_permissions"] = ["agent:invoke"]
    parent = Agent(parent_definition, model=parent_model, tools=registry)

    try:
        result = await parent.arun(
            "Prepare a report",
            context={"private_parent_context": "must not be forwarded"},
            memory={"private_parent_memory": "must not be forwarded"},
        )
    finally:
        await parent.aclose()
        await child.aclose()

    assert result.output == "Research complete"
    assert len(child_model.calls) == 1
    assert "Find primary sources" in str(child_model.calls[0]["messages"])
    assert "private_parent_context" not in str(child_model.calls[0]["messages"])
    assert "private_parent_memory" not in str(child_model.calls[0]["messages"])
    tool_message = next(
        message for message in parent_model.calls[1]["messages"] if message["role"] == "tool"
    )
    delegated = json.loads(tool_message["content"])
    assert delegated["agent"] == "researcher"
    assert delegated["output"] == "Focused findings"
    assert delegated["trace_id"]
    child_request = next(event for _, event in child_tracer.events if event.kind == "request")
    assert child_request.details["parent_trace_id"] == result.trace.trace_id


@pytest.mark.asyncio
async def test_distinct_agents_with_same_name_can_delegate() -> None:
    child_model = ScriptedModel(ModelResponse(content="Child completed"))
    child = Agent(_definition("specialist"), model=child_model)
    parent_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "arguments": json.dumps({"input": "Run child task"}),
                    },
                }
            ]
        ),
        ModelResponse(content="Parent completed"),
    )
    registry = ToolRegistry()
    registry.register(AgentTool(child, name="delegate", description="Delegate one task.").to_tool())
    parent_definition = _definition("specialist", tools=["delegate"])
    parent_definition.policies["allowed_permissions"] = ["agent:invoke"]
    parent = Agent(parent_definition, model=parent_model, tools=registry)

    try:
        result = await parent.arun("Use the child agent")
    finally:
        await parent.aclose()
        await child.aclose()

    assert result.output == "Parent completed"
    assert len(child_model.calls) == 1
    tool_message = next(
        message for message in parent_model.calls[1]["messages"] if message["role"] == "tool"
    )
    delegated = json.loads(tool_message["content"])
    assert delegated["agent"] == "specialist"
    assert delegated["output"] == "Child completed"


@pytest.mark.parametrize("forward_principal", [False, True])
@pytest.mark.asyncio
async def test_agent_tool_forwards_principal_only_when_enabled(forward_principal: bool) -> None:
    principals: list[Principal | None] = []

    def observe_principal(tool_context: ToolContext) -> dict[str, str | None]:
        principals.append(tool_context.principal)
        return {"subject": tool_context.principal.subject if tool_context.principal else None}

    child_tools = ToolRegistry()
    child_tools.register(
        Tool(
            name="observe",
            description="Observe the caller principal.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=observe_principal,
            context_parameter="tool_context",
            output_schema={
                "type": "object",
                "properties": {"subject": {"type": ["string", "null"]}},
                "required": ["subject"],
                "additionalProperties": False,
            },
        )
    )
    child_definition = _definition("child", tools=["observe"])
    child_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "observe-1",
                    "type": "function",
                    "function": {"name": "observe", "arguments": "{}"},
                }
            ]
        ),
        ModelResponse(content="Observed"),
    )
    child = Agent(child_definition, model=child_model, tools=child_tools)
    parent_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "arguments": json.dumps({"input": "Observe caller"}),
                    },
                }
            ]
        ),
        ModelResponse(content="Done"),
    )
    parent_tools = ToolRegistry()
    parent_tools.register(
        AgentTool(
            child,
            name="delegate",
            description="Delegate one task.",
            forward_principal=forward_principal,
        ).to_tool()
    )
    parent_definition = _definition("parent", tools=["delegate"])
    parent_definition.policies["allowed_permissions"] = ["agent:invoke"]
    parent = Agent(parent_definition, model=parent_model, tools=parent_tools)
    principal = Principal(subject="alice", scopes=frozenset({"agent:run"}))

    try:
        await parent.arun("delegate", principal=principal)
    finally:
        await parent.aclose()
        await child.aclose()

    assert principals == [principal if forward_principal else None]


@pytest.mark.asyncio
async def test_agent_tool_rejects_direct_delegation_cycles() -> None:
    agent = Agent(_definition("same-agent"), model=ScriptedModel(ModelResponse(content="unused")))
    tool = AgentTool(agent, name="again", description="Try to call the same agent.").to_tool()
    assert tool.handler is not None
    context = ToolContext(
        agent_name="same-agent",
        run_id="parent-run",
        environment_type="generic",
        environment_description="",
        capabilities=(),
        resources={},
    )

    try:
        with pytest.raises(ToolError, match="delegation cycle"):
            await tool.handler(input="recurse", tool_context=context)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_agent_tool_rejects_indirect_delegation_cycles() -> None:
    class CapturingTracer:
        def __init__(self) -> None:
            self.events: list[Any] = []

        async def on_event(self, *, trace_id: str, agent_name: str, event: Any) -> None:
            self.events.append(event)

    original_model = ScriptedModel(ModelResponse(content="must not run after cycle detection"))
    original = Agent(_definition("shared-name"), model=original_model)

    back_to_original = ToolRegistry()
    back_to_original.register(
        AgentTool(
            original,
            name="back_to_original",
            description="Delegate back to the original agent.",
        ).to_tool()
    )
    intermediate_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "cycle-1",
                    "type": "function",
                    "function": {
                        "name": "back_to_original",
                        "arguments": json.dumps({"input": "This should be rejected"}),
                    },
                }
            ]
        ),
        ModelResponse(content="Cycle was rejected"),
    )
    intermediate_definition = _definition("shared-name", tools=["back_to_original"])
    intermediate_definition.policies["allowed_permissions"] = ["agent:invoke"]
    tracer = CapturingTracer()
    intermediate = Agent(
        intermediate_definition,
        model=intermediate_model,
        tools=back_to_original,
        tracer=tracer,
    )
    delegate_intermediate = AgentTool(
        intermediate,
        name="delegate",
        description="Delegate to the intermediate agent.",
    ).to_tool()

    try:
        assert delegate_intermediate.handler is not None
        result = await delegate_intermediate.handler(
            input="Start the delegation chain",
            tool_context=ToolContext(
                agent_name="shared-name",
                run_id="original-run",
                environment_type="generic",
                environment_description="",
                capabilities=(),
                resources={},
                agent_instance_id=id(original),
            ),
        )
    finally:
        await intermediate.aclose()
        await original.aclose()

    assert result["output"] == "Cycle was rejected"
    assert original_model.calls == []
    assert len(intermediate_model.calls) == 2
    assert any(
        event.kind == "tool_error" and event.details.get("error_code") == "tool_unavailable"
        for event in tracer.events
    )


@pytest.mark.asyncio
async def test_agent_tool_timeout_cancels_the_child_run() -> None:
    class BlockingModel:
        name = "blocking-child"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def complete(self, **_: Any) -> ModelResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            raise AssertionError("blocking model should only finish after cancellation")

    child_model = BlockingModel()
    child = Agent(_definition("slow-specialist"), model=child_model)
    parent_model = ScriptedModel(
        ModelResponse(
            tool_calls=[
                {
                    "id": "delegate-1",
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "arguments": json.dumps({"input": "wait for specialist"}),
                    },
                }
            ]
        ),
        ModelResponse(content="Handled the specialist timeout"),
    )
    parent_tools = ToolRegistry()
    parent_tools.register(
        AgentTool(
            child,
            name="delegate",
            description="Delegate one bounded task.",
            timeout_seconds=0.05,
        ).to_tool()
    )
    parent_definition = _definition("parent", tools=["delegate"])
    parent_definition.policies["allowed_permissions"] = ["agent:invoke"]
    parent = Agent(parent_definition, model=parent_model, tools=parent_tools)

    try:
        result = await parent.arun("delegate once")
    finally:
        await parent.aclose()
        await child.aclose()

    assert child_model.started.is_set()
    assert child_model.cancelled is True
    assert result.output == "Handled the specialist timeout"
    assert any(
        event.kind == "tool_error" and event.details.get("error_code") == "deadline_exceeded"
        for event in result.trace.events
    )
