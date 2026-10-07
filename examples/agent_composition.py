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
"""Run a coordinator agent that delegates a focused task to an offline specialist."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gabby import Agent, AgentDefinition, AgentTool, ModelResponse, ToolRegistry, TraceEvent


class ResearchModel:
    """Offline specialist model that returns a concise finding for its assigned task."""

    name = "offline-research-specialist"

    async def complete(self, *, messages: list[dict[str, Any]], **_: Any) -> ModelResponse:
        task = messages[-1]["content"]
        return ModelResponse(content=f"Finding for '{task}': cite the source and state its limit.")


class CoordinatorModel:
    """Offline parent model that requests a specialist and uses the returned observation."""

    name = "offline-coordinator"

    def __init__(self) -> None:
        self._delegated = False

    async def complete(self, *, messages: list[dict[str, Any]], **_: Any) -> ModelResponse:
        if not self._delegated:
            self._delegated = True
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "research-1",
                        "type": "function",
                        "function": {
                            "name": "research",
                            "arguments": json.dumps(
                                {"input": "Find one source about urban tree cover."}
                            ),
                        },
                    }
                ]
            )
        observation = next(
            message["content"] for message in reversed(messages) if message["role"] == "tool"
        )
        finding = json.loads(observation)["output"]
        return ModelResponse(content=f"Coordinator summary: {finding}")


class CorrelationTracer:
    """Capture child request events to show their parent trace relationship."""

    def __init__(self) -> None:
        self.request_events: list[tuple[str, TraceEvent]] = []

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        if event.kind == "request":
            self.request_events.append((trace_id, event))


async def run_example() -> tuple[str, str, str, str]:
    """Return output, parent trace ID, child trace ID, and the child's parent trace ID."""
    child_tracer = CorrelationTracer()
    researcher = Agent(
        AgentDefinition(
            name="research-specialist",
            model={"provider": "example", "model": "offline"},
            instructions="Return one concise finding for the assigned task.",
        ),
        model=ResearchModel(),
        tracer=child_tracer,
    )
    parent_tools = ToolRegistry()
    parent_tools.register(
        AgentTool(
            researcher,
            name="research",
            description="Ask the research specialist to investigate one focused task.",
        ).to_tool()
    )
    coordinator = Agent(
        AgentDefinition(
            name="research-coordinator",
            model={"provider": "example", "model": "offline"},
            tools=["research"],
            instructions="Delegate a focused research task, then summarize the specialist result.",
            policies={
                "allowed_permissions": ["agent:invoke"],
                "max_steps": 3,
                "timeout_seconds": 5,
            },
        ),
        model=CoordinatorModel(),
        tools=parent_tools,
    )
    async with researcher, coordinator:
        result = await coordinator.arun("Summarize one source about urban tree cover.")

    child_trace_id, child_request = child_tracer.request_events[0]
    return (
        result.output,
        result.trace.trace_id,
        child_trace_id,
        child_request.details["parent_trace_id"],
    )


async def main() -> None:
    """Run and display one parent/child execution and its linked trace IDs."""
    output, parent_trace_id, child_trace_id, linked_parent_trace_id = await run_example()
    print(output)
    print(f"Parent trace ID: {parent_trace_id}")
    print(f"Child trace ID: {child_trace_id}")
    print(f"Child parent trace ID: {linked_parent_trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
