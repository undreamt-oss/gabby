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
"""Show opt-in planner revisions using one deterministic tool observation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gabby import Agent, AgentDefinition, ModelPlanner, ModelResponse, Tool, ToolRegistry


class OfflineProvider:
    """Return scripted plans and reasoning responses without network access."""

    name = "offline-planning-demo"

    def __init__(self) -> None:
        self._requested_lookup = False

    async def complete(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: Any
    ) -> ModelResponse:
        """Distinguish planner requests from the runtime's tool-enabled model calls."""
        if not tools:
            request = json.loads(messages[-1]["content"])
            if request.get("previous_plan") is not None:
                return ModelResponse(
                    content=json.dumps(
                        {
                            "summary": "Report the order status confirmed by the lookup.",
                            "steps": [
                                {
                                    "instruction": "Summarize the observed order status.",
                                    "success_criteria": "The answer matches the tool result.",
                                }
                            ],
                        }
                    )
                )
            return ModelResponse(
                content=json.dumps(
                    {
                        "summary": "Check the order and report its status.",
                        "steps": [
                            {
                                "instruction": "Look up the requested order.",
                                "success_criteria": "The order status is returned.",
                            }
                        ],
                    }
                )
            )
        if not self._requested_lookup:
            self._requested_lookup = True
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "lookup-order-1",
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "arguments": json.dumps({"order_id": "order-123"}),
                        },
                    }
                ]
            )
        observation = next(
            message["content"] for message in reversed(messages) if message["role"] == "tool"
        )
        status = json.loads(observation)["status"]
        return ModelResponse(content=f"Order order-123 is {status}.")


async def main() -> None:
    """Run one stateless request and print plan revision progress."""
    provider = OfflineProvider()
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="lookup_order",
            description="Read the status of one order.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
            handler=lambda order_id: {"status": "shipped"},
        )
    )
    agent = Agent(
        AgentDefinition(
            name="order-status-agent",
            model={"provider": "offline", "model": provider.name},
            tools=["lookup_order"],
            policies={
                "allowed_tools": ["lookup_order"],
                "max_steps": 3,
                "max_replans": 1,
                "timeout_seconds": 5,
            },
        ),
        model=provider,
        tools=tools,
        planner=ModelPlanner(provider, provider.name),
    )
    async with agent:
        async for event in agent.astream("Where is order order-123?"):
            if event.type == "plan_updated":
                print(f"Planner revised its steps: {event.data['plan']['summary']}")
            elif event.type == "completed":
                print(event.data["result"]["output"])


if __name__ == "__main__":
    asyncio.run(main())
