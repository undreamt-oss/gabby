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
"""Offline examples of app-composed support, research, and data agents."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gabby import (
    Agent,
    AgentDefinition,
    Environment,
    ModelResponse,
    SkillDefinition,
    Tool,
    ToolRegistry,
)


class ExampleModel:
    """Deterministic local provider that exercises one tool call and its observation."""

    name = "offline-example"

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self._requested_tool = False

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        timeout_seconds: float = 120,
    ) -> ModelResponse:
        del tools, model, temperature, timeout_seconds
        if not self._requested_tool:
            self._requested_tool = True
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "example-call-1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps(self.arguments),
                        },
                    }
                ]
            )
        observation = next(
            message["content"] for message in reversed(messages) if message["role"] == "tool"
        )
        return ModelResponse(content=f"Example result based on the application tool: {observation}")


def make_tool_agent(
    *,
    name: str,
    domain: str,
    description: str,
    tool: Tool,
    arguments: dict[str, Any],
    skill: SkillDefinition,
    resource: Any,
) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)
    environment = Environment(
        type=domain,
        description=description,
        capabilities=[tool.name],
        resources={"demo_data": resource},
        tools=registry,
        allowed_tools=[tool.name],
    )
    definition = AgentDefinition(
        name=name,
        description=description,
        model={"provider": "example", "model": "offline-script"},
        skills=[skill.name],
        tools=[tool.name],
        instructions="Use registered tools for facts. Be clear that this is sample data.",
        policies={
            "allowed_tools": [tool.name],
            "max_steps": 3,
            "timeout_seconds": 5,
        },
    )
    return Agent(
        definition,
        model=ExampleModel(tool.name, arguments),
        environment=environment,
        skill_registry={skill.name: skill},
    )


async def main() -> None:
    support_tool = Tool(
        name="lookup_order",
        description="Look up one order in the application's support system.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        handler=lambda order_id: {"order_id": order_id, "status": "shipped"},
        output_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "status": {"type": "string"}},
            "required": ["order_id", "status"],
            "additionalProperties": False,
        },
    )
    research_tool = Tool(
        name="search_research_notes",
        description="Search the application's sample research catalog.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=lambda query: [
            {"source": "sample-report.md", "snippet": f"Evidence for {query}: sample finding."}
        ],
        output_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "snippet": {"type": "string"}},
                "required": ["source", "snippet"],
                "additionalProperties": False,
            },
        },
    )
    data_tool = Tool(
        name="summarize_sales",
        description="Return a sample sales summary for one region.",
        parameters={
            "type": "object",
            "properties": {"region": {"type": "string"}},
            "required": ["region"],
            "additionalProperties": False,
        },
        handler=lambda region: {"region": region, "orders": 24, "revenue": 1820.50},
        output_schema={
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "orders": {"type": "integer"},
                "revenue": {"type": "number"},
            },
            "required": ["region", "orders", "revenue"],
            "additionalProperties": False,
        },
    )

    examples = [
        (
            "Support",
            make_tool_agent(
                name="support-example",
                domain="customer_support",
                description="Answer order questions using the support system.",
                tool=support_tool,
                arguments={"order_id": "DEMO-1042"},
                skill=SkillDefinition(
                    name="order_lookup",
                    description="Find the order before answering its status.",
                    instructions="Look up the requested order and report the returned status.",
                    tools=["lookup_order"],
                    triggers=["order", "shipping"],
                ),
                resource={"kind": "mock order service"},
            ),
            "Where is order DEMO-1042?",
        ),
        (
            "Research",
            make_tool_agent(
                name="research-example",
                domain="research",
                description="Find and summarize evidence from an application catalog.",
                tool=research_tool,
                arguments={"query": "urban tree cover"},
                skill=SkillDefinition(
                    name="evidence_search",
                    description="Search sources and retain their attribution.",
                    instructions="Use the returned source label when describing evidence.",
                    tools=["search_research_notes"],
                    triggers=["research", "evidence", "sources"],
                ),
                resource={"kind": "mock research catalog"},
            ),
            "Research evidence about urban tree cover.",
        ),
        (
            "Data",
            make_tool_agent(
                name="data-example",
                domain="data_analysis",
                description="Answer sales questions using an approved data service.",
                tool=data_tool,
                arguments={"region": "north"},
                skill=SkillDefinition(
                    name="sales_summary",
                    description="Query an approved sales dataset and explain the result.",
                    instructions="State the region and include returned totals.",
                    tools=["summarize_sales"],
                    triggers=["sales", "revenue", "region"],
                ),
                resource={"kind": "mock analytics service"},
            ),
            "Summarize sales for the north region.",
        ),
    ]

    for title, agent, task in examples:
        async with agent:
            result = await agent.arun(task)
            skill_names = [
                event.details["name"]
                for event in result.trace.events
                if event.kind == "skill_activation"
            ]
            print(f"{title} agent ({', '.join(skill_names)}): {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
