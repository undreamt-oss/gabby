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
"""Run one offline data-analysis request with Python inside a Docker sandbox."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from gabby import Agent, AgentDefinition, Environment, ModelResponse, ToolRegistry, python_run_tool
from gabby.config import SandboxDefinition


class ExampleModel:
    """Deterministic model that requests Python execution and reads its observation."""

    name = "python-sandbox-example"

    def __init__(self) -> None:
        self._requested = False

    async def complete(self, **kwargs: Any) -> ModelResponse:
        if not self._requested:
            self._requested = True
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "mean-calculation",
                        "type": "function",
                        "function": {
                            "name": "python_run",
                            "arguments": (
                                '{"code":"values = [3, 7, 8]\\nprint(sum(values) / len(values))"}'
                            ),
                        },
                    }
                ]
            )
        observation_json = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "tool"
        )
        observation = json.loads(observation_json)
        return ModelResponse(
            content=f"The mean of 3, 7, and 8 is {observation['stdout'].strip()} "
            f"(Python exit code {observation['exit_code']})."
        )


async def run_example() -> str:
    """Run a stateless calculation in an automatically removed container."""
    tools = ToolRegistry()
    tools.register(python_run_tool(timeout_seconds=20))
    environment = Environment(
        type="data",
        description="Bounded Python analysis in a network-disabled per-run container",
        capabilities=["sandboxed Python execution"],
        tools=tools,
        allowed_tools=["python_run"],
    )
    definition = AgentDefinition(
        name="offline-python-analyst",
        model={"provider": "example", "model": "offline"},
        tools=["python_run"],
        policies={
            "allowed_tools": ["python_run"],
            "allowed_permissions": ["sandbox:python"],
            "require_sandbox": True,
            "max_steps": 2,
            "timeout_seconds": 60,
        },
        sandbox=SandboxDefinition(
            engine="docker",
            image="python:3.14-slim",
            keepalive_argv=("python", "-c", "import time; time.sleep(3600)"),
        ),
    )
    async with Agent(definition, model=ExampleModel(), environment=environment) as agent:
        result = await agent.arun("Find the mean of 3, 7, and 8.")
        return result.output


async def main() -> None:
    print(await run_example())


if __name__ == "__main__":
    asyncio.run(main())
