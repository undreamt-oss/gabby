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
"""Run a stateless data agent against a read-only, application-owned SQLite database."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from gabby import (
    Agent,
    AgentDefinition,
    Environment,
    ModelResponse,
    ToolRegistry,
    sqlite_query_tool,
)


class ExampleModel:
    """Deterministic model that issues one SQL call and summarizes its observation."""

    name = "sqlite-example"

    def __init__(self) -> None:
        self._queried = False

    async def complete(self, **kwargs: Any) -> ModelResponse:
        if not self._queried:
            self._queried = True
            return ModelResponse(
                tool_calls=[
                    {
                        "id": "sales-query",
                        "type": "function",
                        "function": {
                            "name": "sqlite_query",
                            "arguments": json.dumps(
                                {
                                    "sql": "SELECT region, SUM(amount) AS total FROM sales "
                                    "WHERE region = ? GROUP BY region",
                                    "parameters": ["north"],
                                }
                            ),
                        },
                    }
                ]
            )
        observation = next(
            message["content"]
            for message in reversed(kwargs["messages"])
            if message["role"] == "tool"
        )
        return ModelResponse(content=f"The approved sales database returned: {observation}")


async def run_example() -> str:
    """Create sample data, run the agent, then discard the request and database state."""
    with tempfile.TemporaryDirectory(prefix="gabby-sqlite-example-") as directory:
        database = Path(directory) / "sales.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE sales (region TEXT, amount REAL);"
                "INSERT INTO sales VALUES ('north', 15.5), ('north', 24.5), ('south', 7);"
            )

        registry = ToolRegistry()
        registry.register(sqlite_query_tool())
        environment = Environment(
            type="data",
            description="Approved sales reporting database",
            capabilities=["read-only SQL queries"],
            resources={"sqlite_database": database},
            tools=registry,
            allowed_tools=["sqlite_query"],
        )
        definition = AgentDefinition(
            name="sales-analyst",
            model={"provider": "example", "model": "offline"},
            tools=["sqlite_query"],
            policies={
                "allowed_tools": ["sqlite_query"],
                "allowed_permissions": ["database:read"],
                "max_steps": 3,
                "timeout_seconds": 5,
            },
        )
        async with Agent(definition, model=ExampleModel(), environment=environment) as agent:
            result = await agent.arun("Summarize sales in the north region.")
            return result.output


async def main() -> None:
    print(await run_example())


if __name__ == "__main__":
    asyncio.run(main())
