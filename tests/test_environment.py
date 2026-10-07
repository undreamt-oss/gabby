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
"""Environment capability descriptions and input validation."""

from __future__ import annotations

from typing import Any

import pytest

from gabby.environment import Environment
from gabby.tools import ToolRegistry


def test_environment_from_config_preserves_resources_and_tool_registry() -> None:
    resources = {"database": object()}
    tools = ToolRegistry()
    environment = Environment.from_config(
        {
            "type": "data",
            "description": "Read-only test database",
            "capabilities": ["query"],
            "resources": resources,
            "allowed_tools": ["query_database"],
        },
        tools=tools,
    )

    assert environment.type == "data"
    assert environment.resources is resources
    assert environment.tools is tools
    assert environment.allowed_tools == ["query_database"]
    assert environment.summary() == "Type: data\nRead-only test database\nCapabilities: query"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"capabilities": "read"}, "capabilities must be a list"),
        ({"allowed_tools": [1]}, "allowed_tools must be a list"),
        ({"resources": []}, "resources must be a mapping"),
        ({"type": 5}, "environment.type must be a string"),
        ({"description": 5}, "environment.description must be a string"),
    ],
)
def test_environment_rejects_invalid_capability_grants(
    config: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Environment.from_config(config)


def test_environment_summary_omits_optional_details_when_empty() -> None:
    assert Environment().summary() == "Type: generic"
