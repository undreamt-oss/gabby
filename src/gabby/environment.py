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
"""First-class execution environment for tools, resources, and capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .tools import ToolRegistry


@dataclass
class Environment:
    """The operating context around an agent.

    Resource handles stay in the host application and are never copied into model
    context automatically. Capabilities describe the environment; concrete actions
    remain explicit registered tools.
    """

    type: str = "generic"
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    allowed_tools: list[str] | None = None

    @classmethod
    def from_config(
        cls, config: dict[str, Any], *, tools: ToolRegistry | None = None
    ) -> Environment:
        """Build an environment from validated agent configuration values."""
        capabilities = config.get("capabilities", [])
        allowed = config.get("allowed_tools")
        environment_type = config.get("type", "generic")
        description = config.get("description", "")
        resources = config.get("resources", {})
        if not isinstance(capabilities, list) or any(not isinstance(x, str) for x in capabilities):
            raise ValueError("environment.capabilities must be a list of strings")
        if not isinstance(environment_type, str):
            raise ValueError("environment.type must be a string")
        if not isinstance(description, str):
            raise ValueError("environment.description must be a string")
        if not isinstance(resources, dict):
            raise ValueError("environment.resources must be a mapping")
        if allowed is not None and (
            not isinstance(allowed, list) or any(not isinstance(x, str) for x in allowed)
        ):
            raise ValueError("environment.allowed_tools must be a list of tool names")
        return cls(
            type=environment_type,
            description=description,
            capabilities=capabilities,
            resources=resources,
            tools=tools or ToolRegistry(),
            allowed_tools=allowed,
        )

    def summary(self) -> str:
        """Return a model-facing description without serializing host resource handles."""
        values = [f"Type: {self.type}"]
        if self.description:
            values.append(self.description)
        if self.capabilities:
            values.append("Capabilities: " + ", ".join(self.capabilities))
        return "\n".join(values)


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Agent-local immutable snapshot of an environment's declarations."""

    type: str
    description: str
    capabilities: tuple[str, ...]
    resources: Mapping[str, Any]
    tools: ToolRegistry
    allowed_tools: tuple[str, ...] | None

    @classmethod
    def from_environment(cls, environment: Environment, tools: ToolRegistry) -> ResolvedEnvironment:
        """Snapshot environment declarations and bind them to an agent-local tool registry."""
        return cls(
            type=environment.type,
            description=environment.description,
            capabilities=tuple(environment.capabilities),
            resources=MappingProxyType(dict(environment.resources)),
            tools=tools,
            allowed_tools=(
                None if environment.allowed_tools is None else tuple(environment.allowed_tools)
            ),
        )

    def summary(self) -> str:
        """Return the immutable environment's model-facing description."""
        values = [f"Type: {self.type}"]
        if self.description:
            values.append(self.description)
        if self.capabilities:
            values.append("Capabilities: " + ", ".join(self.capabilities))
        return "\n".join(values)
