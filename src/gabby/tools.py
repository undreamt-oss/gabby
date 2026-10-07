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
"""Explicitly registered tools and runtime-enforced tool permissions."""

from __future__ import annotations

import inspect
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, ValidationError

from .auth import Principal


class ToolErrorCode(StrEnum):
    """Stable categories for tool failures observed by models and host consumers."""

    INVALID_ARGUMENTS = "invalid_arguments"
    POLICY_DENIED = "policy_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    TOOL_UNAVAILABLE = "tool_unavailable"
    RESULT_TOO_LARGE = "result_too_large"
    INVALID_RESULT = "invalid_result"
    EXECUTION_FAILED = "execution_failed"


class ToolError(RuntimeError):
    """A sanitized tool failure with a stable machine-readable code."""

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.EXECUTION_FAILED,
    ) -> None:
        if not isinstance(code, ToolErrorCode):
            raise ValueError("ToolError code must be a ToolErrorCode")
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CancellationToken:
    """Read-only cooperative cancellation signal for a tool invocation.

    Synchronous handlers can wait on this token while running in Gabby's callback worker thread.
    Async handlers are also cancelled by Gabby and can inspect ``is_cancelled`` during cleanup.
    """

    _event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @property
    def is_cancelled(self) -> bool:
        """Whether the runtime has cancelled or timed out the tool invocation."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block a synchronous handler until cancellation or its own timeout."""
        return self._event.wait(timeout)

    def _cancel(self) -> None:
        """Signal cancellation to the handler without exposing a public mutation method."""
        self._event.set()


@dataclass(frozen=True)
class ToolContext:
    """Host-only request context with scoped resources and opaque runtime agent identity."""

    agent_name: str
    run_id: str
    environment_type: str
    environment_description: str
    capabilities: tuple[str, ...]
    resources: Mapping[str, Any] = field(repr=False)
    principal: Principal | None = field(default=None, repr=False)
    cancellation: CancellationToken = field(default_factory=CancellationToken, repr=False)
    agent_instance_id: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Tool:
    """Declared agent action with schemas, permissions, and an explicit execution boundary."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] | None = None
    timeout_seconds: float = 30
    output_schema: dict[str, Any] = field(kw_only=True)
    permissions: tuple[str, ...] | list[str] | None = None
    sandbox_action: str | None = None
    execution: Literal["host_trusted", "sandboxed"] = "host_trusted"
    requires_approval: bool = False
    parallel_safe: bool = False
    max_result_bytes: int = 1024 * 1024
    sandbox_command: tuple[str, ...] | None = None
    max_input_bytes: int = 1024 * 1024
    context_parameter: str | None = None
    context_resources: tuple[str, ...] | list[str] = ()
    _input_validator: Draft202012Validator = field(init=False, repr=False)
    _output_validator: Draft202012Validator = field(init=False, repr=False)
    _parameters_snapshot: dict[str, Any] = field(init=False, repr=False)
    _output_schema_snapshot: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("Tool name must use 1-64 letters, digits, underscores, or hyphens")
        if not isinstance(self.description, str):
            raise ValueError(f"Tool {self.name}: description must be a string")
        if self.handler is not None and not callable(self.handler):
            raise ValueError(f"Tool {self.name}: handler must be callable")
        if self.handler is None and self.sandbox_action is None:
            raise ValueError(f"Tool {self.name}: handler must be callable for an in-process tool")
        if self.handler is not None and self.sandbox_action is not None:
            raise ValueError(f"Tool {self.name}: choose either handler or sandbox_action")
        if self.context_parameter is not None and (
            self.handler is None
            or not isinstance(self.context_parameter, str)
            or not self.context_parameter.isidentifier()
        ):
            raise ValueError(
                f"Tool {self.name}: context_parameter must be a Python identifier on a host handler"
            )
        if (
            not isinstance(self.context_resources, (tuple, list))
            or any(not isinstance(name, str) or not name for name in self.context_resources)
            or len(set(self.context_resources)) != len(self.context_resources)
        ):
            raise ValueError(f"Tool {self.name}: context_resources must be unique non-empty names")
        if self.context_resources and self.context_parameter is None:
            raise ValueError(f"Tool {self.name}: context_resources requires context_parameter")
        object.__setattr__(self, "context_resources", tuple(self.context_resources))
        if self.sandbox_action is not None and self.sandbox_action not in {
            "shell.run",
            "filesystem.read_file",
            "filesystem.write_file",
            "filesystem.list_dir",
            "python.run",
            "tool.execute",
        }:
            raise ValueError(f"Tool {self.name}: unknown sandbox action {self.sandbox_action!r}")
        if self.execution not in ("host_trusted", "sandboxed"):
            raise ValueError(f"Tool {self.name}: execution must be 'host_trusted' or 'sandboxed'")
        if (self.execution == "sandboxed") != (self.sandbox_action is not None):
            raise ValueError(
                f"Tool {self.name}: sandboxed tools require a sandbox_action; "
                "tools with sandbox_action must declare execution='sandboxed'"
            )
        if self.sandbox_action in {"tool.execute", "python.run"}:
            if (
                not isinstance(self.sandbox_command, tuple)
                or not self.sandbox_command
                or any(
                    not isinstance(value, str) or not value or "\x00" in value
                    for value in self.sandbox_command
                )
            ):
                raise ValueError(
                    f"Tool {self.name}: {self.sandbox_action} requires a non-empty "
                    "sandbox_command argv"
                )
        elif self.sandbox_command is not None:
            raise ValueError(
                f"Tool {self.name}: sandbox_command is only valid with "
                "sandbox_action='tool.execute' or 'python.run'"
            )
        if not isinstance(self.requires_approval, bool):
            raise ValueError(f"Tool {self.name}: requires_approval must be a boolean")
        if not isinstance(self.parallel_safe, bool):
            raise ValueError(f"Tool {self.name}: parallel_safe must be a boolean")
        if self.parallel_safe and (
            self.handler is None or self.execution != "host_trusted" or self.requires_approval
        ):
            raise ValueError(
                f"Tool {self.name}: parallel_safe requires a host handler without approval"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(f"Tool {self.name}: timeout_seconds must be positive")
        if (
            isinstance(self.max_result_bytes, bool)
            or not isinstance(self.max_result_bytes, int)
            or self.max_result_bytes <= 0
        ):
            raise ValueError(f"Tool {self.name}: max_result_bytes must be a positive integer")
        if (
            isinstance(self.max_input_bytes, bool)
            or not isinstance(self.max_input_bytes, int)
            or self.max_input_bytes <= 0
        ):
            raise ValueError(f"Tool {self.name}: max_input_bytes must be a positive integer")
        if not isinstance(self.parameters, dict) or self.parameters.get("type") != "object":
            raise ValueError(f"Tool {self.name}: parameters must be an object JSON Schema")
        parameters = deepcopy(self.parameters)
        properties = parameters.get("properties", {})
        if (
            self.context_parameter is not None
            and isinstance(properties, Mapping)
            and self.context_parameter in properties
        ):
            raise ValueError(
                f"Tool {self.name}: context_parameter cannot be a model input property"
            )
        object.__setattr__(self, "_parameters_snapshot", parameters)
        object.__setattr__(self, "parameters", deepcopy(parameters))
        self._validate_schema(parameters, "parameters")
        object.__setattr__(self, "_input_validator", Draft202012Validator(parameters))
        if not isinstance(self.output_schema, dict):
            raise ValueError(f"Tool {self.name}: output_schema must be a JSON Schema object")
        output_schema = deepcopy(self.output_schema)
        object.__setattr__(self, "_output_schema_snapshot", output_schema)
        object.__setattr__(self, "output_schema", deepcopy(output_schema))
        self._validate_schema(output_schema, "output_schema")
        object.__setattr__(self, "_output_validator", Draft202012Validator(output_schema))
        if self.context_parameter is not None:
            assert self.handler is not None
            try:
                handler_parameters = tuple(inspect.signature(self.handler).parameters.values())
            except (TypeError, ValueError):
                handler_parameters = ()
            accepts_named_context = any(
                parameter.name == self.context_parameter
                and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                for parameter in handler_parameters
            )
            accepts_kwargs = any(
                parameter.kind == parameter.VAR_KEYWORD for parameter in handler_parameters
            )
            if handler_parameters and not (accepts_named_context or accepts_kwargs):
                raise ValueError(
                    f"Tool {self.name}: handler does not accept context_parameter "
                    f"{self.context_parameter!r}"
                )
        if self.permissions is None:
            object.__setattr__(self, "permissions", ())
        elif any(not isinstance(value, str) or not value for value in self.permissions):
            raise ValueError(f"Tool {self.name}: permissions must be non-empty strings")
        else:
            object.__setattr__(self, "permissions", tuple(self.permissions))

    def snapshot(self) -> Tool:
        """Copy tool metadata while retaining its explicitly trusted handler reference."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=deepcopy(self._parameters_snapshot),
            handler=self.handler,
            timeout_seconds=self.timeout_seconds,
            max_result_bytes=self.max_result_bytes,
            output_schema=deepcopy(self._output_schema_snapshot),
            permissions=self.permissions,
            sandbox_action=self.sandbox_action,
            execution=self.execution,
            requires_approval=self.requires_approval,
            parallel_safe=self.parallel_safe,
            context_parameter=self.context_parameter,
            context_resources=self.context_resources,
            sandbox_command=self.sandbox_command,
            max_input_bytes=self.max_input_bytes,
        )

    def _validate_schema(self, schema: dict[str, Any], label: str) -> None:
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise ValueError(f"Tool {self.name}: invalid {label} JSON Schema: {exc}") from exc

    def validate_arguments(self, arguments: Any) -> dict[str, Any]:
        """Validate model-supplied arguments against the declared input JSON Schema."""
        if not isinstance(arguments, dict):
            raise ToolError(
                "Tool arguments must decode to an object",
                code=ToolErrorCode.INVALID_ARGUMENTS,
            )
        try:
            self._input_validator.validate(arguments)
        except ValidationError as exc:
            location = ".".join(str(item) for item in exc.absolute_path) or "<root>"
            raise ToolError(
                f"Invalid arguments for {self.name} at {location} (schema rule: {exc.validator})",
                code=ToolErrorCode.INVALID_ARGUMENTS,
            ) from exc
        return arguments

    def validate_result(self, result: Any) -> Any:
        """Validate a handler result when an output schema was declared."""
        try:
            self._output_validator.validate(result)
        except ValidationError as exc:
            location = ".".join(str(item) for item in exc.absolute_path) or "<root>"
            raise ToolError(
                f"Tool {self.name} returned an invalid result at {location} "
                f"(schema rule: {exc.validator})",
                code=ToolErrorCode.INVALID_RESULT,
            ) from exc
        return result

    def as_model_tool(self) -> dict[str, Any]:
        """Return the provider-neutral function tool declaration sent to a model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    self.description
                    if not self.requires_approval
                    else f"{self.description} Human approval is required before execution."
                ),
                "parameters": deepcopy(self._parameters_snapshot),
            },
        }


class ToolRegistry:
    """Named collection of tool contracts, frozen when resolved into an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._frozen = False

    def register(self, tool: Tool) -> None:
        """Register a unique tool unless the registry has been frozen."""
        if self._frozen:
            raise RuntimeError("Tool registry is frozen after agent construction")
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"Tool name is empty or already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def freeze(self) -> None:
        """Prevent capability changes after the registry is resolved into an agent."""
        self._frozen = True

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise ``ToolError`` for an unknown name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(
                f"Tool is not registered: {name}", code=ToolErrorCode.TOOL_UNAVAILABLE
            ) from exc

    def names(self) -> list[str]:
        """Return registered tool names in stable sorted order."""
        return sorted(self._tools)


class PolicyEngine:
    """Fail-closed tool authorization derived from an agent definition."""

    def __init__(
        self,
        declared_tools: list[str],
        policies: Mapping[str, Any],
        environment_allowed: Sequence[str] | None = None,
        allowed_permissions: Sequence[str] | None = None,
    ) -> None:
        self.declared_tools = set(declared_tools)
        configured = policies.get("allowed_tools", declared_tools)
        if (
            isinstance(configured, str)
            or not isinstance(configured, Sequence)
            or any(not isinstance(x, str) for x in configured)
        ):
            raise ValueError("policies.allowed_tools must be a list of tool names")
        self.allowed_tools = set(configured)
        if environment_allowed is not None:
            self.allowed_tools.intersection_update(environment_allowed)
        if allowed_permissions is None:
            allowed_permissions = policies.get("allowed_permissions", [])
        if (
            isinstance(allowed_permissions, str)
            or not isinstance(allowed_permissions, Sequence)
            or any(not isinstance(x, str) for x in allowed_permissions)
        ):
            raise ValueError("policies.allowed_permissions must be a list of permission names")
        self.allowed_permissions = set(allowed_permissions)

    async def authorize_tool(self, name: str) -> None:
        """Raise a policy denial when the declared tool is not granted."""
        if name not in self.declared_tools or name not in self.allowed_tools:
            raise ToolError(f"Policy denied tool: {name}", code=ToolErrorCode.POLICY_DENIED)

    async def authorize_permissions(self, name: str, permissions: list[str]) -> None:
        """Raise a policy denial when any requested permission is not granted."""
        missing = set(permissions) - self.allowed_permissions
        if missing:
            raise ToolError(
                f"Policy denied tool {name}; missing permissions: {', '.join(sorted(missing))}",
                code=ToolErrorCode.POLICY_DENIED,
            )


class PolicyEngineProtocol(Protocol):
    """Authorization operations required by the agent runtime."""

    async def authorize_tool(self, name: str) -> None:
        """Authorize one declared tool or raise a typed denial."""
        ...

    async def authorize_permissions(self, name: str, permissions: list[str]) -> None:
        """Authorize every permission required by one tool or raise a typed denial."""
        ...


class PolicyEngineFactory(Protocol):
    """Asynchronously create a run-scoped policy engine from grants and identity."""

    async def create(
        self,
        *,
        declared_tools: list[str],
        policies: Mapping[str, Any],
        environment_allowed: Sequence[str] | None,
        principal: Principal | None,
    ) -> PolicyEngineProtocol:
        """Build the run-scoped authorization engine from the active agent grants."""
        ...


@dataclass(frozen=True)
class DefaultPolicyEngineFactory:
    """Construct Gabby's built-in fail-closed tool authorization engine."""

    async def create(
        self,
        *,
        declared_tools: list[str],
        policies: Mapping[str, Any],
        environment_allowed: Sequence[str] | None,
        principal: Principal | None,
    ) -> PolicyEngineProtocol:
        """Build the default engine from the active grants, ignoring caller identity."""
        del principal  # The default policy engine uses only configured tool and permission grants.
        return PolicyEngine(declared_tools, policies, environment_allowed)
