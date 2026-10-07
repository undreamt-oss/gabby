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
"""Tool schema, registry, and policy behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from gabby.tools import PolicyEngine, Tool, ToolError, ToolErrorCode, ToolRegistry


def _tool(**changes: Any) -> Tool:
    values: dict[str, Any] = {
        "name": "lookup",
        "description": "Look up a record.",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
        "handler": lambda **_: {"found": True},
        "output_schema": {"type": "object"},
    }
    values.update(changes)
    return Tool(**values)


def test_tool_rejects_invalid_definition_values() -> None:
    with pytest.raises(TypeError, match="output_schema"):
        Tool(  # type: ignore[call-arg]
            name="missing_output_schema",
            description="A tool without a declared result contract.",
            parameters={"type": "object"},
            handler=lambda: {},
        )
    with pytest.raises(ValueError, match="Tool name"):
        _tool(name="contains spaces")
    with pytest.raises(ValueError, match="description must be a string"):
        _tool(description=3)
    with pytest.raises(ValueError, match="handler must be callable"):
        _tool(handler=None)
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        _tool(timeout_seconds=float("inf"))
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        _tool(timeout_seconds=True)
    with pytest.raises(ValueError, match="max_result_bytes must be a positive integer"):
        _tool(max_result_bytes=0)
    with pytest.raises(ValueError, match="max_result_bytes must be a positive integer"):
        _tool(max_result_bytes=True)
    with pytest.raises(ValueError, match="parameters must be an object JSON Schema"):
        _tool(parameters={"type": "array"})
    with pytest.raises(ValueError, match="invalid parameters JSON Schema"):
        _tool(parameters={"type": "object", "required": "id"})
    with pytest.raises(ValueError, match="invalid output_schema JSON Schema"):
        _tool(output_schema={"type": "unknown"})
    with pytest.raises(ValueError, match="permissions must be non-empty strings"):
        _tool(permissions=[""])
    with pytest.raises(ValueError, match="sandboxed tools require a sandbox_action"):
        _tool(execution="sandboxed")
    with pytest.raises(ValueError, match="requires a non-empty sandbox_command argv"):
        _tool(handler=None, execution="sandboxed", sandbox_action="tool.execute")
    with pytest.raises(ValueError, match="max_input_bytes must be a positive integer"):
        _tool(max_input_bytes=0)


def test_tool_rejects_conflicting_execution_and_context_contracts() -> None:
    with pytest.raises(ValueError, match="handler must be callable"):
        _tool(handler=42)
    with pytest.raises(ValueError, match="choose either handler or sandbox_action"):
        _tool(sandbox_action="shell.run")
    with pytest.raises(ValueError, match="context_parameter must be a Python identifier"):
        _tool(context_parameter="not a name")
    with pytest.raises(ValueError, match="context_resources must be unique"):
        _tool(context_parameter="context", context_resources=("db", "db"))
    with pytest.raises(ValueError, match="context_resources requires context_parameter"):
        _tool(context_resources=("db",))
    with pytest.raises(ValueError, match="unknown sandbox action"):
        _tool(handler=None, execution="sandboxed", sandbox_action="host.exec")
    with pytest.raises(ValueError, match="execution must be"):
        _tool(execution="remote")
    with pytest.raises(ValueError, match="sandbox_command is only valid"):
        _tool(sandbox_command=("echo", "hello"))
    with pytest.raises(ValueError, match="requires_approval must be a boolean"):
        _tool(requires_approval=1)
    with pytest.raises(ValueError, match="parallel_safe must be a boolean"):
        _tool(parallel_safe=1)
    with pytest.raises(ValueError, match="parallel_safe requires"):
        _tool(parallel_safe=True, requires_approval=True)
    with pytest.raises(ValueError, match="max_input_bytes must be a positive integer"):
        _tool(max_input_bytes=True)
    with pytest.raises(ValueError, match="context_parameter cannot be a model input property"):
        _tool(
            context_parameter="context",
            parameters={"type": "object", "properties": {"context": {}}},
        )
    with pytest.raises(ValueError, match="output_schema must be a JSON Schema object"):
        _tool(output_schema=[])

    def handler(value: int) -> dict[str, bool]:
        return {"found": bool(value)}

    with pytest.raises(ValueError, match="handler does not accept context_parameter"):
        _tool(
            handler=handler,
            context_parameter="gabby_context",
            context_resources=("database",),
        )


def test_sandboxed_custom_tool_contract_survives_registry_snapshot() -> None:
    tool = Tool(
        name="classify_document",
        description="Classify one document.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        output_schema={"type": "object"},
        sandbox_action="tool.execute",
        execution="sandboxed",
        sandbox_command=("/usr/local/bin/classify-document", "--json-file"),
    )

    snapshot = tool.snapshot()

    assert snapshot.sandbox_command == ("/usr/local/bin/classify-document", "--json-file")
    assert snapshot.max_input_bytes == 1024 * 1024
    assert snapshot.execution == "sandboxed"


def test_tool_validates_arguments_results_and_model_schema() -> None:
    tool = _tool(
        output_schema={"type": "object", "required": ["found"]},
        permissions=["records.read"],
    )

    assert tool.validate_arguments({"id": 4}) == {"id": 4}
    assert tool.validate_result({"found": True}) == {"found": True}
    assert tool.as_model_tool()["function"]["parameters"] == tool.parameters

    with pytest.raises(ToolError, match="decode to an object") as invalid_shape:
        tool.validate_arguments([])
    assert invalid_shape.value.code is ToolErrorCode.INVALID_ARGUMENTS
    with pytest.raises(ToolError, match="Invalid arguments.*schema rule: type"):
        tool.validate_arguments({"id": "four"})
    with pytest.raises(ToolError, match="invalid result.*schema rule: required") as invalid_result:
        tool.validate_result({})
    assert invalid_result.value.code is ToolErrorCode.INVALID_RESULT


def test_tool_error_requires_and_exposes_a_stable_code() -> None:
    assert ToolError("failed").code is ToolErrorCode.EXECUTION_FAILED
    assert ToolError("failed", code=ToolErrorCode.POLICY_DENIED).code is (
        ToolErrorCode.POLICY_DENIED
    )
    with pytest.raises(ValueError, match="must be a ToolErrorCode"):
        ToolError("failed", code="policy_denied")  # type: ignore[arg-type]


def test_tool_schema_snapshot_and_tool_values_are_immutable() -> None:
    schema: dict[str, Any] = {"type": "object", "properties": {"id": {"type": "integer"}}}
    tool = _tool(parameters=schema)

    schema["properties"]["id"]["type"] = "string"
    model_schema = tool.as_model_tool()["function"]["parameters"]
    model_schema["properties"]["id"]["type"] = "boolean"

    assert tool.as_model_tool()["function"]["parameters"]["properties"]["id"]["type"] == ("integer")
    assert tool.max_result_bytes == 1024 * 1024
    assert tool.snapshot().max_result_bytes == tool.max_result_bytes
    with pytest.raises(ToolError, match="schema rule: type"):
        tool.validate_arguments({"id": "four"})
    with pytest.raises(FrozenInstanceError):
        tool.handler = None  # type: ignore[misc]


def test_tool_registry_rejects_duplicates_and_reports_missing_names() -> None:
    registry = ToolRegistry()
    tool = _tool()
    registry.register(tool)
    assert registry.get("lookup") is tool
    assert registry.names() == ["lookup"]

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)
    with pytest.raises(ToolError, match="not registered: missing") as unavailable:
        registry.get("missing")
    assert unavailable.value.code is ToolErrorCode.TOOL_UNAVAILABLE

    registry.freeze()
    with pytest.raises(RuntimeError, match="registry is frozen"):
        registry.register(_tool(name="later"))


@pytest.mark.asyncio
async def test_policy_engine_intersects_grants_and_rejects_malformed_policy() -> None:
    policy = PolicyEngine(
        ["read", "write"],
        {"allowed_tools": ["read", "write"], "allowed_permissions": ["records.read"]},
        environment_allowed=["read"],
    )
    await policy.authorize_tool("read")
    await policy.authorize_permissions("read", ["records.read"])

    with pytest.raises(ToolError, match="Policy denied tool: write") as denied:
        await policy.authorize_tool("write")
    assert denied.value.code is ToolErrorCode.POLICY_DENIED
    with pytest.raises(ToolError, match="missing permissions: records.write"):
        await policy.authorize_permissions("read", ["records.write"])
    with pytest.raises(ToolError, match="Policy denied tool: undeclared"):
        await policy.authorize_tool("undeclared")
    with pytest.raises(ValueError, match="allowed_tools must be a list"):
        PolicyEngine(["read"], {"allowed_tools": "read"})
    with pytest.raises(ValueError, match="allowed_permissions must be a list"):
        PolicyEngine(["read"], {"allowed_permissions": [1]})
