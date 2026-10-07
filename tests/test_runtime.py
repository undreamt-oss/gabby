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
"""Behavioral contracts for stateless agent runs and tool enforcement."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from gabby.agent import Agent
from gabby.approval import ApprovalDecision, ApprovalRequest
from gabby.approval_sqlite import AuditedApprovalHandler, SQLiteApprovalAudit
from gabby.auth import Principal
from gabby.config import AgentDefinition, ConfigError
from gabby.environment import Environment
from gabby.models import ModelProvider, ModelResponse, ModelStreamDelta
from gabby.runtime import (
    MAX_RUN_INPUT_CHARS,
    AgentRuntimeError,
    RunRequest,
    Runtime,
    _add_usage,
    _bounded_json_dumps,
    _format_retrieved_documents,
    _invoke,
    _parse_structured_output,
    _planning_observations,
)
from gabby.skill_trust import SkillRevokedError
from gabby.tools import Tool, ToolContext, ToolError, ToolErrorCode, ToolRegistry

_ANY_JSON_SCHEMA = {"type": ["object", "array", "string", "number", "boolean", "null"]}


class FakeModel:
    """Deterministic async provider for runtime contract tests."""

    name = "fake"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class StreamingFakeModel(FakeModel):
    async def stream(self, **kwargs: Any) -> AsyncIterator[ModelStreamDelta]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if response.content:
            yield ModelStreamDelta(content_delta=response.content)


def make_agent(
    model: ModelProvider,
    *,
    tools: ToolRegistry | None = None,
    policies: dict[str, Any] | None = None,
    declared_tools: list[str] | None = None,
    approval_handler: Any | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Agent:
    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies=policies or {"max_steps": 4, "timeout_seconds": 2},
        tools=declared_tools or [],
        output_schema=output_schema,
    )
    return Agent(definition, model=model, tools=tools, approval_handler=approval_handler)


@pytest.mark.asyncio
async def test_structured_output_is_prompted_validated_and_returned_as_data() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"category": {"type": "string"}},
        "required": ["category"],
        "additionalProperties": False,
    }
    model = FakeModel([ModelResponse(content='{"category":"billing"}')])
    agent = make_agent(model, output_schema=schema)
    schema["properties"]["category"]["type"] = "number"
    schema["required"].clear()

    result = await agent.arun("Classify this request")

    assert result.output == '{"category":"billing"}'
    assert result.metadata["structured_output"] == {"category": "billing"}
    assert "Final response JSON Schema" in model.calls[0]["messages"][0]["content"]
    validation = [event for event in result.trace.events if event.kind == "output_validation"]
    assert len(validation) == 1
    assert validation[0].details["passed"] is True
    await agent.aclose()


@pytest.mark.asyncio
async def test_structured_stream_withholds_unvalidated_text_and_emits_validated_result() -> None:
    schema = {"type": "object", "required": ["answer"]}
    model = StreamingFakeModel([ModelResponse(content='{"answer":42}')])
    agent = make_agent(model, output_schema=schema)

    events = [event async for event in agent.astream("Return an answer")]

    text_events = [event for event in events if event.type == "text_delta"]
    assert len(text_events) == 1
    assert text_events[0].data["text"] == '{"answer":42}'
    completed = next(event for event in events if event.type == "completed")
    assert completed.data["result"]["metadata"]["structured_output"] == {"answer": 42}
    await agent.aclose()


@pytest.mark.asyncio
async def test_invalid_structured_output_fails_without_streaming_unvalidated_content() -> None:
    schema = {"type": "object", "required": ["answer"]}
    model = StreamingFakeModel([ModelResponse(content='{"unexpected":true}')])
    agent = make_agent(model, output_schema=schema)
    events = []

    with pytest.raises(AgentRuntimeError, match="configured output_schema"):
        async for event in agent.astream("Return an answer"):
            events.append(event)

    assert not any(event.type == "text_delta" for event in events)
    await agent.aclose()


@pytest.mark.parametrize(
    ("clock_values", "revoked", "expected_checks"),
    [
        ([2.0], False, 0),
        ([0.0, 2.0], False, 0),
        ([0.0, 0.0, 0.0, 2.0], True, 1),
    ],
)
@pytest.mark.asyncio
async def test_skill_revocation_monitor_does_not_outlive_run_deadline(
    monkeypatch: pytest.MonkeyPatch,
    clock_values: list[float],
    revoked: bool,
    expected_checks: int,
) -> None:
    class Checker:
        def __init__(self) -> None:
            self.checks = 0

        async def check_not_revoked(self, _key_ids: frozenset[str]) -> None:
            self.checks += 1
            if revoked:
                raise SkillRevokedError("publisher revoked")
            return None

    agent = make_agent(FakeModel([ModelResponse(content="unused")]))
    checker = Checker()
    agent._trusted_skill_signers = frozenset({"publisher"})
    agent.skill_revocation_checker = checker
    timestamps = iter(clock_values)

    async def no_wait(_delay: float) -> None:
        return None

    monkeypatch.setattr("gabby.runtime.time.perf_counter", lambda: next(timestamps, 2.0))
    monkeypatch.setattr("gabby.runtime.asyncio.sleep", no_wait)
    failure: asyncio.Future[Exception] = asyncio.get_running_loop().create_future()
    current = asyncio.current_task()
    assert current is not None

    await Runtime(agent)._monitor_skill_revocations(
        run_task=current,
        deadline=1.0,
        failure=failure,
    )

    assert checker.checks == expected_checks
    assert not failure.done()
    await agent.aclose()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"input": ""}, ValueError),
        ({"input": "x" * (MAX_RUN_INPUT_CHARS + 1)}, ValueError),
        ({"input": None}, TypeError),
        ({"context": []}, TypeError),
        ({"memory": {1: "bad key"}}, TypeError),
        ({"metadata": {"bad": float("nan")}}, ValueError),
    ],
)
def test_run_request_rejects_invalid_embedded_payloads(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    payload: dict[str, Any] = {"input": "valid"}
    payload.update(kwargs)
    with pytest.raises(error):
        RunRequest(**payload)


def test_run_request_snapshots_nested_caller_data() -> None:
    original_context = {"source": {"label": "before"}}
    original_memory = {"hint": ["before"]}
    original_metadata = {"request": {"tag": "before"}}

    request = RunRequest(
        input="task",
        context=original_context,
        memory=original_memory,
        metadata=original_metadata,
    )
    original_context["source"]["label"] = "after"
    original_memory["hint"].append("after")
    original_metadata["request"]["tag"] = "after"

    assert request.context == {"source": {"label": "before"}}
    assert request.memory == {"hint": ["before"]}
    assert request.metadata == {"request": {"tag": "before"}}


def test_run_request_uses_agent_request_byte_policy_for_context_snapshot() -> None:
    with pytest.raises(ValueError, match="max_model_request_bytes"):
        RunRequest(
            input="task",
            context={"payload": "x" * 256},
            max_model_request_bytes=128,
        )


def test_planning_observations_drop_bad_shapes_limit_count_and_truncate_utf8() -> None:
    calls: list[dict[str, Any]] = [{"function": {"name": f"tool-{index}"}} for index in range(17)]
    messages: list[dict[str, Any]] = [{"content": f"result-{index}"} for index in range(17)]
    calls[1] = {"function": None}
    messages[2] = {"content": ["not text"]}
    messages[-1] = {"content": "a" * 4095 + "é"}

    observations = _planning_observations(calls, messages)

    assert len(observations) == 14
    assert all(item.tool_name != "tool-0" for item in observations)
    assert observations[-1].truncated is True
    assert observations[-1].original_bytes == 4097
    assert observations[-1].content == "a" * 4095


@pytest.mark.parametrize(
    "output",
    ['{"x":1,"x":2}', '{"x":NaN}'],
)
def test_structured_output_parser_rejects_duplicate_keys_and_nonstandard_constants(
    output: str,
) -> None:
    with pytest.raises(ValueError):
        _parse_structured_output(output, ())


def test_request_usage_aggregation_ignores_invalid_and_overflowing_values() -> None:
    total: dict[str, int | float] = {"tokens": 2}

    _add_usage(
        total,
        {
            "tokens": 3,
            "": 1,
            "x" * 65: 1,
            "boolean": True,
            "negative": -1,
            "text": "4",
            "nan": float("nan"),
            "infinite": float("inf"),
        },
    )
    _add_usage({"large": 1e308}, {"large": 1e308})

    assert total == {"tokens": 5}


def test_retrieved_document_context_validates_shape_and_byte_budget() -> None:
    assert _format_retrieved_documents([], limit=2, max_context_bytes=64) == ("", [])
    with pytest.raises(AgentRuntimeError, match="invalid document collection"):
        _format_retrieved_documents((), limit=2, max_context_bytes=64)
    with pytest.raises(AgentRuntimeError, match="more documents than requested"):
        _format_retrieved_documents([object(), object()], limit=1, max_context_bytes=64)
    with pytest.raises(AgentRuntimeError, match="invalid document"):
        _format_retrieved_documents([object()], limit=1, max_context_bytes=64)


@pytest.mark.asyncio
async def test_extension_invocation_supports_sync_async_and_sync_returned_awaitables() -> None:
    async def async_callback(value: str) -> str:
        return f"async:{value}"

    async def awaitable_result() -> str:
        return "awaitable"

    assert await _invoke(lambda value: f"sync:{value}", "value") == "sync:value"
    assert await _invoke(async_callback, "value") == "async:value"
    assert await _invoke(lambda: awaitable_result()) == "awaitable"


@pytest.mark.asyncio
async def test_arun_builds_fresh_request_context_for_each_invocation() -> None:
    model = FakeModel([ModelResponse(content="first"), ModelResponse(content="second")])
    agent = make_agent(model)

    first = await agent.arun("one", context={"caller": "a"})
    second = await agent.arun("two", context={"caller": "b"})

    assert first.output == "first"
    assert second.output == "second"
    assert model.calls[0]["messages"] != model.calls[1]["messages"]
    assert "one" in model.calls[0]["messages"][-1]["content"]
    assert "two" not in str(model.calls[0]["messages"])
    assert first.trace.trace_id != second.trace.trace_id


@pytest.mark.asyncio
async def test_arun_rejects_invalid_context_before_calling_provider() -> None:
    model = FakeModel([ModelResponse(content="must not run")])
    agent = make_agent(model)

    with pytest.raises(TypeError, match="context must be a dictionary"):
        await agent.arun("task", context=["bad context"])  # type: ignore[arg-type]

    assert model.calls == []


@pytest.mark.asyncio
async def test_concurrent_runs_keep_contexts_and_traces_isolated() -> None:
    class ConcurrentModel:
        name = "concurrent-fake"

        def __init__(self) -> None:
            self.started = 0
            self.both_started = asyncio.Event()
            self.calls: list[dict[str, Any]] = []

        async def complete(self, **kwargs: Any) -> ModelResponse:
            self.calls.append(kwargs)
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=1)
            prompt = kwargs["messages"][-1]["content"]
            return ModelResponse(content=f"answer:{prompt}")

    model = ConcurrentModel()
    agent = make_agent(model)

    first, second = await asyncio.gather(
        agent.arun("request-alpha", context={"marker": "context-alpha"}),
        agent.arun("request-beta", context={"marker": "context-beta"}),
    )

    assert first.output == "answer:request-alpha"
    assert second.output == "answer:request-beta"
    assert first.trace.trace_id != second.trace.trace_id
    assert len(model.calls) == 2
    alpha_call = next(call for call in model.calls if "request-alpha" in str(call["messages"]))
    beta_call = next(call for call in model.calls if "request-beta" in str(call["messages"]))
    assert "context-alpha" in str(alpha_call["messages"])
    assert "context-beta" not in str(alpha_call["messages"])
    assert "context-beta" in str(beta_call["messages"])
    assert "context-alpha" not in str(beta_call["messages"])


@pytest.mark.asyncio
async def test_tool_arguments_are_schema_checked_before_handler_execution() -> None:
    called = False

    def handler(count: int) -> dict[str, int]:
        nonlocal called
        called = True
        return {"count": count}

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="count_items",
            description="Count items",
            handler=handler,
            parameters={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "count_items", "arguments": '{"count":"many"}'},
                    }
                ]
            )
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={
            "max_steps": 1,
            "timeout_seconds": 2,
            "allowed_tools": ["count_items"],
        },
        declared_tools=["count_items"],
    )

    with pytest.raises(AgentRuntimeError, match="max_steps"):
        await agent.arun("count")

    assert called is False


@pytest.mark.asyncio
async def test_parallel_safe_tool_batch_runs_concurrently_and_preserves_result_order() -> None:
    active = 0
    maximum_active = 0
    both_started = asyncio.Event()

    async def handler(value: str) -> str:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_started.set()
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1)
            await asyncio.sleep(0)
            return value
        finally:
            active -= 1

    registry = ToolRegistry()
    for tool_name in ("first", "second"):
        registry.register(
            Tool(
                name=tool_name,
                description=tool_name,
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                output_schema={"type": "string"},
                handler=handler,
                parallel_safe=True,
            )
        )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-first",
                        "type": "function",
                        "function": {"name": "first", "arguments": '{"value":"alpha"}'},
                    },
                    {
                        "id": "call-second",
                        "type": "function",
                        "function": {"name": "second", "arguments": '{"value":"beta"}'},
                    },
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        declared_tools=["first", "second"],
        policies={"max_steps": 3, "timeout_seconds": 2, "max_parallel_tool_calls": 2},
    )

    result = await agent.arun("Run both independent checks")

    assert result.output == "done"
    assert maximum_active == 2
    tool_messages = model.calls[1]["messages"][-2:]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-first",
        "call-second",
    ]
    await agent.aclose()


def test_parallel_safe_requires_unapproved_host_handler() -> None:
    with pytest.raises(ValueError, match="parallel_safe requires a host handler"):
        Tool(
            name="unsafe-parallel",
            description="Approval-gated operation",
            parameters={"type": "object", "properties": {}},
            output_schema={"type": "null"},
            handler=lambda: None,
            requires_approval=True,
            parallel_safe=True,
        )


@pytest.mark.asyncio
async def test_environment_resources_are_injected_only_into_opted_in_tools() -> None:
    database = object()
    observed_contexts: list[ToolContext] = []

    def query_database(timeout: int, gabby_context: ToolContext) -> dict[str, str]:
        assert timeout == 7
        observed_contexts.append(gabby_context)
        return {"status": "queried"}

    tool = Tool(
        output_schema=_ANY_JSON_SCHEMA,
        name="query_database",
        description="Query the configured database.",
        handler=query_database,
        context_parameter="gabby_context",
        context_resources=("database",),
        parameters={
            "type": "object",
            "properties": {"timeout": {"type": "integer"}},
            "required": ["timeout"],
            "additionalProperties": False,
        },
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-query",
                        "type": "function",
                        "function": {
                            "name": "query_database",
                            "arguments": '{"timeout":7}',
                        },
                    }
                ]
            ),
            ModelResponse(content="queried"),
        ]
    )
    registry = ToolRegistry()
    registry.register(tool)
    definition = AgentDefinition(
        name="data-agent",
        model={"provider": "fake", "model": "test-model"},
        environment={"type": "data", "description": "Read-only reporting database"},
        tools=["query_database"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 2,
            "allowed_tools": ["query_database"],
        },
    )
    agent = Agent(
        definition,
        model=model,
        environment=Environment(
            type="data",
            description="Read-only reporting database",
            capabilities=["query"],
            resources={"database": database, "audit": object()},
            tools=registry,
        ),
    )
    principal = Principal(subject="report-reader")

    result = await agent.arun("count reports", principal=principal)

    assert result.output == "queried"
    assert len(observed_contexts) == 1
    injected = observed_contexts[0]
    assert injected.agent_name == "data-agent"
    assert injected.run_id == result.trace.trace_id
    assert injected.environment_type == "data"
    assert injected.capabilities == ("query",)
    assert injected.resources["database"] is database
    assert tuple(injected.resources) == ("database",)
    assert injected.principal == principal
    assert "report-reader" not in repr(injected)
    assert repr(database) not in repr(injected)
    properties = tool.as_model_tool()["function"]["parameters"]["properties"]
    assert set(properties) == {"timeout"}
    with pytest.raises(ConfigError, match="unavailable environment resources: 'database'"):
        Agent(definition, model=FakeModel([]), environment=Environment(tools=registry))


@pytest.mark.asyncio
async def test_sync_tool_handler_runs_without_blocking_async_model_contract() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Return a value",
            handler=lambda value: {"value": value},
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            output_schema={"type": "object", "required": ["value"]},
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value":"ok"}'},
                    }
                ]
            ),
            ModelResponse(content="done"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={
            "max_steps": 2,
            "timeout_seconds": 2,
            "allowed_tools": ["echo"],
        },
        declared_tools=["echo"],
    )

    result = await agent.arun("echo")

    assert result.output == "done"
    assert any(
        event.kind == "tool_call" and event.details["name"] == "echo"
        for event in result.trace.events
    )


@pytest.mark.asyncio
async def test_oversized_tool_result_is_rejected_before_entering_model_context() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="large_result",
            description="Return a value that exceeds the result limit.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "é",
            max_result_bytes=3,
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-large",
                        "type": "function",
                        "function": {"name": "large_result", "arguments": "{}"},
                    }
                ]
            ),
            ModelResponse(content="handled"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["large_result"]},
        declared_tools=["large_result"],
    )

    result = await agent.arun("call the tool")

    assert result.output == "handled"
    observation = next(
        message
        for message in model.calls[1]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "call-large"
    )
    error = json.loads(observation["content"])
    assert error["error_type"] == "ToolError"
    assert error["error_code"] == "result_too_large"
    assert "max_result_bytes=3" in error["error"]
    assert "é" not in observation["content"]
    assert any(
        event.kind == "tool_error" and event.details["name"] == "large_result"
        for event in result.trace.events
    )
    assert not any(
        event.kind == "tool_call" and event.details["name"] == "large_result"
        for event in result.trace.events
    )


def test_large_nested_tool_string_is_rejected() -> None:
    with pytest.raises(ToolError) as error:
        _bounded_json_dumps(
            {"nested": ["\x00" * 100]},
            max_bytes=16,
            tool_name="large_result",
        )

    assert error.value.code is ToolErrorCode.RESULT_TOO_LARGE


def test_bounded_json_dumps_counts_json_escaped_and_unicode_string_bytes() -> None:
    shared_list = ["é", "中", "🦊"]
    shared_mapping = {"quoted": '"\\', "controls": "\x00\n", "ascii": "text"}
    value = {
        "strings": shared_list,
        "strings_again": shared_list,
        "mapping": shared_mapping,
        "mapping_again": shared_mapping,
    }

    encoded = _bounded_json_dumps(value, max_bytes=1024, tool_name="unicode_result")

    assert json.loads(encoded) == value


def test_bounded_json_dumps_rejects_escaped_string_over_byte_limit() -> None:
    with pytest.raises(ToolError) as error:
        _bounded_json_dumps("\x00" * 8, max_bytes=32, tool_name="escaped_result")

    assert error.value.code is ToolErrorCode.RESULT_TOO_LARGE


@pytest.mark.asyncio
async def test_oversized_tool_arguments_are_rejected_before_handler_invocation() -> None:
    arguments = '{"text":"é"}'
    invoked: list[dict[str, Any]] = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="bounded_input",
            description="Accept a bounded text payload.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda **value: invoked.append(value),
            max_input_bytes=len(arguments.encode("utf-8")) - 1,
            output_schema={"type": "null"},
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-bounded-input",
                        "type": "function",
                        "function": {"name": "bounded_input", "arguments": arguments},
                    }
                ]
            ),
            ModelResponse(content="input was rejected"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={
            "max_steps": 2,
            "timeout_seconds": 2,
            "allowed_tools": ["bounded_input"],
        },
        declared_tools=["bounded_input"],
    )

    result = await agent.arun("submit a payload")

    assert result.output == "input was rejected"
    assert invoked == []
    observation = next(
        message
        for message in model.calls[1]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "call-bounded-input"
    )
    error = json.loads(observation["content"])
    assert error["error_type"] == "ToolError"
    assert error["error_code"] == "invalid_arguments"
    assert error["error"] == (
        "Tool 'bounded_input' arguments exceeded max_input_bytes="
        f"{len(arguments.encode('utf-8')) - 1}"
    )


@pytest.mark.asyncio
async def test_tool_argument_json_rejects_nonstandard_numeric_constants() -> None:
    called = False

    def handler(value: float) -> float:
        nonlocal called
        called = True
        return value

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="number",
            description="Return a number.",
            handler=handler,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-nan",
                        "type": "function",
                        "function": {"name": "number", "arguments": '{"value":NaN}'},
                    }
                ]
            ),
            ModelResponse(content="handled"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["number"]},
        declared_tools=["number"],
    )

    result = await agent.arun("call number tool")

    assert result.output == "handled"
    assert called is False
    observation = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    error = json.loads(observation["content"])
    assert error["error_code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_tool_result_json_rejects_nonstandard_numeric_constants() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="number",
            description="Return a non-finite number.",
            handler=lambda: float("nan"),
            parameters={"type": "object", "properties": {}},
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-nan-result",
                        "type": "function",
                        "function": {"name": "number", "arguments": "{}"},
                    }
                ]
            ),
            ModelResponse(content="handled"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["number"]},
        declared_tools=["number"],
    )

    result = await agent.arun("call number tool")

    assert result.output == "handled"
    observation = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    error = json.loads(observation["content"])
    assert error["error_code"] == "invalid_result"
    assert "NaN" not in observation["content"]


@pytest.mark.asyncio
async def test_run_deadline_times_out_async_provider() -> None:
    class SlowModel:
        name = "slow"

        async def complete(self, **_: Any) -> ModelResponse:
            await asyncio.sleep(1)
            return ModelResponse(content="late")

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "slow", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 0.01},
    )
    agent = Agent(definition, model=SlowModel())

    with pytest.raises(AgentRuntimeError, match="deadline"):
        await agent.arun("wait")


@pytest.mark.asyncio
async def test_sync_wrapper_rejects_use_inside_running_event_loop() -> None:
    agent = make_agent(FakeModel([ModelResponse(content="unused")]))

    with pytest.raises(RuntimeError, match="Agent.arun"):
        agent.run("nested")
    with pytest.raises(RuntimeError, match="await aclose"):
        agent.close()


@pytest.mark.asyncio
async def test_sensitive_tool_requires_host_approval_before_execution() -> None:
    called = False
    requests: list[ApprovalRequest] = []

    class Approver:
        async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
            requests.append(request)
            return ApprovalDecision(approved=True)

    def handler(destination: str) -> dict[str, str]:
        nonlocal called
        called = True
        return {"sent_to": destination}

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="send_report",
            description="Send a report.",
            parameters={
                "type": "object",
                "properties": {"destination": {"type": "string"}},
                "required": ["destination"],
                "additionalProperties": False,
            },
            handler=handler,
            requires_approval=True,
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-sensitive",
                        "type": "function",
                        "function": {
                            "name": "send_report",
                            "arguments": '{"destination":"ops@example.test"}',
                        },
                    }
                ]
            ),
            ModelResponse(content="sent"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["send_report"]},
        declared_tools=["send_report"],
        approval_handler=Approver(),
    )

    result = await agent.arun("send the report", principal=Principal("test-user"))

    assert called
    assert requests[0].tool_name == "send_report"
    assert requests[0].arguments == {"destination": "ops@example.test"}
    assert requests[0].principal == Principal("test-user")
    assert "test-user" not in str(model.calls[0]["messages"])
    assert any(
        event.kind == "approval" and event.details["approved"] for event in result.trace.events
    )
    assert "approval" in model.calls[0]["tools"][0]["function"]["description"]


@pytest.mark.asyncio
async def test_sensitive_tool_without_approval_handler_is_denied() -> None:
    called = False

    def handler() -> dict[str, bool]:
        nonlocal called
        called = True
        return {"sent": True}

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="send_report",
            description="Send a report.",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            requires_approval=True,
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-sensitive",
                        "type": "function",
                        "function": {"name": "send_report", "arguments": "{}"},
                    }
                ]
            ),
            ModelResponse(content="request denied"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["send_report"]},
        declared_tools=["send_report"],
    )

    result = await agent.arun("send the report")

    assert not called
    observation = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    assert json.loads(observation["content"]) == {
        "error": "Tool requires approval, but no approval handler is configured",
        "error_type": "ToolError",
        "error_code": "approval_unavailable",
    }
    assert not any(event.kind == "approval" for event in result.trace.events)


@pytest.mark.asyncio
async def test_sqlite_audit_is_committed_before_approved_tool_execution(tmp_path: Path) -> None:
    database = tmp_path / "approval.sqlite3"
    audit = SQLiteApprovalAudit(database)

    async def review(_: ApprovalRequest) -> bool:
        return True

    def send_report() -> dict[str, bool]:
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT approved FROM gabby_tool_approval_audit WHERE call_id = 'call-audit'"
            ).fetchone()
        assert row == (1,)
        return {"sent": True}

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="send_report",
            description="Send a report.",
            parameters={"type": "object", "properties": {}},
            handler=send_report,
            requires_approval=True,
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-audit",
                        "type": "function",
                        "function": {"name": "send_report", "arguments": "{}"},
                    }
                ]
            ),
            ModelResponse(content="report sent"),
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={"max_steps": 2, "timeout_seconds": 5, "allowed_tools": ["send_report"]},
        declared_tools=["send_report"],
        approval_handler=AuditedApprovalHandler(review, audit),
    )
    try:
        result = await agent.arun("send the report")
    finally:
        await agent.aclose()

    assert result.output == "report sent"
    record = (await audit.list_records())[0]
    assert record.run_id == result.trace.trace_id
    assert record.call_id == "call-audit"
    assert record.approved


@pytest.mark.asyncio
async def test_cancelled_approval_wait_never_invokes_sensitive_tool() -> None:
    approval_started = asyncio.Event()
    approval_cancelled = asyncio.Event()
    tool_executed = False

    class WaitingApproval:
        async def approve(self, _: ApprovalRequest) -> ApprovalDecision:
            approval_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                approval_cancelled.set()
            return ApprovalDecision(approved=True)

    def handler() -> dict[str, bool]:
        nonlocal tool_executed
        tool_executed = True
        return {"executed": True}

    registry = ToolRegistry()
    registry.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="commit_change",
            description="Commit the prepared change.",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            requires_approval=True,
        )
    )
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    {
                        "id": "call-commit",
                        "type": "function",
                        "function": {"name": "commit_change", "arguments": "{}"},
                    }
                ]
            )
        ]
    )
    agent = make_agent(
        model,
        tools=registry,
        policies={
            "max_steps": 2,
            "timeout_seconds": 30,
            "allowed_tools": ["commit_change"],
        },
        declared_tools=["commit_change"],
        approval_handler=WaitingApproval(),
    )
    run = asyncio.create_task(agent.arun("commit the change"))

    await asyncio.wait_for(approval_started.wait(), timeout=1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await asyncio.wait_for(approval_cancelled.wait(), timeout=1)

    assert not tool_executed
