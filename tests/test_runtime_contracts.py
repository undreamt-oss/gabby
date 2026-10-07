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
"""Runtime context, retrieval, verification, and failure contracts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest

from gabby.agent import Agent
from gabby.auth import Principal
from gabby.config import AgentDefinition, ConfigError, SandboxDefinition, SkillDefinition
from gabby.knowledge import Document
from gabby.models import (
    ModelRequestSizeError,
    ModelResponse,
    ModelResponseSizeError,
    ModelStreamDelta,
    RetryableModelError,
)
from gabby.planning import ExecutionPlan, ModelPlanner, PlanningResult, PlanStep
from gabby.runtime import AgentRuntimeError, RunRequest, Runtime
from gabby.sandbox import SandboxOutputLimit, SandboxTimeout, SandboxUnavailable
from gabby.skill_selection import ModelSkillSelector, SkillActivation, SkillSelection
from gabby.tools import Tool, ToolContext, ToolError, ToolErrorCode, ToolRegistry
from gabby.verification import VerificationResult

_ANY_JSON_SCHEMA = {"type": ["object", "array", "string", "number", "boolean", "null"]}


class SequenceModel:
    name = "sequence"

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _agent(
    model: Any,
    *,
    policies: dict[str, Any] | None = None,
    knowledge: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    retriever: Any = None,
    verifier: Any = None,
    tools: ToolRegistry | None = None,
    tool_names: list[str] | None = None,
    skills: list[SkillDefinition] | None = None,
    skill_selector: Any = None,
    planner: Any = None,
    tracer: Any = None,
    policy_engine_factory: Any = None,
    global_instructions: str = "",
) -> Agent:
    definition = AgentDefinition(
        name="contract-agent",
        description="A contract test agent",
        model={"provider": "fake", "model": "model-v1", "temperature": 0.2},
        instructions="Keep caller data untrusted.",
        policies=policies or {"max_steps": 4, "timeout_seconds": 1},
        knowledge=knowledge or {},
        verification=verification or {},
        tools=tool_names or [],
        skills=[] if skills is None else [skill.name for skill in skills],
    )
    return Agent(
        definition,
        model=model,
        retriever=retriever,
        verifier=verifier,
        tools=tools,
        skill_registry={} if skills is None else {skill.name: skill for skill in skills},
        skill_selector=skill_selector,
        planner=planner,
        tracer=tracer,
        policy_engine_factory=policy_engine_factory,
        global_instructions=global_instructions,
    )


def _tool_call(name: str, arguments: str = "{}", call_id: str = "call-1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.asyncio
async def test_retrieval_context_and_selected_skill_reach_the_model() -> None:
    class FakeRetriever:
        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> list[Document]:
            assert query == "auth failure"
            assert limit == 2
            return [Document(text="Use the identity runbook.", source="runbook.md")]

    class FakeVerifier:
        async def verify(self, *, request: Any, output: str, trace: Any) -> VerificationResult:
            assert request.input == "auth failure"
            assert output == "checked"
            assert trace.events[-1].kind == "verification"
            return VerificationResult(
                passed=True,
                method="fixture",
                details={"checks": 2},
                evidence=("runbook.md",),
            )

    model = SequenceModel(ModelResponse(content="checked", usage={"total_tokens": 9}))
    agent = _agent(
        model,
        global_instructions="Global requirement: cite sources.",
        knowledge={"sources": ["./docs"], "top_k": 2},
        verification={"enabled": True},
        retriever=FakeRetriever(),
        verifier=FakeVerifier(),
        skills=[
            SkillDefinition(
                name="always-on",
                instructions="Always apply this procedure.",
                examples="Use the source label when citing evidence.",
                triggers=[],
                constraints=["Do not expose secrets."],
                verification=["Check the result."],
            ),
            SkillDefinition(name="authentication", triggers=["auth"]),
            SkillDefinition(
                name="unrelated",
                triggers=["billing"],
                examples="Inactive examples must stay out of this run.",
            ),
        ],
    )
    result = await agent.arun(
        "auth failure", context={"tenant": "local"}, memory={"hint": "short-lived"}
    )

    first = model.calls[0]["messages"][0]["content"]
    all_content = "\n".join(message.get("content", "") for message in model.calls[0]["messages"])
    assert "Agent purpose" in first
    assert "Agent instructions" in first
    assert "Global requirement: cite sources." in first
    assert "Skill: always-on" in first and "Skill: authentication" in first
    assert "Use the source label when citing evidence." in first
    assert "Skill: unrelated" not in first
    assert "Inactive examples must stay out of this run." not in first
    assert "untrusted reference data" in all_content and "runbook.md" in all_content
    assert first.index("Global instructions") < first.index("Agent instructions")
    assert first.index("Agent instructions") < first.index("Skill: always-on")
    assert '"tenant": "local"' in all_content and '"hint": "short-lived"' in all_content
    assert result.metadata["usage"] == {"total_tokens": 9}
    assert result.metadata["verification"] == {
        "passed": True,
        "method": "fixture",
        "details": {"checks": 2},
        "evidence": ["runbook.md"],
    }
    verification_event = next(
        event for event in result.trace.events if event.kind == "verification"
    )
    assert verification_event.details["result"] == result.metadata["verification"]
    assert [event.kind for event in result.trace.events].count("skill_activation") == 2
    assert any(event.kind == "retrieval" for event in result.trace.events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("documents", "knowledge", "message"),
    [
        (
            [Document(text="one", source="one"), Document(text="two", source="two")],
            {"top_k": 1},
            "more documents than requested",
        ),
        (
            [Document(text="fact " * 20, source="source")],
            {"max_context_bytes": 96},
            "max_context_bytes",
        ),
        ("not-a-document-list", {}, "invalid document collection"),
        ([object()], {}, "invalid document"),
    ],
)
async def test_retrieval_results_are_validated_and_bounded(
    documents: object,
    knowledge: dict[str, Any],
    message: str,
) -> None:
    class UntrustedRetriever:
        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> Any:
            return documents

    model = SequenceModel(ModelResponse(content="should not be called"))
    agent = _agent(model, knowledge=knowledge, retriever=UntrustedRetriever())

    with pytest.raises(AgentRuntimeError, match=message):
        await agent.arun("find relevant knowledge")

    assert model.calls == []


@pytest.mark.asyncio
async def test_optional_planner_is_advisory_and_records_capabilities_and_usage() -> None:
    class FakePlanner:
        async def plan(self, **kwargs: Any) -> PlanningResult:
            self.kwargs = kwargs
            return PlanningResult(
                plan=ExecutionPlan(
                    summary="Review the request.",
                    steps=(PlanStep("Inspect context.", "Relevant facts are listed."),),
                ),
                model="planner-v1",
                usage={"total_tokens": 4},
            )

    planner = FakePlanner()
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="lookup",
            description="Look up an approved record.",
            handler=lambda: {"status": "found"},
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(ModelResponse(content="The record was found.", usage={"total_tokens": 9}))
    agent = _agent(
        model,
        tools=tools,
        tool_names=["lookup"],
        skills=[SkillDefinition(name="triage", description="Triage a report.")],
        planner=planner,
    )

    result = await agent.arun("Check record status.")

    assert [capability.name for capability in planner.kwargs["capabilities"]] == [
        "triage",
        "lookup",
    ]
    assert result.output == "The record was found."
    assert result.metadata["usage"] == {"total_tokens": 13}
    plan_message = next(
        message
        for message in model.calls[0]["messages"]
        if "Optional model-generated" in message.get("content", "")
    )
    assert "Review the request." in plan_message["content"]
    planning_event = next(event for event in result.trace.events if event.kind == "planning")
    assert planning_event.details["model"] == "planner-v1"
    assert planning_event.details["usage"] == {"total_tokens": 4}


@pytest.mark.asyncio
async def test_model_planner_executes_through_tool_observation_to_final_response() -> None:
    summary = "Check the record and report its status."
    plan = {
        "summary": summary,
        "steps": [
            {
                "instruction": "Look up the requested record.",
                "success_criteria": "The record status is available.",
            },
            {
                "instruction": "Summarize the observed status.",
                "success_criteria": "The response reports the returned status.",
            },
        ],
    }
    model = SequenceModel(
        ModelResponse(
            content=json.dumps(plan),
            usage={"total_tokens": 3},
        ),
        ModelResponse(
            tool_calls=[_tool_call("lookup", '{"record_id":"r-17"}')],
            usage={"total_tokens": 5},
        ),
        ModelResponse(content="Record r-17 is active.", usage={"total_tokens": 7}),
    )
    observed: list[str] = []

    def lookup_record(record_id: str) -> dict[str, str]:
        observed.append(record_id)
        return {"status": "active"}

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="lookup",
            description="Look up one approved record.",
            parameters={
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
            output_schema=_ANY_JSON_SCHEMA,
            handler=lookup_record,
        )
    )
    planner = ModelPlanner(model, "model-v1")
    agent = _agent(
        model,
        tools=tools,
        tool_names=["lookup"],
        planner=planner,
        policies={"max_steps": 3, "timeout_seconds": 2},
    )

    result = await agent.arun("Is record r-17 active?")

    assert observed == ["r-17"]
    assert result.output == "Record r-17 is active."
    assert result.metadata["usage"] == {"total_tokens": 15}
    assert len(model.calls) == 3
    reasoning_messages = model.calls[1]["messages"]
    plan_message = next(
        message
        for message in reasoning_messages
        if isinstance(message.get("content"), str)
        and "Optional model-generated execution plan" in message["content"]
    )
    plan_text = plan_message.get("content")
    assert isinstance(plan_text, str)
    assert summary in plan_text
    final_messages = model.calls[2]["messages"]
    tool_observation = next(message for message in final_messages if message.get("role") == "tool")
    assert json.loads(tool_observation["content"]) == {"status": "active"}
    trace_kinds = [event.kind for event in result.trace.events]
    assert "planning" in trace_kinds
    assert "tool_call" in trace_kinds


@pytest.mark.asyncio
async def test_model_planner_replans_from_bounded_tool_observations() -> None:
    initial_plan = {
        "summary": "Check the record.",
        "steps": [
            {
                "instruction": "Look up the requested record.",
                "success_criteria": "A record status is returned.",
            }
        ],
    }
    revised_plan = {
        "summary": "Report the verified record status.",
        "steps": [
            {
                "instruction": "Summarize the returned status.",
                "success_criteria": "The answer matches the tool observation.",
            }
        ],
    }
    model = SequenceModel(
        ModelResponse(content=json.dumps(initial_plan), usage={"total_tokens": 2}),
        ModelResponse(tool_calls=[_tool_call("lookup", '{"record_id":"r-17"}')]),
        ModelResponse(content=json.dumps(revised_plan), usage={"total_tokens": 3}),
        ModelResponse(content="Record r-17 is active."),
    )
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="lookup",
            description="Look up one approved record.",
            parameters={
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
            output_schema=_ANY_JSON_SCHEMA,
            handler=lambda record_id: {"status": "active", "evidence": "x" * 5000},
        )
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["lookup"],
        planner=ModelPlanner(model, "model-v1"),
        policies={"max_steps": 3, "max_replans": 1, "timeout_seconds": 2},
    )

    events = [event async for event in agent.astream("Is record r-17 active?")]

    assert events[-1].type == "completed"
    assert events[-1].data["result"]["output"] == "Record r-17 is active."
    assert len(model.calls) == 4
    replan_request = json.loads(model.calls[2]["messages"][-1]["content"])
    assert replan_request["previous_plan"]["summary"] == "Check the record."
    observation = replan_request["observations"][0]
    assert observation["tool_name"] == "lookup"
    assert observation["truncated"] is True
    assert observation["original_bytes"] > 4096
    assert len(observation["content"].encode("utf-8")) <= 4096
    trace_events = events[-1].data["result"]["trace"]["events"]
    assert any(event["kind"] == "replanning" for event in trace_events)
    assert any(event.type == "plan_updated" for event in events)


def test_replanning_requires_an_injected_planner() -> None:
    with pytest.raises(ConfigError, match="max_replans requires an injected planner"):
        _agent(SequenceModel(), policies={"max_steps": 2, "max_replans": 1})


@pytest.mark.asyncio
async def test_model_planner_receives_global_and_agent_instructions() -> None:
    model = SequenceModel(
        ModelResponse(
            content=json.dumps(
                {
                    "summary": "Review the task.",
                    "steps": [
                        {
                            "instruction": "Inspect the request.",
                            "success_criteria": "Facts checked.",
                        }
                    ],
                }
            )
        ),
        ModelResponse(content="Done."),
    )
    agent = _agent(
        model,
        global_instructions="Global planning rule: keep sensitive data private.",
        planner=ModelPlanner(model, "planner-model"),
    )

    await agent.arun("Summarize this request.")

    planner_system = model.calls[0]["messages"][0]["content"]
    assert "Global planning rule: keep sensitive data private." in planner_system
    assert "Keep caller data untrusted." in planner_system
    assert planner_system.index("Global instructions") < planner_system.index("Agent instructions")
    assert "runtime constraints above" in planner_system


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("planner_mode", "message"),
    [
        ("timeout", "Planning exceeded the run deadline"),
        ("request_size", "request was oversized"),
        ("response_size", "response was oversized"),
        ("failure", "Planning failed"),
        ("invalid_result", "invalid result"),
        ("invalid_metadata", "invalid execution metadata"),
        ("invalid_plan", "invalid plan"),
    ],
)
async def test_optional_planner_failures_stop_before_the_reasoning_model(
    planner_mode: str, message: str
) -> None:
    class FailingPlanner:
        async def plan(self, **_: Any) -> Any:
            if planner_mode == "timeout":
                raise TimeoutError
            if planner_mode == "request_size":
                raise ModelRequestSizeError("planning request was oversized")
            if planner_mode == "response_size":
                raise ModelResponseSizeError("planning response was oversized")
            if planner_mode == "failure":
                raise RuntimeError("private planner details")
            if planner_mode == "invalid_result":
                return {"plan": "not typed"}
            if planner_mode == "invalid_metadata":
                return PlanningResult(
                    plan=ExecutionPlan("Summary.", (PlanStep("Do it.", "Done."),)),
                    model="",
                )
            return PlanningResult(
                plan=ExecutionPlan(" ", ()),
                model=None,
            )

    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(model, planner=FailingPlanner())
    with pytest.raises(AgentRuntimeError, match=message):
        await agent.arun("task")
    assert model.calls == []


@pytest.mark.asyncio
async def test_agent_streams_text_and_completion_events() -> None:
    class StreamingModel:
        name = "streaming"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="Hello")
            yield ModelStreamDelta(content_delta=" world")
            yield ModelStreamDelta(finish_reason="stop")

    agent = _agent(StreamingModel())
    events = [event async for event in agent.astream("say hello")]

    assert events[0].type == "run_started"
    assert [event.data["text"] for event in events if event.type == "text_delta"] == [
        "Hello",
        " world",
    ]
    assert events[-1].type == "completed"
    assert events[-1].data["result"]["output"] == "Hello world"


@pytest.mark.asyncio
async def test_runtime_retries_explicitly_transient_completion_within_deadline() -> None:
    class FlakyModel:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **_: Any) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                raise RetryableModelError("upstream busy", retry_after_seconds=0)
            return ModelResponse(content="recovered")

    model = FlakyModel()
    agent = _agent(
        model,
        policies={"max_steps": 1, "timeout_seconds": 1, "max_model_retries": 1},
    )

    result = await agent.arun("respond")

    assert result.output == "recovered"
    assert model.calls == 2
    retry = next(event for event in result.trace.events if event.kind == "model_retry")
    assert retry.details == {
        "attempt": 2,
        "purpose": "reasoning",
        "delay_ms": 0,
        "error_type": "RetryableModelError",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("retries", "expected_calls"), [(0, 1), (2, 3)])
async def test_runtime_bounds_retry_count_and_defaults_to_no_retries(
    retries: int, expected_calls: int
) -> None:
    class AlwaysBusyModel:
        name = "always-busy"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **_: Any) -> ModelResponse:
            self.calls += 1
            raise RetryableModelError("provider busy", retry_after_seconds=0)

    model = AlwaysBusyModel()
    agent = _agent(
        model,
        policies={"max_steps": 1, "timeout_seconds": 1, "max_model_retries": retries},
    )

    with pytest.raises(RetryableModelError):
        await agent.arun("respond")

    assert model.calls == expected_calls


@pytest.mark.asyncio
async def test_runtime_retry_policy_covers_planning_stage() -> None:
    class FlakyPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def plan(self, **_: Any) -> PlanningResult:
            self.calls += 1
            if self.calls == 1:
                raise RetryableModelError("planner upstream busy", retry_after_seconds=0)
            return PlanningResult(ExecutionPlan("answer", (PlanStep("answer", "respond"),)))

    planner = FlakyPlanner()
    model = SequenceModel(ModelResponse(content="planned answer"))
    agent = _agent(
        model,
        planner=planner,
        policies={"max_steps": 1, "timeout_seconds": 1, "max_model_retries": 1},
    )

    result = await agent.arun("respond")

    assert result.output == "planned answer"
    assert planner.calls == 2
    retry = next(event for event in result.trace.events if event.kind == "model_retry")
    assert retry.details["error_type"] == "RetryableModelError"
    assert retry.details["purpose"] == "planning"


@pytest.mark.asyncio
async def test_stream_retries_before_output_but_never_replays_partial_text() -> None:
    class FlakyStream:
        name = "flaky-stream"

        def __init__(self, *, fail_after_text: bool) -> None:
            self.calls = 0
            self.fail_after_text = fail_after_text

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                if self.fail_after_text:
                    yield ModelStreamDelta(content_delta="partial")
                raise RetryableModelError("temporarily unavailable", retry_after_seconds=0)
            yield ModelStreamDelta(content_delta="complete")

    before_output = FlakyStream(fail_after_text=False)
    agent = _agent(
        before_output,
        policies={"max_steps": 1, "timeout_seconds": 1, "max_model_retries": 1},
    )
    events = [event async for event in agent.astream("respond")]
    assert before_output.calls == 2
    assert [event.data["text"] for event in events if event.type == "text_delta"] == ["complete"]
    retry_event = next(event for event in events if event.type == "model_retry")
    assert retry_event.data["purpose"] == "reasoning"

    after_output = FlakyStream(fail_after_text=True)
    agent = _agent(
        after_output,
        policies={"max_steps": 1, "timeout_seconds": 1, "max_model_retries": 1},
    )
    events = []
    with pytest.raises(RetryableModelError):
        async for event in agent.astream("respond"):
            events.append(event)
    assert after_output.calls == 1
    assert [event.data["text"] for event in events if event.type == "text_delta"] == ["partial"]
    assert not any(event.type == "model_retry" for event in events)


@pytest.mark.asyncio
async def test_agent_stream_drains_a_full_event_queue_before_finishing() -> None:
    class ManyDeltasModel:
        name = "many-deltas"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            for _delta_index in range(63):
                yield ModelStreamDelta(content_delta="x")

    agent = _agent(ManyDeltasModel())
    events = [event async for event in agent.astream("emit many deltas")]

    assert len([event for event in events if event.type == "text_delta"]) == 63
    assert events[-1].type == "completed"


@pytest.mark.asyncio
async def test_agent_stream_cancellation_stops_execution() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowStreamingModel:
        name = "slow-streaming"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            yield ModelStreamDelta(content_delta="unreachable")

    agent = _agent(SlowStreamingModel())
    stream = agent.astream("wait")
    assert (await anext(stream)).type == "run_started"
    await started.wait()
    await stream.aclose()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_agent_stream_assembles_tool_deltas_and_reports_tool_progress() -> None:
    class ToolStreamingModel:
        name = "tool-streaming"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamDelta(tool_call_index=0, tool_call_id="call-1")
                yield ModelStreamDelta(tool_call_index=0, tool_name_delta="echo")
                yield ModelStreamDelta(tool_call_index=0, tool_arguments_delta='{"value":')
                yield ModelStreamDelta(tool_call_index=0, tool_arguments_delta='"hello"}')
                yield ModelStreamDelta(finish_reason="tool_calls")
            else:
                yield ModelStreamDelta(content_delta="done")

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="echo",
            description="Echo a value.",
            handler=lambda value: value,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    )
    model = ToolStreamingModel()
    agent = _agent(model, tools=tools, tool_names=["echo"])
    events = [event async for event in agent.astream("echo hello")]

    assert model.calls == 2
    assert [event.type for event in events if event.type.startswith("tool_")] == [
        "tool_started",
        "tool_completed",
    ]
    assert events[-1].data["result"]["output"] == "done"


@pytest.mark.asyncio
async def test_agent_stream_tool_failure_includes_stable_error_code() -> None:
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("echo", arguments="not-json")]),
        ModelResponse(content="handled"),
    )
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="echo",
            description="Echo a value.",
            handler=lambda value: value,
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["echo"],
        policies={"max_steps": 2, "timeout_seconds": 1, "allowed_tools": ["echo"]},
    )

    events = [event async for event in agent.astream("call echo")]

    failed = next(event for event in events if event.type == "tool_failed")
    assert failed.data["error_code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_selected_skill_activates_dependencies_even_when_their_triggers_do_not_match() -> (
    None
):
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="dependency_tool",
            description="A tool required by the dependency skill.",
            handler=lambda: "ready",
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(ModelResponse(content="ready"))
    agent = _agent(
        model,
        tools=tools,
        skills=[
            SkillDefinition(
                name="dependency",
                instructions="Apply the dependency procedure.",
                triggers=["unmatched-trigger"],
                tools=["dependency_tool"],
            ),
            SkillDefinition(
                name="parent",
                instructions="Apply the parent procedure.",
                triggers=["upgrade"],
                dependencies=["dependency"],
            ),
        ],
    )

    result = await agent.arun("upgrade this package")

    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Apply the dependency procedure." in system_prompt
    assert "Apply the parent procedure." in system_prompt
    assert [tool["function"]["name"] for tool in model.calls[0]["tools"]] == ["dependency_tool"]
    dependency_event = next(
        event
        for event in result.trace.events
        if event.kind == "skill_activation" and event.details["name"] == "dependency"
    )
    assert dependency_event.details["method"] == "dependency"


@pytest.mark.asyncio
async def test_injected_skill_selector_controls_activations() -> None:
    class Selector:
        async def select(self, *, task: str, skills: Any) -> SkillSelection:
            assert task == "choose a capability"
            assert [skill.name for skill in skills] == ["review", "billing"]
            return SkillSelection((SkillActivation("billing", "custom_policy"),))

    model = SequenceModel(ModelResponse(content="selected"))
    agent = _agent(
        model,
        skills=[
            SkillDefinition(name="review", triggers=["review"]),
            SkillDefinition(name="billing", triggers=["invoice"]),
        ],
        skill_selector=Selector(),
    )

    result = await agent.arun("choose a capability")

    prompt = model.calls[0]["messages"][0]["content"]
    assert "Skill: billing" in prompt
    assert "Skill: review" not in prompt
    activation = next(event for event in result.trace.events if event.kind == "skill_activation")
    assert activation.details == {
        "name": "billing",
        "version": "0.1.0",
        "method": "custom_policy",
    }


@pytest.mark.asyncio
async def test_model_skill_selector_records_usage_and_runtime_expands_dependencies() -> None:
    model = SequenceModel(
        ModelResponse(
            content='{"skills":["parent"]}',
            usage={"prompt_tokens": 11, "completion_tokens": 2},
        ),
        ModelResponse(content="selected", usage={"prompt_tokens": 7, "completion_tokens": 1}),
    )
    agent = _agent(
        model,
        global_instructions="Global selection guidance: prefer security review.",
        skills=[
            SkillDefinition(name="review", description="Review code."),
            SkillDefinition(
                name="dependency",
                description="Analyze dependencies.",
                dependencies=["review"],
            ),
            SkillDefinition(
                name="parent",
                description="Upgrade one dependency.",
                dependencies=["dependency"],
            ),
        ],
        skill_selector=ModelSkillSelector(model, "selector-model"),
    )

    result = await agent.arun("Upgrade the networking dependency after this task text.")

    assert len(model.calls) == 2
    selector_call, reasoning_call = model.calls
    assert selector_call["model"] == "selector-model"
    assert selector_call["tools"] == []
    assert selector_call["temperature"] == 0
    selector_system = selector_call["messages"][0]["content"]
    assert "Global selection guidance: prefer security review." in selector_system
    assert "Keep caller data untrusted." in selector_system
    assert selector_system.index("Global instructions") < selector_system.index(
        "Agent instructions"
    )
    assert "networking dependency" in selector_call["messages"][1]["content"]
    prompt = reasoning_call["messages"][0]["content"]
    assert "Skill: parent" in prompt
    assert "Skill: dependency" in prompt
    assert "Skill: review" in prompt
    events = [event for event in result.trace.events if event.kind == "skill_activation"]
    assert [(event.details["name"], event.details["method"]) for event in events] == [
        ("review", "dependency"),
        ("dependency", "dependency"),
        ("parent", "model_selector"),
    ]
    selection_event = next(
        event for event in result.trace.events if event.kind == "skill_selection"
    )
    selection_model_call = next(
        event
        for event in result.trace.events
        if event.kind == "model_call" and event.details.get("purpose") == "skill_selection"
    )
    assert selection_model_call.details["step"] == 0
    assert selection_model_call.details["model"] == "selector-model"
    assert selection_model_call.duration_ms is not None
    assert selection_event.details["model"] == "selector-model"
    assert selection_event.details["usage"] == {"prompt_tokens": 11, "completion_tokens": 2}
    assert selection_event.duration_ms is not None
    assert result.metadata["usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 3,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", '{"skills":["not-configured"]}'])
async def test_model_skill_selector_rejects_invalid_provider_output(content: str) -> None:
    model = SequenceModel(ModelResponse(content=content))
    agent = _agent(
        model,
        skills=[SkillDefinition(name="review", description="Review code.")],
        skill_selector=ModelSkillSelector(model, "selector-model"),
    )

    with pytest.raises(AgentRuntimeError, match="Skill selection failed"):
        await agent.arun("review")

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_model_skill_selector_obeys_the_agent_run_deadline() -> None:
    class SlowProvider:
        name = "slow-selector"

        def __init__(self) -> None:
            self.cancelled = False

        async def complete(self, **_: Any) -> ModelResponse:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return ModelResponse(content='{"skills":[]}')

    model = SlowProvider()
    agent = _agent(
        model,
        policies={"max_steps": 4, "timeout_seconds": 0.02},
        skills=[SkillDefinition(name="review", description="Review code.")],
        skill_selector=ModelSkillSelector(model, "selector-model", timeout_seconds=2),
    )

    with pytest.raises(AgentRuntimeError, match="Skill selection exceeded the run deadline"):
        await agent.arun("review this")

    assert model.cancelled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_id": ""}, "model_id"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_model_request_bytes": True}, "max_model_request_bytes"),
        ({"max_model_response_bytes": 0}, "max_model_response_bytes"),
    ],
)
def test_model_skill_selector_rejects_invalid_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    model = SequenceModel(ModelResponse(content='{"skills":[]}'))
    values: dict[str, Any] = {"model_id": "selector-model"}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ModelSkillSelector(model, **values)


@pytest.mark.asyncio
async def test_model_skill_selector_skips_provider_when_no_skills_are_configured() -> None:
    model = SequenceModel()
    agent = _agent(model)
    selector = ModelSkillSelector(model, "selector-model")

    selection = await selector.select(task="anything", skills=agent.skills)

    assert selection.activations == ()
    assert selection.model == "selector-model"
    assert model.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        ModelResponse(content=None),
        ModelResponse(content="{}"),
        ModelResponse(content='{"skills":"review"}'),
        ModelResponse(content='{"skills":[1]}'),
        ModelResponse(content='{"skills":["review","review"]}'),
        ModelResponse(content='{"skills":[]}', tool_calls=[_tool_call("anything")]),
        ModelResponse(content='{"skills":[]}', usage=[]),  # type: ignore[arg-type]
    ],
)
async def test_model_skill_selector_rejects_malformed_response_contracts(
    response: ModelResponse,
) -> None:
    model = SequenceModel(response)
    agent = _agent(
        model,
        skills=[SkillDefinition(name="review", description="Review changes.")],
    )
    selector = ModelSkillSelector(model, "selector-model")

    with pytest.raises(ValueError):
        await selector.select(task="review", skills=agent.skills)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "activations",
    [
        (SkillActivation("missing", "custom_policy"),),
        (SkillActivation("review", "custom_policy"), SkillActivation("review", "again")),
    ],
)
async def test_skill_selector_cannot_activate_unknown_or_duplicate_skills(
    activations: tuple[SkillActivation, ...],
) -> None:
    class Selector:
        async def select(self, *, task: str, skills: Any) -> SkillSelection:
            return SkillSelection(activations)

    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(
        model,
        skills=[SkillDefinition(name="review", triggers=["review"])],
        skill_selector=Selector(),
    )

    with pytest.raises(AgentRuntimeError, match="Skill selector"):
        await agent.arun("review this")
    assert model.calls == []


@pytest.mark.asyncio
async def test_require_sandbox_rejects_host_trusted_tools_before_model_call() -> None:
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="host_action",
            description="An injected host handler.",
            handler=lambda: "should not run",
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(
        model,
        policies={"max_steps": 4, "timeout_seconds": 1, "require_sandbox": True},
        tools=tools,
        tool_names=["host_action"],
    )

    with pytest.raises(AgentRuntimeError, match="Policy requires sandboxed execution"):
        await agent.arun("run the action")

    assert model.calls == []


def test_sandbox_tool_requires_agent_sandbox_at_construction() -> None:
    model = SequenceModel(ModelResponse(content="unused"))
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="sandbox_only",
            description="Execute inside the configured container.",
            output_schema=_ANY_JSON_SCHEMA,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            sandbox_action="tool.execute",
            execution="sandboxed",
            sandbox_command=("sandbox-only",),
        )
    )
    definition = AgentDefinition(
        name="missing-sandbox-agent",
        model={"provider": "fake", "model": "model-v1"},
        tools=["sandbox_only"],
    )

    with pytest.raises(ConfigError, match="require a sandbox configuration: sandbox_only"):
        Agent(definition, model=model, tools=tools)
    assert model.calls == []


@pytest.mark.asyncio
async def test_expired_run_deadline_before_execution_has_accurate_error() -> None:
    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(model)
    started = time.perf_counter()

    with pytest.raises(AgentRuntimeError, match="Agent run deadline expired before execution"):
        await Runtime(agent)._run_in_environment(
            RunRequest("task"),
            started=started,
            deadline=started - 1,
            event_sink=None,
            skill_integrity_check_ms=None,
            skill_revocation_check_ms=None,
        )

    assert model.calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_active_skill_input_schema_rejects_request_before_reasoning_call() -> None:
    skill = SkillDefinition(
        name="triage",
        triggers=["triage"],
        input_schema={
            "type": "object",
            "properties": {"context": {"type": "object", "required": ["account_id"]}},
            "required": ["context"],
        },
    )
    model = SequenceModel(ModelResponse(content='{"category":"billing"}'))
    agent = _agent(model, skills=[skill])

    with pytest.raises(AgentRuntimeError, match="input_schema"):
        await agent.arun("triage this request", context={})

    assert model.calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_active_skill_output_schema_is_validated_and_returned_as_data() -> None:
    skill = SkillDefinition(
        name="triage",
        triggers=["triage"],
        output_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
    )
    model = SequenceModel(ModelResponse(content='{"category":"billing"}'))
    agent = _agent(model, skills=[skill])

    result = await agent.arun("triage this request")

    system_message = model.calls[0]["messages"][0]["content"]
    assert "Final response must also satisfy this skill output schema" in system_message
    assert result.metadata["structured_output"] == {"category": "billing"}
    assert any(event.kind == "output_validation" for event in result.trace.events)
    await agent.aclose()


@pytest.mark.asyncio
async def test_active_skill_output_schema_buffers_stream_until_validation() -> None:
    class StreamingModel:
        name = "streaming"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider should be used")

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta='{"category":')
            yield ModelStreamDelta(content_delta='"billing"}')
            yield ModelStreamDelta(finish_reason="stop")

    skill = SkillDefinition(
        name="triage",
        triggers=["triage"],
        output_schema={"type": "object", "required": ["category"]},
    )
    agent = _agent(StreamingModel(), skills=[skill])

    events = [event async for event in agent.astream("triage this request")]

    text_events = [event for event in events if event.type == "text_delta"]
    assert [event.data["text"] for event in text_events] == ['{"category":"billing"}']
    completed = next(event for event in events if event.type == "completed")
    assert completed.data["result"]["metadata"]["structured_output"] == {"category": "billing"}
    await agent.aclose()


@pytest.mark.asyncio
async def test_injected_policy_engine_receives_principal_and_can_deny_before_model_call() -> None:
    captured: dict[str, Any] = {}

    class DenyingPolicy:
        async def authorize_tool(self, name: str) -> None:
            raise ToolError(
                "private policy detail",
                code=ToolErrorCode.POLICY_DENIED,
            )

        async def authorize_permissions(self, name: str, permissions: list[str]) -> None:
            raise AssertionError("tool denial should short-circuit permission checks")

    class Factory:
        async def create(self, **kwargs: Any) -> DenyingPolicy:
            captured.update(kwargs)
            return DenyingPolicy()

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="read_record",
            description="Read one record.",
            parameters={"type": "object", "properties": {}},
            output_schema=_ANY_JSON_SCHEMA,
            handler=lambda: {"ok": True},
        )
    )
    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(
        model,
        tools=tools,
        tool_names=["read_record"],
        policy_engine_factory=Factory(),
    )
    principal = Principal(subject="alice", scopes=frozenset({"records.read"}))

    with pytest.raises(ToolError, match="Policy denied access") as denied:
        await agent.arun("read the record", principal=principal)

    assert "private policy detail" not in str(denied.value)
    assert captured["declared_tools"] == ["read_record"]
    assert captured["principal"] is principal
    assert model.calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_injected_policy_factory_failure_is_sanitized() -> None:
    class BrokenFactory:
        async def create(self, **_: Any) -> Any:
            raise RuntimeError("secret policy backend detail")

    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(model, policy_engine_factory=BrokenFactory())

    with pytest.raises(AgentRuntimeError, match="Policy engine could not be constructed") as error:
        await agent.arun("do not call the model")

    assert "secret policy backend detail" not in str(error.value)
    assert model.calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_injected_policy_engine_authorizes_before_exposure_and_execution() -> None:
    checks: list[str] = []

    class AllowingPolicy:
        async def authorize_tool(self, name: str) -> None:
            checks.append(f"tool:{name}")

        async def authorize_permissions(self, name: str, permissions: list[str]) -> None:
            checks.append(f"permissions:{name}")

    class Factory:
        async def create(self, **_: Any) -> AllowingPolicy:
            return AllowingPolicy()

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="read_record",
            description="Read one record.",
            parameters={"type": "object", "properties": {}},
            output_schema=_ANY_JSON_SCHEMA,
            handler=lambda: {"ok": True},
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("read_record")]),
        ModelResponse(content="record read"),
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["read_record"],
        policy_engine_factory=Factory(),
    )

    result = await agent.arun("read the record")

    assert result.output == "record read"
    assert checks == [
        "tool:read_record",
        "permissions:read_record",
        "tool:read_record",
        "permissions:read_record",
    ]
    await agent.aclose()


@pytest.mark.asyncio
async def test_injected_policy_authorization_shares_the_run_deadline() -> None:
    class SlowPolicy:
        async def authorize_tool(self, name: str) -> None:
            await asyncio.sleep(1)

        async def authorize_permissions(self, name: str, permissions: list[str]) -> None:
            return None

    class Factory:
        async def create(self, **_: Any) -> SlowPolicy:
            return SlowPolicy()

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="read_record",
            description="Read one record.",
            parameters={"type": "object", "properties": {}},
            output_schema=_ANY_JSON_SCHEMA,
            handler=lambda: {"ok": True},
        )
    )
    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(
        model,
        policies={"max_steps": 2, "timeout_seconds": 0.05},
        tools=tools,
        tool_names=["read_record"],
        policy_engine_factory=Factory(),
    )

    with pytest.raises(AgentRuntimeError, match="Policy authorization exceeded the run deadline"):
        await agent.arun("read the record")

    assert model.calls == []
    await agent.aclose()


@pytest.mark.parametrize(
    ("sandbox_error", "expected_code"),
    [
        (SandboxTimeout("private timeout detail"), "deadline_exceeded"),
        (SandboxUnavailable("private engine detail"), "sandbox_unavailable"),
        (SandboxOutputLimit("private output detail"), "result_too_large"),
    ],
)
@pytest.mark.asyncio
async def test_sandbox_tool_errors_keep_stable_codes_in_stream_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    sandbox_error: Exception,
    expected_code: str,
) -> None:
    class FakeSession:
        async def invoke(
            self, _action: str, _arguments: dict[str, object], *, timeout: float
        ) -> object:
            raise sandbox_error

    @asynccontextmanager
    async def fake_open_sandbox(*_args: Any, **_kwargs: Any) -> Any:
        yield FakeSession()

    monkeypatch.setattr("gabby.runtime.open_sandbox", fake_open_sandbox)
    model = SequenceModel(
        ModelResponse(
            tool_calls=[_tool_call("sandbox_action", arguments='{"argv":["echo","hi"]}')]
        ),
        ModelResponse(content="recovered"),
    )
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="sandbox_action",
            description="Run an isolated action.",
            parameters={
                "type": "object",
                "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
                "required": ["argv"],
                "additionalProperties": False,
            },
            output_schema=_ANY_JSON_SCHEMA,
            sandbox_action="shell.run",
            execution="sandboxed",
        )
    )
    definition = AgentDefinition(
        name="sandbox-error-agent",
        model={"provider": "fake", "model": "model-v1"},
        tools=["sandbox_action"],
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["sandbox_action"]},
        sandbox=SandboxDefinition(
            engine="docker",
            image="example/test:latest",
            keepalive_argv=("sleep", "infinity"),
        ),
    )
    agent = Agent(definition, model=model, tools=tools)

    events = [event async for event in agent.astream("run isolated action")]

    failed = next(event for event in events if event.type == "tool_failed")
    assert failed.data["error_code"] == expected_code
    assert "private" not in str(failed.data)
    completed = events[-1]
    assert completed.type == "completed"
    result = completed.data["result"]
    assert any(
        event["kind"] == "tool_error" and event["details"]["error_code"] == expected_code
        for event in result["trace"]["events"]
    )
    assert result["output"] == "recovered"
    await agent.aclose()


@pytest.mark.asyncio
async def test_retrieval_failure_is_reported_as_a_runtime_error() -> None:
    class BrokenRetriever:
        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> list[Document]:
            raise OSError("private storage detail")

    agent = _agent(SequenceModel(ModelResponse(content="unused")), retriever=BrokenRetriever())

    with pytest.raises(AgentRuntimeError, match="Knowledge retrieval failed") as error:
        await agent.arun("lookup")

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_verification_rejection_stops_the_run() -> None:
    class RejectingVerifier:
        async def verify(self, **_: Any) -> VerificationResult:
            return VerificationResult(passed=False, method="fixture")

    agent = _agent(
        SequenceModel(ModelResponse(content="unverified")),
        verification={"enabled": True},
        verifier=RejectingVerifier(),
    )

    with pytest.raises(AgentRuntimeError, match="verification did not pass"):
        await agent.arun("check this")


@pytest.mark.asyncio
async def test_verification_rejects_unstructured_results() -> None:
    class LegacyVerifier:
        async def verify(self, **_: Any) -> dict[str, bool]:
            return {"passed": True}

    agent = _agent(
        SequenceModel(ModelResponse(content="unchecked")),
        verification={"enabled": True},
        verifier=LegacyVerifier(),
    )

    with pytest.raises(AgentRuntimeError, match="invalid result"):
        await agent.arun("check this")


@pytest.mark.asyncio
async def test_sync_verifier_uses_the_sync_callback_bridge() -> None:
    class SyncVerifier:
        def verify(self, **_: Any) -> VerificationResult:
            return VerificationResult(passed=True, method="sync_fixture")

    agent = _agent(
        SequenceModel(ModelResponse(content="checked")),
        verification={"enabled": True},
        verifier=SyncVerifier(),
    )

    result = await agent.arun("check this")

    assert result.metadata["verification"]["method"] == "sync_fixture"


@pytest.mark.asyncio
async def test_verifier_exception_is_normalized_and_keeps_its_cause() -> None:
    class BrokenVerifier:
        async def verify(self, **_: Any) -> VerificationResult:
            raise OSError("private verifier detail")

    agent = _agent(
        SequenceModel(ModelResponse(content="unchecked")),
        verification={"enabled": True},
        verifier=BrokenVerifier(),
    )

    with pytest.raises(AgentRuntimeError, match="Configured verifier failed") as error:
        await agent.arun("check this")

    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_runtime_rejects_malformed_provider_responses_and_invalid_budgets() -> None:
    malformed = _agent(SequenceModel(ModelResponse(usage=[])))  # type: ignore[arg-type]
    with pytest.raises(AgentRuntimeError, match="malformed response"):
        await malformed.arun("bad usage")

    malformed_calls = _agent(
        SequenceModel(ModelResponse(tool_calls=[{}])),
        policies={"max_steps": 1, "timeout_seconds": 1},
    )
    with pytest.raises(AgentRuntimeError, match="malformed tool calls"):
        await malformed_calls.arun("bad call")

    with pytest.raises(ConfigError, match="policies.max_steps must be a positive integer"):
        _agent(
            SequenceModel(ModelResponse(content="unused")),
            policies={"max_steps": 0, "timeout_seconds": 1},
        )

    with pytest.raises(ConfigError, match="policies.max_tool_calls must be an integer"):
        _agent(
            SequenceModel(ModelResponse(content="unused")),
            policies={"max_tool_calls": 1025, "timeout_seconds": 1},
        )

    with pytest.raises(ConfigError, match="policies.timeout_seconds must be a finite positive"):
        _agent(
            SequenceModel(ModelResponse(content="unused")),
            policies={"max_steps": 1, "timeout_seconds": float("inf")},
        )


@pytest.mark.asyncio
async def test_runtime_enforces_model_request_byte_limit_before_provider_call() -> None:
    model = SequenceModel(ModelResponse(content="unused"))
    agent = _agent(
        model,
        policies={
            "max_steps": 1,
            "timeout_seconds": 1,
            "max_model_request_bytes": 512,
        },
    )

    with pytest.raises(AgentRuntimeError, match="max_model_request_bytes=512"):
        await agent.arun("x" * 1024)

    assert model.calls == []


@pytest.mark.asyncio
async def test_runtime_maps_oversized_model_response_to_agent_runtime_error() -> None:
    agent = _agent(
        SequenceModel(ModelResponse(content="response exceeds cap")),
        policies={
            "max_steps": 1,
            "timeout_seconds": 1,
            "max_model_response_bytes": 8,
        },
    )

    with pytest.raises(AgentRuntimeError, match="max_model_response_bytes=8"):
        await agent.arun("small request")


@pytest.mark.asyncio
async def test_tool_batch_exceeding_run_budget_is_rejected_before_execution() -> None:
    invoked: list[str] = []
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="record",
            description="Record a call.",
            handler=lambda: invoked.append("called"),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
    )
    model = SequenceModel(
        ModelResponse(
            tool_calls=[
                _tool_call("record", call_id="first"),
                _tool_call("record", call_id="second"),
            ]
        )
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["record"],
        policies={
            "max_steps": 2,
            "max_tool_calls": 1,
            "timeout_seconds": 1,
            "allowed_tools": ["record"],
        },
    )

    with pytest.raises(AgentRuntimeError, match="max_tool_calls=1"):
        await agent.arun("record it twice")

    assert invoked == []


@pytest.mark.asyncio
async def test_tool_permission_denial_is_observed_without_running_handler() -> None:
    calls: list[int] = []
    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="write_record",
            description="Write a record",
            handler=lambda: calls.append(1),
            parameters={"type": "object", "properties": {}},
            permissions=["database.write"],
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("write_record")]),
        ModelResponse(content="The action was denied."),
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["write_record"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 1,
            "allowed_tools": ["write_record"],
            "allowed_permissions": [],
        },
    )

    with pytest.raises(ToolError, match="database.write"):
        await agent.arun("write a record")

    assert calls == []
    assert model.calls == []


@pytest.mark.asyncio
async def test_async_retriever_timeout_stops_before_the_model_call() -> None:
    class SlowRetriever:
        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> list[Document]:
            await asyncio.sleep(1)
            return []

    agent = _agent(
        SequenceModel(ModelResponse(content="unused")),
        policies={"max_steps": 1, "timeout_seconds": 0.01},
        retriever=SlowRetriever(),
    )

    with pytest.raises(AgentRuntimeError, match="Knowledge retrieval exceeded"):
        await agent.arun("slow lookup")


@pytest.mark.asyncio
async def test_sync_tool_may_return_an_awaitable() -> None:
    async def complete() -> dict[str, bool]:
        return {"ok": True}

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="hybrid",
            description="Return an awaitable from a sync adapter.",
            handler=lambda: complete(),
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("hybrid")]), ModelResponse(content="finished")
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["hybrid"],
        policies={"max_steps": 2, "timeout_seconds": 1, "allowed_tools": ["hybrid"]},
    )

    result = await agent.arun("hybrid action")

    tool_observation = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    assert json.loads(tool_observation["content"]) == {"ok": True}
    assert result.output == "finished"


@pytest.mark.asyncio
async def test_sync_tool_exception_is_not_returned_to_the_model() -> None:
    def fail() -> None:
        raise RuntimeError("private host detail")

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="fail",
            description="A failing tool.",
            handler=fail,
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("fail")]), ModelResponse(content="finished")
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["fail"],
        policies={"max_steps": 2, "timeout_seconds": 1, "allowed_tools": ["fail"]},
    )

    result = await agent.arun("call tool")

    tool_observation = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    assert "private host detail" not in tool_observation["content"]
    assert "Tool execution failed" in tool_observation["content"]
    assert json.loads(tool_observation["content"])["error_code"] == "execution_failed"
    assert any(
        event.kind == "tool_error" and event.details["error_type"] == "ToolExecutionError"
        for event in result.trace.events
    )


@pytest.mark.asyncio
async def test_custom_tool_error_exposes_code_and_redacts_handler_message() -> None:
    def fail() -> None:
        raise ToolError("database password is private", code=ToolErrorCode.POLICY_DENIED)

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="fail",
            description="A failing tool.",
            handler=fail,
            parameters={"type": "object", "properties": {}},
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("fail")]), ModelResponse(content="finished")
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["fail"],
        policies={"max_steps": 2, "timeout_seconds": 1, "allowed_tools": ["fail"]},
    )

    result = await agent.arun("call tool")

    observation = next(
        message for message in model.calls[1]["messages"] if message.get("role") == "tool"
    )
    payload = json.loads(observation["content"])
    assert payload["error_code"] == "policy_denied"
    assert payload["error"] == "Tool execution failed"
    assert "database password" not in observation["content"]
    assert any(
        event.kind == "tool_error" and event.details["error_code"] == "policy_denied"
        for event in result.trace.events
    )


@pytest.mark.asyncio
async def test_tracer_receives_ordered_events_and_failure_is_fail_open() -> None:
    class RecordingTracer:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, str]] = []

        async def on_event(self, *, trace_id: str, agent_name: str, event: Any) -> None:
            self.events.append((trace_id, agent_name, event.kind))

    tracer = RecordingTracer()
    agent = _agent(SequenceModel(ModelResponse(content="finished")), tracer=tracer)
    result = await agent.arun("record this run")

    assert result.output == "finished"
    assert tracer.events
    assert {trace_id for trace_id, _, _ in tracer.events} == {result.trace.trace_id}
    assert {agent_name for _, agent_name, _ in tracer.events} == {"contract-agent"}
    assert [kind for _, _, kind in tracer.events] == [event.kind for event in result.trace.events]

    class BrokenTracer:
        calls = 0

        async def on_event(self, **_: Any) -> None:
            self.calls += 1
            raise OSError("private telemetry endpoint detail")

    broken = BrokenTracer()
    result = await _agent(
        SequenceModel(ModelResponse(content="still finished")), tracer=broken
    ).arun("tracer fails")
    assert result.output == "still finished"
    assert broken.calls == 1
    errors = [event for event in result.trace.events if event.kind == "tracer_error"]
    assert len(errors) == 1
    assert errors[0].details == {"error_type": "TracerError"}
    assert "private telemetry endpoint detail" not in str(result.trace.as_dict())


@pytest.mark.asyncio
async def test_tracer_timeout_is_bounded_and_does_not_fail_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gabby.runtime._TRACER_EVENT_TIMEOUT_SECONDS", 0.001)

    class SlowTracer:
        async def on_event(self, **_: Any) -> None:
            await asyncio.sleep(1)

    result = await _agent(
        SequenceModel(ModelResponse(content="still finished")), tracer=SlowTracer()
    ).arun("slow telemetry")

    assert result.output == "still finished"
    errors = [event for event in result.trace.events if event.kind == "tracer_error"]
    assert len(errors) == 1
    assert errors[0].details == {"error_type": "TimeoutError"}


@pytest.mark.asyncio
async def test_timed_out_sync_tool_gets_cooperative_cancellation_signal() -> None:
    worker_started = threading.Event()
    cancellation_observed = threading.Event()

    def wait_for_cancellation(*, context: ToolContext) -> dict[str, bool]:
        worker_started.set()
        if context.cancellation.wait(timeout=1):
            cancellation_observed.set()
        return {"cancelled": context.cancellation.is_cancelled}

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="wait_for_cancellation",
            description="Wait cooperatively for the execution runtime to cancel the call.",
            handler=wait_for_cancellation,
            parameters={"type": "object", "properties": {}},
            timeout_seconds=0.2,
            context_parameter="context",
        )
    )
    model = SequenceModel(
        ModelResponse(tool_calls=[_tool_call("wait_for_cancellation")]),
        ModelResponse(content="finished after timeout"),
    )
    agent = _agent(
        model,
        tools=tools,
        tool_names=["wait_for_cancellation"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 1,
            "allowed_tools": ["wait_for_cancellation"],
        },
    )

    result = await agent.arun("perform the slow tool")
    assert result.output == "finished after timeout"
    assert worker_started.is_set()
    for _ in range(100):
        if cancellation_observed.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancellation_observed.is_set()


@pytest.mark.asyncio
async def test_cancelled_run_signals_cooperative_sync_tool() -> None:
    worker_started = threading.Event()
    cancellation_observed = threading.Event()

    def wait_for_cancellation(*, context: ToolContext) -> dict[str, bool]:
        worker_started.set()
        if context.cancellation.wait(timeout=1):
            cancellation_observed.set()
        return {"cancelled": context.cancellation.is_cancelled}

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="wait_for_cancellation",
            description="Wait cooperatively for the execution runtime to cancel the call.",
            handler=wait_for_cancellation,
            parameters={"type": "object", "properties": {}},
            context_parameter="context",
        )
    )
    model = SequenceModel(ModelResponse(tool_calls=[_tool_call("wait_for_cancellation")]))
    agent = _agent(
        model,
        tools=tools,
        tool_names=["wait_for_cancellation"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 1,
            "allowed_tools": ["wait_for_cancellation"],
        },
    )

    execution = asyncio.create_task(agent.arun("start the slow tool"))
    for _ in range(100):
        if worker_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert worker_started.is_set()
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution
    for _ in range(100):
        if cancellation_observed.is_set():
            break
        await asyncio.sleep(0.01)
    assert cancellation_observed.is_set()
