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
"""Deterministic stateless agent evaluation contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from gabby import (
    Agent,
    AgentDefinition,
    EvaluationCase,
    EvaluationError,
    EvaluationReport,
    ExecutionResult,
    ExecutionTrace,
    ModelResponse,
    evaluate_agent,
    load_evaluation_dataset,
)


class FakeAgent:
    """Agent stub recording independent requests for the evaluation runner."""

    def __init__(self, outputs: list[str | Exception], tools: tuple[str, ...] = ()) -> None:
        self.outputs = list(outputs)
        self.tools = tools
        self.requests: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    async def arun(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        self.requests.append((input, context or {}, memory or {}, metadata or {}))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        trace = ExecutionTrace()
        for tool in self.tools:
            trace.add("tool_call", name=tool)
        return ExecutionResult(output, trace)


def test_dataset_loader_reads_cases_and_generates_stable_line_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '\n{"input":"hello","expected_contains":["world"]}\n'
        '{"id":"exact","input":"task","expected_output":"done",'
        '"context":{"region":"west"}}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_dataset(path)

    assert [case.id for case in cases] == ["case-0002", "exact"]
    assert cases[0].expected_contains == ("world",)
    assert cases[1].context == {"region": "west"}


def test_dataset_loader_reads_structured_expectations_including_null(tmp_path: Path) -> None:
    path = tmp_path / "structured.jsonl"
    path.write_text(
        '{"input":"classify","expected_structured_output":{"label":"billing"}}\n'
        '{"input":"empty","expected_structured_output":null}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_dataset(path)

    assert cases[0].expected_structured_output == {"label": "billing"}
    assert cases[1].expected_structured_output is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"input":"task"}\n', "at least one expectation"),
        ('{"input":"task","expected_output":"x","extra":1}\n', "unknown field"),
        (
            '{"id":"same","input":"one","expected_output":"x"}\n'
            '{"id":"same","input":"two","expected_output":"y"}\n',
            "repeats case id",
        ),
        ('{"id":"x","id":"y","input":"task","expected_output":"x"}\n', "repeat keys"),
    ],
)
def test_dataset_loader_rejects_malformed_cases(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EvaluationError, match=message):
        load_evaluation_dataset(path)


@pytest.mark.asyncio
async def test_evaluation_scores_output_and_tool_expectations_without_shared_state() -> None:
    agent = FakeAgent(["The refund takes five days.", "done"], tools=("lookup_policy",))
    cases = (
        EvaluationCase(
            id="refund",
            input="How long?",
            context={"case": "one"},
            expected_contains=("five days",),
            required_tools=("lookup_policy",),
            forbidden_tools=("send_payment",),
        ),
        EvaluationCase(
            id="exact",
            input="Finish",
            memory={"caller": "two"},
            expected_output="done",
        ),
    )

    report = await evaluate_agent(agent, cases)

    assert isinstance(report, EvaluationReport)
    assert report.passed_count == 2
    assert report.score == 1.0
    assert [result.case_id for result in report.results] == ["refund", "exact"]
    assert report.results[0].tool_calls == ("lookup_policy",)
    assert agent.requests == [
        ("How long?", {"case": "one"}, {}, {}),
        ("Finish", {}, {"caller": "two"}, {}),
    ]
    assert "The refund" not in json.dumps(report.as_dict())
    assert report.results[0].as_dict()["tool_call_count"] == 1
    assert report.results[0].as_dict()["tool_calls_truncated"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actual", "expected", "passed"),
    [
        ({"label": "billing", "urgent": False}, {"urgent": False, "label": "billing"}, True),
        ({"count": True}, {"count": 1}, False),
        ({"label": "billing"}, None, False),
    ],
)
async def test_evaluation_checks_structured_output_without_reporting_values(
    actual: Any, expected: Any, passed: bool
) -> None:
    class StructuredAgent(FakeAgent):
        async def arun(self, *args: Any, **kwargs: Any) -> ExecutionResult:
            result = await super().arun(*args, **kwargs)
            return ExecutionResult(
                '{"provider formatting may vary":true}',
                result.trace,
                {"structured_output": actual},
            )

    case = EvaluationCase(
        id="structured",
        input="classify",
        expected_structured_output=expected,
    )
    report = await evaluate_agent(StructuredAgent(["ignored raw response"]), [case])

    assert report.results[0].checks[0].name == "structured_output"
    assert report.results[0].checks[0].passed is passed
    assert "billing" not in json.dumps(report.as_dict())


@pytest.mark.asyncio
async def test_structured_expectation_fails_when_result_has_no_parsed_value() -> None:
    report = await evaluate_agent(
        FakeAgent(["{}"]),
        [EvaluationCase(id="structured", input="task", expected_structured_output={})],
    )

    assert report.results[0].passed is False
    assert report.results[0].checks[0].name == "structured_output"
    assert report.results[0].checks[0].passed is False


def test_evaluation_bounds_expectations_per_case() -> None:
    with pytest.raises(EvaluationError, match="exceeds 1000 expectations"):
        EvaluationCase(
            id="too-many",
            input="task",
            expected_contains=("x",) * 1_000,
            expected_output="x",
        )


@pytest.mark.asyncio
async def test_evaluation_bounds_suite_expectations_before_running() -> None:
    cases = [
        EvaluationCase(
            id=f"case-{index}",
            input="task",
            expected_contains=("x",) * 1_000,
        )
        for index in range(10)
    ]
    cases.append(EvaluationCase(id="extra", input="task", expected_output="x"))

    with pytest.raises(EvaluationError, match="exceeds 10000 expectations"):
        await evaluate_agent(FakeAgent([]), cases)


@pytest.mark.asyncio
async def test_evaluation_report_bounds_tool_call_names_and_preserves_checking() -> None:
    names = tuple(f"tool_{index}" for index in range(25))
    agent = FakeAgent(["done"], tools=names)
    case = EvaluationCase(
        id="many-tools",
        input="task",
        expected_output="done",
        required_tools=(names[-1],),
    )

    report = await evaluate_agent(agent, [case])
    result = report.results[0]

    assert result.passed
    assert result.tool_call_count == 25
    assert result.tool_calls == names[:20]
    assert result.tool_calls_truncated is True
    assert result.as_dict()["tool_calls_truncated"] is True


@pytest.mark.asyncio
async def test_evaluation_records_sanitized_case_errors_and_continues() -> None:
    agent = FakeAgent([RuntimeError("private provider details"), "okay"])
    cases = (
        EvaluationCase(id="fails", input="first", expected_output="expected"),
        EvaluationCase(id="passes", input="second", expected_output="okay"),
    )

    report = await evaluate_agent(agent, cases)

    assert report.passed_count == 1
    assert report.results[0].error_type == "RuntimeError"
    assert report.results[0].checks == ()
    assert "private provider details" not in json.dumps(report.as_dict())


@pytest.mark.asyncio
async def test_evaluation_propagates_cancellation_and_rejects_duplicate_ids() -> None:
    class CancelledAgent:
        async def arun(self, *_args: Any, **_kwargs: Any) -> ExecutionResult:
            raise asyncio.CancelledError

    case = EvaluationCase(id="cancel", input="stop", expected_contains=("x",))
    with pytest.raises(asyncio.CancelledError):
        await evaluate_agent(CancelledAgent(), [case])
    with pytest.raises(EvaluationError, match="unique"):
        await evaluate_agent(FakeAgent(["x", "x"]), [case, case])


@pytest.mark.asyncio
async def test_evaluation_runs_against_real_stateless_agent_runtime() -> None:
    class Model:
        name = "evaluation-fixture"

        async def complete(self, **_kwargs: Any) -> ModelResponse:
            return ModelResponse(content="Approved for processing.")

    agent = Agent(
        AgentDefinition(
            name="evaluation-fixture",
            model={"provider": "fixture", "model": "fixture"},
            policies={"max_steps": 2, "timeout_seconds": 2},
        ),
        model=Model(),
    )
    async with agent:
        report = await evaluate_agent(
            agent,
            [
                EvaluationCase(
                    id="approval-wording",
                    input="Respond with the approved decision.",
                    expected_contains=("Approved", "processing"),
                )
            ],
        )

    assert report.passed_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": ""}, "case id"),
        ({"id": "x" * 257}, "case id"),
        ({"input": ""}, "input must"),
        ({"input": "x" * 100_001}, "input must"),
        ({"context": []}, "context must be a JSON object"),
        ({"memory": {"bad": float("nan")}}, "finite JSON values"),
        ({"metadata": {"bad": object()}}, "finite JSON values"),
        ({"context": {"blob": "x" * (1024 * 1024 + 1)}}, "exceeds 1048576 bytes"),
        ({"expected_output": 1}, "expected_output must be a string"),
        ({"expected_structured_output": float("inf")}, "finite JSON values"),
        (
            {"expected_structured_output": {"blob": "x" * (1024 * 1024 + 1)}},
            "expected_structured_output exceeds",
        ),
        ({"expected_contains": ["x"]}, "expected_contains must be a tuple"),
        ({"required_tools": ("​", 2)}, "required_tools must be a tuple"),
        ({"forbidden_tools": ("",)}, "forbidden_tools must be a tuple"),
        ({}, "at least one expectation"),
        (
            {"context": {"blob": "x" * 900_000}, "expected_output": "y" * 200_000},
            "evaluation case exceeds",
        ),
    ],
)
def test_evaluation_case_validates_inputs_and_expectations(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {"id": "case", "input": "request"}
    values.update(kwargs)
    with pytest.raises(EvaluationError, match=message):
        EvaluationCase(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not-json\n", "not valid JSON"),
        (b"NaN\n", "finite JSON values"),
        (b"[]\n", "must be an object"),
        (b'{"input":"x","input":"y","expected_output":"z"}\n', "repeat keys"),
        (
            b'{"input":"x","expected_output":"z","unexpected":1}\n',
            "unknown field",
        ),
        (b'{"expected_output":"z"}\n', "requires a string input"),
        (b'{"input":1,"expected_output":"z"}\n', "requires a string input"),
        (
            b'{"input":"x","expected_output":"z","required_tools":"tool"}\n',
            "required_tools must be an array",
        ),
        (
            b'{"input":"x","expected_output":"z","context":[]}\n',
            "context must be a JSON object",
        ),
    ],
)
def test_evaluation_dataset_rejects_invalid_records(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(content)
    with pytest.raises(EvaluationError, match=message):
        load_evaluation_dataset(path)


def test_evaluation_dataset_handles_file_and_dataset_bounds(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    with pytest.raises(EvaluationError, match="unavailable"):
        load_evaluation_dataset(path)

    path.write_bytes(b"\xff")
    with pytest.raises(EvaluationError, match="UTF-8"):
        load_evaluation_dataset(path)

    path.write_bytes(b"\n \t\n")
    with pytest.raises(EvaluationError, match="at least one case"):
        load_evaluation_dataset(path)

    path.write_bytes(b" " * (10 * 1024 * 1024 + 1))
    with pytest.raises(EvaluationError, match="exceeds 10485760 bytes"):
        load_evaluation_dataset(path)

    path.write_text('{"input":"x","expected_output":"y"}' + " " * 1_048_600)
    with pytest.raises(EvaluationError, match="exceeds 1048576 bytes"):
        load_evaluation_dataset(path)

    record = json.dumps({"input": "x", "expected_output": "y"})
    path.write_text((record + "\n") * 1001, encoding="utf-8")
    with pytest.raises(EvaluationError, match="exceeds 1000 cases"):
        load_evaluation_dataset(path)


@pytest.mark.asyncio
async def test_evaluation_validates_suite_shape_size_and_duplicate_ids() -> None:
    case = EvaluationCase(id="one", input="task", expected_output="done")
    with pytest.raises(TypeError, match="sequence"):
        await evaluate_agent(FakeAgent([]), None)  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="1 through 1000"):
        await evaluate_agent(FakeAgent([]), [])
    with pytest.raises(TypeError, match="EvaluationCase"):
        await evaluate_agent(FakeAgent([]), [object()])  # type: ignore[list-item]
    with pytest.raises(EvaluationError, match="unique"):
        await evaluate_agent(FakeAgent([]), [case, case])

    large_cases = [
        EvaluationCase(
            id=f"case-{index}",
            input="task",
            context={"blob": "x" * 900_000},
            expected_output="done",
        )
        for index in range(12)
    ]
    with pytest.raises(EvaluationError, match="suite exceeds"):
        await evaluate_agent(FakeAgent([]), large_cases)


def test_evaluation_result_reports_empty_suite_and_compares_json_types() -> None:
    empty = EvaluationReport((), 0.0)
    assert empty.score == 0.0
    assert empty.passed_count == 0
    assert empty.as_dict()["case_count"] == 0

    class TypedAgent(FakeAgent):
        async def arun(self, *args: Any, **kwargs: Any) -> ExecutionResult:
            result = await super().arun(*args, **kwargs)
            return ExecutionResult(
                "ok",
                result.trace,
                {"structured_output": {"values": [1, True], "enabled": True}},
            )

    report = asyncio.run(
        evaluate_agent(
            TypedAgent(["ok"]),
            [
                EvaluationCase(
                    id="nested",
                    input="task",
                    expected_structured_output={"enabled": True, "values": [1, True]},
                )
            ],
        )
    )
    assert report.results[0].passed


@pytest.mark.asyncio
async def test_evaluation_reports_long_tool_names_and_ignores_invalid_trace_names() -> None:
    class TraceAgent:
        async def arun(self, *_args: Any, **_kwargs: Any) -> ExecutionResult:
            trace = ExecutionTrace()
            trace.add("tool_call", name="t" * 80)
            trace.add("tool_call", name=7)
            return ExecutionResult("done", trace)

    report = await evaluate_agent(
        TraceAgent(),
        [EvaluationCase(id="trace", input="task", expected_output="done")],
    )
    result = report.results[0]
    assert result.passed
    assert result.tool_call_count == 1
    assert result.tool_calls == ("t" * 64,)
    assert result.tool_calls_truncated
    assert report.results[0].trace_id is not None
