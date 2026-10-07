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
"""Deterministic, stateless agent regression evaluations."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .agent import Agent
from .runtime import ExecutionResult

MAX_EVALUATION_CASES = 1_000
MAX_EVALUATION_DATASET_BYTES = 10 * 1024 * 1024
MAX_EVALUATION_CASE_BYTES = 1024 * 1024
MAX_EVALUATION_EXPECTATIONS_PER_CASE = 1_000
MAX_EVALUATION_EXPECTATIONS = 10_000
MAX_REPORTED_TOOL_CALLS = 20
MAX_REPORTED_TOOL_NAME_CHARS = 64
_NO_STRUCTURED_EXPECTATION = object()
_CASE_FIELDS = frozenset(
    {
        "id",
        "input",
        "context",
        "memory",
        "metadata",
        "expected_output",
        "expected_structured_output",
        "expected_contains",
        "required_tools",
        "forbidden_tools",
    }
)


class EvaluationError(ValueError):
    """An evaluation dataset or suite is invalid."""


@dataclass(frozen=True)
class EvaluationCase:
    """One isolated run request and its deterministic regression expectations."""

    id: str
    input: str
    context: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_output: str | None = None
    expected_contains: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_structured_output: Any = field(default=_NO_STRUCTURED_EXPECTATION, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 256:
            raise EvaluationError("evaluation case id must contain 1 through 256 characters")
        if not isinstance(self.input, str) or not 1 <= len(self.input) <= 100_000:
            raise EvaluationError("evaluation case input must contain 1 through 100000 characters")
        for name in ("context", "memory", "metadata"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise EvaluationError(f"evaluation case {name} must be a JSON object")
            try:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
                raise EvaluationError(
                    f"evaluation case {name} must contain finite JSON values"
                ) from None
            if len(encoded) > MAX_EVALUATION_CASE_BYTES:
                raise EvaluationError(
                    f"evaluation case {name} exceeds {MAX_EVALUATION_CASE_BYTES} bytes"
                )
            try:
                snapshot = json.loads(encoded)
            except (json.JSONDecodeError, RecursionError):
                raise EvaluationError(f"evaluation case {name} is not valid JSON") from None
            object.__setattr__(self, name, snapshot)
        if self.expected_output is not None and not isinstance(self.expected_output, str):
            raise EvaluationError("expected_output must be a string when provided")
        if self.expected_structured_output is not _NO_STRUCTURED_EXPECTATION:
            try:
                structured_bytes = json.dumps(
                    self.expected_structured_output,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(structured_bytes) > MAX_EVALUATION_CASE_BYTES:
                    raise EvaluationError(
                        f"expected_structured_output exceeds {MAX_EVALUATION_CASE_BYTES} bytes"
                    )
                structured_snapshot = json.loads(structured_bytes)
            except EvaluationError:
                raise
            except (
                TypeError,
                ValueError,
                UnicodeEncodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                raise EvaluationError(
                    "expected_structured_output must contain finite JSON values"
                ) from None
            object.__setattr__(self, "expected_structured_output", structured_snapshot)
        for name in ("expected_contains", "required_tools", "forbidden_tools"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise EvaluationError(f"{name} must be a tuple of non-empty strings")
        if not (
            self.expected_output is not None
            or self.expected_contains
            or self.expected_structured_output is not _NO_STRUCTURED_EXPECTATION
            or self.required_tools
            or self.forbidden_tools
        ):
            raise EvaluationError("evaluation case must define at least one expectation")
        expectation_count = (
            int(self.expected_output is not None)
            + int(self.expected_structured_output is not _NO_STRUCTURED_EXPECTATION)
            + len(self.expected_contains)
            + len(self.required_tools)
            + len(self.forbidden_tools)
        )
        if expectation_count > MAX_EVALUATION_EXPECTATIONS_PER_CASE:
            raise EvaluationError(
                f"evaluation case exceeds {MAX_EVALUATION_EXPECTATIONS_PER_CASE} expectations"
            )
        if _case_size(self) > MAX_EVALUATION_CASE_BYTES:
            raise EvaluationError(f"evaluation case exceeds {MAX_EVALUATION_CASE_BYTES} bytes")


@dataclass(frozen=True)
class EvaluationCheck:
    """Outcome of one named deterministic expectation."""

    name: str
    passed: bool


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Bounded metrics and checks for one evaluation case."""

    case_id: str
    checks: tuple[EvaluationCheck, ...]
    duration_ms: float
    trace_id: str | None
    tool_calls: tuple[str, ...]
    tool_call_count: int = 0
    tool_calls_truncated: bool = False
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the run completed and every configured expectation passed."""
        return self.error_type is None and all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result without model output or exception details."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "checks": [{"name": check.name, "passed": check.passed} for check in self.checks],
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "tool_calls": list(self.tool_calls),
            "tool_call_count": self.tool_call_count,
            "tool_calls_truncated": self.tool_calls_truncated,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate results for a bounded stateless evaluation suite."""

    results: tuple[EvaluationCaseResult, ...]
    duration_ms: float

    @property
    def passed_count(self) -> int:
        """Number of cases that passed all expectations."""
        return sum(result.passed for result in self.results)

    @property
    def score(self) -> float:
        """Fraction of passing cases, or zero for an empty suite."""
        return self.passed_count / len(self.results) if self.results else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return aggregate metrics and per-case checks as JSON-ready data."""
        return {
            "case_count": len(self.results),
            "passed_count": self.passed_count,
            "score": self.score,
            "duration_ms": self.duration_ms,
            "results": [result.as_dict() for result in self.results],
        }


class EvaluationAgent(Protocol):
    """Async stateless agent interface consumed by the evaluation runner."""

    async def arun(
        self,
        input: str,
        *,
        context: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute one isolated evaluation case."""
        ...


def load_evaluation_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load a bounded JSON Lines dataset with unique case IDs and strict JSON values."""
    dataset_path = Path(path).expanduser()
    try:
        with dataset_path.open("rb") as stream:
            content = stream.read(MAX_EVALUATION_DATASET_BYTES + 1)
    except OSError:
        raise EvaluationError("evaluation dataset is unavailable") from None
    if len(content) > MAX_EVALUATION_DATASET_BYTES:
        raise EvaluationError(f"evaluation dataset exceeds {MAX_EVALUATION_DATASET_BYTES} bytes")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise EvaluationError("evaluation dataset must use UTF-8") from None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationError("evaluation dataset objects must not repeat keys")
            result[key] = value
        return result

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_EVALUATION_CASE_BYTES:
            raise EvaluationError(
                f"evaluation case on line {line_number} exceeds {MAX_EVALUATION_CASE_BYTES} bytes"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    EvaluationError("evaluation dataset must contain finite JSON values")
                ),
            )
        except EvaluationError:
            raise
        except (json.JSONDecodeError, RecursionError):
            raise EvaluationError(
                f"evaluation dataset line {line_number} is not valid JSON"
            ) from None
        if not isinstance(value, dict):
            raise EvaluationError(f"evaluation dataset line {line_number} must be an object")
        unknown_fields = set(value) - _CASE_FIELDS
        if unknown_fields:
            raise EvaluationError(
                f"evaluation dataset line {line_number} has unknown field(s): "
                + ", ".join(sorted(unknown_fields))
            )
        case_input = value.get("input")
        if not isinstance(case_input, str):
            raise EvaluationError(f"evaluation dataset line {line_number} requires a string input")
        case_id = value.get("id", f"case-{line_number:04d}")
        try:
            case = EvaluationCase(
                id=case_id,
                input=case_input,
                context=value.get("context", {}),
                memory=value.get("memory", {}),
                metadata=value.get("metadata", {}),
                expected_output=value.get("expected_output"),
                expected_structured_output=value.get(
                    "expected_structured_output", _NO_STRUCTURED_EXPECTATION
                ),
                expected_contains=_string_tuple(
                    value.get("expected_contains", []), "expected_contains"
                ),
                required_tools=_string_tuple(value.get("required_tools", []), "required_tools"),
                forbidden_tools=_string_tuple(value.get("forbidden_tools", []), "forbidden_tools"),
            )
        except EvaluationError as exc:
            raise EvaluationError(f"evaluation dataset line {line_number}: {exc}") from None
        if case.id in seen_ids:
            raise EvaluationError(f"evaluation dataset repeats case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
        if len(cases) > MAX_EVALUATION_CASES:
            raise EvaluationError(f"evaluation dataset exceeds {MAX_EVALUATION_CASES} cases")
    if not cases:
        raise EvaluationError("evaluation dataset must contain at least one case")
    return tuple(cases)


async def evaluate_agent(
    agent: EvaluationAgent | Agent,
    cases: Sequence[EvaluationCase],
) -> EvaluationReport:
    """Run cases sequentially without sharing conversation or memory between them."""
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of EvaluationCase values")
    if not 1 <= len(cases) <= MAX_EVALUATION_CASES:
        raise EvaluationError(f"cases must contain 1 through {MAX_EVALUATION_CASES} entries")
    if any(not isinstance(case, EvaluationCase) for case in cases):
        raise TypeError("cases must contain EvaluationCase values")
    if len({case.id for case in cases}) != len(cases):
        raise EvaluationError("evaluation case IDs must be unique")
    suite_bytes = 0
    suite_expectations = 0
    for case in cases:
        suite_bytes += _case_size(case)
        if suite_bytes > MAX_EVALUATION_DATASET_BYTES:
            raise EvaluationError(
                f"evaluation suite exceeds {MAX_EVALUATION_DATASET_BYTES} serialized bytes"
            )
        suite_expectations += (
            int(case.expected_output is not None)
            + int(case.expected_structured_output is not _NO_STRUCTURED_EXPECTATION)
            + len(case.expected_contains)
            + len(case.required_tools)
            + len(case.forbidden_tools)
        )
        if suite_expectations > MAX_EVALUATION_EXPECTATIONS:
            raise EvaluationError(
                f"evaluation suite exceeds {MAX_EVALUATION_EXPECTATIONS} expectations"
            )

    started = time.perf_counter()
    results: list[EvaluationCaseResult] = []
    for case in cases:
        case_started = time.perf_counter()
        try:
            result = await agent.arun(
                case.input,
                context=case.context,
                memory=case.memory,
                metadata=case.metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            results.append(
                EvaluationCaseResult(
                    case_id=case.id,
                    checks=(),
                    duration_ms=(time.perf_counter() - case_started) * 1000,
                    trace_id=None,
                    tool_calls=(),
                    error_type=type(exc).__name__,
                )
            )
            continue

        all_tool_calls = tuple(
            event.details["name"]
            for event in result.trace.events
            if event.kind == "tool_call" and isinstance(event.details.get("name"), str)
        )
        checks: list[EvaluationCheck] = []
        if case.expected_output is not None:
            checks.append(EvaluationCheck("exact_output", result.output == case.expected_output))
        if case.expected_structured_output is not _NO_STRUCTURED_EXPECTATION:
            checks.append(
                EvaluationCheck(
                    "structured_output",
                    "structured_output" in result.metadata
                    and _json_values_equal(
                        result.metadata["structured_output"], case.expected_structured_output
                    ),
                )
            )
        checks.extend(
            EvaluationCheck(f"contains:{index}", expected in result.output)
            for index, expected in enumerate(case.expected_contains)
        )
        checks.extend(
            EvaluationCheck(f"required_tool:{index}", name in all_tool_calls)
            for index, name in enumerate(case.required_tools)
        )
        checks.extend(
            EvaluationCheck(f"forbidden_tool:{index}", name not in all_tool_calls)
            for index, name in enumerate(case.forbidden_tools)
        )
        results.append(
            EvaluationCaseResult(
                case_id=case.id,
                checks=tuple(checks),
                duration_ms=(time.perf_counter() - case_started) * 1000,
                trace_id=result.trace.trace_id,
                tool_calls=tuple(
                    name[:MAX_REPORTED_TOOL_NAME_CHARS]
                    for name in all_tool_calls[:MAX_REPORTED_TOOL_CALLS]
                ),
                tool_call_count=len(all_tool_calls),
                tool_calls_truncated=(
                    len(all_tool_calls) > MAX_REPORTED_TOOL_CALLS
                    or any(len(name) > MAX_REPORTED_TOOL_NAME_CHARS for name in all_tool_calls)
                ),
            )
        )
    return EvaluationReport(tuple(results), (time.perf_counter() - started) * 1000)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise EvaluationError(f"{name} must be an array of non-empty strings")
    return tuple(value)


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without treating booleans as numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _case_size(case: EvaluationCase) -> int:
    """Measure one case's canonical serialized size for programmatic suite bounds."""
    try:
        return len(
            json.dumps(
                {
                    "id": case.id,
                    "input": case.input,
                    "context": case.context,
                    "memory": case.memory,
                    "metadata": case.metadata,
                    "expected_output": case.expected_output,
                    **(
                        {"expected_structured_output": case.expected_structured_output}
                        if case.expected_structured_output is not _NO_STRUCTURED_EXPECTATION
                        else {}
                    ),
                    "expected_contains": case.expected_contains,
                    "required_tools": case.required_tools,
                    "forbidden_tools": case.forbidden_tools,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise EvaluationError("evaluation case contains invalid JSON values") from None
