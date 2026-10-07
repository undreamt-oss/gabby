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
"""Structured planner contracts and bounded provider interactions."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gabby.models import ModelRequestSizeError, ModelResponse, ModelResponseSizeError
from gabby.planning import (
    ExecutionPlan,
    ModelPlanner,
    PlanningCapability,
    PlanStep,
    validate_execution_plan,
)


class StubProvider:
    name = "planner-test"
    supports_tool_choice = False

    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return self.response


def _planner(response: ModelResponse, **kwargs: Any) -> tuple[ModelPlanner, StubProvider]:
    provider = StubProvider(response)
    return ModelPlanner(provider=provider, model_id="plan-model", **kwargs), provider


def _json_response(content: str) -> ModelResponse:
    return ModelResponse(content=content, usage={"prompt_tokens": 12, "completion_tokens": 8})


@pytest.mark.asyncio
async def test_model_planner_returns_validated_plan_and_bounded_request() -> None:
    content = json.dumps(
        {
            "summary": "Check the reported failure.",
            "steps": [
                {
                    "instruction": "Inspect the error context.",
                    "success_criteria": "The failing component is identified.",
                }
            ],
        }
    )
    planner, provider = _planner(_json_response(content))
    capabilities = [PlanningCapability("skill", "debugging", "Diagnose failures.")]

    result = await planner.plan(
        task="Why did this fail?",
        context={"source": "service"},
        memory={"case": "one-run-only"},
        capabilities=capabilities,
    )

    assert result.plan.as_dict() == json.loads(content)
    assert result.model == "plan-model"
    assert result.usage == {"prompt_tokens": 12, "completion_tokens": 8}
    call = provider.calls[0]
    assert call["model"] == "plan-model"
    assert call["temperature"] == 0
    assert call["timeout_seconds"] == 20
    assert call["tools"] == []
    payload = json.loads(call["messages"][1]["content"])
    assert payload == {
        "task": "Why did this fail?",
        "context": {"source": "service"},
        "memory": {"case": "one-run-only"},
        "capabilities": [
            {"kind": "skill", "name": "debugging", "description": "Diagnose failures."}
        ],
    }


@pytest.mark.asyncio
async def test_model_planner_rejects_oversized_request_before_provider_call() -> None:
    planner, provider = _planner(
        _json_response('{"summary":"ok","steps":[]}'), max_model_request_bytes=1
    )
    with pytest.raises(ModelRequestSizeError):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])
    assert provider.calls == []


@pytest.mark.asyncio
async def test_model_planner_enforces_provider_response_bound() -> None:
    planner, _ = _planner(_json_response('{"summary":"ok","steps":[]}'), max_model_response_bytes=8)
    with pytest.raises(ModelResponseSizeError):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        ModelResponse(content=None),
        ModelResponse(content="{}", tool_calls=[{"id": "call"}]),
    ],
)
async def test_model_planner_requires_plain_text_without_tool_calls(
    response: ModelResponse,
) -> None:
    planner, _ = _planner(response)
    with pytest.raises(ValueError, match="JSON text response"):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "invalid JSON"),
        ('{"summary":"a","summary":"b","steps":[]}', "invalid JSON"),
        ('{"summary":"a","steps":[],"value":NaN}', "invalid JSON"),
    ],
)
async def test_model_planner_rejects_invalid_json(content: str, message: str) -> None:
    planner, _ = _planner(_json_response(content))
    with pytest.raises(ValueError, match=message):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "plan shape"),
        ({"summary": "ok"}, "plan shape"),
        ({"summary": " ", "steps": []}, "plan fields"),
        ({"summary": "ok", "steps": "step"}, "plan fields"),
        ({"summary": "ok", "steps": []}, "number of steps"),
        (
            {"summary": "ok", "steps": [{"instruction": "do it"}]},
            "step shape",
        ),
        (
            {
                "summary": "ok",
                "steps": [{"instruction": " ", "success_criteria": "done"}],
            },
            "step fields",
        ),
    ],
)
async def test_model_planner_rejects_invalid_plan_fields(payload: object, message: str) -> None:
    planner, _ = _planner(_json_response(json.dumps(payload)))
    with pytest.raises(ValueError, match=message):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.asyncio
async def test_model_planner_rejects_too_many_steps_and_oversized_plan_text() -> None:
    step = {"instruction": "do", "success_criteria": "done"}
    too_many = _planner(_json_response(json.dumps({"summary": "ok", "steps": [step] * 17})))[0]
    with pytest.raises(ValueError, match="number of steps"):
        await too_many.plan(task="task", context={}, memory={}, capabilities=[])

    too_long = _planner(_json_response(json.dumps({"summary": "x" * 1201, "steps": [step]})))[0]
    with pytest.raises(ValueError, match="plan fields"):
        await too_long.plan(task="task", context={}, memory={}, capabilities=[])

    too_many_chars = _planner(_json_response("x" * (16 * 1024 + 1)))[0]
    with pytest.raises(ValueError, match="response exceeded the size limit"):
        await too_many_chars.plan(task="task", context={}, memory={}, capabilities=[])

    too_many_utf8_bytes = _planner(_json_response("é" * 9000))[0]
    with pytest.raises(ValueError, match="response exceeded the size limit"):
        await too_many_utf8_bytes.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.asyncio
async def test_model_planner_rejects_invalid_usage_shape() -> None:
    response = _json_response(
        '{"summary":"ok","steps":[{"instruction":"do","success_criteria":"done"}]}'
    )
    response.usage = []  # type: ignore[assignment]
    planner, _ = _planner(response)
    with pytest.raises(ValueError, match="invalid token usage"):
        await planner.plan(task="task", context={}, memory={}, capabilities=[])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": " "},
        {"timeout_seconds": True},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": 0},
        {"max_model_request_bytes": True},
        {"max_model_request_bytes": 0},
        {"max_model_response_bytes": False},
        {"max_model_response_bytes": 0},
    ],
)
def test_model_planner_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    provider = StubProvider(_json_response("{}"))
    config: dict[str, Any] = {"model_id": "plan-model", **kwargs}
    with pytest.raises(ValueError):
        ModelPlanner(provider=provider, **config)


def test_validate_execution_plan_returns_a_validated_snapshot() -> None:
    original = ExecutionPlan(
        summary="A valid summary.",
        steps=(PlanStep("Inspect the result.", "A finding is recorded."),),
    )
    validated = validate_execution_plan(original)
    assert validated == original
    assert validated is not original

    with pytest.raises(ValueError, match="invalid plan"):
        validate_execution_plan({"summary": "bad", "steps": []})

    invalid = ExecutionPlan(summary=" ", steps=original.steps)
    with pytest.raises(ValueError, match="plan fields"):
        validate_execution_plan(invalid)
