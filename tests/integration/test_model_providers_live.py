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
"""Opt-in acceptance checks against configured hosted and local model providers."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from gabby.agent import Agent
from gabby.config import AgentDefinition
from gabby.models import TransformersProvider


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "run_flag", "model_env"),
    [
        ("ollama", "GABBY_RUN_OLLAMA_INTEGRATION", "GABBY_OLLAMA_MODEL"),
        ("huggingface", "GABBY_RUN_HUGGINGFACE_INTEGRATION", "GABBY_HUGGINGFACE_MODEL"),
        ("anthropic", "GABBY_RUN_ANTHROPIC_INTEGRATION", "GABBY_ANTHROPIC_MODEL"),
    ],
)
async def test_live_model_provider_completion_and_stream(
    provider: str, run_flag: str, model_env: str
) -> None:
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to call the live {provider} provider")
    model_id = os.environ.get(model_env)
    if not model_id:
        pytest.fail(f"set {model_env} to a model available to {provider}")
    if provider == "huggingface" and not os.environ.get("HF_TOKEN"):
        pytest.fail("set HF_TOKEN in the environment before calling Hugging Face")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("set ANTHROPIC_API_KEY in the environment before calling Anthropic")

    model: dict[str, Any] = {"provider": provider, "model": model_id}
    endpoint_env = {
        "ollama": "GABBY_OLLAMA_BASE_URL",
        "huggingface": "GABBY_HUGGINGFACE_BASE_URL",
        "anthropic": "GABBY_ANTHROPIC_BASE_URL",
    }[provider]
    if base_url := os.environ.get(endpoint_env):
        model["base_url"] = base_url

    definition = AgentDefinition(
        name=f"{provider}-acceptance",
        model=model,
        instructions="Reply briefly in plain text.",
        policies={"max_steps": 1, "timeout_seconds": 180},
    )
    agent = Agent(definition)
    try:
        result = await agent.arun("Reply with a short greeting.")
        assert result.output.strip(), "provider returned an empty completion"

        events = [event async for event in agent.astream("Reply with a short greeting.")]
        text_deltas = [
            event.data.get("text")
            for event in events
            if event.type == "text_delta" and isinstance(event.data.get("text"), str)
        ]
        completions = [
            event.data.get("result")
            for event in events
            if event.type == "completed" and isinstance(event.data.get("result"), dict)
        ]
        assert any(isinstance(text, str) and text.strip() for text in text_deltas), (
            "provider stream emitted no text deltas"
        )
        assert len(completions) == 1, "agent stream did not emit one completion event"
        completion = completions[0]
        assert isinstance(completion, dict), "stream completion result was malformed"
        output = completion.get("output")
        assert isinstance(output, str) and output.strip(), "stream completion output was empty"
    finally:
        await agent.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_local_transformers_model_completion() -> None:
    run_flag = "GABBY_RUN_TRANSFORMERS_INTEGRATION"
    model_env = "GABBY_TRANSFORMERS_MODEL"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to load a local Transformers model")
    model_id = os.environ.get(model_env)
    if not model_id:
        pytest.fail(f"set {model_env} to a local path or Hugging Face model ID")

    model: dict[str, Any] = {
        "provider": "transformers",
        "model": model_id,
        "device": os.environ.get("GABBY_TRANSFORMERS_DEVICE", "cpu"),
        "local_files_only": os.environ.get("GABBY_TRANSFORMERS_LOCAL_FILES_ONLY", "1") != "0",
        "max_input_tokens": 16384,
        "max_new_tokens": 128,
    }
    if revision := os.environ.get("GABBY_TRANSFORMERS_REVISION"):
        model["revision"] = revision
    definition = AgentDefinition(
        name="transformers-acceptance",
        model=model,
        instructions="Reply briefly in plain text.",
        policies={"max_steps": 1, "timeout_seconds": 300},
    )
    agent = Agent(definition)
    try:
        result = await agent.arun("Reply with a short greeting.")
        assert result.output.strip(), "local Transformers model returned an empty completion"
    finally:
        await agent.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_local_transformers_structured_tool_call() -> None:
    run_flag = "GABBY_RUN_TRANSFORMERS_TOOL_INTEGRATION"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to verify local Transformers tool calling")
    model_id = os.environ.get("GABBY_TRANSFORMERS_MODEL")
    if not model_id:
        pytest.fail("set GABBY_TRANSFORMERS_MODEL to a local path or Hugging Face model ID")

    model: dict[str, Any] = {
        "provider": "transformers",
        "model": model_id,
        "device": os.environ.get("GABBY_TRANSFORMERS_DEVICE", "cpu"),
        "local_files_only": os.environ.get("GABBY_TRANSFORMERS_LOCAL_FILES_ONLY", "1") != "0",
        "max_input_tokens": 16384,
        "max_new_tokens": 128,
    }
    if revision := os.environ.get("GABBY_TRANSFORMERS_REVISION"):
        model["revision"] = revision
    if template := os.environ.get("GABBY_TRANSFORMERS_TOOL_RESPONSE_TEMPLATE"):
        model["tool_response_template"] = json.loads(template)

    provider = TransformersProvider.from_config(model)
    tool_schema = {
        "type": "function",
        "function": {
            "name": "sum_values",
            "description": "Add two integers and return their sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    response = await provider.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You must call the sum_values tool exactly once for arithmetic. "
                    "Return only the tool call and no other content."
                ),
            },
            {"role": "user", "content": "Calculate 2 + 3."},
        ],
        tools=[tool_schema],
        model=model_id,
        timeout_seconds=300,
    )
    assert len(response.tool_calls) == 1, f"expected one structured tool call: {response!r}"
    function = response.tool_calls[0]["function"]
    assert function["name"] == "sum_values"
    assert json.loads(function["arguments"]) == {"a": 2, "b": 3}
