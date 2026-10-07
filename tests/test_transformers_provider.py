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
"""Local Transformers adapter contracts using a deterministic fake backend."""

from __future__ import annotations

import asyncio
import builtins
import sys
import threading
import time
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from gabby import Agent, AgentDefinition, TransformersProvider
from gabby.config import ConfigError
from gabby.models import (
    ModelError,
    _transformers_messages,
    _transformers_response,
    _unwrap_transformers_json_fence,
)


class _Tensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))
        self.device = "cpu"

    def __getitem__(self, key: int | slice) -> _Tensor | list[int]:
        if isinstance(key, int):
            return _Tensor(self.values)
        return self.values[key]

    def tolist(self) -> list[int]:
        return list(self.values)


class _Inputs(dict[str, _Tensor]):
    def to(self, device: str) -> _Inputs:
        assert device == "cpu"
        return self


class _Output:
    def __init__(self, prefix: list[int]) -> None:
        self.values = [*prefix, 10, 11]

    def __getitem__(self, index: int) -> _Tensor:
        assert index == 0
        return _Tensor(self.values)


class _StopCriteria:
    pass


class _Torch:
    bool = bool

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()

    @staticmethod
    def full(
        shape: tuple[int, ...], value: builtins.bool, *, dtype: Any, device: str
    ) -> list[builtins.bool]:
        assert shape == (1,)
        assert dtype is bool
        assert device == "cpu"
        return [value]


class _Tokenizer:
    def __init__(self, parsed: dict[str, Any] | None = None) -> None:
        self.parsed = parsed
        self.request: dict[str, Any] | None = None
        self.parse_response: Any = self._parse_response

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Inputs:
        self.request = {"messages": messages, **kwargs}
        return _Inputs(input_ids=_Tensor([1, 2, 3] if kwargs.get("tools") else [1, 2]))

    def decode(self, generated: list[int], *, skip_special_tokens: bool) -> str:
        assert generated == [10, 11]
        assert skip_special_tokens is (not bool(self.request and self.request.get("tools")))
        return "raw generated text"

    def _parse_response(
        self,
        content: str,
        *,
        prefix: _Tensor,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert content == "raw generated text"
        assert prefix.tolist() == [1, 2, 3]
        assert tools
        assert self.parsed is not None
        return self.parsed


class _Model:
    device = "cpu"

    def generate(self, **kwargs: Any) -> _Output:
        assert kwargs["max_new_tokens"] == 32
        assert kwargs["do_sample"] is False
        assert kwargs["stopping_criteria"][0](_Tensor([1]), None) == [False]
        return _Output(kwargs["input_ids"].tolist())


class _LoadableModel(_Model):
    def __init__(self) -> None:
        self.loaded_device: str | None = None
        self.evaluated = False

    def to(self, device: str) -> _LoadableModel:
        self.loaded_device = device
        return self

    def eval(self) -> None:
        self.evaluated = True


async def _provider_response(
    tokenizer: _Tokenizer,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    provider = TransformersProvider(model_id="local/test", max_new_tokens=32)
    provider._load_model = lambda: (tokenizer, _Model(), _Torch, _StopCriteria)  # type: ignore[method-assign]
    return await provider.complete(
        messages=[{"role": "user", "content": "hello"}],
        tools=tools or [],
        model="local/test",
        temperature=0,
        timeout_seconds=2,
    )


@pytest.mark.asyncio
async def test_transformers_provider_completes_off_loop_with_bounded_generation() -> None:
    tokenizer = _Tokenizer()
    response = await _provider_response(tokenizer)

    assert response.content == "raw generated text"
    assert response.tool_calls == []
    assert tokenizer.request is not None
    assert tokenizer.request["add_generation_prompt"] is True


@pytest.mark.asyncio
async def test_transformers_provider_parses_and_normalizes_tool_calls() -> None:
    tokenizer = _Tokenizer(
        {"content": None, "tool_calls": [{"name": "sum", "arguments": {"value": 7}}]}
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "sum",
                "description": "Add a value.",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
            },
        }
    ]
    response = await _provider_response(tokenizer, tools=tools)

    assert response.content is None
    assert response.tool_calls[0]["function"]["name"] == "sum"
    assert response.tool_calls[0]["function"]["arguments"] == '{"value": 7}'
    assert response.tool_calls[0]["id"].startswith("transformers-")


def test_transformers_provider_configuration_is_lazy_and_rejects_inline_secrets() -> None:
    agent = Agent(
        AgentDefinition(
            name="local-agent",
            model={"provider": "transformers", "model": "local/test", "local_files_only": True},
        )
    )
    assert isinstance(agent.model, TransformersProvider)
    assert agent.model.local_files_only is True

    with pytest.raises(ConfigError, match="api_key"):
        TransformersProvider.from_config(
            {"provider": "transformers", "model": "local/test", "api_key": "secret"}
        )


def test_transformers_provider_snapshots_bounded_tool_response_template() -> None:
    template = {
        "start_anchor": "<|im_start|>assistant",
        "fields": {
            "tool_calls": {
                "open": "<tool_call>",
                "close": "</tool_call>",
                "content": "json",
                "repeats": True,
            }
        },
    }
    provider = TransformersProvider.from_config(
        {
            "provider": "transformers",
            "model": "local/test",
            "tool_response_template": template,
        }
    )
    template["start_anchor"] = "mutated"

    assert provider.tool_response_template is not None
    assert provider.tool_response_template["start_anchor"] == "<|im_start|>assistant"
    with pytest.raises(ConfigError, match="tool_response_template must be a JSON object"):
        TransformersProvider.from_config(
            {
                "provider": "transformers",
                "model": "local/test",
                "tool_response_template": [],
            }
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"api_key_env": "bad name"},
        {"revision": 7},
        {"adapter_id": ""},
        {"adapter_revision": "abc"},
        {"adapter_revision": 7},
        {"cache_dir": 7},
        {"device": 7},
        {"local_files_only": "yes"},
        {"max_new_tokens": 0},
        {"max_input_tokens": True},
    ],
)
def test_transformers_provider_rejects_invalid_configuration(settings: dict[str, Any]) -> None:
    config = {"provider": "transformers", "model": "local/test", **settings}
    with pytest.raises(ConfigError):
        TransformersProvider.from_config(config)


def test_transformers_provider_loads_safely_and_caches_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    tokenizer = _Tokenizer()
    model = _LoadableModel()
    torch_module = ModuleType("torch")
    torch_module.__dict__.update(
        {"bool": bool, "inference_mode": _Torch.inference_mode, "full": _Torch.full}
    )
    transformers_module = ModuleType("transformers")

    def load_tokenizer(model_id: str, **options: Any) -> _Tokenizer:
        calls.append(("tokenizer", model_id, options))
        return tokenizer

    def load_model(model_id: str, **options: Any) -> _LoadableModel:
        calls.append(("model", model_id, options))
        return model

    transformers_module.__dict__.update(
        {
            "AutoTokenizer": SimpleNamespace(from_pretrained=load_tokenizer),
            "AutoModelForCausalLM": SimpleNamespace(from_pretrained=load_model),
            "StoppingCriteria": _StopCriteria,
        }
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setenv("TEST_HF_TOKEN", "secret-test-token")
    provider = TransformersProvider(
        model_id="org/model",
        token_env="TEST_HF_TOKEN",
        revision="a" * 40,
        cache_dir="/tmp/model-cache",
        local_files_only=True,
        device="cpu",
    )

    loaded = provider._load_model()
    assert loaded == (tokenizer, model, torch_module, _StopCriteria)
    assert provider._load_model() == loaded
    assert len(calls) == 2
    assert calls[0][1] == "org/model"
    assert calls[0][2] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": "secret-test-token",
        "revision": "a" * 40,
        "cache_dir": "/tmp/model-cache",
    }
    assert calls[1][2]["dtype"] == "auto"
    assert calls[1][2]["use_safetensors"] is True
    assert model.loaded_device == "cpu"
    assert model.evaluated
    provider._clear_model()
    assert provider._model is None
    assert provider._tokenizer is None


def test_transformers_provider_loads_peft_adapter_for_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _Tokenizer()
    base_model = _LoadableModel()
    adapter_calls: list[tuple[Any, str, dict[str, Any]]] = []
    torch_module = ModuleType("torch")
    transformers_module = ModuleType("transformers")
    transformers_module.__dict__.update(
        {
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
            "AutoModelForCausalLM": SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: base_model
            ),
            "StoppingCriteria": _StopCriteria,
        }
    )
    peft_module = ModuleType("peft")

    def load_adapter(model: Any, adapter_id: str, **options: Any) -> Any:
        adapter_calls.append((model, adapter_id, options))
        return model

    peft_module.__dict__["PeftModel"] = SimpleNamespace(from_pretrained=load_adapter)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setenv("TEST_HF_TOKEN", "adapter-test-token")
    provider = TransformersProvider(
        model_id="org/base-model",
        token_env="TEST_HF_TOKEN",
        revision="base-revision",
        adapter_id="org/adapter",
        adapter_revision="adapter-revision",
        cache_dir="/tmp/model-cache",
        local_files_only=True,
    )

    _tokenizer, loaded_model, _torch, _stopping = provider._load_model()

    assert loaded_model is base_model
    assert adapter_calls == [
        (
            base_model,
            "org/adapter",
            {
                "is_trainable": False,
                "local_files_only": True,
                "use_safetensors": True,
                "token": "adapter-test-token",
                "revision": "adapter-revision",
                "cache_dir": "/tmp/model-cache",
            },
        )
    ]
    assert base_model.loaded_device == "cpu"
    assert base_model.evaluated


def test_transformers_provider_reports_missing_peft_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_module = ModuleType("torch")
    transformers_module = ModuleType("transformers")
    transformers_module.__dict__.update(
        {
            "AutoTokenizer": SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: _Tokenizer()
            ),
            "AutoModelForCausalLM": SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: _LoadableModel()
            ),
            "StoppingCriteria": _StopCriteria,
        }
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", None)
    provider = TransformersProvider(model_id="org/base", adapter_id="org/adapter")

    with pytest.raises(ModelError, match="'transformers-adapters'"):
        provider._load_model()


def test_transformers_provider_load_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TransformersProvider(model_id="org/model")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ModelError, match="optional 'transformers'"):
        provider._load_model()

    torch_module = ModuleType("torch")
    transformers_module = ModuleType("transformers")

    def fail_load(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("private repository detail")

    transformers_module.__dict__.update(
        {
            "AutoTokenizer": SimpleNamespace(from_pretrained=fail_load),
            "AutoModelForCausalLM": SimpleNamespace(from_pretrained=fail_load),
            "StoppingCriteria": _StopCriteria,
        }
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    with pytest.raises(ModelError, match="could not be loaded") as error:
        provider._load_model()
    assert "private repository detail" not in str(error.value)


def test_transformers_message_adapter_normalizes_prior_tool_history() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "lookup", "arguments": '{"query": "x"}'}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    normalized = _transformers_messages(messages)
    assert normalized[0]["tool_calls"] == [{"name": "lookup", "arguments": {"query": "x"}}]
    assert normalized[1]["tool_call_id"] == "call-1"
    with pytest.raises(ModelError, match="invalid JSON"):
        _transformers_messages(
            cast(
                list[dict[str, Any]],
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "x", "arguments": "{"}}],
                    }
                ],
            )
        )
    with pytest.raises(ModelError, match="not compatible"):
        _transformers_messages([{"role": "assistant", "tool_calls": [None]}])


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {"content": 7},
        {"content": None, "tool_calls": "invalid"},
        {"content": None, "tool_calls": [{"name": "other", "arguments": {}}]},
        {"content": None, "tool_calls": [{"name": "lookup", "arguments": []}]},
    ],
)
def test_transformers_tool_response_parser_rejects_invalid_shapes(parsed: Any) -> None:
    class Parser:
        def parse_response(
            self,
            content: str,
            *,
            prefix: str,
            tools: list[dict[str, Any]],
        ) -> Any:
            del content, prefix, tools
            return parsed

    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with pytest.raises(ModelError):
        _transformers_response(Parser(), "generated", tools, prefix="prompt")


def test_transformers_tool_response_parser_sanitizes_parser_errors() -> None:
    class Parser:
        def parse_response(
            self,
            content: str,
            *,
            prefix: str,
            tools: list[dict[str, Any]],
        ) -> Any:
            del content, prefix, tools
            raise RuntimeError("private parser detail")

    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with pytest.raises(ModelError, match="could not be parsed") as error:
        _transformers_response(Parser(), "generated", tools, prefix="prompt")
    assert "private parser detail" not in str(error.value)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('```json\n{"name":"sum_values"}\n```<|im_end|>', '{"name":"sum_values"}<|im_end|>'),
        ('```\n{"name":"sum_values"}\n```', '{"name":"sum_values"}'),
        ("ordinary text", "ordinary text"),
        (
            '```json\n{"name":"sum_values"}\n``` trailing text',
            '```json\n{"name":"sum_values"}\n``` trailing text',
        ),
    ],
)
def test_transformers_json_fence_unwrap_is_narrow(content: str, expected: str) -> None:
    assert _unwrap_transformers_json_fence(content) == expected


@pytest.mark.asyncio
async def test_transformers_provider_rejects_tools_without_structured_parser() -> None:
    class NoParser(_Tokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.parse_response = None

    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with pytest.raises(ModelError, match="does not provide structured"):
        await _provider_response(NoParser(), tools=tools)


@pytest.mark.asyncio
async def test_transformers_provider_rejects_chat_templates_that_ignore_tools() -> None:
    class IgnoresTools(_Tokenizer):
        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Inputs:
            self.request = {"messages": messages, **kwargs}
            return _Inputs(input_ids=_Tensor([1, 2, 3]))

    class NeverGenerate(_Model):
        def __init__(self) -> None:
            self.called = False

        def generate(self, **kwargs: Any) -> _Output:
            self.called = True
            return super().generate(**kwargs)

    tokenizer = IgnoresTools()
    model = NeverGenerate()
    provider = TransformersProvider(model_id="local/test", max_new_tokens=32)
    provider._load_model = lambda: (tokenizer, model, _Torch, _StopCriteria)  # type: ignore[method-assign]
    tools = [
        {
            "type": "function",
            "function": {"name": "sum", "description": "Add.", "parameters": {}},
        }
    ]

    with pytest.raises(ModelError, match="did not include the configured tool schemas"):
        await provider.complete(
            messages=[{"role": "user", "content": "Call sum."}],
            tools=tools,
            model="local/test",
            temperature=0,
            timeout_seconds=2,
        )

    assert not model.called


@pytest.mark.asyncio
async def test_transformers_provider_timeout_requests_generation_cancellation() -> None:
    stopped = threading.Event()

    class SlowModel(_Model):
        def generate(self, **kwargs: Any) -> _Output:
            criterion = kwargs["stopping_criteria"][0]
            while True:
                if criterion(_Tensor([1]), None) == [True]:
                    stopped.set()
                    return _Output([1])
                time.sleep(0.005)

    provider = TransformersProvider(model_id="local/test", max_new_tokens=32)
    provider._load_model = lambda: (  # type: ignore[method-assign]
        _Tokenizer(),
        SlowModel(),
        _Torch,
        _StopCriteria,
    )
    with pytest.raises(ModelError, match="exceeded its timeout"):
        await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            model="local/test",
            timeout_seconds=0.02,
        )
    for _ in range(100):
        if stopped.is_set():
            break
        await asyncio.sleep(0.01)
    assert stopped.is_set()
