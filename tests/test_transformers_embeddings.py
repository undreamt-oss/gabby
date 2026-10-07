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
"""Contracts for the optional local Transformers embedding provider."""

from __future__ import annotations

import asyncio
import math
import sys
import threading
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gabby import EmbeddingError, TransformersEmbeddingProvider


class _Tensor:
    def __init__(self, values: Any) -> None:
        self.values = values
        self.shape = self._shape(values)
        self.dtype = "float32"

    @classmethod
    def _shape(cls, values: Any) -> tuple[int, ...]:
        if not isinstance(values, list):
            return ()
        return (len(values), *cls._shape(values[0])) if values else (0,)

    def __getitem__(self, key: tuple[slice, int, slice]) -> _Tensor:
        _, token_index, _ = key
        return _Tensor([row[token_index] for row in self.values])

    def unsqueeze(self, dimension: int) -> _Tensor:
        assert dimension == -1
        return _Tensor([[[value] for value in row] for row in self.values])

    def to(self, *_args: Any, **_kwargs: Any) -> _Tensor:
        return self

    def __mul__(self, other: _Tensor) -> _Tensor:
        return _Tensor(
            [
                [
                    [value * weight[0] for value in token]
                    for token, weight in zip(row, other_row, strict=True)
                ]
                for row, other_row in zip(self.values, other.values, strict=True)
            ]
        )

    def sum(self, *, dim: int) -> _Tensor:
        assert dim == 1
        if len(self.shape) == 3:
            return _Tensor(
                [
                    [sum(token[column] for token in row) for column in range(self.shape[2])]
                    for row in self.values
                ]
            )
        return _Tensor([[sum(token[0] for token in row)] for row in self.values])

    def clamp(self, *, min: float) -> _Tensor:
        return _Tensor([[max(value, min) for value in row] for row in self.values])

    def __truediv__(self, other: _Tensor) -> _Tensor:
        return _Tensor(
            [
                [value / denominator[0] for value in row]
                for row, denominator in zip(self.values, other.values, strict=True)
            ]
        )

    def detach(self) -> _Tensor:
        return self

    def tolist(self) -> Any:
        return self.values


class _Batch(dict[str, _Tensor]):
    def to(self, _device: str) -> _Batch:
        return self


class _Tokenizer:
    def __init__(self) -> None:
        self.options: dict[str, Any] | None = None

    def __call__(self, texts: list[str], **options: Any) -> _Batch:
        self.options = options
        return _Batch(
            input_ids=_Tensor([[index + 1, index + 2] for index in range(len(texts))]),
            attention_mask=_Tensor([[1, 1], [1, 0]][: len(texts)]),
        )


class _Encoder:
    def __call__(self, **inputs: _Tensor) -> SimpleNamespace:
        batch_size = len(inputs["input_ids"].values)
        return SimpleNamespace(
            last_hidden_state=_Tensor(
                [
                    [[1.0, 0.0], [3.0, 4.0]],
                    [[0.0, 2.0], [9.0, 9.0]],
                ][:batch_size]
            )
        )


class _Torch:
    float32 = "float32"

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()

    class nn:
        class functional:
            @staticmethod
            def normalize(tensor: _Tensor, *, p: int, dim: int) -> _Tensor:
                assert p == 2 and dim == 1
                return _Tensor(
                    [
                        [value / math.sqrt(sum(item * item for item in row)) for value in row]
                        for row in tensor.values
                    ]
                )


def _provider(**kwargs: Any) -> TransformersEmbeddingProvider:
    provider = TransformersEmbeddingProvider(model_id="sentence/local", **kwargs)
    provider._load_model = lambda: (_Tokenizer(), _Encoder(), _Torch)  # type: ignore[method-assign]
    return provider


@pytest.mark.asyncio
async def test_local_transformers_provider_mean_pools_masks_and_normalizes() -> None:
    provider = _provider()
    vectors = await provider.embed(["first", "second"])
    await provider.aclose()

    assert vectors[0] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_local_transformers_provider_converts_low_precision_pooling_before_normalization() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    requested_dtypes: list[Any] = []

    class LowPrecisionTensor(_Tensor):
        def to(self, *_args: Any, **kwargs: Any) -> _Tensor:
            requested_dtypes.append(kwargs.get("dtype"))
            return _Tensor(self.values)

    provider._load_model = lambda: (_Tokenizer(), _Encoder(), _Torch)  # type: ignore[method-assign]
    provider._pool = lambda _hidden, _inputs: LowPrecisionTensor([[3.0, 4.0]])  # type: ignore[assignment]

    vectors = provider._embed_sync(["first"], threading.Event())

    assert requested_dtypes == [_Torch.float32]
    assert vectors[0] == pytest.approx([0.6, 0.8])


@pytest.mark.asyncio
async def test_local_transformers_provider_supports_cls_pooling_and_no_normalization() -> None:
    provider = _provider(pooling="cls", normalize=False)
    assert await provider.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 2.0]]
    await provider.aclose()


@pytest.mark.asyncio
async def test_local_transformers_provider_batches_with_bounded_tokenization() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local", batch_size=1)
    calls: list[list[str]] = []

    class RecordingTokenizer(_Tokenizer):
        def __call__(self, texts: list[str], **options: Any) -> _Batch:
            calls.append(texts)
            return super().__call__(texts, **options)

    tokenizer = RecordingTokenizer()
    provider._load_model = lambda: (tokenizer, _Encoder(), _Torch)  # type: ignore[method-assign]
    vectors = await provider.embed(["first", "second"])
    assert len(calls) == 2
    assert tokenizer.options == {
        "padding": True,
        "truncation": True,
        "max_length": 512,
        "return_tensors": "pt",
    }
    assert vectors[0] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    await provider.aclose()


@pytest.mark.asyncio
async def test_local_transformers_provider_bounds_inputs_and_returns_empty_without_loading() -> (
    None
):
    provider = TransformersEmbeddingProvider(model_id="sentence/local", max_input_bytes=3)
    assert await provider.embed([]) == []
    assert provider._model is None
    with pytest.raises(EmbeddingError, match="max_input_bytes=3"):
        await provider.embed(["four"])
    with pytest.raises(ValueError, match="non-empty strings"):
        await provider.embed([""])


@pytest.mark.asyncio
async def test_local_transformers_provider_bounds_vector_materialization() -> None:
    provider = _provider(max_output_bytes=15)
    with pytest.raises(EmbeddingError, match="max_output_bytes=15"):
        await provider.embed(["first", "second"])
    await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("texts", "error", "message"),
    [
        ("single string", TypeError, "sequence of strings"),
        (["one", "two"], EmbeddingError, "max_texts=1"),
        (["\ud800"], ValueError, "UTF-8 encodable"),
    ],
)
async def test_local_transformers_provider_rejects_invalid_input_sequences(
    texts: Any, error: type[Exception], message: str
) -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local", max_texts=1)
    with pytest.raises(error, match=message):
        await provider.embed(texts)


@pytest.mark.asyncio
async def test_local_transformers_provider_timeout_signals_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gabby.transformers_embeddings as module

    provider = TransformersEmbeddingProvider(model_id="sentence/local", timeout_seconds=0.01)
    observed: list[threading.Event] = []

    async def wait_for_timeout(
        _callback: Any, _texts: list[str], stop_event: threading.Event
    ) -> list[list[float]]:
        observed.append(stop_event)
        await asyncio.sleep(1)
        return [[1.0]]

    monkeypatch.setattr(module, "run_sync_callback", wait_for_timeout)
    with pytest.raises(EmbeddingError, match="exceeded its timeout"):
        await provider.embed(["query"])
    assert observed[0].is_set()


@pytest.mark.asyncio
async def test_local_transformers_provider_cancellation_signals_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gabby.transformers_embeddings as module

    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    observed: list[threading.Event] = []

    async def cancel_callback(
        _callback: Any, _texts: list[str], stop_event: threading.Event
    ) -> list[list[float]]:
        observed.append(stop_event)
        raise asyncio.CancelledError

    monkeypatch.setattr(module, "run_sync_callback", cancel_callback)
    with pytest.raises(asyncio.CancelledError):
        await provider.embed(["query"])
    assert observed[0].is_set()


@pytest.mark.asyncio
async def test_local_transformers_provider_rejects_invalid_callback_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gabby.transformers_embeddings as module

    async def invalid_result(*_args: Any) -> str:
        return "invalid"

    monkeypatch.setattr(module, "run_sync_callback", invalid_result)
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    with pytest.raises(EmbeddingError, match="invalid result"):
        await provider.embed(["query"])


@pytest.mark.parametrize(
    ("rows", "expected_count", "dimensions", "message"),
    [
        ([], 1, 2, "unexpected number"),
        ([[1.0]], 1, 2, "invalid embedding vector"),
        ([[True, 1.0]], 1, 2, "non-numeric"),
        ([[float("inf"), 1.0]], 1, 2, "non-finite"),
        ([[0.0, 0.0]], 1, 2, "zero or invalid"),
    ],
)
def test_local_transformers_provider_rejects_invalid_vectors(
    rows: Any, expected_count: int, dimensions: int, message: str
) -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    with pytest.raises(EmbeddingError, match=message):
        provider._validate_vectors(rows, expected_count, dimensions)


def test_local_transformers_provider_stops_before_the_next_batch() -> None:
    provider = _provider(batch_size=1)
    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(EmbeddingError, match="was cancelled"):
        provider._embed_sync(["query"], stop_event)


def test_local_transformers_provider_rejects_missing_attention_mask() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    with pytest.raises(EmbeddingError, match="did not return an attention mask"):
        provider._pool(_Tensor([[[1.0, 2.0]]]), {})


def test_local_transformers_provider_rejects_invalid_model_vector_size() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local")

    class Encoder:
        def __call__(self, **_inputs: Any) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=SimpleNamespace(shape=(1, 1, 65_537)))

    provider._load_model = lambda: (_Tokenizer(), Encoder(), _Torch)  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="invalid vector size"):
        provider._embed_sync(["query"], threading.Event())


def test_local_transformers_provider_rejects_dimension_changes_between_batches() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local", batch_size=1)

    class ChangingEncoder:
        calls = 0

        def __call__(self, **_inputs: Any) -> SimpleNamespace:
            self.calls += 1
            size = self.calls + 1
            hidden = [[[1.0] * size, [2.0] * size]]
            return SimpleNamespace(last_hidden_state=_Tensor(hidden))

    provider._load_model = lambda: (_Tokenizer(), ChangingEncoder(), _Torch)  # type: ignore[method-assign]
    with pytest.raises(EmbeddingError, match="inconsistent vector dimensions"):
        provider._embed_sync(["first", "second"], threading.Event())


def test_local_transformers_provider_sanitizes_inference_failures() -> None:
    provider = TransformersEmbeddingProvider(model_id="sentence/local")

    class FailingTokenizer:
        def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("private input and model details")

    provider._load_model = lambda: (  # type: ignore[method-assign]
        FailingTokenizer(),
        _Encoder(),
        _Torch,
    )
    with pytest.raises(EmbeddingError, match="embedding failed") as error:
        provider._embed_sync(["query"], threading.Event())
    assert "private input" not in str(error.value)


def test_local_transformers_provider_handles_missing_torch_after_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    provider._tokenizer = _Tokenizer()
    provider._model = _Encoder()
    with pytest.raises(EmbeddingError, match="Install Gabby's optional"):
        provider._load_model()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pooling": "max"}, "pooling"),
        ({"normalize": 1}, "normalize"),
        ({"model_id": " "}, "model_id"),
        ({"revision": " "}, "revision"),
        ({"cache_dir": " "}, "cache_dir"),
        ({"local_files_only": 1}, "local_files_only"),
        ({"device": " "}, "device"),
        ({"max_input_tokens": 8193}, "max_input_tokens"),
        ({"batch_size": 65}, "batch_size"),
        ({"max_texts": 10001}, "max_texts"),
        ({"max_input_bytes": 0}, "max_input_bytes"),
        ({"max_output_bytes": True}, "max_output_bytes"),
        ({"token_env": "BAD-NAME"}, "token_env"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
    ],
)
def test_local_transformers_provider_validates_resource_configuration(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TransformersEmbeddingProvider(**{"model_id": "sentence/local", **options})


def test_local_transformers_provider_loads_safetensors_without_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Loadable:
        def to(self, device: str) -> Loadable:
            assert device == "cpu"
            return self

        def eval(self) -> None:
            return None

    def tokenizer_load(_model_id: str, **options: Any) -> _Tokenizer:
        calls.append(("tokenizer", options))
        return _Tokenizer()

    def model_load(_model_id: str, **options: Any) -> Loadable:
        calls.append(("model", options))
        return Loadable()

    monkeypatch.setenv("HF_TOKEN", "host-secret")
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_load)  # type: ignore[attr-defined]
    transformers.AutoModel = SimpleNamespace(from_pretrained=model_load)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    provider = TransformersEmbeddingProvider(
        model_id="sentence/local",
        revision="0123456789abcdef",
        cache_dir="/tmp/model-cache",
        local_files_only=True,
    )
    provider._load_model()

    assert calls[0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": "host-secret",
        "revision": "0123456789abcdef",
        "cache_dir": "/tmp/model-cache",
    }
    assert calls[1][1]["use_safetensors"] is True
    assert calls[1][1]["trust_remote_code"] is False


def test_local_transformers_provider_reuses_loaded_model_and_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    tokenizer = _Tokenizer()
    model = _Encoder()
    provider._tokenizer = tokenizer
    provider._model = model

    loaded_tokenizer, loaded_model, loaded_torch = provider._load_model()

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert loaded_torch is torch
    provider._clear_model()
    assert provider._tokenizer is None
    assert provider._model is None


def test_local_transformers_provider_sanitizes_model_loading_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = ModuleType("transformers")

    def fail_load(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private endpoint and credential details")

    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=fail_load)  # type: ignore[attr-defined]
    transformers.AutoModel = SimpleNamespace(from_pretrained=fail_load)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    with pytest.raises(EmbeddingError, match="could not be loaded") as error:
        provider._load_model()
    assert "private endpoint" not in str(error.value)


def test_local_transformers_provider_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    provider = TransformersEmbeddingProvider(model_id="sentence/local")
    with pytest.raises(EmbeddingError, match="Install Gabby's optional"):
        provider._load_model()
