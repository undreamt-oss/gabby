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
"""Validation and HTTP contracts for OpenAI-compatible embedding providers."""

from __future__ import annotations

import json
import traceback
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from gabby import (
    Document,
    EmbeddingError,
    EmbeddingResponseSizeError,
    GeminiEmbeddingProvider,
    HuggingFaceFeatureExtractionProvider,
    HybridIndexCoordinator,
    OpenAICompatibleEmbeddingProvider,
    SQLiteFTS5Store,
    SQLiteGenerationManifestStore,
    SQLiteVectorStore,
)

_ASYNC_CLIENT = httpx.AsyncClient


def _provider(
    handler: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> OpenAICompatibleEmbeddingProvider:
    transport = httpx.MockTransport(handler)

    def make_client(**client_kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **client_kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return OpenAICompatibleEmbeddingProvider(model="embed-test", **kwargs)


def _huggingface_provider(
    handler: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> HuggingFaceFeatureExtractionProvider:
    transport = httpx.MockTransport(handler)

    def make_client(**client_kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **client_kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return HuggingFaceFeatureExtractionProvider(model="org/embed-test", **kwargs)


@pytest.mark.asyncio
async def test_huggingface_provider_batches_and_pools_token_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[[[1, 2], [3, 4]], [[2, 4], [4, 6]]],
        )

    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    provider = _huggingface_provider(
        respond,
        monkeypatch,
        batch_size=2,
        truncate=True,
        truncation_direction="right",
        prompt_name="passage",
    )
    try:
        vectors = await provider.embed(["first", "second"])
    finally:
        await provider.aclose()

    assert vectors == [[2.0, 3.0], [3.0, 5.0]]
    assert requests[0].url == (
        "https://router.huggingface.co/hf-inference/models/org/embed-test/"
        "pipeline/feature-extraction"
    )
    assert requests[0].headers["authorization"] == "Bearer test-hf-token"
    assert json.loads(requests[0].content) == {
        "inputs": ["first", "second"],
        "truncate": True,
        "prompt_name": "passage",
        "truncation_direction": "right",
    }


def _gemini_provider(
    handler: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> GeminiEmbeddingProvider:
    transport = httpx.MockTransport(handler)

    def make_client(**client_kwargs: Any) -> httpx.AsyncClient:
        return _ASYNC_CLIENT(transport=transport, **client_kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return GeminiEmbeddingProvider(**kwargs)


@pytest.mark.asyncio
async def test_gemini_embeddings_batch_task_config_and_host_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        count = len(payload["requests"])
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": [float(i), 1.0, *([0.0] * 766)]} for i in range(count)]
            },
        )

    monkeypatch.setenv("GEMINI_API_KEY", "host-gemini-key")
    provider = _gemini_provider(
        respond,
        monkeypatch,
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        dimensions=768,
        batch_size=2,
    )
    try:
        vectors = await provider.embed(["first", "second", "third"])
    finally:
        await provider.aclose()

    assert len(vectors) == 3
    assert [vector[:2] for vector in vectors] == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert len(requests) == 2
    assert requests[0].url.path == "/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert requests[0].headers["x-goog-api-key"] == "host-gemini-key"
    assert json.loads(requests[0].content) == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "first"}]},
                "embedContentConfig": {
                    "taskType": "RETRIEVAL_QUERY",
                    "outputDimensionality": 768,
                },
            },
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "second"}]},
                "embedContentConfig": {
                    "taskType": "RETRIEVAL_QUERY",
                    "outputDimensionality": 768,
                },
            },
        ]
    }


@pytest.mark.asyncio
async def test_gemini_role_specific_tasks_work_through_hybrid_index_and_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed_task_types: list[str | None] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        vectors = []
        for item in payload["requests"]:
            observed_task_types.append(item.get("embedContentConfig", {}).get("taskType"))
            text = item["content"]["parts"][0]["text"].casefold()
            vectors.append({"values": [1.0, 0.0] if "red" in text else [0.0, 1.0]})
        return httpx.Response(200, json={"embeddings": vectors})

    embeddings = _gemini_provider(
        respond,
        monkeypatch,
        model="gemini-embedding-001",
        query_task_type="RETRIEVAL_QUERY",
        document_task_type="RETRIEVAL_DOCUMENT",
    )
    coordinator = HybridIndexCoordinator(
        SQLiteFTS5Store(tmp_path / "gemini-lexical.db"),
        embeddings,
        SQLiteVectorStore(tmp_path / "gemini-vectors.db"),
        SQLiteGenerationManifestStore(tmp_path / "gemini-manifest.db"),
    )
    try:
        await coordinator.replace_source(
            "catalog.md",
            [
                Document("red apples", "catalog.md", id="red"),
                Document("blue berries", "catalog.md", id="blue"),
            ],
        )
        results = await coordinator.retrieve("red", limit=2)
    finally:
        await embeddings.aclose()

    assert [document.id for document in results] == ["red", "blue"]
    assert observed_task_types == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]


@pytest.mark.asyncio
async def test_gemini_embedding_2_uses_batch_items_and_empty_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 2.0, *([0.0] * 766)]}]})

    provider = _gemini_provider(respond, monkeypatch, dimensions=768)
    try:
        assert await provider.embed([]) == []
        vectors = await provider.embed(["one"])
    finally:
        await provider.aclose()

    assert len(requests) == 1
    assert len(vectors[0]) == 768
    assert json.loads(requests[0].content)["requests"][0]["embedContentConfig"] == {
        "outputDimensionality": 768
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_type": "RETRIEVAL_QUERY"}, "does not support task_type"),
        ({"model": "gemini-embedding-001", "task_type": "UNKNOWN"}, "task_type"),
        ({"model": "gemini-embedding-001", "title": "Document"}, "title requires"),
        ({"dimensions": 64}, "dimensions"),
        ({"batch_size": 101}, "batch_size"),
        ({"base_url": "http://remote.example/v1beta"}, "requires HTTPS"),
    ],
)
def test_gemini_embedding_provider_validates_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GeminiEmbeddingProvider(**kwargs)


def test_gemini_embedding_config_rejects_inline_credentials() -> None:
    with pytest.raises(Exception, match="model.api_key"):
        GeminiEmbeddingProvider.from_config({"api_key": "secret"})


@pytest.mark.asyncio
async def test_gemini_embedding_request_and_response_bodies_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GeminiEmbeddingProvider(max_request_bytes=32)
    with pytest.raises(EmbeddingError, match="max_request_bytes"):
        await provider.embed(["x" * 100])

    oversized = _gemini_provider(
        lambda _request: httpx.Response(200, content=b'{"embeddings": []}'),
        monkeypatch,
        max_response_bytes=8,
    )
    try:
        with pytest.raises(EmbeddingResponseSizeError, match="max_response_bytes"):
            await oversized.embed(["one"])
    finally:
        await oversized.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": []},
        {"embeddings": [{"values": [1.0]}, {"values": [1.0]}]},
        {"embeddings": [{"values": [True]}]},
        {"embeddings": [{"values": [float("nan")]}]},
        {"embeddings": [{"values": []}]},
    ],
)
async def test_gemini_embedding_rejects_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    provider = _gemini_provider(lambda _request: httpx.Response(200, json=body), monkeypatch)
    try:
        with pytest.raises(EmbeddingError):
            await provider.embed(["one"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "response object"),
        ({"embeddings": []}, "unexpected number"),
        ({"embeddings": [{"values": [1.0]}, {"values": [1.0, 2.0]}]}, "inconsistent"),
    ],
)
async def test_gemini_provider_rejects_invalid_response_contracts(
    monkeypatch: pytest.MonkeyPatch, body: Any, message: str
) -> None:
    provider = _gemini_provider(lambda _: httpx.Response(200, json=body), monkeypatch)
    texts = (
        ["one", "two"]
        if isinstance(body, dict) and len(body.get("embeddings", [])) == 2
        else ["one"]
    )
    try:
        with pytest.raises(EmbeddingError, match=message):
            await provider.embed(texts)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_query_rejects_a_non_string_prefix() -> None:
    provider = GeminiEmbeddingProvider()
    with pytest.raises(TypeError, match="prefix must be a string"):
        await provider.embed_queries("query", prefix=1)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_kind", "message"),
    [
        ("openai", "non-finite vector value"),
        ("huggingface", "non-finite feature value"),
        ("gemini", "non-finite embedding value"),
    ],
)
async def test_embedding_adapters_reject_values_that_overflow_float(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, message: str
) -> None:
    if provider_kind == "openai":
        provider = _provider(
            lambda _: httpx.Response(200, json={"data": [{"index": 0, "embedding": [10**400]}]}),
            monkeypatch,
        )
    elif provider_kind == "huggingface":
        provider = _huggingface_provider(
            lambda _: httpx.Response(200, json=[[10**400]]), monkeypatch
        )
    else:
        provider = _gemini_provider(
            lambda _: httpx.Response(200, json={"embeddings": [{"values": [10**400]}]}),
            monkeypatch,
        )
    try:
        with pytest.raises(EmbeddingError, match=message):
            await provider.embed(["query"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_gemini_embedding_sanitizes_http_and_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _gemini_provider(
        lambda _request: httpx.Response(403, text="private upstream response"), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="HTTP 403") as error:
            await provider.embed(["one"])
        assert "private upstream response" not in str(error.value)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_huggingface_provider_supports_cls_pooling_and_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[1, 2], [3, 4]])

    provider = _huggingface_provider(respond, monkeypatch, pooling="cls")
    try:
        assert await provider.embed([]) == []
        assert await provider.embed(["one token sequence"]) == [[1.0, 2.0]]
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": "../outside"}, ValueError, "model ID"),
        ({"model": "org/model?token=x"}, ValueError, "model ID"),
        ({"pooling": "max"}, ValueError, "pooling"),
        ({"truncation_direction": "sideways"}, ValueError, "truncation_direction"),
        ({"truncation_direction": "left"}, ValueError, "requires truncate=True"),
        ({"base_url": "http://remote.test/models"}, ValueError, "requires HTTPS"),
    ],
)
def test_huggingface_provider_validates_configuration(
    kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    config: dict[str, Any] = {"model": "org/embed-test", **kwargs}
    with pytest.raises(error, match=message):
        HuggingFaceFeatureExtractionProvider(**config)


def test_huggingface_provider_config_rejects_inline_credentials() -> None:
    with pytest.raises(ValueError, match="model.api_key"):
        HuggingFaceFeatureExtractionProvider.from_config(
            {"model": "org/embed-test", "api_key": "inline-secret"}
        )

    provider = HuggingFaceFeatureExtractionProvider.from_config(
        {"model": "org/embed-test", "pooling": "cls", "batch_size": 4}
    )
    assert provider.api_key_env == "HF_TOKEN"
    assert provider.pooling == "cls"
    assert provider.batch_size == 4


@pytest.mark.asyncio
async def test_huggingface_provider_bounds_requests_and_streamed_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=_Chunks([b"[[1,2]]", b"  "]))

    provider = _huggingface_provider(
        respond,
        monkeypatch,
        max_request_bytes=30,
        max_response_bytes=8,
    )
    try:
        with pytest.raises(EmbeddingError, match="max_request_bytes"):
            await provider.embed(["a" * 100])
        with pytest.raises(EmbeddingResponseSizeError, match="max_response_bytes=8"):
            await provider.embed(["query"])
        assert len(requests) == 1
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_data", "message"),
    [
        ({"bad": "shape"}, "invalid feature response"),
        ([None], "invalid feature vector"),
        ([[]], "invalid feature vector"),
        ([[[1, 2], [3]]], "ragged token vectors"),
        ([[[True, 2]]], "non-numeric"),
        ([[[1e400]]], "non-finite"),
    ],
)
async def test_huggingface_provider_rejects_invalid_feature_responses(
    monkeypatch: pytest.MonkeyPatch, response_data: Any, message: str
) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        if message == "non-finite":
            return httpx.Response(200, content=b"[[[1e400]]]")
        return httpx.Response(200, json=response_data)

    provider = _huggingface_provider(respond, monkeypatch)
    try:
        with pytest.raises(EmbeddingError, match=message):
            await provider.embed(["query"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_huggingface_provider_rejects_wrong_batch_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _huggingface_provider(
        lambda _: httpx.Response(200, json=[[1.0, 2.0]]), monkeypatch, batch_size=2
    )
    try:
        with pytest.raises(EmbeddingError, match="unexpected number of vectors"):
            await provider.embed(["one", "two"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_huggingface_provider_sanitizes_http_and_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _huggingface_provider(
        lambda _: httpx.Response(503, text="private upstream response"), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="HTTP 503") as error:
            await provider.embed(["query"])
        assert "private upstream response" not in str(error.value)
    finally:
        await provider.aclose()

    provider = _huggingface_provider(
        lambda _: (_ for _ in ()).throw(httpx.ConnectError("private endpoint detail")),
        monkeypatch,
    )
    try:
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed(["query"])
        assert "private endpoint detail" not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__
        assert "private endpoint detail" not in "".join(traceback.format_exception(error.value))
    finally:
        await provider.aclose()

    provider = _huggingface_provider(
        lambda _: httpx.Response(200, content=b'{"private response excerpt": invalid}'), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed(["query"])
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__
        assert "private response excerpt" not in "".join(traceback.format_exception(error.value))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_batches_and_restores_input_order_with_environment_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == "embed-test"
        assert payload["dimensions"] == 2
        inputs = payload["input"]
        offset = 0 if inputs[0] == "first" else 2
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, offset + 2]},
                    {"index": 0, "embedding": [offset + 1, 0]},
                ]
            },
        )

    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "key-from-env")
    provider = _provider(
        respond,
        monkeypatch,
        api_key_env="TEST_EMBEDDINGS_KEY",
        base_url="https://embeddings.test/v1",
        dimensions=2,
        batch_size=2,
        extra_headers={"X-Client": "gabby"},
    )
    try:
        vectors = await provider.embed(["first", "second", "third", "fourth"])
    finally:
        await provider.aclose()

    assert vectors == [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]
    assert len(requests) == 2
    assert requests[0].url == "https://embeddings.test/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer key-from-env"
    assert requests[0].headers["x-client"] == "gabby"


@pytest.mark.asyncio
async def test_provider_returns_empty_for_empty_input_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise AssertionError("empty batches must not call the provider")

    provider = _provider(fail, monkeypatch)
    try:
        assert await provider.embed([]) == []
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_texts", ["query", b"query", None, ["query", 1]])
async def test_embedding_providers_reject_invalid_input_sequence(
    monkeypatch: pytest.MonkeyPatch, bad_texts: Any
) -> None:
    providers = (
        _provider(lambda _: httpx.Response(200, json={}), monkeypatch),
        _huggingface_provider(lambda _: httpx.Response(200, json=[]), monkeypatch),
        _gemini_provider(lambda _: httpx.Response(200, json={}), monkeypatch),
    )
    for provider in providers:
        try:
            with pytest.raises(
                (TypeError, ValueError), match="sequence of strings|non-empty strings"
            ):
                await provider.embed(bad_texts)
        finally:
            await provider.aclose()


@pytest.mark.asyncio
async def test_embedding_adapters_reject_unencodable_request_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for provider in (
        _provider(lambda _: httpx.Response(200, json={}), monkeypatch),
        _huggingface_provider(lambda _: httpx.Response(200, json=[]), monkeypatch),
        _gemini_provider(lambda _: httpx.Response(200, json={}), monkeypatch),
    ):
        try:
            with pytest.raises(EmbeddingError, match="not valid JSON"):
                await provider.embed(["\ud800"])
        finally:
            await provider.aclose()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model": ""}, ValueError, "model"),
        ({"api_key_env": "bad-name"}, ValueError, "api_key_env"),
        ({"dimensions": True}, ValueError, "dimensions"),
        ({"batch_size": 0}, ValueError, "batch_size"),
        ({"max_request_bytes": False}, ValueError, "max_request_bytes"),
        ({"max_response_bytes": 0}, ValueError, "max_response_bytes"),
        ({"timeout_seconds": float("inf")}, ValueError, "timeout_seconds"),
        ({"extra_headers": {"x": 1}}, TypeError, "extra_headers"),
        ({"base_url": "http://remote.test/v1"}, ValueError, "requires HTTPS"),
    ],
)
def test_provider_validates_configuration(
    kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    config: dict[str, Any] = {"model": "embed-test", **kwargs}
    with pytest.raises(error, match=message):
        OpenAICompatibleEmbeddingProvider(**config)


def test_from_config_rejects_inline_secret_and_reads_supported_fields() -> None:
    with pytest.raises(ValueError, match="model.api_key"):
        OpenAICompatibleEmbeddingProvider.from_config({"model": "embed-test", "api_key": "secret"})

    provider = OpenAICompatibleEmbeddingProvider.from_config(
        {"model": "embed-test", "base_url": "http://localhost:8000/v1", "batch_size": 3}
    )
    assert provider.model == "embed-test"
    assert provider.api_key_env == "OPENAI_API_KEY"
    assert provider.batch_size == 3


@pytest.mark.asyncio
async def test_provider_rejects_invalid_text_and_oversized_requests_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})

    provider = _provider(respond, monkeypatch, max_request_bytes=20)
    try:
        with pytest.raises(ValueError, match="non-empty strings"):
            await provider.embed([""])
        with pytest.raises(EmbeddingError, match="max_request_bytes"):
            await provider.embed(["a" * 100])
        assert requests == []
    finally:
        await provider.aclose()


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_provider_enforces_response_limit_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_Chunks([b'{"data":', b"[]}]"]))

    provider = _provider(respond, monkeypatch, max_response_bytes=10)
    try:
        with pytest.raises(EmbeddingResponseSizeError, match="max_response_bytes=10"):
            await provider.embed(["query"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "response object"),
        ({"data": "bad"}, "response object"),
        ({"data": []}, "unexpected number"),
        ({"data": [None]}, "vector entry"),
        ({"data": [{"index": True, "embedding": [1]}]}, "vector index"),
        ({"data": [{"index": 1, "embedding": [1]}]}, "vector index"),
        (
            {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]},
            "duplicate vector index",
        ),
        ({"data": [{"index": 0, "embedding": []}]}, "invalid vector"),
        ({"data": [{"index": 0, "embedding": [True]}]}, "non-numeric"),
        ({"data": [{"index": 0, "embedding": [float("inf")]}]}, "non-finite"),
        (
            {"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [1, 2]}]},
            "inconsistent dimensions",
        ),
    ],
)
async def test_provider_rejects_invalid_response_shapes(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], message: str
) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        if message == "non-finite":
            return httpx.Response(
                200,
                content=b'{"data":[{"index":0,"embedding":[1e400]}]}',
            )
        return httpx.Response(200, json=body)

    provider = _provider(respond, monkeypatch, batch_size=2)
    texts = ["one", "two"] if len(body.get("data", [])) == 2 else ["one"]
    try:
        with pytest.raises(EmbeddingError, match=message):
            await provider.embed(texts)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_rejects_dimension_changes_between_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        vector = [1, 0] if calls == 1 else [1]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector}]})

    provider = _provider(respond, monkeypatch, batch_size=1)
    try:
        with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
            await provider.embed(["one", "two"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_reports_http_and_transport_errors_without_returning_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(lambda _: httpx.Response(429, text="quota exceeded"), monkeypatch)
    try:
        with pytest.raises(EmbeddingError, match="HTTP 429") as error:
            await provider.embed(["query"])
        assert "quota exceeded" not in str(error.value)
    finally:
        await provider.aclose()

    provider = _provider(
        lambda _: (_ for _ in ()).throw(httpx.ConnectError("private endpoint detail")), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed(["query"])
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__
        assert "private endpoint detail" not in "".join(traceback.format_exception(error.value))
    finally:
        await provider.aclose()

    provider = _provider(
        lambda _: httpx.Response(200, content=b'{"private response excerpt": invalid}'), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed(["query"])
        assert error.value.__cause__ is None
        assert error.value.__suppress_context__
        assert "private response excerpt" not in "".join(traceback.format_exception(error.value))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_embedding_provider_runs_coordinated_sqlite_hybrid_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        vectors = [[1, 0] if "red" in text.casefold() else [0, 1] for text in payload["input"]]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": vector}
                    for index, vector in reversed(list(enumerate(vectors)))
                ]
            },
        )

    embeddings = _provider(respond, monkeypatch, base_url="https://embeddings.test/v1")
    coordinator = HybridIndexCoordinator(
        SQLiteFTS5Store(tmp_path / "lexical.db"),
        embeddings,
        SQLiteVectorStore(tmp_path / "vectors.db"),
        SQLiteGenerationManifestStore(tmp_path / "manifest.db"),
    )
    try:
        await coordinator.replace_source(
            "catalog.md",
            [
                Document("red apples", "catalog.md", id="red"),
                Document("blue berries", "catalog.md", id="blue"),
            ],
        )
        results = await coordinator.retrieve("red", limit=2)
    finally:
        await embeddings.aclose()

    assert [document.id for document in results] == ["red", "blue"]


@pytest.mark.asyncio
async def test_huggingface_embeddings_run_coordinated_sqlite_hybrid_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        vectors = [[1, 0] if "red" in text.casefold() else [0, 1] for text in payload["inputs"]]
        return httpx.Response(200, json=vectors)

    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    embeddings = _huggingface_provider(respond, monkeypatch)
    coordinator = HybridIndexCoordinator(
        SQLiteFTS5Store(tmp_path / "hf-lexical.db"),
        embeddings,
        SQLiteVectorStore(tmp_path / "hf-vectors.db"),
        SQLiteGenerationManifestStore(tmp_path / "hf-manifest.db"),
    )
    try:
        await coordinator.replace_source(
            "catalog.md",
            [
                Document("red apples", "catalog.md", id="red"),
                Document("blue berries", "catalog.md", id="blue"),
            ],
        )
        results = await coordinator.retrieve("red", limit=2)
    finally:
        await embeddings.aclose()

    assert [document.id for document in results] == ["red", "blue"]

    provider = _provider(
        lambda _: (_ for _ in ()).throw(httpx.ConnectError("private endpoint detail")), monkeypatch
    )
    try:
        with pytest.raises(EmbeddingError, match="request failed") as error:
            await provider.embed(["query"])
        assert "private endpoint detail" not in str(error.value)
    finally:
        await provider.aclose()
