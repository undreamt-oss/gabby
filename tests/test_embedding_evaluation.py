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
"""Embedding ranking benchmark contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from gabby import (
    EmbeddingCandidate,
    EmbeddingEvaluationCase,
    EmbeddingEvaluationError,
    EmbeddingInputFormat,
    evaluate_embeddings,
    load_embedding_evaluation_dataset,
)
from gabby.embedding_evaluation import _cosine_similarities


class FixtureEmbeddings:
    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]


def _case() -> EmbeddingEvaluationCase:
    return EmbeddingEvaluationCase(
        id="policy",
        query="refund query",
        documents=(
            EmbeddingCandidate("high", "high relevance", 3),
            EmbeddingCandidate("zero", "irrelevant", 0),
            EmbeddingCandidate("medium", "partly relevant", 2),
        ),
        limit=3,
    )


@pytest.mark.asyncio
async def test_embedding_evaluation_measures_cosine_rankings() -> None:
    provider = FixtureEmbeddings(
        {
            "refund query": [1, 0],
            "high relevance": [1, 0],
            "irrelevant": [0, 1],
            "partly relevant": [0.8, 0.6],
        }
    )

    report = await evaluate_embeddings(provider, [_case()])

    result = report.results[0]
    assert result.passed
    assert result.ndcg == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.pairwise_accuracy == 1.0
    assert result.compared_pairs == 3
    assert result.top_document_ids == ("high", "medium", "zero")
    assert provider.calls == [("refund query", "high relevance", "irrelevant", "partly relevant")]
    assert json.loads(json.dumps(report.as_dict()))["mean_ndcg"] == 1.0


@pytest.mark.asyncio
async def test_embedding_evaluation_supports_asymmetric_query_document_prefixes() -> None:
    provider = FixtureEmbeddings(
        {
            "query: refund query": [1, 0],
            "passage: high relevance": [1, 0],
            "passage: irrelevant": [0, 1],
            "passage: partly relevant": [0.8, 0.6],
        }
    )

    report = await evaluate_embeddings(
        provider,
        [_case()],
        input_format=EmbeddingInputFormat(query_prefix="query: ", document_prefix="passage: "),
    )

    assert report.results[0].ndcg == 1.0
    assert provider.calls[0][0] == "query: refund query"
    assert provider.calls[0][1:] == (
        "passage: high relevance",
        "passage: irrelevant",
        "passage: partly relevant",
    )


@pytest.mark.asyncio
async def test_embedding_evaluation_uses_asymmetric_provider_roles() -> None:
    class RoleProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[str, ...], str | None]] = []

        async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise AssertionError("role-specific methods should be selected")

        async def embed_queries(
            self, texts: Sequence[str], *, prefix: str | None = None
        ) -> Sequence[Sequence[float]]:
            self.calls.append(("query", tuple(texts), prefix))
            return [[1, 0]]

        async def embed_documents(
            self, texts: Sequence[str], *, prefix: str | None = None
        ) -> Sequence[Sequence[float]]:
            self.calls.append(("documents", tuple(texts), prefix))
            vectors: dict[str, list[float]] = {
                "high relevance": [1, 0],
                "irrelevant": [0, 1],
                "partly relevant": [0.8, 0.6],
            }
            return [vectors[text] for text in texts]

    provider = RoleProvider()
    report = await evaluate_embeddings(
        provider,
        [_case()],
        input_format=EmbeddingInputFormat(query_prefix="query: ", document_prefix="passage: "),
    )

    assert report.results[0].ndcg == 1.0
    assert provider.calls == [
        ("query", ("refund query",), "query: "),
        (
            "documents",
            ("high relevance", "irrelevant", "partly relevant"),
            "passage: ",
        ),
    ]


@pytest.mark.asyncio
async def test_embedding_evaluation_normalizes_large_finite_vectors_safely() -> None:
    provider = FixtureEmbeddings(
        {
            "refund query": [1e308, 0],
            "high relevance": [1e308, 0],
            "irrelevant": [0, 1e308],
            "partly relevant": [8e307, 6e307],
        }
    )

    report = await evaluate_embeddings(provider, [_case()])

    assert report.results[0].ndcg == 1.0
    assert report.results[0].pairwise_accuracy == 1.0


@pytest.mark.asyncio
async def test_embedding_evaluation_reports_invalid_provider_output_safely() -> None:
    provider = FixtureEmbeddings(
        {
            "refund query": [1, 0],
            "high relevance": [1],
            "irrelevant": [0, 1],
            "partly relevant": [0.8, 0.6],
        }
    )

    report = await evaluate_embeddings(provider, [_case()])

    assert not report.results[0].passed
    assert report.results[0].error_type == "EmbeddingEvaluationError"
    assert "vector" not in json.dumps(report.as_dict())


@pytest.mark.asyncio
async def test_embedding_evaluation_propagates_cancellation() -> None:
    class CancelledEmbeddings:
        async def embed(self, _texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_embeddings(CancelledEmbeddings(), [_case()])


@pytest.mark.asyncio
async def test_embedding_evaluation_bounds_provider_waits() -> None:
    class SlowEmbeddings:
        async def embed(self, _texts: Sequence[str]) -> Sequence[Sequence[float]]:
            await asyncio.sleep(1)
            return []

    report = await evaluate_embeddings(SlowEmbeddings(), [_case()], timeout_seconds=0.001)
    assert not report.results[0].passed
    assert report.results[0].error_type == "TimeoutError"


@pytest.mark.parametrize(
    "vectors",
    [
        None,
        [[1, 0]],
        [[1, 0], "not-a-vector", [0, 1], [0, 1]],
        [[], [], [], []],
        [[1, 0], [1], [0, 1], [0, 1]],
        [[True, 0], [1, 0], [0, 1], [0, 1]],
        [[float("nan"), 0], [1, 0], [0, 1], [0, 1]],
    ],
)
@pytest.mark.asyncio
async def test_embedding_evaluation_rejects_invalid_vectors(vectors: Any) -> None:
    class InvalidProvider:
        async def embed(self, _texts: Sequence[str]) -> Any:
            return vectors

    report = await evaluate_embeddings(InvalidProvider(), [_case()])
    assert not report.results[0].passed
    assert report.mean_ndcg == 0


def test_cosine_similarity_handles_zero_vectors_and_invalid_collections() -> None:
    assert _cosine_similarities([[0, 0], [0, 1], [1, 0]], 3) == (1.0, 0.0, 0.0)
    for vectors, count in (("bad", 1), ([[1, 0]], 2), ([[1, 0], object()], 2)):
        with pytest.raises(EmbeddingEvaluationError):
            _cosine_similarities(vectors, count)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embedding_evaluation_validates_public_arguments_and_formatted_bytes() -> None:
    valid = _case()
    invalid_arguments: list[tuple[Any, Any, Any, type[Exception]]] = [
        (True, [valid], None, ValueError),
        (float("inf"), [valid], None, ValueError),
        (1, [valid], object(), TypeError),
        (1, "not-cases", None, TypeError),
        (1, [], None, EmbeddingEvaluationError),
        (1, [object()], None, TypeError),
        (1, [valid, valid], None, EmbeddingEvaluationError),
    ]
    for timeout, cases, input_format, error in invalid_arguments:
        with pytest.raises(error):
            await evaluate_embeddings(
                FixtureEmbeddings({}), cases, timeout_seconds=timeout, input_format=input_format
            )

    large = EmbeddingEvaluationCase(
        "large-formatted",
        "q",
        tuple(
            EmbeddingCandidate(f"doc-{index}", "x" * 84_000, int(index == 0)) for index in range(12)
        ),
    )
    report = await evaluate_embeddings(
        FixtureEmbeddings({}),
        [large],
        input_format=EmbeddingInputFormat(document_prefix="p" * 4096),
    )
    assert report.results[0].error_type == "EmbeddingEvaluationError"


def test_embedding_dataset_is_strict_and_bounded(tmp_path: Path) -> None:
    dataset = tmp_path / "embedding.jsonl"
    dataset.write_text(
        '{"query":"refund query","documents":['
        '{"id":"a","text":"good","relevance":3},'
        '{"id":"b","text":"other","relevance":0}]}\n',
        encoding="utf-8",
    )
    cases = load_embedding_evaluation_dataset(dataset)
    assert cases[0].id == "case-0001"
    assert cases[0].limit == 10

    dataset.write_text('{"query":"refund query","extra":true,"documents":[]}\n', encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="unknown field"):
        load_embedding_evaluation_dataset(dataset)


def test_embedding_case_rejects_invalid_judgments() -> None:
    with pytest.raises(EmbeddingEvaluationError, match="positive relevance"):
        EmbeddingEvaluationCase(
            "empty-labels",
            "query",
            (EmbeddingCandidate("a", "one", 0), EmbeddingCandidate("b", "two", 0)),
        )
    with pytest.raises(EmbeddingEvaluationError, match="unique"):
        EmbeddingEvaluationCase(
            "duplicate",
            "query",
            (EmbeddingCandidate("a", "one", 1), EmbeddingCandidate("a", "two", 0)),
        )


def test_embedding_case_rejects_bad_identity_text_and_size() -> None:
    candidates = (EmbeddingCandidate("a", "x", 1), EmbeddingCandidate("b", "y", 0))
    for case_id, query in (("", "q"), ("x" * 257, "q"), ("ok", ""), ("ok", "x" * 100_001)):
        with pytest.raises(EmbeddingEvaluationError):
            EmbeddingEvaluationCase(case_id, query, candidates)
    large_documents = tuple(
        EmbeddingCandidate(f"id-{index}", "x" * 100_000, int(index == 0)) for index in range(11)
    )
    with pytest.raises(EmbeddingEvaluationError, match="1 MiB"):
        EmbeddingEvaluationCase("large", "query", large_documents)


def test_embedding_input_and_candidate_contract_bounds() -> None:
    for prefix in (None, "\ud800", "x" * 4097):
        with pytest.raises(ValueError):
            EmbeddingInputFormat(query_prefix=prefix)  # type: ignore[arg-type]
    for candidate in (
        ("", "text", 1),
        ("x" * 257, "text", 1),
        ("ok", "", 1),
        ("ok", "text", True),
        ("ok", "text", 6),
    ):
        with pytest.raises(EmbeddingEvaluationError):
            EmbeddingCandidate(*candidate)
    valid = EmbeddingCandidate("valid", "text", 1)
    for documents, limit in (
        ((valid,), 1),
        ([valid, EmbeddingCandidate("two", "text", 0)], 1),
        ((object(), valid), 1),
        ((valid, EmbeddingCandidate("two", "text", 0)), True),
        ((valid, EmbeddingCandidate("two", "text", 0)), 101),
    ):
        with pytest.raises(EmbeddingEvaluationError):
            EmbeddingEvaluationCase("invalid", "query", documents, limit)  # type: ignore[arg-type]


def test_embedding_dataset_parser_rejects_malformed_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    invalid_lines = (
        '{"query":"q","query":"duplicate","documents":[]}',
        '{"query":"q","documents":[],"value":NaN}',
        "not-json",
        "[]",
        '{"query":"q","documents":[]}\n{"query":"q","documents":[]}\n',
        '{"query":"q","documents":[{"id":"a","text":"x","relevance":1}]}',
        '{"query":"q","documents":[{"id":"a","text":"x","relevance":1},{"id":"a","text":"y","relevance":0}]}',
    )
    for line in invalid_lines:
        dataset.write_text(line, encoding="utf-8")
        with pytest.raises(EmbeddingEvaluationError):
            load_embedding_evaluation_dataset(dataset)
    dataset.write_bytes(b"\xff")
    with pytest.raises(EmbeddingEvaluationError, match="UTF-8"):
        load_embedding_evaluation_dataset(dataset)
    with pytest.raises(EmbeddingEvaluationError, match="unavailable"):
        load_embedding_evaluation_dataset(tmp_path / "missing.jsonl")
    dataset.write_text("\n   \n", encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="at least one"):
        load_embedding_evaluation_dataset(dataset)
    duplicate_id = json.dumps(
        {
            "id": "same",
            "query": "q",
            "documents": [
                {"id": "a", "text": "x", "relevance": 1},
                {"id": "b", "text": "y", "relevance": 0},
            ],
        }
    )
    dataset.write_text(f"{duplicate_id}\n{duplicate_id}", encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="repeats case id"):
        load_embedding_evaluation_dataset(dataset)
    dataset.write_text(json.dumps({"query": "q", "documents": []}) + "\n", encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="documents must contain"):
        load_embedding_evaluation_dataset(dataset)
    dataset.write_text(
        json.dumps({"query": "q", "documents": [{"id": "a"}, {"id": "b"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EmbeddingEvaluationError, match="exactly id"):
        load_embedding_evaluation_dataset(dataset)


def test_embedding_dataset_enforces_dataset_and_case_byte_caps(tmp_path: Path) -> None:
    dataset = tmp_path / "large.jsonl"
    dataset.write_bytes(b" " * (10 * 1024 * 1024 + 1))
    with pytest.raises(EmbeddingEvaluationError, match="exceeds 10 MiB"):
        load_embedding_evaluation_dataset(dataset)
    dataset.write_text('"' + ("x" * (1024 * 1024)) + '"', encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="line 1 exceeds"):
        load_embedding_evaluation_dataset(dataset)
    one_case = json.dumps(
        {
            "query": "q",
            "documents": [
                {"id": "a", "text": "x", "relevance": 1},
                {"id": "b", "text": "y", "relevance": 0},
            ],
        }
    )
    dataset.write_text((one_case + "\n") * 1001, encoding="utf-8")
    with pytest.raises(EmbeddingEvaluationError, match="exceeds 1000 cases"):
        load_embedding_evaluation_dataset(dataset)
