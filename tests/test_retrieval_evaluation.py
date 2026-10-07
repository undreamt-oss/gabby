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
"""Retrieval ranking benchmark contracts."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from gabby import (
    Document,
    RetrievalEvaluationCase,
    RetrievalEvaluationError,
    evaluate_retriever,
    load_retrieval_evaluation_dataset,
)


class FixedRetriever:
    def __init__(self, sources: tuple[str, ...]) -> None:
        self.sources = sources
        self.calls: list[tuple[str, int, dict[str, object] | None]] = []

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[Document]:
        self.calls.append((query, limit, filters))
        return [Document(text="fixture", source=source) for source in self.sources[:limit]]


@pytest.mark.asyncio
async def test_retrieval_evaluation_computes_source_ranking_metrics() -> None:
    retriever = FixedRetriever(("irrelevant", "policy", "policy", "faq"))
    cases = [
        RetrievalEvaluationCase(
            id="refund",
            query="refund time",
            relevant_sources=("policy", "faq"),
            filters={"region": "west"},
            limit=5,
        )
    ]

    report = await evaluate_retriever(retriever, cases)

    result = report.results[0]
    assert result.retrieved_sources == ("irrelevant", "policy", "faq")
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.reciprocal_rank == 0.5
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = 1 + 1 / math.log2(3)
    assert result.ndcg == pytest.approx(expected_dcg / ideal_dcg)
    assert report.mean_recall == 1.0
    assert retriever.calls == [("refund time", 5, {"region": "west"})]
    assert json.loads(json.dumps(report.as_dict()))["case_count"] == 1


@pytest.mark.asyncio
async def test_retrieval_evaluation_bounds_reported_sources() -> None:
    sources = tuple(f"source-{index}" for index in range(30))
    report = await evaluate_retriever(
        FixedRetriever(sources),
        [RetrievalEvaluationCase("many", "query", (sources[0],), limit=30)],
    )
    result = report.results[0]
    assert len(result.retrieved_sources) == 20
    assert result.retrieved_source_count == 30
    assert result.retrieved_sources_truncated


def test_retrieval_dataset_loads_and_rejects_ambiguous_rows(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text('{"query":"refund","relevant_sources":["policy"]}\n', encoding="utf-8")
    cases = load_retrieval_evaluation_dataset(path)
    assert cases[0].id == "case-0001"
    assert cases[0].limit == 5

    path.write_text(
        '{"query":"refund","query":"other","relevant_sources":["policy"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(RetrievalEvaluationError, match="repeat keys"):
        load_retrieval_evaluation_dataset(path)


def test_retrieval_case_validates_relevance_and_result_limit() -> None:
    with pytest.raises(RetrievalEvaluationError, match="unique source names"):
        RetrievalEvaluationCase("dup", "query", ("policy", "policy"))
    with pytest.raises(RetrievalEvaluationError, match="limit must be"):
        RetrievalEvaluationCase("limit", "query", ("policy",), limit=101)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": ""}, "case id"),
        ({"id": "x" * 257}, "case id"),
        ({"query": ""}, "query must"),
        ({"query": "q" * 100_001}, "query must"),
        ({"relevant_sources": ()}, "unique source names"),
        ({"relevant_sources": ["policy"]}, "unique source names"),
        ({"relevant_sources": ("x" * 513,)}, "unique source names"),
        ({"filters": []}, "filters must be a JSON object"),
        ({"filters": {"score": float("nan")}}, "finite JSON values"),
        ({"filters": {"value": object()}}, "finite JSON values"),
        ({"filters": {"blob": "x" * (1024 * 1024 + 1)}}, "1 MiB case limit"),
        ({"filters": {"blob": "x" * 1_048_500}}, "evaluation case exceeds"),
        ({"limit": True}, "limit must be"),
        ({"limit": 0}, "limit must be"),
        ({"limit": 101}, "limit must be"),
    ],
)
def test_retrieval_case_rejects_invalid_fields(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "id": "case",
        "query": "query",
        "relevant_sources": ("policy",),
    }
    values.update(kwargs)
    with pytest.raises(RetrievalEvaluationError, match=message):
        RetrievalEvaluationCase(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not-json\n", "not valid JSON"),
        (b"NaN\n", "finite JSON values"),
        (b"[]\n", "must be a JSON object"),
        (b'{"query":"x","query":"y"}\n', "repeat keys"),
        (b'{"query":"x","relevant_sources":["a"],"other":1}\n', "unknown field"),
        (b'{"query":"x","relevant_sources":"a"}\n', "requires relevant_sources"),
        (b'{"query":"x","relevant_sources":[1]}\n', "requires relevant_sources"),
        (b'{"relevant_sources":["a"]}\n', "requires a string query"),
        (b'{"query":1,"relevant_sources":["a"]}\n', "requires a string query"),
        (b'{"query":"x","relevant_sources":["a"],"filters":[]}\n', "filters must be"),
    ],
)
def test_retrieval_dataset_rejects_invalid_lines(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(content)
    with pytest.raises(RetrievalEvaluationError, match=message):
        load_retrieval_evaluation_dataset(path)


def test_retrieval_dataset_handles_empty_missing_invalid_utf8_and_size_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(RetrievalEvaluationError, match="unavailable"):
        load_retrieval_evaluation_dataset(tmp_path / "missing.jsonl")

    path = tmp_path / "dataset.jsonl"
    path.write_bytes(b" \n\t\n")
    with pytest.raises(RetrievalEvaluationError, match="at least one case"):
        load_retrieval_evaluation_dataset(path)

    path.write_bytes(b"\xff")
    with pytest.raises(RetrievalEvaluationError, match="UTF-8"):
        load_retrieval_evaluation_dataset(path)

    path.write_bytes(b" " * (10 * 1024 * 1024 + 1))
    with pytest.raises(RetrievalEvaluationError, match="exceeds 10 MiB"):
        load_retrieval_evaluation_dataset(path)


def test_retrieval_dataset_bounds_case_count_and_line_size(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    record = json.dumps({"query": "x", "relevant_sources": ["policy"]})
    path.write_text((record + "\n") * 1001, encoding="utf-8")
    with pytest.raises(RetrievalEvaluationError, match="exceeds 1000 cases"):
        load_retrieval_evaluation_dataset(path)

    oversized_case = json.dumps({"query": "x", "relevant_sources": ["p"], "id": "x" * 1_048_600})
    path.write_text(oversized_case, encoding="utf-8")
    with pytest.raises(RetrievalEvaluationError, match="1 MiB case limit"):
        load_retrieval_evaluation_dataset(path)


@pytest.mark.asyncio
async def test_retrieval_evaluation_validates_suites_and_records_backend_errors() -> None:
    valid = RetrievalEvaluationCase("ok", "query", ("policy",))
    with pytest.raises(TypeError, match="sequence"):
        await evaluate_retriever(FixedRetriever(()), None)  # type: ignore[arg-type]
    with pytest.raises(RetrievalEvaluationError, match="1 through 1000"):
        await evaluate_retriever(FixedRetriever(()), [])
    with pytest.raises(TypeError, match="RetrievalEvaluationCase"):
        await evaluate_retriever(FixedRetriever(()), [object()])  # type: ignore[list-item]
    with pytest.raises(RetrievalEvaluationError, match="unique"):
        await evaluate_retriever(FixedRetriever(()), [valid, valid])

    large_cases = [
        RetrievalEvaluationCase(f"case-{index}", "q" * 100_000, ("policy",)) for index in range(105)
    ]
    with pytest.raises(RetrievalEvaluationError, match="exceeds 10 MiB"):
        await evaluate_retriever(FixedRetriever(()), large_cases)

    class BrokenRetriever:
        def __init__(self, result: object) -> None:
            self.result = result

        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> Any:
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    invalid_results: tuple[object, ...] = (
        (Document("x", source="policy"),),
        [Document("x", source="policy"), Document("y", source="faq")],
        [object()],
        [Document("x", source="")],
        [Document("x", source="s" * 513)],
        RuntimeError("private backend detail"),
    )
    bounded_case = RetrievalEvaluationCase("bounded", "query", ("policy",), limit=1)
    for backend_result in invalid_results:
        report = await evaluate_retriever(BrokenRetriever(backend_result), [bounded_case])
        result = report.results[0]
        assert not result.passed
        assert result.precision == result.recall == result.reciprocal_rank == result.ndcg == 0.0
        assert result.error_type in {"RetrievalEvaluationError", "RuntimeError"}
        assert "private backend detail" not in json.dumps(result.as_dict())


@pytest.mark.asyncio
async def test_retrieval_evaluation_propagates_cancellation_and_handles_no_hits() -> None:
    case = RetrievalEvaluationCase("empty", "query", ("policy",))

    class CancelledRetriever:
        async def retrieve(self, *_args: object, **_kwargs: object) -> list[Document]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_retriever(CancelledRetriever(), [case])

    report = await evaluate_retriever(FixedRetriever(()), [case])
    assert report.results[0].passed
    assert report.mean_precision == report.mean_recall == 0.0
    assert report.mean_reciprocal_rank == report.mean_ndcg == 0.0
