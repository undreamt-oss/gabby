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
"""Bounded source-level retrieval benchmarks for replaceable knowledge backends."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .knowledge import MAX_RETRIEVAL_DOCUMENTS, Document

MAX_RETRIEVAL_EVALUATION_CASES = 1_000
MAX_RETRIEVAL_EVALUATION_DATASET_BYTES = 10 * 1024 * 1024
MAX_RETRIEVAL_EVALUATION_CASE_BYTES = 1024 * 1024
MAX_RELEVANT_SOURCES_PER_CASE = 100
MAX_REPORTED_RETRIEVED_SOURCES = 20
MAX_REPORTED_SOURCE_CHARS = 256
_CASE_FIELDS = frozenset({"id", "query", "relevant_sources", "filters", "limit"})


class RetrievalEvaluationError(ValueError):
    """A retrieval evaluation dataset or suite is invalid."""


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """A query with source-level relevance judgments."""

    id: str
    query: str
    relevant_sources: tuple[str, ...]
    filters: dict[str, Any] = field(default_factory=dict)
    limit: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 256:
            raise RetrievalEvaluationError("case id must contain 1 through 256 characters")
        if not isinstance(self.query, str) or not 1 <= len(self.query) <= 100_000:
            raise RetrievalEvaluationError("query must contain 1 through 100000 characters")
        if (
            not isinstance(self.relevant_sources, tuple)
            or not 1 <= len(self.relevant_sources) <= MAX_RELEVANT_SOURCES_PER_CASE
            or any(
                not isinstance(source, str) or not source or len(source) > 512
                for source in self.relevant_sources
            )
            or len(set(self.relevant_sources)) != len(self.relevant_sources)
        ):
            raise RetrievalEvaluationError(
                "relevant_sources must contain 1 through 100 unique source names "
                "of at most 512 characters"
            )
        if not isinstance(self.filters, dict):
            raise RetrievalEvaluationError("filters must be a JSON object")
        try:
            encoded_filters = json.dumps(
                self.filters, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            filters_snapshot = json.loads(encoded_filters)
        except (TypeError, ValueError, UnicodeEncodeError, json.JSONDecodeError, RecursionError):
            raise RetrievalEvaluationError("filters must contain finite JSON values") from None
        if len(encoded_filters) > MAX_RETRIEVAL_EVALUATION_CASE_BYTES:
            raise RetrievalEvaluationError("filters exceed the 1 MiB case limit")
        object.__setattr__(self, "filters", filters_snapshot)
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= MAX_RETRIEVAL_DOCUMENTS
        ):
            raise RetrievalEvaluationError(
                f"limit must be an integer from 1 through {MAX_RETRIEVAL_DOCUMENTS}"
            )
        if _case_size(self) > MAX_RETRIEVAL_EVALUATION_CASE_BYTES:
            raise RetrievalEvaluationError("evaluation case exceeds the 1 MiB case limit")


@dataclass(frozen=True)
class RetrievalCaseResult:
    """Ranking metrics for one judged query."""

    case_id: str
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float
    retrieved_sources: tuple[str, ...]
    retrieved_source_count: int
    retrieved_sources_truncated: bool
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        """Whether retrieval completed without an exception."""
        return self.error_type is None

    def as_dict(self) -> dict[str, Any]:
        """Return machine-readable metrics without backend exception details."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "precision": self.precision,
            "recall": self.recall,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg": self.ndcg,
            "retrieved_sources": list(self.retrieved_sources),
            "retrieved_source_count": self.retrieved_source_count,
            "retrieved_sources_truncated": self.retrieved_sources_truncated,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Per-query retrieval metrics and their macro averages."""

    results: tuple[RetrievalCaseResult, ...]
    duration_ms: float

    @property
    def passed_count(self) -> int:
        """Number of queries successfully retrieved."""
        return sum(result.passed for result in self.results)

    def _average(self, metric: str) -> float:
        successful = [result for result in self.results if result.passed]
        return (
            sum(getattr(result, metric) for result in successful) / len(successful)
            if successful
            else 0.0
        )

    @property
    def mean_precision(self) -> float:
        """Macro-average source precision across successful queries."""
        return self._average("precision")

    @property
    def mean_recall(self) -> float:
        """Macro-average source recall across successful queries."""
        return self._average("recall")

    @property
    def mean_reciprocal_rank(self) -> float:
        """Macro-average reciprocal rank across successful queries."""
        return self._average("reciprocal_rank")

    @property
    def mean_ndcg(self) -> float:
        """Macro-average normalized discounted cumulative gain."""
        return self._average("ndcg")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "case_count": len(self.results),
            "passed_count": self.passed_count,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "mean_ndcg": self.mean_ndcg,
            "duration_ms": self.duration_ms,
            "results": [result.as_dict() for result in self.results],
        }


class RetrievalEvaluator(Protocol):
    """Async retriever contract used by :func:`evaluate_retriever`."""

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve documents for one query."""
        ...


def load_retrieval_evaluation_dataset(
    path: str | Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Load a bounded JSON Lines dataset with unique case IDs."""
    try:
        with Path(path).expanduser().open("rb") as stream:
            content = stream.read(MAX_RETRIEVAL_EVALUATION_DATASET_BYTES + 1)
    except OSError:
        raise RetrievalEvaluationError("retrieval evaluation dataset is unavailable") from None
    if len(content) > MAX_RETRIEVAL_EVALUATION_DATASET_BYTES:
        raise RetrievalEvaluationError("retrieval evaluation dataset exceeds 10 MiB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise RetrievalEvaluationError("retrieval evaluation dataset must use UTF-8") from None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RetrievalEvaluationError("dataset objects must not repeat keys")
            result[key] = value
        return result

    cases: list[RetrievalEvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line, object_pairs_hook=unique_object, parse_constant=_reject_constant
            )
        except RetrievalEvaluationError:
            raise
        except (json.JSONDecodeError, RecursionError):
            raise RetrievalEvaluationError(f"line {line_number} is not valid JSON") from None
        if not isinstance(value, dict):
            raise RetrievalEvaluationError(f"line {line_number} must be a JSON object")
        unknown = set(value) - _CASE_FIELDS
        if unknown:
            raise RetrievalEvaluationError(
                f"line {line_number} has unknown field(s): " + ", ".join(sorted(unknown))
            )
        if len(line.encode("utf-8")) > MAX_RETRIEVAL_EVALUATION_CASE_BYTES:
            raise RetrievalEvaluationError(f"line {line_number} exceeds the 1 MiB case limit")
        sources = value.get("relevant_sources")
        if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
            raise RetrievalEvaluationError(f"line {line_number} requires relevant_sources array")
        query = value.get("query")
        if not isinstance(query, str):
            raise RetrievalEvaluationError(f"line {line_number} requires a string query")
        case = RetrievalEvaluationCase(
            id=value.get("id", f"case-{line_number:04d}"),
            query=query,
            relevant_sources=tuple(sources),
            filters=value.get("filters", {}),
            limit=value.get("limit", 5),
        )
        if case.id in seen_ids:
            raise RetrievalEvaluationError(f"dataset repeats case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
        if len(cases) > MAX_RETRIEVAL_EVALUATION_CASES:
            raise RetrievalEvaluationError("dataset exceeds 1000 cases")
    if not cases:
        raise RetrievalEvaluationError("dataset must contain at least one case")
    return tuple(cases)


async def evaluate_retriever(
    retriever: RetrievalEvaluator, cases: Sequence[RetrievalEvaluationCase]
) -> RetrievalEvaluationReport:
    """Measure source ranking quality without model calls or shared query state."""
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of RetrievalEvaluationCase values")
    if not 1 <= len(cases) <= MAX_RETRIEVAL_EVALUATION_CASES:
        raise RetrievalEvaluationError("cases must contain 1 through 1000 entries")
    if any(not isinstance(case, RetrievalEvaluationCase) for case in cases):
        raise TypeError("cases must contain RetrievalEvaluationCase values")
    if len({case.id for case in cases}) != len(cases):
        raise RetrievalEvaluationError("evaluation case IDs must be unique")
    total_bytes = sum(_case_size(case) for case in cases)
    if total_bytes > MAX_RETRIEVAL_EVALUATION_DATASET_BYTES:
        raise RetrievalEvaluationError("evaluation suite exceeds 10 MiB")

    started = time.perf_counter()
    results: list[RetrievalCaseResult] = []
    for case in cases:
        try:
            documents = await retriever.retrieve(case.query, limit=case.limit, filters=case.filters)
            if (
                not isinstance(documents, list)
                or len(documents) > case.limit
                or any(
                    not isinstance(document, Document)
                    or not isinstance(document.source, str)
                    or not document.source
                    or len(document.source) > 512
                    for document in documents
                )
            ):
                raise RetrievalEvaluationError("retriever returned an invalid result")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            results.append(
                RetrievalCaseResult(case.id, 0.0, 0.0, 0.0, 0.0, (), 0, False, type(exc).__name__)
            )
            continue

        retrieved_sources = _unique_sources(documents)
        relevant = set(case.relevant_sources)
        hits = [source in relevant for source in retrieved_sources]
        relevant_count = sum(hits)
        reciprocal_rank = next(
            (1 / index for index, is_relevant in enumerate(hits, start=1) if is_relevant), 0.0
        )
        dcg = sum(
            1 / math.log2(rank + 1) for rank, is_relevant in enumerate(hits, start=1) if is_relevant
        )
        ideal_hits = min(len(relevant), case.limit)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        results.append(
            RetrievalCaseResult(
                case.id,
                relevant_count / len(retrieved_sources) if retrieved_sources else 0.0,
                relevant_count / len(relevant),
                reciprocal_rank,
                dcg / ideal_dcg if ideal_dcg else 0.0,
                tuple(
                    source[:MAX_REPORTED_SOURCE_CHARS]
                    for source in retrieved_sources[:MAX_REPORTED_RETRIEVED_SOURCES]
                ),
                len(retrieved_sources),
                len(retrieved_sources) > MAX_REPORTED_RETRIEVED_SOURCES
                or any(len(source) > MAX_REPORTED_SOURCE_CHARS for source in retrieved_sources),
            )
        )
    return RetrievalEvaluationReport(tuple(results), (time.perf_counter() - started) * 1000)


def _unique_sources(documents: Sequence[Document]) -> tuple[str, ...]:
    sources: list[str] = []
    seen: set[str] = set()
    for document in documents:
        if document.source not in seen:
            sources.append(document.source)
            seen.add(document.source)
    return tuple(sources)


def _reject_constant(_value: str) -> None:
    raise RetrievalEvaluationError("dataset must contain finite JSON values")


def _case_size(case: RetrievalEvaluationCase) -> int:
    try:
        return len(
            json.dumps(
                {
                    "id": case.id,
                    "query": case.query,
                    "relevant_sources": case.relevant_sources,
                    "filters": case.filters,
                    "limit": case.limit,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise RetrievalEvaluationError("case contains invalid JSON values") from None
