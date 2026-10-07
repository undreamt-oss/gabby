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
"""Labeled query/document evaluation for replaceable embedding providers."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MAX_EMBEDDING_EVALUATION_CASES = 1_000
MAX_EMBEDDING_EVALUATION_DATASET_BYTES = 10 * 1024 * 1024
MAX_EMBEDDING_EVALUATION_CASE_BYTES = 1024 * 1024
MAX_EMBEDDING_CANDIDATES = 100
MAX_EMBEDDING_PREVIEW_IDS = 10
MAX_EMBEDDING_DIMENSIONS = 65_536
MAX_EMBEDDING_INPUT_PREFIX_BYTES = 4_096
_CASE_FIELDS = frozenset({"id", "query", "documents", "limit"})


class EmbeddingEvaluationError(ValueError):
    """An embedding evaluation dataset or provider response is invalid."""


@dataclass(frozen=True)
class EmbeddingInputFormat:
    """Optional query and document prefixes for asymmetric embedding models."""

    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        for name in ("query_prefix", "document_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            try:
                size = len(value.encode("utf-8"))
            except UnicodeEncodeError:
                raise ValueError(f"{name} must be valid UTF-8") from None
            if size > MAX_EMBEDDING_INPUT_PREFIX_BYTES:
                raise ValueError(f"{name} exceeds {MAX_EMBEDDING_INPUT_PREFIX_BYTES} UTF-8 bytes")


@dataclass(frozen=True)
class EmbeddingCandidate:
    """A labeled candidate document for one query."""

    id: str
    text: str
    relevance: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 256:
            raise EmbeddingEvaluationError("candidate id must contain 1 through 256 characters")
        if not isinstance(self.text, str) or not 1 <= len(self.text) <= 100_000:
            raise EmbeddingEvaluationError(
                "candidate text must contain 1 through 100000 characters"
            )
        if isinstance(self.relevance, bool) or not isinstance(self.relevance, int):
            raise EmbeddingEvaluationError(
                "candidate relevance must be an integer from 0 through 5"
            )
        if not 0 <= self.relevance <= 5:
            raise EmbeddingEvaluationError(
                "candidate relevance must be an integer from 0 through 5"
            )


@dataclass(frozen=True)
class EmbeddingEvaluationCase:
    """A query, judged candidate set, and ranking cutoff."""

    id: str
    query: str
    documents: tuple[EmbeddingCandidate, ...]
    limit: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or len(self.id) > 256:
            raise EmbeddingEvaluationError("case id must contain 1 through 256 characters")
        if not isinstance(self.query, str) or not 1 <= len(self.query) <= 100_000:
            raise EmbeddingEvaluationError("query must contain 1 through 100000 characters")
        if (
            not isinstance(self.documents, tuple)
            or not 2 <= len(self.documents) <= MAX_EMBEDDING_CANDIDATES
            or any(not isinstance(document, EmbeddingCandidate) for document in self.documents)
        ):
            raise EmbeddingEvaluationError(
                f"documents must contain 2 through {MAX_EMBEDDING_CANDIDATES} candidates"
            )
        if len({document.id for document in self.documents}) != len(self.documents):
            raise EmbeddingEvaluationError("candidate IDs must be unique within a case")
        if not any(document.relevance > 0 for document in self.documents):
            raise EmbeddingEvaluationError("at least one candidate must have positive relevance")
        if (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or not 1 <= self.limit <= MAX_EMBEDDING_CANDIDATES
        ):
            raise EmbeddingEvaluationError(
                f"limit must be an integer from 1 through {MAX_EMBEDDING_CANDIDATES}"
            )
        if _case_size(self) > MAX_EMBEDDING_EVALUATION_CASE_BYTES:
            raise EmbeddingEvaluationError("evaluation case exceeds the 1 MiB case limit")


@dataclass(frozen=True)
class EmbeddingCaseResult:
    """Ranking metrics for one query and its judged documents."""

    case_id: str
    ndcg: float
    reciprocal_rank: float
    pairwise_accuracy: float
    compared_pairs: int
    top_document_ids: tuple[str, ...]
    top_documents_truncated: bool
    error_type: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the embedding call and result validation succeeded."""
        return self.error_type is None

    def as_dict(self) -> dict[str, Any]:
        """Return bounded machine-readable metrics without provider error details."""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "ndcg": self.ndcg,
            "reciprocal_rank": self.reciprocal_rank,
            "pairwise_accuracy": self.pairwise_accuracy,
            "compared_pairs": self.compared_pairs,
            "top_document_ids": list(self.top_document_ids),
            "top_documents_truncated": self.top_documents_truncated,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class EmbeddingEvaluationReport:
    """Per-query embedding metrics and macro averages."""

    results: tuple[EmbeddingCaseResult, ...]
    duration_ms: float

    @property
    def passed_count(self) -> int:
        """Number of cases with valid embedding output."""
        return sum(result.passed for result in self.results)

    def _average(self, name: str) -> float:
        successful = [result for result in self.results if result.passed]
        return (
            sum(getattr(result, name) for result in successful) / len(successful)
            if successful
            else 0.0
        )

    @property
    def mean_ndcg(self) -> float:
        """Macro-average normalized discounted cumulative gain."""
        return self._average("ndcg")

    @property
    def mean_reciprocal_rank(self) -> float:
        """Macro-average reciprocal rank of the first relevant document."""
        return self._average("reciprocal_rank")

    @property
    def mean_pairwise_accuracy(self) -> float:
        """Macro-average pairwise ordering accuracy, excluding relevance ties."""
        return self._average("pairwise_accuracy")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result with bounded candidate previews."""
        return {
            "case_count": len(self.results),
            "passed_count": self.passed_count,
            "mean_ndcg": self.mean_ndcg,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "mean_pairwise_accuracy": self.mean_pairwise_accuracy,
            "duration_ms": self.duration_ms,
            "results": [result.as_dict() for result in self.results],
        }


class EmbeddingEvaluator(Protocol):
    """Async embedding provider contract consumed by :func:`evaluate_embeddings`."""

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector for each input text, in input order."""
        ...


def load_embedding_evaluation_dataset(
    path: str | Path,
) -> tuple[EmbeddingEvaluationCase, ...]:
    """Load a bounded JSON Lines dataset with relevance-graded candidate documents."""
    try:
        with Path(path).expanduser().open("rb") as stream:
            content = stream.read(MAX_EMBEDDING_EVALUATION_DATASET_BYTES + 1)
    except OSError:
        raise EmbeddingEvaluationError("embedding evaluation dataset is unavailable") from None
    if len(content) > MAX_EMBEDDING_EVALUATION_DATASET_BYTES:
        raise EmbeddingEvaluationError("embedding evaluation dataset exceeds 10 MiB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise EmbeddingEvaluationError("embedding evaluation dataset must use UTF-8") from None

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EmbeddingEvaluationError("dataset objects must not repeat keys")
            result[key] = value
        return result

    cases: list[EmbeddingEvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_EMBEDDING_EVALUATION_CASE_BYTES:
            raise EmbeddingEvaluationError(f"line {line_number} exceeds the 1 MiB case limit")
        try:
            value = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=_reject_constant,
            )
        except EmbeddingEvaluationError:
            raise
        except (json.JSONDecodeError, RecursionError):
            raise EmbeddingEvaluationError(f"line {line_number} is not valid JSON") from None
        if not isinstance(value, dict):
            raise EmbeddingEvaluationError(f"line {line_number} must be a JSON object")
        unknown = set(value) - _CASE_FIELDS
        if unknown:
            raise EmbeddingEvaluationError(
                f"line {line_number} has unknown field(s): " + ", ".join(sorted(unknown))
            )
        query = value.get("query")
        candidates = value.get("documents")
        if not isinstance(query, str) or not isinstance(candidates, list):
            raise EmbeddingEvaluationError(
                f"line {line_number} requires a string query and documents array"
            )
        documents: list[EmbeddingCandidate] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {"id", "text", "relevance"}:
                raise EmbeddingEvaluationError(
                    f"line {line_number} candidates require exactly id, text, and relevance"
                )
            documents.append(
                EmbeddingCandidate(
                    id=candidate["id"],
                    text=candidate["text"],
                    relevance=candidate["relevance"],
                )
            )
        case = EmbeddingEvaluationCase(
            id=value.get("id", f"case-{line_number:04d}"),
            query=query,
            documents=tuple(documents),
            limit=value.get("limit", 10),
        )
        if case.id in seen_ids:
            raise EmbeddingEvaluationError(f"dataset repeats case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
        if len(cases) > MAX_EMBEDDING_EVALUATION_CASES:
            raise EmbeddingEvaluationError("dataset exceeds 1000 cases")
    if not cases:
        raise EmbeddingEvaluationError("dataset must contain at least one case")
    return tuple(cases)


async def evaluate_embeddings(
    provider: EmbeddingEvaluator,
    cases: Sequence[EmbeddingEvaluationCase],
    *,
    timeout_seconds: float = 120.0,
    input_format: EmbeddingInputFormat | None = None,
) -> EmbeddingEvaluationReport:
    """Rank labeled documents by cosine similarity from one injected embedding provider.

    Each provider call has a bounded timeout; failed or timed-out cases are included in the report
    and the remaining cases continue.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a finite positive number")
    if input_format is not None and not isinstance(input_format, EmbeddingInputFormat):
        raise TypeError("input_format must be an EmbeddingInputFormat value")
    formatter = input_format or EmbeddingInputFormat()
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of EmbeddingEvaluationCase values")
    if not 1 <= len(cases) <= MAX_EMBEDDING_EVALUATION_CASES:
        raise EmbeddingEvaluationError("cases must contain 1 through 1000 entries")
    if any(not isinstance(case, EmbeddingEvaluationCase) for case in cases):
        raise TypeError("cases must contain EmbeddingEvaluationCase values")
    if len({case.id for case in cases}) != len(cases):
        raise EmbeddingEvaluationError("evaluation case IDs must be unique")
    if sum(_case_size(case) for case in cases) > MAX_EMBEDDING_EVALUATION_DATASET_BYTES:
        raise EmbeddingEvaluationError("evaluation suite exceeds 10 MiB")

    started = time.perf_counter()
    results: list[EmbeddingCaseResult] = []
    for case in cases:
        try:
            texts = [
                formatter.query_prefix + case.query,
                *(formatter.document_prefix + doc.text for doc in case.documents),
            ]
            input_bytes = sum(len(text.encode("utf-8")) for text in texts)
            if input_bytes > MAX_EMBEDDING_EVALUATION_CASE_BYTES:
                raise EmbeddingEvaluationError("formatted model inputs exceed the 1 MiB case limit")
            async with asyncio.timeout(timeout_seconds):
                vectors: Sequence[Sequence[float]]
                embed_queries = getattr(provider, "embed_queries", None)
                embed_documents = getattr(provider, "embed_documents", None)
                if callable(embed_queries) and callable(embed_documents):
                    query_prefix = formatter.query_prefix if input_format is not None else None
                    document_prefix = (
                        formatter.document_prefix if input_format is not None else None
                    )
                    query_vectors = await embed_queries([case.query], prefix=query_prefix)
                    document_vectors = await embed_documents(
                        [document.text for document in case.documents], prefix=document_prefix
                    )
                    vectors = [*query_vectors, *document_vectors]
                else:
                    vectors = await provider.embed(texts)
            similarities = _cosine_similarities(vectors, len(case.documents) + 1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__[:128] or "Exception"
            results.append(EmbeddingCaseResult(case.id, 0.0, 0.0, 0.0, 0, (), False, error_type))
            continue

        ranked = sorted(
            range(len(case.documents)),
            key=lambda index: (-similarities[index + 1], index),
        )
        top = ranked[: case.limit]
        relevance = [case.documents[index].relevance for index in top]
        gains = [(2**grade) - 1 for grade in relevance]
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sorted((doc.relevance for doc in case.documents), reverse=True)[: case.limit]
        ideal_dcg = sum(((2**grade) - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
        first_relevant_rank = next(
            (rank for rank, grade in enumerate(relevance, start=1) if grade > 0), None
        )
        pairwise_correct = 0.0
        compared_pairs = 0
        for left in range(len(case.documents)):
            for right in range(left + 1, len(case.documents)):
                left_grade = case.documents[left].relevance
                right_grade = case.documents[right].relevance
                if left_grade == right_grade:
                    continue
                compared_pairs += 1
                similarity_delta = similarities[left + 1] - similarities[right + 1]
                pairwise_correct += (
                    1.0
                    if (left_grade > right_grade and similarity_delta > 0)
                    or (right_grade > left_grade and similarity_delta < 0)
                    else 0.5
                    if similarity_delta == 0
                    else 0.0
                )
        top_ids = tuple(case.documents[index].id for index in top[:MAX_EMBEDDING_PREVIEW_IDS])
        results.append(
            EmbeddingCaseResult(
                case_id=case.id,
                ndcg=dcg / ideal_dcg if ideal_dcg else 0.0,
                reciprocal_rank=1 / first_relevant_rank if first_relevant_rank else 0.0,
                pairwise_accuracy=pairwise_correct / compared_pairs if compared_pairs else 0.0,
                compared_pairs=compared_pairs,
                top_document_ids=top_ids,
                top_documents_truncated=len(top) > MAX_EMBEDDING_PREVIEW_IDS,
            )
        )
    return EmbeddingEvaluationReport(tuple(results), (time.perf_counter() - started) * 1000)


def _cosine_similarities(
    vectors: Sequence[Sequence[float]], expected_count: int
) -> tuple[float, ...]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise EmbeddingEvaluationError("embedding provider returned an invalid vector collection")
    if len(vectors) != expected_count:
        raise EmbeddingEvaluationError("embedding provider returned the wrong number of vectors")
    if any(
        not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) for vector in vectors
    ):
        raise EmbeddingEvaluationError("embedding provider returned an invalid vector")
    dimensions = len(vectors[0])
    if (
        dimensions == 0
        or dimensions > MAX_EMBEDDING_DIMENSIONS
        or any(len(vector) != dimensions for vector in vectors)
    ):
        raise EmbeddingEvaluationError("embedding provider returned inconsistent vector dimensions")
    validated: list[tuple[float, ...]] = []
    for vector in vectors:
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingEvaluationError("embedding vectors must contain finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise EmbeddingEvaluationError("embedding vectors must contain finite numbers")
            values.append(number)
        validated.append(tuple(values))
    query = _unit_vector(validated[0])
    similarities: list[float] = [1.0]
    for document in validated[1:]:
        document_unit = _unit_vector(document)
        if query is None or document_unit is None:
            similarities.append(0.0)
            continue
        score = math.fsum(left * right for left, right in zip(query, document_unit, strict=True))
        similarities.append(max(-1.0, min(1.0, score)))
    return tuple(similarities)


def _unit_vector(vector: Sequence[float]) -> tuple[float, ...] | None:
    scale = max(abs(value) for value in vector)
    if scale == 0:
        return None
    scaled = tuple(value / scale for value in vector)
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _reject_constant(_value: str) -> None:
    raise EmbeddingEvaluationError("dataset must contain finite JSON values")


def _case_size(case: EmbeddingEvaluationCase) -> int:
    try:
        return len(
            json.dumps(
                {
                    "id": case.id,
                    "query": case.query,
                    "documents": [
                        {"id": doc.id, "text": doc.text, "relevance": doc.relevance}
                        for doc in case.documents
                    ],
                    "limit": case.limit,
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        raise EmbeddingEvaluationError("case contains invalid JSON values") from None
