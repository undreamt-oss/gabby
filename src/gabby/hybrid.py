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
"""Hybrid lexical and semantic retrieval with deterministic reciprocal-rank fusion."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from copy import deepcopy
from inspect import Parameter, signature
from typing import Any, cast

from .knowledge import (
    MAX_RETRIEVAL_DOCUMENTS,
    ActiveGenerationReader,
    Document,
    EmbeddingProvider,
    GenerationAwareRetriever,
    GenerationAwareVectorStore,
    Reranker,
    Retriever,
    VectorStore,
)


def _supports_generation_filter(store: Any, method_name: str) -> bool:
    """Check the runtime methods needed to keep manifest reads generation-consistent."""
    if not callable(getattr(store, "stage_source", None)):
        return False
    method = getattr(store, method_name, None)
    if not callable(method):
        return False
    try:
        parameters = signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        (
            parameter.name == "generations"
            and parameter.kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
        )
        or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _validated_vectors(
    vectors: Sequence[Sequence[float]], *, expected_count: int
) -> list[list[float]]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise ValueError("embedding provider must return a sequence of vectors")
    if len(vectors) != expected_count:
        raise ValueError("embedding provider returned an unexpected number of vectors")
    normalized: list[list[float]] = []
    dimensions: int | None = None
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            raise ValueError("embedding vectors must be non-empty numeric sequences")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("embedding values must be finite numbers")
            try:
                converted = float(value)
            except OverflowError as exc:
                raise ValueError("embedding values must be finite numbers") from exc
            if not math.isfinite(converted):
                raise ValueError("embedding values must be finite numbers")
            values.append(converted)
        if dimensions is None:
            dimensions = len(values)
        elif len(values) != dimensions:
            raise ValueError("embedding vectors must have a consistent dimension")
        normalized.append(values)
    return normalized


def _identity(document: Document) -> str:
    if not isinstance(document, Document):
        raise TypeError("retrievers must return Document instances")
    if not isinstance(document.text, str) or not isinstance(document.source, str):
        raise TypeError("retrieved documents must have string text and source fields")
    if document.id is not None:
        if not isinstance(document.id, str) or not document.id:
            raise ValueError("retrieved document IDs must be non-empty strings")
        return document.id
    return f"{document.source}\0{document.text}"


class HybridRetriever:
    """Combine lexical and semantic candidates with reciprocal-rank fusion.

    The lexical and vector indexes are supplied independently. Indexing remains explicit through
    their respective store interfaces; this retriever embeds each query once and runs both searches
    concurrently. Both indexes should use the same stable document IDs for reliable de-duplication.
    """

    def __init__(
        self,
        lexical: Retriever | GenerationAwareRetriever,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        *,
        candidate_limit: int = 20,
        rrf_constant: float = 60.0,
        manifest: ActiveGenerationReader | None = None,
    ) -> None:
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer from 1 through 100")
        if not 1 <= candidate_limit <= MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError("candidate_limit must be an integer from 1 through 100")
        if (
            isinstance(rrf_constant, bool)
            or not isinstance(rrf_constant, (int, float))
            or not math.isfinite(rrf_constant)
            or rrf_constant <= 0
        ):
            raise ValueError("rrf_constant must be a finite positive number")
        if manifest is not None:
            if not _supports_generation_filter(lexical, "retrieve"):
                raise TypeError(
                    "manifest-filtered retrieval requires a generation-aware lexical store"
                )
            if not _supports_generation_filter(vectors, "search"):
                raise TypeError(
                    "manifest-filtered retrieval requires a generation-aware vector store"
                )
        self.lexical = lexical
        self.embeddings = embeddings
        self.vectors = vectors
        self.candidate_limit = candidate_limit
        self.rrf_constant = float(rrf_constant)
        self.manifest = manifest

    async def retrieve(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """Fetch both candidate lists and return the top fused documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 0 <= limit <= MAX_RETRIEVAL_DOCUMENTS
        ):
            raise ValueError("limit must be an integer from 0 through 100")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        if limit == 0 or not query.strip():
            return []

        generations = (
            await self.manifest.active_generations() if self.manifest is not None else None
        )
        if generations is not None and not generations:
            return []
        if generations is not None:
            if not _supports_generation_filter(self.lexical, "retrieve"):
                raise TypeError(
                    "manifest-filtered retrieval requires a generation-aware lexical store"
                )
            if not _supports_generation_filter(self.vectors, "search"):
                raise TypeError(
                    "manifest-filtered retrieval requires a generation-aware vector store"
                )
        embed_query = getattr(self.embeddings, "embed_queries", None)
        if callable(embed_query):
            vector_output = await embed_query([query])
        else:
            vector_output = await self.embeddings.embed([query])
        query_vectors = _validated_vectors(vector_output, expected_count=1)
        candidate_limit = max(limit, self.candidate_limit)
        async with asyncio.TaskGroup() as group:
            if generations is None:
                lexical_task = group.create_task(
                    self.lexical.retrieve(query, limit=candidate_limit, filters=filters)
                )
            else:
                lexical = cast(GenerationAwareRetriever, self.lexical)
                lexical_task = group.create_task(
                    lexical.retrieve(
                        query,
                        limit=candidate_limit,
                        filters=filters,
                        generations=generations,
                    )
                )
            if generations is None:
                vector_task = group.create_task(
                    self.vectors.search(query_vectors[0], limit=candidate_limit, filters=filters)
                )
            else:
                vectors = cast(GenerationAwareVectorStore, self.vectors)
                vector_task = group.create_task(
                    vectors.search(
                        query_vectors[0],
                        limit=candidate_limit,
                        filters=filters,
                        generations=generations,
                    )
                )
        lexical_documents = lexical_task.result()
        vector_documents = vector_task.result()
        if not isinstance(lexical_documents, list) or not isinstance(vector_documents, list):
            raise TypeError("retrievers must return lists of documents")
        if len(lexical_documents) > candidate_limit or len(vector_documents) > candidate_limit:
            raise ValueError("retriever returned more documents than requested")

        scores: dict[str, float] = {}
        documents_by_id: dict[str, Document] = {}
        order: dict[str, int] = {}
        next_order = 0
        for ranked_documents in (lexical_documents, vector_documents):
            seen_in_list: set[str] = set()
            for rank, document in enumerate(ranked_documents, start=1):
                document_id = _identity(document)
                if document_id in seen_in_list:
                    continue
                seen_in_list.add(document_id)
                if document_id not in documents_by_id:
                    documents_by_id[document_id] = document
                    order[document_id] = next_order
                    next_order += 1
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                    self.rrf_constant + rank
                )

        ranked_ids = sorted(scores, key=lambda item: (-scores[item], order[item]))[:limit]
        return [deepcopy(documents_by_id[document_id]) for document_id in ranked_ids]


class RerankingRetriever:
    """Apply a second-stage reranker to a bounded candidate set from any retriever.

    The reranker may reorder candidates but cannot add, duplicate, or rewrite retrieved documents.
    Use this wrapper around a lexical or hybrid retriever when a workload needs second-stage
    ranking.
    """

    _MAX_CANDIDATES = MAX_RETRIEVAL_DOCUMENTS

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_limit: int = 20,
    ) -> None:
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int):
            raise TypeError("candidate_limit must be an integer from 1 through 100")
        if not 1 <= candidate_limit <= self._MAX_CANDIDATES:
            raise ValueError("candidate_limit must be an integer from 1 through 100")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Fetch bounded candidates, rerank them, then return trusted copies in the new order."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        if limit == 0 or not query.strip():
            return []

        requested_candidates = max(limit, self.candidate_limit)
        candidates = await self.retriever.retrieve(
            query,
            limit=requested_candidates,
            filters=filters,
        )
        if not isinstance(candidates, list):
            raise TypeError("retriever must return a list of documents")
        if len(candidates) > requested_candidates:
            raise ValueError("retriever returned more documents than requested")

        candidates_by_id: dict[str, Document] = {}
        for document in candidates:
            document_id = _identity(document)
            if not isinstance(document.text, str) or not isinstance(document.source, str):
                raise TypeError("retrieved documents must have string text and source fields")
            candidates_by_id.setdefault(document_id, deepcopy(document))
        if not candidates_by_id:
            return []

        reranked = await self.reranker.rerank(
            query,
            deepcopy(list(candidates_by_id.values())),
            limit=limit,
        )
        if not isinstance(reranked, list):
            raise TypeError("reranker must return a list of documents")
        if len(reranked) > limit:
            raise ValueError("reranker returned more documents than requested")
        ranked_ids: list[str] = []
        seen: set[str] = set()
        for document in reranked:
            document_id = _identity(document)
            if document_id not in candidates_by_id:
                raise ValueError("reranker returned a document outside the candidate set")
            if document_id in seen:
                raise ValueError("reranker returned a duplicate document")
            seen.add(document_id)
            ranked_ids.append(document_id)
        return [deepcopy(candidates_by_id[document_id]) for document_id in ranked_ids]
