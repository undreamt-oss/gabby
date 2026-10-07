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
"""Contract coverage for vector adapters and reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from gabby.hybrid import HybridRetriever, RerankingRetriever
from gabby.knowledge import Document


class FakeEmbeddingProvider:
    def __init__(self, vectors: Any = ((0.1, 0.2),)) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: Any) -> Any:
        self.calls.append(list(texts))
        return self.vectors


class FakeLexicalRetriever:
    def __init__(self, documents: Any) -> None:
        self.documents = documents
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> Any:
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        return self.documents


class FakeVectorStore:
    def __init__(self, documents: Any) -> None:
        self.documents = documents
        self.calls: list[dict[str, Any]] = []

    async def upsert(
        self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]
    ) -> int:
        return len(documents)

    async def replace_source(
        self,
        source: str,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        return len(documents)

    async def delete_source(self, source: str) -> int:
        return 0

    async def search(
        self,
        embedding: Any,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append({"embedding": list(embedding), "limit": limit, "filters": filters})
        return self.documents


def _doc(name: str, *, text: str | None = None) -> Document:
    return Document(text=text or name, source="guide.md", id=name)


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_lexical_and_vector_ranks() -> None:
    lexical = FakeLexicalRetriever([_doc("a"), _doc("b"), _doc("c")])
    vectors = FakeVectorStore([_doc("b"), _doc("c"), _doc("d")])
    embeddings = FakeEmbeddingProvider()
    retriever = HybridRetriever(lexical, embeddings, vectors, candidate_limit=7, rrf_constant=10)
    filters = {"department": "support"}

    result = await retriever.retrieve("reset account", limit=3, filters=filters)

    assert [document.id for document in result] == ["b", "c", "a"]
    assert embeddings.calls == [["reset account"]]
    assert lexical.calls == [{"query": "reset account", "limit": 7, "filters": filters}]
    assert vectors.calls == [{"embedding": [0.1, 0.2], "limit": 7, "filters": filters}]


@pytest.mark.asyncio
async def test_hybrid_retriever_uses_requested_limit_and_stable_tie_order() -> None:
    lexical = FakeLexicalRetriever([_doc("first"), _doc("second")])
    vectors = FakeVectorStore([_doc("second"), _doc("first")])
    retriever = HybridRetriever(lexical, FakeEmbeddingProvider(), vectors, rrf_constant=10)

    result = await retriever.retrieve("query", limit=1)

    assert [document.id for document in result] == ["first"]
    assert lexical.calls[0]["limit"] == 20


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_copies_of_index_documents() -> None:
    source = _doc("only", text="original")
    retriever = HybridRetriever(
        FakeLexicalRetriever([source]),
        FakeEmbeddingProvider(),
        FakeVectorStore([]),
    )

    results = await retriever.retrieve("query")
    results[0].metadata["changed"] = True

    assert source.metadata == {}


@pytest.mark.asyncio
async def test_hybrid_retriever_skips_backends_for_empty_query_or_zero_limit() -> None:
    lexical = FakeLexicalRetriever([])
    vectors = FakeVectorStore([])
    embeddings = FakeEmbeddingProvider()
    retriever = HybridRetriever(lexical, embeddings, vectors)

    assert await retriever.retrieve("   ") == []
    assert await retriever.retrieve("query", limit=0) == []
    assert embeddings.calls == []
    assert lexical.calls == []
    assert vectors.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        [],
        "not-a-sequence",
        ((1.0, 2.0), (3.0, 4.0)),
        ((),),
        ((1.0, float("nan")),),
        ((1.0, float("inf")),),
        ((True, 0.2),),
        ((1.0, "bad"),),
        ((1.0,), (1.0, 2.0)),
    ],
)
async def test_hybrid_retriever_rejects_invalid_query_embeddings(vectors: Any) -> None:
    lexical = FakeLexicalRetriever([])
    vector_store = FakeVectorStore([])
    retriever = HybridRetriever(lexical, FakeEmbeddingProvider(vectors), vector_store)

    with pytest.raises(ValueError, match="embedding"):
        await retriever.retrieve("query")
    assert lexical.calls == []
    assert vector_store.calls == []


@pytest.mark.asyncio
async def test_hybrid_retriever_normalizes_embedding_integer_overflow() -> None:
    retriever = HybridRetriever(
        FakeLexicalRetriever([]),
        FakeEmbeddingProvider(((10**10_000,),)),
        FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="finite numbers"):
        await retriever.retrieve("query")


@pytest.mark.asyncio
async def test_hybrid_retriever_requires_document_lists_from_backends() -> None:
    retriever = HybridRetriever(
        FakeLexicalRetriever(()), FakeEmbeddingProvider(), FakeVectorStore([])
    )

    with pytest.raises(TypeError, match="must return lists"):
        await retriever.retrieve("query")


@pytest.mark.asyncio
@pytest.mark.parametrize("over_returning_backend", ["lexical", "vector"])
async def test_hybrid_retriever_rejects_backend_results_over_candidate_limit(
    over_returning_backend: str,
) -> None:
    lexical = FakeLexicalRetriever([_doc("lexical-a"), _doc("lexical-b")])
    vectors = FakeVectorStore([_doc("vector-a"), _doc("vector-b")])
    if over_returning_backend == "lexical":
        vectors.documents = []
    else:
        lexical.documents = []
    retriever = HybridRetriever(
        lexical,
        FakeEmbeddingProvider(),
        vectors,
        candidate_limit=1,
    )

    with pytest.raises(ValueError, match="more documents than requested"):
        await retriever.retrieve("query", limit=1)


@pytest.mark.asyncio
async def test_hybrid_retriever_caps_direct_requested_result_count() -> None:
    lexical = FakeLexicalRetriever([])
    vectors = FakeVectorStore([])
    embeddings = FakeEmbeddingProvider()
    retriever = HybridRetriever(lexical, embeddings, vectors)

    with pytest.raises(ValueError, match="limit must be an integer from 0 through 100"):
        await retriever.retrieve("query", limit=101)
    assert embeddings.calls == []
    assert lexical.calls == []
    assert vectors.calls == []


@pytest.mark.asyncio
async def test_hybrid_retriever_rejects_invalid_documents() -> None:
    retriever = HybridRetriever(
        FakeLexicalRetriever([object()]), FakeEmbeddingProvider(), FakeVectorStore([])
    )

    with pytest.raises(TypeError, match="Document instances"):
        await retriever.retrieve("query")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "limit", "filters", "error", "message"),
    [
        (None, 5, None, TypeError, "query must be a string"),
        ("query", True, None, ValueError, "limit must be an integer"),
        ("query", 5, [], TypeError, "filters must be a mapping"),
        ("query", 5, {1: "invalid"}, TypeError, "filter keys must be strings"),
    ],
)
async def test_hybrid_retriever_validates_query_and_filter_arguments(
    query: Any,
    limit: Any,
    filters: Any,
    error: type[Exception],
    message: str,
) -> None:
    lexical = FakeLexicalRetriever([])
    vectors = FakeVectorStore([])
    embeddings = FakeEmbeddingProvider()
    retriever = HybridRetriever(lexical, embeddings, vectors)

    with pytest.raises(error, match=message):
        await retriever.retrieve(query, limit=limit, filters=filters)
    assert embeddings.calls == []
    assert lexical.calls == []
    assert vectors.calls == []


@pytest.mark.asyncio
async def test_hybrid_retriever_deduplicates_documents_without_explicit_ids() -> None:
    duplicate = Document("same text", source="guide.md")
    retriever = HybridRetriever(
        FakeLexicalRetriever([duplicate, duplicate]),
        FakeEmbeddingProvider(),
        FakeVectorStore([]),
    )

    results = await retriever.retrieve("query")

    assert len(results) == 1
    assert (results[0].source, results[0].text) == ("guide.md", "same text")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_store", ["lexical", "vector"])
async def test_manifest_retrieval_requires_generation_aware_stores_at_construction(
    missing_store: str,
) -> None:
    class Manifest:
        async def active_generations(self) -> dict[str, str]:
            return {"guide.md": "generation-1"}

    class GenerationAwareLexical(FakeLexicalRetriever):
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
            generations: dict[str, str] | None = None,
        ) -> Any:
            del query, limit, filters, generations
            return self.documents

    lexical: Any = FakeLexicalRetriever([])
    vectors: Any = FakeVectorStore([])
    if missing_store == "vector":
        lexical = GenerationAwareLexical([])

    with pytest.raises(TypeError, match=f"generation-aware {missing_store} store"):
        HybridRetriever(lexical, FakeEmbeddingProvider(), vectors, manifest=Manifest())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_filter", ["lexical", "vector", "lexical-positional", "vector-positional"]
)
async def test_manifest_retrieval_requires_generation_filter_keyword_at_construction(
    missing_filter: str,
) -> None:
    class Manifest:
        async def active_generations(self) -> dict[str, str]:
            return {"guide.md": "generation-1"}

    class GenerationAwareLexical(FakeLexicalRetriever):
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
            generations: dict[str, str] | None = None,
        ) -> Any:
            del query, limit, filters, generations
            return self.documents

    class GenerationAwareVectors(FakeVectorStore):
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def search(
            self,
            embedding: Any,
            *,
            limit: int,
            filters: dict[str, Any] | None = None,
            generations: dict[str, str] | None = None,
        ) -> Any:
            del embedding, limit, filters, generations
            return self.documents

    class LexicalWithoutGenerationFilter(FakeLexicalRetriever):
        async def stage_source(self, *_: Any) -> int:
            return 0

    class VectorsWithoutGenerationFilter(FakeVectorStore):
        async def stage_source(self, *_: Any) -> int:
            return 0

    class LexicalPositionalGenerationFilter:
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def retrieve(
            self,
            query: str,
            generations: dict[str, str] | None,
            /,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
        ) -> Any:
            del query, generations, limit, filters
            return []

    class VectorsPositionalGenerationFilter:
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def search(
            self,
            embedding: Any,
            generations: dict[str, str] | None,
            /,
            *,
            limit: int,
            filters: dict[str, Any] | None = None,
        ) -> Any:
            del embedding, generations, limit, filters
            return []

    lexical: Any = GenerationAwareLexical([])
    vectors: Any = GenerationAwareVectors([])
    if missing_filter == "lexical":
        lexical = LexicalWithoutGenerationFilter([])
    elif missing_filter == "vector":
        vectors = VectorsWithoutGenerationFilter([])
    elif missing_filter == "lexical-positional":
        lexical = LexicalPositionalGenerationFilter()
    else:
        vectors = VectorsPositionalGenerationFilter()

    store_name = missing_filter.split("-")[0]
    with pytest.raises(TypeError, match=f"generation-aware {store_name} store"):
        HybridRetriever(lexical, FakeEmbeddingProvider(), vectors, manifest=Manifest())


@pytest.mark.asyncio
async def test_hybrid_retriever_filters_both_indexes_by_active_generation() -> None:
    generations = {"guide.md": "generation-3"}

    class Manifest:
        async def active_generations(self) -> dict[str, str]:
            return generations

    class GenerationAwareLexical(FakeLexicalRetriever):
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def retrieve(
            self,
            query: str,
            *,
            limit: int = 5,
            filters: dict[str, Any] | None = None,
            generations: dict[str, str] | None = None,
        ) -> Any:
            self.calls.append(
                {"query": query, "limit": limit, "filters": filters, "generations": generations}
            )
            return self.documents

    class GenerationAwareVectors(FakeVectorStore):
        async def stage_source(self, *_: Any) -> int:
            return 0

        async def search(
            self,
            embedding: Any,
            *,
            limit: int,
            filters: dict[str, Any] | None = None,
            generations: dict[str, str] | None = None,
        ) -> Any:
            self.calls.append(
                {
                    "embedding": list(embedding),
                    "limit": limit,
                    "filters": filters,
                    "generations": generations,
                }
            )
            return self.documents

    lexical = GenerationAwareLexical([_doc("current")])
    vectors = GenerationAwareVectors([_doc("current")])
    retriever = HybridRetriever(
        lexical,
        FakeEmbeddingProvider(),
        vectors,
        manifest=Manifest(),
    )

    results = await retriever.retrieve("query")

    assert [document.id for document in results] == ["current"]
    assert lexical.calls[0]["generations"] == generations
    assert vectors.calls[0]["generations"] == generations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document", "error", "message"),
    [
        (Document(cast(str, 5)), TypeError, "string text and source"),
        (Document("text", source=cast(str, 3)), TypeError, "string text and source"),
        (Document("text", id=""), ValueError, "IDs must be non-empty"),
        (Document("text", id=cast(str, 4)), ValueError, "IDs must be non-empty"),
    ],
)
async def test_hybrid_retriever_rejects_invalid_document_identity(
    document: Document,
    error: type[Exception],
    message: str,
) -> None:
    retriever = HybridRetriever(
        FakeLexicalRetriever([document]), FakeEmbeddingProvider(), FakeVectorStore([])
    )

    with pytest.raises(error, match=message):
        await retriever.retrieve("query")


@pytest.mark.asyncio
async def test_reranking_retriever_composes_with_any_retriever_and_bounds_candidates() -> None:
    class ReverseReranker:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def rerank(
            self, query: str, documents: Sequence[Document], *, limit: int
        ) -> list[Document]:
            self.calls.append({"query": query, "documents": list(documents), "limit": limit})
            documents[-1].text = "rewritten by the reranker"
            documents[-1].metadata["changed"] = True
            return list(reversed(documents))[:limit]

    lexical = FakeLexicalRetriever([_doc("a"), _doc("b"), _doc("c")])
    reranker = ReverseReranker()
    retriever = RerankingRetriever(lexical, reranker, candidate_limit=3)

    results = await retriever.retrieve("account recovery", limit=2)

    assert [document.id for document in results] == ["c", "b"]
    assert lexical.calls == [{"query": "account recovery", "limit": 3, "filters": None}]
    assert reranker.calls[0]["limit"] == 2
    assert [document.id for document in reranker.calls[0]["documents"]] == ["a", "b", "c"]
    assert results[0].text == "c"
    results[0].metadata["changed"] = True
    assert lexical.documents[2].metadata == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "limit", "filters", "error", "message"),
    [
        (None, 5, None, TypeError, "query must be a string"),
        ("query", True, None, ValueError, "limit must be an integer"),
        ("query", 5, [], TypeError, "filters must be a mapping"),
        ("query", 5, {1: "invalid"}, TypeError, "filter keys must be strings"),
    ],
)
async def test_reranking_retriever_validates_query_and_filter_arguments(
    query: Any,
    limit: Any,
    filters: Any,
    error: type[Exception],
    message: str,
) -> None:
    class NeverReranker:
        async def rerank(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("invalid retrieval arguments must fail before reranking")

    lexical = FakeLexicalRetriever([])
    retriever = RerankingRetriever(lexical, NeverReranker())

    with pytest.raises(error, match=message):
        await retriever.retrieve(query, limit=limit, filters=filters)
    assert lexical.calls == []


@pytest.mark.asyncio
async def test_reranking_retriever_skips_provider_for_empty_candidates() -> None:
    class NeverReranker:
        async def rerank(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("empty candidate lists must not invoke reranking")

    retriever = RerankingRetriever(FakeLexicalRetriever([]), NeverReranker())

    assert await retriever.retrieve("query") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("documents", "error", "message"),
    [
        ("not-a-list", TypeError, "retriever must return a list"),
        ([Document("a"), Document("b"), Document("c")], ValueError, "more documents"),
    ],
)
async def test_reranking_retriever_rejects_invalid_candidate_lists(
    documents: Any,
    error: type[Exception],
    message: str,
) -> None:
    class FixedRetriever:
        async def retrieve(self, *_: Any, **__: Any) -> Any:
            return documents

    class NeverReranker:
        async def rerank(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("invalid candidate results must fail before reranking")

    retriever = RerankingRetriever(FixedRetriever(), NeverReranker(), candidate_limit=2)

    with pytest.raises(error, match=message):
        await retriever.retrieve("query", limit=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ranked", "message"),
    [
        ([_doc("outside")], "outside the candidate set"),
        ([_doc("candidate"), _doc("candidate")], "duplicate document"),
        (
            [_doc("candidate"), _doc("other"), _doc("third")],
            "more documents than requested",
        ),
        ("not-a-list", "must return a list"),
    ],
)
async def test_reranking_retriever_rejects_invalid_reranker_results(
    ranked: Any, message: str
) -> None:
    class FixedReranker:
        async def rerank(self, query: str, documents: Sequence[Document], *, limit: int) -> Any:
            return ranked

    retriever = RerankingRetriever(
        FakeLexicalRetriever([_doc("candidate"), _doc("other"), _doc("third")]),
        FixedReranker(),
        candidate_limit=3,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        await retriever.retrieve("query", limit=2)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"candidate_limit": 0}, ValueError),
        ({"candidate_limit": 101}, ValueError),
        ({"candidate_limit": True}, TypeError),
        ({"rrf_constant": 0}, ValueError),
        ({"rrf_constant": float("nan")}, ValueError),
    ],
)
def test_hybrid_retriever_validates_configuration(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        HybridRetriever(
            FakeLexicalRetriever([]), FakeEmbeddingProvider(), FakeVectorStore([]), **kwargs
        )
