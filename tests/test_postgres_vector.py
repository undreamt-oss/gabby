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
"""SDK-independent contract tests for the PostgreSQL pgvector adapter."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from gabby import Document, KnowledgeStoreError, PostgresVectorStore


class _Connection:
    def __init__(
        self,
        *,
        fetch: list[list[dict[str, Any]]] | None = None,
        dimensions: int = 3,
        configured: bool = False,
        fences: list[dict[str, Any] | None] | None = None,
        deletes: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fence_results = deque(fences or [])
        self.delete_results = deque(deletes or [])
        self.dimensions = dimensions
        self.configured = configured
        self.fail = fail
        self.executions: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL vector details")
        if "INSERT INTO gabby_vector_store_config" in query and not self.configured:
            self.dimensions = int(args[0])
            self.configured = True
        if query.lstrip().startswith("DELETE"):
            return self.delete_results.popleft() if self.delete_results else "DELETE 0"
        return "OK"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executions.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL vector details")
        if "SELECT dimensions" in query:
            return {"dimensions": self.dimensions} if self.configured else None
        return self.fence_results.popleft() if self.fence_results else {"fencing_token": 1}

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL vector details")
        return self.fetch_results.popleft() if self.fetch_results else []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Connection]:
        yield self.connection


def _store(connection: _Connection, *, dimensions: int = 3) -> PostgresVectorStore:
    return PostgresVectorStore(_Pool(connection), dimensions=dimensions)


def _document(text: str = "auth failure", *, source: str = "docs", **kwargs: Any) -> Document:
    return Document(text=text, source=source, **kwargs)


def test_constructor_and_vector_validation() -> None:
    with pytest.raises(TypeError, match="acquire"):
        PostgresVectorStore(object(), dimensions=3)  # type: ignore[arg-type]
    for dimensions in (True, 1.5, "3"):
        with pytest.raises(TypeError, match="dimensions"):
            PostgresVectorStore(_Pool(_Connection()), dimensions=dimensions)  # type: ignore[arg-type]
    for dimensions in (0, 2_001):
        with pytest.raises(ValueError, match="dimensions"):
            PostgresVectorStore(_Pool(_Connection()), dimensions=dimensions)
    for setting in (True, 1.5):
        with pytest.raises(TypeError):
            PostgresVectorStore(_Pool(_Connection()), dimensions=3, ef_search=setting)  # type: ignore[arg-type]
    for setting in (0, 1_001):
        with pytest.raises(ValueError):
            PostgresVectorStore(_Pool(_Connection()), dimensions=3, ef_search=setting)
    with pytest.raises(ValueError):
        PostgresVectorStore(_Pool(_Connection()), dimensions=3, max_scan_tuples=0)

    store = _store(_Connection())
    for vector in ((1, 2), (0, 0, 0), (1, float("inf"), 2), (True, 0, 1), "bad"):
        with pytest.raises((TypeError, ValueError)):
            store._validate_vector(vector)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_upsert_persists_stable_ids_vectors_and_dimension_contract() -> None:
    connection = _Connection()
    store = _store(connection)
    document = _document(metadata={"team": "runtime"})

    assert await store.upsert([document], [[1, 0, 0]]) == 1
    statements = [query for query, _ in connection.executions]
    assert "SELECT dimensions" in statements[0]
    assert "gabby_vector_store_config" in statements[1]
    assert "SELECT dimensions" in statements[2]
    assert "pg_advisory_xact_lock" in statements[3]
    upsert_query, args = connection.executions[4]
    assert "ON CONFLICT (storage_id) DO UPDATE" in upsert_query
    assert args[0] == args[1]
    assert args[4] == '{"team":"runtime"}'
    assert args[-1] == "[1.0,0.0,0.0]"
    assert await store.upsert([], []) == 0

    with pytest.raises(ValueError, match="dimension"):
        await store.upsert([_document()], [[1, 0]])
    with pytest.raises(ValueError, match="one embedding"):
        await store.upsert([_document()], [])


@pytest.mark.asyncio
async def test_replace_and_delete_source_validate_and_return_counts() -> None:
    connection = _Connection(deletes=["DELETE 2", "DELETE 1"])
    store = _store(connection)
    with pytest.raises(ValueError, match="requested source"):
        await store.replace_source("docs", [_document(source="other")], [[1, 0, 0]])
    assert not connection.executions

    assert await store.replace_source("docs", [_document("new")], [[0, 1, 0]]) == 1
    assert any(
        "DELETE FROM gabby_vector_documents WHERE source" in query
        for query, _ in connection.executions
    )
    assert await store.delete_source("docs") == 1


@pytest.mark.asyncio
async def test_search_uses_cosine_distance_metadata_and_generation_filters() -> None:
    connection = _Connection(
        fetch=[
            [
                {
                    "document_id": "logical-id",
                    "text": "authentication issue",
                    "source": "docs",
                    "metadata_json": '{"team":"runtime"}',
                    "generation": "g2",
                }
            ]
        ]
    )
    store = _store(connection)
    docs = await store.search(
        [1, 0, 0],
        limit=7,
        filters={"team": "runtime"},
        generations={"docs": "g2"},
    )
    assert docs == [
        Document(
            text="authentication issue",
            source="docs",
            metadata={"team": "runtime"},
            id="logical-id",
            generation="g2",
        )
    ]
    query, args = connection.fetches[0]
    search_settings = [
        (query, args) for query, args in connection.executions if "set_config('hnsw." in query
    ]
    assert search_settings == [
        ("SELECT set_config('hnsw.iterative_scan', $1, TRUE)", ("strict_order",)),
        (
            "SELECT set_config('hnsw.ef_search', $1, TRUE), "
            "set_config('hnsw.max_scan_tuples', $2, TRUE)",
            ("40", "20000"),
        ),
    ]
    assert "embedding <=> $1::text::vector" in query
    assert "metadata -> $2::text = $3::jsonb" in query
    assert "source = $4::text AND generation = $5::text" in query
    assert query.endswith("LIMIT $6")
    assert args == ("[1.0,0.0,0.0]", "team", '"runtime"', "docs", "g2", 7)

    for kwargs in (
        {"limit": True},
        {"limit": 101},
        {"filters": {1: "invalid"}},
        {"generations": {"docs": ""}},
    ):
        with pytest.raises((TypeError, ValueError)):
            await store.search([1, 0, 0], **kwargs)
    assert await store.search([1, 0, 0], limit=0) == []
    assert await store.search([1, 0, 0], limit=5, generations={}) == []


@pytest.mark.asyncio
async def test_search_rejects_mismatched_persisted_dimension() -> None:
    with pytest.raises(KnowledgeStoreError, match="dimension"):
        await _store(_Connection(dimensions=4, configured=True)).search([1, 0, 0], limit=1)


@pytest.mark.asyncio
async def test_stage_fences_source_and_preserves_logical_id() -> None:
    connection = _Connection(fences=[{"fencing_token": 4}])
    store = _store(connection)
    original = _document(id="stable-id")

    assert await store.stage_source("docs", "generation-4", 4, [original], [[1, 0, 0]]) == 1
    fence_query, fence_args = next(
        (query, args) for query, args in connection.executions if "RETURNING fencing_token" in query
    )
    insert_query, args = connection.executions[-1]
    assert "gabby_vector_source_fences" in fence_query
    assert fence_args == ("docs", 4)
    assert "INSERT INTO gabby_vector_documents" in insert_query
    assert args[0] != args[1]
    assert args[1] == "stable-id"
    assert args[-1] == "[1.0,0.0,0.0]"
    assert original.generation is None


@pytest.mark.asyncio
async def test_stale_writer_and_generation_cleanup_contract() -> None:
    connection = _Connection(
        fences=[None, {"fencing_token": 5}, {"fencing_token": 6}],
        deletes=["DELETE 1", "DELETE 0"],
    )
    store = _store(connection)
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source("docs", "stale", 4, [_document()], [[1, 0, 0]])
    assert not any(
        "DELETE FROM gabby_vector_documents" in query for query, _ in connection.executions
    )
    assert await store.discard_generation("docs", "incomplete", 5) == 1
    await store.advance_fence("docs", 6)
    assert await store.delete_generation("docs", "retired") == 0


@pytest.mark.asyncio
async def test_database_failures_are_redacted() -> None:
    with pytest.raises(KnowledgeStoreError, match="operation failed") as captured:
        await _store(_Connection(fail=True)).upsert([_document()], [[1, 0, 0]])
    assert "private PostgreSQL vector details" not in str(captured.value)
