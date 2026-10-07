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
"""SDK-independent contract tests for the PostgreSQL knowledge adapter."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from gabby import Document, KnowledgeStoreError, PostgresKnowledgeStore


class _Connection:
    def __init__(
        self,
        *,
        fetch: list[list[dict[str, Any]]] | None = None,
        fetchrow: list[dict[str, Any] | None] | None = None,
        execute: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fetchrow_results = deque(fetchrow or [])
        self.execute_results = deque(execute or [])
        self.executions: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = fail

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL details")
        if query.lstrip().startswith("DELETE"):
            return self.execute_results.popleft() if self.execute_results else "DELETE 0"
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL details")
        return self.fetch_results.popleft() if self.fetch_results else []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executions.append((query, args))
        if self.fail:
            raise RuntimeError("private PostgreSQL details")
        return self.fetchrow_results.popleft() if self.fetchrow_results else None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Connection]:
        yield self.connection


def _store(connection: _Connection) -> PostgresKnowledgeStore:
    return PostgresKnowledgeStore(_Pool(connection))


def _document(text: str = "auth failure", *, source: str = "docs", **kwargs: Any) -> Document:
    return Document(text=text, source=source, **kwargs)


def test_constructor_and_document_validation() -> None:
    with pytest.raises(TypeError, match="acquire"):
        PostgresKnowledgeStore(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Document"):
        PostgresKnowledgeStore(_Pool(_Connection()))._prepare(object())  # type: ignore[arg-type]
    for document in (
        Document(text="", source="s"),
        Document(text="ok", source="s", metadata={1: "bad"}),  # type: ignore[dict-item]
        Document(text="ok", source="s", metadata={"n": float("nan")}),
        Document(text="ok", source="s", generation=" "),
        Document(text="ok", source="s", id=" "),
    ):
        with pytest.raises((TypeError, ValueError)):
            PostgresKnowledgeStore._prepare(document)


@pytest.mark.asyncio
async def test_ingest_uses_stable_logical_ids_and_source_locks() -> None:
    connection = _Connection()
    store = _store(connection)
    document = _document(metadata={"team": "runtime"})

    assert await store.ingest([document]) == 1
    lock_query, lock_args = connection.executions[0]
    insert_query, insert_args = connection.executions[1]
    assert "pg_advisory_xact_lock" in lock_query
    assert lock_args == ("docs",)
    assert "ON CONFLICT (storage_id) DO UPDATE" in insert_query
    assert insert_args[0] == insert_args[1]
    assert insert_args[2:4] == ("auth failure", "docs")
    assert insert_args[4] == '{"team":"runtime"}'
    assert await store.ingest([]) == 0


@pytest.mark.asyncio
async def test_replace_and_delete_source_are_atomic_and_validated() -> None:
    connection = _Connection(execute=["DELETE 2", "DELETE 1"])
    store = _store(connection)
    with pytest.raises(ValueError, match="requested source"):
        await store.replace_source("docs", [_document(source="other")])
    assert not connection.executions

    assert await store.replace_source("docs", [_document("new snapshot")]) == 1
    assert "DELETE FROM gabby_knowledge_documents WHERE source" in connection.executions[1][0]
    assert await store.delete_source("docs") == 1


@pytest.mark.asyncio
async def test_retrieve_applies_metadata_generation_filters_and_result_bound() -> None:
    connection = _Connection(
        fetch=[
            [
                {
                    "document_id": "logical-id",
                    "text": "authentication failure",
                    "source": "docs",
                    "metadata_json": '{"team":"runtime"}',
                    "generation": "g2",
                }
            ]
        ]
    )
    store = _store(connection)
    docs = await store.retrieve(
        "Authentication failed!",
        limit=6,
        filters={"team": "runtime"},
        generations={"docs": "g2"},
    )
    assert docs == [
        Document(
            text="authentication failure",
            source="docs",
            metadata={"team": "runtime"},
            id="logical-id",
            generation="g2",
        )
    ]
    query, args = connection.fetches[0]
    assert "metadata -> $2::text = $3::jsonb" in query
    assert "source = $4::text AND generation = $5::text" in query
    assert query.endswith("LIMIT $6")
    assert args == (["authentication", "failed"], "team", '"runtime"', "docs", "g2", 6)

    for kwargs in (
        {"limit": True},
        {"limit": 101},
        {"generations": {"docs": ""}},
        {"filters": {1: "bad"}},
    ):
        with pytest.raises((TypeError, ValueError)):
            await store.retrieve("query", **kwargs)
    assert await store.retrieve("---") == []
    assert await store.retrieve("query", limit=0) == []
    assert await store.retrieve("query", generations={}) == []


@pytest.mark.asyncio
async def test_stage_fences_generation_and_preserves_logical_document_id() -> None:
    connection = _Connection(fetchrow=[{"fencing_token": 4}])
    store = _store(connection)
    document = _document(id="stable-id")

    assert await store.stage_source("docs", "generation-4", 4, [document]) == 1
    fence_query, fence_args = connection.executions[1]
    delete_query, delete_args = connection.executions[2]
    insert_query, insert_args = connection.executions[3]
    assert "gabby_knowledge_source_fences" in fence_query
    assert fence_args == ("docs", 4)
    assert "generation = $2" in delete_query
    assert delete_args == ("docs", "generation-4")
    assert "INSERT INTO gabby_knowledge_documents" in insert_query
    assert insert_args[0] != insert_args[1]
    assert insert_args[1] == "stable-id"
    assert insert_args[-1] == "generation-4"
    assert document.generation is None


@pytest.mark.asyncio
async def test_stale_fence_rejects_before_generation_mutation() -> None:
    connection = _Connection(fetchrow=[None])
    store = _store(connection)
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source("docs", "old", 3, [_document()])
    assert not any(
        "DELETE FROM gabby_knowledge_documents" in query for query, _ in connection.executions
    )


@pytest.mark.asyncio
async def test_generation_cleanup_and_fence_operations_validate_and_serialize() -> None:
    connection = _Connection(
        fetchrow=[{"fencing_token": 5}, {"fencing_token": 6}, None],
        execute=["DELETE 1", "DELETE 0"],
    )
    store = _store(connection)
    assert await store.discard_generation("docs", "incomplete", 5) == 1
    await store.advance_fence("docs", 6)
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.advance_fence("docs", 4)
    assert await store.delete_generation("docs", "retired") == 0
    with pytest.raises(ValueError):
        await store.discard_generation("", "g", 1)
    with pytest.raises(TypeError):
        await store.advance_fence("docs", True)


@pytest.mark.asyncio
async def test_database_failures_are_redacted() -> None:
    with pytest.raises(KnowledgeStoreError, match="operation failed") as captured:
        await _store(_Connection(fail=True)).ingest([_document()])
    assert "private PostgreSQL details" not in str(captured.value)
