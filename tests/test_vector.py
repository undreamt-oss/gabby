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
"""Persistence, validation, and generation contracts for the SQLite vector backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from gabby import (
    Document,
    HybridIndexCoordinator,
    KnowledgeStoreError,
    SQLiteFTS5Store,
    SQLiteGenerationManifestStore,
    SQLiteVectorStore,
)


@pytest.mark.parametrize(
    ("path", "busy_timeout", "error"),
    [
        (":memory:", 1.0, ValueError),
        ("vectors.db", True, ValueError),
        ("vectors.db", float("inf"), ValueError),
    ],
)
def test_sqlite_vector_store_validates_construction(
    path: str, busy_timeout: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        SQLiteVectorStore(path, busy_timeout_seconds=busy_timeout)


def test_sqlite_vector_store_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    import asyncio
    import sqlite3

    path = tmp_path / "vectors.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE gabby_vector_schema (singleton INTEGER PRIMARY KEY, version INTEGER)"
        )
        connection.execute("INSERT INTO gabby_vector_schema VALUES (1, 99)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(KnowledgeStoreError, match="schema version"):
        asyncio.run(SQLiteVectorStore(path)._run(lambda _: 1))


@pytest.mark.asyncio
async def test_sqlite_vector_store_searches_by_cosine_and_persists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vectors.db"
    store = SQLiteVectorStore(path)
    documents = [
        Document("north", "guide.md", {"kind": "direction"}, id="north"),
        Document("northeast", "guide.md", {"kind": "direction"}, id="northeast"),
        Document("east", "guide.md", {"kind": "direction"}, id="east"),
    ]
    assert await store.upsert(documents, [[1, 0], [0.8, 0.6], [0, 1]]) == 3

    results = await store.search([7, 7], limit=3)
    assert [document.id for document in results] == ["northeast", "east", "north"]
    filtered = await store.search([1, 0], limit=5, filters={"kind": "direction"})
    assert [document.id for document in filtered] == ["north", "northeast", "east"]
    assert await store.search([1, 0], limit=5, filters={"kind": True}) == []

    reopened = SQLiteVectorStore(path)
    persisted = await reopened.search([1, 0], limit=1)
    assert [(item.id, item.text, item.source) for item in persisted] == [
        ("north", "north", "guide.md")
    ]


@pytest.mark.asyncio
async def test_sqlite_vector_store_replaces_sources_atomically_and_fixes_dimension(
    tmp_path: Path,
) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    first = Document("old", "faq.md", id="old")
    await store.replace_source("faq.md", [first], [[1, 0]])

    with pytest.raises(ValueError, match="embedding dimension"):
        await store.replace_source("faq.md", [Document("new", "faq.md")], [[1, 0, 0]])
    with pytest.raises(ValueError, match="embedding dimension"):
        await store.search([1, 0, 0], limit=1)
    assert [item.id for item in await store.search([1, 0], limit=5)] == ["old"]

    assert (
        await store.replace_source("faq.md", [Document("new", "faq.md", id="new")], [[0, 1]]) == 1
    )
    assert [item.id for item in await store.search([0, 1], limit=5)] == ["new"]
    assert await store.delete_source("faq.md") == 1
    assert await store.search([0, 1], limit=5) == []


@pytest.mark.asyncio
async def test_sqlite_vector_store_supports_empty_and_upsert_paths(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    assert await store.upsert([], []) == 0
    assert await store.replace_source("empty.md", [], []) == 0
    assert await store.upsert([Document("old", "guide.md", id="same")], [[1, 0]]) == 1
    assert await store.upsert([Document("new", "guide.md", id="same")], [[0, 1]]) == 1
    assert [item.text for item in await store.search([0, 1], limit=1)] == ["new"]

    with pytest.raises(ValueError, match="requested source"):
        await store.replace_source("other.md", [Document("wrong", "guide.md")], [[1, 0]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("documents", "embeddings", "error", "message"),
    [
        ([Document("one", "a")], [], ValueError, "count must match"),
        ([Document("one", "a")], [[0, 0]], ValueError, "non-zero norm"),
        ([Document("one", "a")], [[float("nan"), 0]], ValueError, "finite numbers"),
        ([Document("one", "a")], [[True, 0]], ValueError, "finite numbers"),
        ([Document("one", "a")], [[10**1000, 0]], ValueError, "finite numbers"),
        ([Document("one", "a")], [[1, 0], [0, 1]], ValueError, "count must match"),
        (
            [Document("one", "a"), Document("two", "b")],
            [[1, 0], [0, 0, 1]],
            ValueError,
            "consistent dimension",
        ),
    ],
)
async def test_sqlite_vector_store_rejects_invalid_batches_without_writing(
    tmp_path: Path,
    documents: list[Document],
    embeddings: list[list[float]],
    error: type[Exception],
    message: str,
) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    with pytest.raises(error, match=message):
        await store.upsert(documents, embeddings)
    assert await store.search([1, 0], limit=5) == []


@pytest.mark.asyncio
async def test_sqlite_vector_store_fences_and_filters_generations(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    doc = Document("old guidance", "guide.md", id="guidance")
    await store.stage_source("guide.md", "old", 3, [doc], [[1, 0]])
    await store.stage_source("guide.md", "new", 4, [doc], [[0, 1]])

    old = await store.search([1, 0], limit=5, generations={"guide.md": "old"})
    active = await store.search([1, 0], limit=5, generations={"guide.md": "new"})
    assert [(item.id, item.generation) for item in old] == [("guidance", "old")]
    assert [(item.id, item.generation) for item in active] == [("guidance", "new")]

    await store.advance_fence("guide.md", 7)
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source("guide.md", "stale", 6, [doc], [[1, 0]])
    assert await store.discard_generation("guide.md", "old", 7) == 1
    assert await store.delete_generation("guide.md", "old") == 0
    assert await store.search([1, 0], limit=5, generations={"guide.md": "old"}) == []


@pytest.mark.asyncio
async def test_sqlite_vector_store_validates_public_query_and_writer_inputs(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")

    with pytest.raises(ValueError, match="source"):
        await store.replace_source(" ", [], [])
    with pytest.raises(ValueError, match="generation"):
        await store.stage_source("guide.md", " ", 1, [], [])
    with pytest.raises(TypeError, match="fencing_token"):
        await store.stage_source("guide.md", "one", True, [], [])
    with pytest.raises(ValueError, match="requested source"):
        await store.stage_source("guide.md", "one", 1, [Document("x", "other.md")], [[1, 0]])
    with pytest.raises(ValueError, match="limit"):
        await store.search([1, 0], limit=True)
    with pytest.raises(TypeError, match="filters"):
        await store.search([1, 0], limit=1, filters=[("a", 1)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keys"):
        await store.search([1, 0], limit=1, filters={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="generations"):
        await store.search([1, 0], limit=1, generations=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="generation IDs"):
        await store.search([1, 0], limit=1, generations={"guide.md": 1})  # type: ignore[dict-item]
    assert await store.search([1, 0], limit=0) == []
    assert await store.search([], limit=0) == []
    assert await store.search([1, 0], limit=1, generations={}) == []

    with pytest.raises(ValueError, match="numeric sequences"):
        await store.search("not a vector", limit=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimension must be"):
        await store.search([1.0] * 65_537, limit=1)

    with pytest.raises(TypeError, match="sequences"):
        await store.upsert(None, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Document instances"):
        await store.upsert([object()], [[1, 0]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty string"):
        await store.upsert([Document("", "guide.md")], [[1, 0]])
    with pytest.raises(TypeError, match="source must be a string"):
        await store.upsert([Document("x", 1)], [[1, 0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string keys"):
        await store.upsert([Document("x", "guide.md", {1: "x"})], [[1, 0]])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite JSON"):
        await store.upsert([Document("x", "guide.md", {"bad": object()})], [[1, 0]])
    with pytest.raises(ValueError, match="non-empty string"):
        await store.upsert([Document("x", "guide.md", id=" ")], [[1, 0]])


@pytest.mark.asyncio
async def test_sqlite_vector_store_wraps_database_errors(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path)
    with pytest.raises(KnowledgeStoreError, match="SQLite vector operation failed"):
        await store.search([1, 0], limit=1)


@pytest.mark.asyncio
async def test_sqlite_vector_store_rejects_corrupt_metadata(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "vectors.db"
    store = SQLiteVectorStore(path)
    await store.upsert([Document("evidence", "guide.md", id="evidence")], [[1, 0]])
    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE gabby_vectors SET metadata_json = '[]'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(KnowledgeStoreError, match="metadata is invalid"):
        await store.search([1, 0], limit=1)


@pytest.mark.asyncio
async def test_hybrid_index_coordinator_uses_sqlite_vector_generations(tmp_path: Path) -> None:
    class Embeddings:
        async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [[1.0, 0.0] if "north" in text else [0.0, 1.0] for text in texts]

    lexical = SQLiteFTS5Store(tmp_path / "lexical.db")
    vectors = SQLiteVectorStore(tmp_path / "vectors.db")
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    coordinator = HybridIndexCoordinator(lexical, Embeddings(), vectors, manifest)

    await coordinator.replace_source(
        "directions.md",
        [
            Document("north route", "directions.md", {"section": 1}, id="north"),
            Document("east route", "directions.md", {"section": 2}, id="east"),
        ],
    )
    results = await coordinator.retrieve("north", limit=2)
    assert [item.id for item in results] == ["north", "east"]
    assert all(item.generation is not None for item in results)

    await coordinator.replace_source(
        "directions.md", [Document("new east route", "directions.md", id="east")]
    )
    assert [item.id for item in await coordinator.retrieve("north", limit=5)] == ["east"]
