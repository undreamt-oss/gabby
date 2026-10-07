# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.
"""Reference lexical retrieval behavior."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from gabby.knowledge import (
    Document,
    InMemoryBM25Retriever,
    KnowledgeStoreError,
    SQLiteFTS5Store,
)


@pytest.mark.asyncio
async def test_bm25_retriever_ranks_lexical_matches_and_respects_metadata_filters() -> None:
    runbook = Document(
        "Authentication token validation fails after key rotation.",
        source="auth.md",
        metadata={"team": "identity", "version": 2},
    )
    unrelated = Document(
        "The build pipeline compiles the application.",
        source="build.md",
        metadata={"team": "platform"},
    )
    retriever = InMemoryBM25Retriever(
        [
            Document("Token checks use the configured issuer.", source="guide.md"),
            Document("token token authentication and key rotation", source="error.md"),
            runbook,
            unrelated,
        ]
    )

    results = await retriever.retrieve("AUTH token rotation", limit=2)
    filtered = await retriever.retrieve(
        "authentication token rotation",
        limit=4,
        filters={"team": "identity", "version": 2},
    )

    assert [document.source for document in results] == ["error.md", "auth.md"]
    assert [document.source for document in filtered] == ["auth.md"]


@pytest.mark.asyncio
async def test_bm25_retriever_returns_no_results_for_empty_or_unmatched_queries() -> None:
    retriever = InMemoryBM25Retriever([Document("alpha beta", source="one")])

    assert await retriever.retrieve("  !!! ") == []
    assert await retriever.retrieve("missing term") == []
    assert await retriever.retrieve("alpha", limit=0) == []


@pytest.mark.asyncio
async def test_bm25_retriever_indexes_a_snapshot_of_documents() -> None:
    document = Document("alpha", metadata={"team": "one"})
    retriever = InMemoryBM25Retriever([document])
    document.text = "beta"
    document.metadata["team"] = "two"

    result = await retriever.retrieve("alpha", filters={"team": "one"})
    assert [document.source for document in result] == [""]
    assert await retriever.retrieve("beta") == []
    result[0].metadata["team"] = "changed"
    assert await retriever.retrieve("alpha", filters={"team": "one"})


@pytest.mark.parametrize(
    ("documents", "kwargs", "message"),
    [
        ([Document("ok")], {"k1": 0}, "k1"),
        ([Document("ok")], {"b": 1.1}, "b"),
        ([object()], {}, "Document instances"),
        ([Document(3)], {}, "Document text"),  # type: ignore[arg-type]
        ([Document("ok", metadata=None)], {}, "metadata"),  # type: ignore[arg-type]
    ],
)
def test_bm25_retriever_rejects_invalid_index_configuration(
    documents: list[object], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        InMemoryBM25Retriever(documents, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bm25_retriever_rejects_invalid_query_options() -> None:
    retriever = InMemoryBM25Retriever([Document("alpha")])

    with pytest.raises(ValueError, match="limit"):
        await retriever.retrieve("alpha", limit=-1)
    with pytest.raises(TypeError, match="filters"):
        await retriever.retrieve("alpha", filters=["not", "a", "mapping"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sqlite_fts5_store_persists_and_filters_documents(tmp_path: Path) -> None:
    database = tmp_path / "knowledge" / "gabby.db"
    store = SQLiteFTS5Store(database)
    await store.ingest(
        [
            Document(
                "Authentication fails after signing key rotation.",
                source="runbook/auth.md",
                metadata={"team": "identity", "revision": 3},
                id="auth-runbook",
            ),
            Document(
                "The build uses a package cache.",
                source="runbook/build.md",
                metadata={"team": "platform"},
            ),
        ]
    )

    reopened = SQLiteFTS5Store(database)
    results = await reopened.retrieve("AUTH key rotation")
    filtered = await reopened.retrieve(
        "authentication rotation", filters={"team": "identity", "revision": 3}
    )

    assert results[0].id == "auth-runbook"
    assert results[0].source == "runbook/auth.md"
    assert filtered == results[:1]
    assert await reopened.retrieve('" OR * : (', limit=5) == []


@pytest.mark.asyncio
async def test_sqlite_fts5_store_migrates_v1_documents_without_changing_ids(
    tmp_path: Path,
) -> None:
    import sqlite3

    database = tmp_path / "legacy.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.executescript(
            """CREATE TABLE gabby_documents (
                   id INTEGER PRIMARY KEY,
                   external_id TEXT NOT NULL UNIQUE,
                   text TEXT NOT NULL,
                   source TEXT NOT NULL,
                   metadata_json TEXT NOT NULL
               );
               CREATE TABLE gabby_document_metadata (
                   document_id INTEGER NOT NULL REFERENCES gabby_documents(id) ON DELETE CASCADE,
                   key TEXT NOT NULL,
                   value_json TEXT NOT NULL,
                   PRIMARY KEY(document_id, key)
               );
               CREATE VIRTUAL TABLE gabby_documents_fts USING fts5(
                   text, content='gabby_documents', content_rowid='id'
               );
               CREATE TRIGGER gabby_documents_ai AFTER INSERT ON gabby_documents BEGIN
                   INSERT INTO gabby_documents_fts(rowid, text) VALUES (new.id, new.text);
               END;
               INSERT INTO gabby_documents(external_id, text, source, metadata_json)
                   VALUES ('legacy-id', 'legacy authentication guide', 'guide.md', '{}');
               PRAGMA user_version = 1;"""
        )

    results = await SQLiteFTS5Store(database).retrieve("legacy authentication")

    assert [(document.id, document.text) for document in results] == [
        ("legacy-id", "legacy authentication guide")
    ]


@pytest.mark.asyncio
async def test_sqlite_store_stages_fenced_generations_and_discards_safely(
    tmp_path: Path,
) -> None:
    store = SQLiteFTS5Store(tmp_path / "generations.db")
    old = Document("current authentication policy", "policy.md", id="policy")
    staged = Document("replacement authentication policy", "policy.md", id="policy")
    assert await store.ingest([old]) == 1
    assert await store.stage_source("policy.md", "generation-2", 4, [staged]) == 1
    assert [
        doc.id
        for doc in await store.retrieve("replacement", generations={"policy.md": "generation-2"})
    ] == ["policy"]
    assert [
        doc.id for doc in await store.retrieve("current", generations={"policy.md": "generation-2"})
    ] == []
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source("policy.md", "generation-1", 3, [old])
    assert await store.discard_generation("policy.md", "generation-2", 5) == 1
    assert await store.discard_generation("policy.md", "generation-2", 5) == 0
    assert await store.retrieve("replacement", generations={"policy.md": "generation-2"}) == []


@pytest.mark.asyncio
async def test_sqlite_stage_and_discard_validate_fence_and_source_inputs(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "invalid-generations.db")
    document = Document("valid document", "source")
    with pytest.raises(TypeError, match="fencing_token"):
        await store.stage_source("source", "generation", True, [document])
    with pytest.raises(ValueError, match="fencing_token"):
        await store.stage_source("source", "generation", -1, [document])
    with pytest.raises(ValueError, match="requested source"):
        await store.stage_source("source", "generation", 1, [Document("wrong", "other")])
    with pytest.raises(ValueError, match="source"):
        await store.discard_generation(" ", "generation", 1)
    with pytest.raises(ValueError, match="generation"):
        await store.discard_generation("source", " ", 1)
    with pytest.raises(ValueError, match="source"):
        await store.replace_source(" ", [])
    with pytest.raises(ValueError, match="generation"):
        await store.stage_source("source", " ", 1, [document])
    with pytest.raises(ValueError, match="generation"):
        await store.delete_generation("source", " ")
    with pytest.raises(ValueError, match="source"):
        await store.delete_source(" ")
    with pytest.raises(TypeError, match="Document instances"):
        await store.ingest([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-empty string"):
        await store.ingest([Document("", "source")])
    with pytest.raises(TypeError, match="source must be a string"):
        await store.ingest([Document("text", 1)])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_sqlite_store_migrates_v2_generation_schema(tmp_path: Path) -> None:
    import sqlite3

    database = tmp_path / "v2.db"
    store = SQLiteFTS5Store(database)
    await store.ingest([Document("existing v2 document", "guide.md", id="v2")])
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DROP TABLE gabby_source_fences")
        connection.execute("PRAGMA user_version = 2")

    assert [doc.id for doc in await store.retrieve("existing v2")] == ["v2"]
    assert await store.stage_source("guide.md", "new-generation", 1, []) == 0


@pytest.mark.asyncio
async def test_sqlite_generation_filter_validates_mapping_and_empty_values(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "filter.db")
    await store.ingest([Document("filtered generation content", "guide.md")])
    with pytest.raises(TypeError, match="generations"):
        await store.retrieve("content", generations=["invalid"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source strings"):
        await store.retrieve("content", generations={1: "generation"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="non-empty generation"):
        await store.retrieve("content", generations={"guide.md": ""})
    assert await store.retrieve("content", generations={}) == []
    assert await store.retrieve("!!!") == []


@pytest.mark.asyncio
async def test_sqlite_fts5_store_upserts_and_replaces_source_atomically(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    await store.ingest(
        [
            Document("old authentication instructions", source="auth.md", id="auth"),
            Document("keep this unrelated guide", source="other.md", id="other"),
        ]
    )
    await store.ingest(
        [
            Document(
                "new authentication procedure",
                source="auth.md",
                metadata={"status": "current"},
                id="auth",
            )
        ]
    )
    assert await store.retrieve("old instructions") == []
    assert (await store.retrieve("new procedure"))[0].metadata == {"status": "current"}

    replaced = await store.replace_source(
        "auth.md", [Document("rotated credential recovery", source="auth.md", id="recovery")]
    )
    assert replaced == 1
    assert await store.retrieve("new procedure") == []
    assert [document.id for document in await store.retrieve("credential recovery")] == ["recovery"]
    assert await store.delete_source("auth.md") == 1
    assert await store.delete_source("auth.md") == 0
    assert [document.id for document in await store.retrieve("unrelated guide")] == ["other"]


@pytest.mark.asyncio
async def test_sqlite_source_replacement_rolls_back_on_duplicate_ids(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    await store.ingest([Document("original policy", source="policy.md", id="original")])

    with pytest.raises(KnowledgeStoreError, match="SQLite FTS5"):
        await store.replace_source(
            "policy.md",
            [
                Document("replacement one", source="policy.md", id="duplicate"),
                Document("replacement two", source="policy.md", id="duplicate"),
            ],
        )

    assert [document.id for document in await store.retrieve("original policy")] == ["original"]
    assert await store.retrieve("replacement") == []


@pytest.mark.asyncio
async def test_sqlite_fts5_store_uses_stable_generated_ids_and_validates_inputs(
    tmp_path: Path,
) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    document = Document("stable content", source="guide.md", metadata={"labels": ["a", "b"]})
    assert await store.ingest([document]) == 1
    assert await store.ingest([document]) == 1
    results = await store.retrieve("stable")
    assert len(results) == 1
    assert results[0].id is not None

    assert await store.ingest([]) == 0
    assert await store.replace_source("guide.md", []) == 0
    assert await store.retrieve("stable") == []
    with pytest.raises(ValueError, match="finite JSON"):
        await store.ingest([Document("bad metadata", metadata={"value": object()})])
    with pytest.raises(ValueError, match="requested source"):
        await store.replace_source("one.md", [Document("wrong source", source="two.md")])
    with pytest.raises(TypeError, match="keys must be strings"):
        await store.ingest([Document("bad key", metadata={1: "value"})])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="limit"):
        await store.retrieve("stable", limit=-1)
    with pytest.raises(TypeError, match="filters"):
        await store.retrieve("stable", filters=["invalid"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="persistent database"):
        SQLiteFTS5Store(":memory:")
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        SQLiteFTS5Store(tmp_path / "invalid.db", busy_timeout_seconds=0)
    with pytest.raises(TypeError, match="query must be a string"):
        await store.retrieve(5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="filter keys must be strings"):
        await store.retrieve("stable", filters={1: "invalid"})  # type: ignore[dict-item]


@pytest.mark.asyncio
async def test_sqlite_fts5_store_reports_newer_schema_and_backend_errors(tmp_path: Path) -> None:
    import sqlite3

    database = tmp_path / "future.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA user_version = 4")
    store = SQLiteFTS5Store(database)
    with pytest.raises(KnowledgeStoreError, match="newer than supported"):
        await store.retrieve("query")
