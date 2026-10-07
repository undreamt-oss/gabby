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
"""Replaceable retrieval interfaces; storage remains outside run state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from ._sync import run_sync_callback

MAX_RETRIEVAL_DOCUMENTS = 100


@dataclass
class Document:
    """Text returned by a retriever, with source attribution and filterable metadata."""

    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    generation: str | None = None


class Retriever(Protocol):
    """Asynchronous interface for returning relevant documents for one run."""

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return at most ``limit`` relevant documents matching optional metadata filters."""
        ...


class GenerationAwareRetriever(Retriever, Protocol):
    """Retriever that can limit results to one active generation per source."""

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return documents whose source generation matches the supplied active map."""
        ...

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
    ) -> int:
        """Atomically stage a source snapshot and reject stale fencing tokens."""
        ...

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Advance the source fence and atomically discard an incomplete generation."""
        ...

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Persist a newer source fence without changing any staged generation data."""
        ...

    async def delete_generation(self, source: str, generation: str) -> int:
        """Delete one staged or retired generation; repeated deletion must be safe."""
        ...


class ActiveGenerationReader(Protocol):
    """Read the committed generation selected for each indexed source."""

    async def active_generations(self) -> Mapping[str, str]:
        """Return the durable source-to-generation map used by retrieval."""
        ...


class SourceKnowledgeWriter(Protocol):
    """Replace or remove all indexed documents for one named source."""

    async def replace_source(self, source: str, documents: Sequence[Document]) -> int:
        """Atomically replace a source snapshot and return its document count."""
        ...

    async def delete_source(self, source: str) -> int:
        """Delete one source and return the number of documents removed."""
        ...


class KnowledgeStore(Retriever, Protocol):
    """Persistent or external knowledge store with document ingestion operations."""

    async def ingest(self, documents: Sequence[Document]) -> int:
        """Insert or update documents, returning the number processed."""
        ...

    async def replace_source(self, source: str, documents: Sequence[Document]) -> int:
        """Atomically replace all documents attributed to ``source``."""
        ...

    async def delete_source(self, source: str) -> int:
        """Delete all documents attributed to ``source`` and return the count removed."""
        ...


class EmbeddingProvider(Protocol):
    """Replaceable semantic embedding interface, independent of storage and model choice."""

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one embedding vector per input text."""
        ...


class AsymmetricEmbeddingProvider(EmbeddingProvider, Protocol):
    """Embedding provider with role-specific query and document behavior."""

    async def embed_queries(
        self, texts: Sequence[str], *, prefix: str | None = None
    ) -> Sequence[Sequence[float]]:
        """Embed retrieval queries, optionally overriding the configured query prefix."""
        ...

    async def embed_documents(
        self, texts: Sequence[str], *, prefix: str | None = None
    ) -> Sequence[Sequence[float]]:
        """Embed indexed documents, optionally overriding the configured document prefix."""
        ...


class VectorStore(Protocol):
    """Replaceable document-vector index with ordered similarity search."""

    async def upsert(
        self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert or update documents and their vectors, returning the count processed."""
        ...

    async def replace_source(
        self,
        source: str,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Atomically replace one source's documents and vectors."""
        ...

    async def delete_source(self, source: str) -> int:
        """Delete one source and return the number of vector records removed."""
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return nearest documents first, with stable IDs and matching metadata."""
        ...


class GenerationAwareVectorStore(VectorStore, Protocol):
    """Vector store that stages and filters fenced source generations."""

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Stage one generation and reject tokens older than the source's latest fence."""
        ...

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Advance the source fence and atomically discard an incomplete generation."""
        ...

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Persist a newer source fence without changing any staged generation data."""
        ...

    async def delete_generation(self, source: str, generation: str) -> int:
        """Delete one staged or retired generation; repeated deletion must be safe."""
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return nearest documents first, limited to the active generation map."""
        ...


class Reranker(Protocol):
    """Replaceable second-stage ranking interface for retrieved documents."""

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return up to ``limit`` documents ordered by relevance."""
        ...


class KnowledgeStoreError(RuntimeError):
    """A knowledge backend could not complete a persistence or retrieval operation."""


_SQLiteResult = TypeVar("_SQLiteResult")


class SQLiteFTS5Store:
    """Persistent lexical knowledge store backed by SQLite FTS5.

    Every database operation runs in a worker thread and opens a short-lived connection,
    so retrieval and ingestion do not block Gabby's async runtime or keep a loop-bound
    connection alive. The database file can be shared across store instances and processes.
    """

    _token_pattern = re.compile(r"\w+", re.UNICODE)
    _schema_version = 3

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 10.0) -> None:
        if isinstance(path, str) and path == ":memory:":
            raise ValueError("SQLiteFTS5Store requires a persistent database file, not :memory:")
        if (
            isinstance(busy_timeout_seconds, bool)
            or not isinstance(busy_timeout_seconds, (int, float))
            or not math.isfinite(busy_timeout_seconds)
            or busy_timeout_seconds <= 0
        ):
            raise ValueError("busy_timeout_seconds must be a finite positive number")
        self.path = Path(path).expanduser()
        self.busy_timeout_seconds = float(busy_timeout_seconds)
        self._schema_lock = threading.Lock()

    async def ingest(self, documents: Sequence[Document]) -> int:
        """Insert or update documents in one transaction, preserving explicit IDs."""
        prepared = [self._prepare(document) for document in documents]
        if not prepared:
            return 0

        def write(connection: sqlite3.Connection) -> int:
            with connection:
                for external_id, document, metadata_json, metadata_values in prepared:
                    connection.execute(
                        """INSERT INTO gabby_documents
                           (external_id, logical_id, generation, text, source, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(external_id) DO UPDATE SET
                             logical_id=excluded.logical_id,
                             generation=excluded.generation,
                             text=excluded.text,
                             source=excluded.source,
                             metadata_json=excluded.metadata_json""",
                        (
                            self._stored_id(external_id, document.generation),
                            external_id,
                            document.generation,
                            document.text,
                            document.source,
                            metadata_json,
                        ),
                    )
                    row = connection.execute(
                        "SELECT id FROM gabby_documents WHERE external_id = ?",
                        (self._stored_id(external_id, document.generation),),
                    ).fetchone()
                    if row is None:
                        raise KnowledgeStoreError("SQLite did not return the ingested document")
                    row_id = int(row[0])
                    connection.execute(
                        "DELETE FROM gabby_document_metadata WHERE document_id = ?", (row_id,)
                    )
                    connection.executemany(
                        """INSERT INTO gabby_document_metadata (document_id, key, value_json)
                           VALUES (?, ?, ?)""",
                        [(row_id, key, value) for key, value in metadata_values],
                    )
            return len(prepared)

        return await self._run(write)

    async def replace_source(self, source: str, documents: Sequence[Document]) -> int:
        """Atomically replace one source's indexed documents with a new snapshot."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        prepared = [self._prepare(document) for document in documents]
        if any(document.source != source for _, document, _, _ in prepared):
            raise ValueError("every replacement document must have the requested source")

        def replace(connection: sqlite3.Connection) -> int:
            with connection:
                connection.execute("DELETE FROM gabby_documents WHERE source = ?", (source,))
                for external_id, document, metadata_json, metadata_values in prepared:
                    connection.execute(
                        """INSERT INTO gabby_documents
                           (external_id, logical_id, generation, text, source, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self._stored_id(external_id, document.generation),
                            external_id,
                            document.generation,
                            document.text,
                            document.source,
                            metadata_json,
                        ),
                    )
                    row = connection.execute(
                        "SELECT id FROM gabby_documents WHERE external_id = ?",
                        (self._stored_id(external_id, document.generation),),
                    ).fetchone()
                    if row is None:
                        raise KnowledgeStoreError("SQLite did not return the ingested document")
                    connection.executemany(
                        """INSERT INTO gabby_document_metadata (document_id, key, value_json)
                           VALUES (?, ?, ?)""",
                        [(int(row[0]), key, value) for key, value in metadata_values],
                    )
            return len(prepared)

        return await self._run(replace)

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
    ) -> int:
        """Atomically stage one source generation while retaining its other generations."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if fencing_token < 0:
            raise ValueError("fencing_token must be a non-negative integer")
        prepared = [self._prepare(document) for document in documents]
        if any(document.source != source for _, document, _, _ in prepared):
            raise ValueError("every staged document must have the requested source")
        for _, document, _, _ in prepared:
            document.generation = generation

        def stage(connection: sqlite3.Connection) -> int:
            with connection:
                self._advance_source_fence(connection, source, fencing_token)
                connection.execute(
                    "DELETE FROM gabby_documents WHERE source = ? AND generation = ?",
                    (source, generation),
                )
                for logical_id, document, metadata_json, metadata_values in prepared:
                    storage_id = self._stored_id(logical_id, generation)
                    connection.execute(
                        """INSERT INTO gabby_documents
                           (external_id, logical_id, generation, text, source, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            storage_id,
                            logical_id,
                            generation,
                            document.text,
                            document.source,
                            metadata_json,
                        ),
                    )
                    row = connection.execute(
                        "SELECT id FROM gabby_documents WHERE external_id = ?", (storage_id,)
                    ).fetchone()
                    if row is None:
                        raise KnowledgeStoreError("SQLite did not return the staged document")
                    connection.executemany(
                        """INSERT INTO gabby_document_metadata (document_id, key, value_json)
                           VALUES (?, ?, ?)""",
                        [(int(row[0]), key, value) for key, value in metadata_values],
                    )
            return len(prepared)

        return await self._run(stage)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Fence older writers and remove one incomplete generation atomically."""
        self._validate_fencing_token(fencing_token)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

        def discard(connection: sqlite3.Connection) -> int:
            with connection:
                self._advance_source_fence(connection, source, fencing_token)
                cursor = connection.execute(
                    "DELETE FROM gabby_documents WHERE source = ? AND generation = ?",
                    (source, generation),
                )
            return cursor.rowcount

        return await self._run(discard)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Fence stale writers while preserving every staged generation."""
        self._validate_fencing_token(fencing_token)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        def advance(connection: sqlite3.Connection) -> None:
            with connection:
                self._advance_source_fence(connection, source, fencing_token)

        await self._run(advance)

    async def delete_generation(self, source: str, generation: str) -> int:
        """Delete one staged or retired generation idempotently."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

        def delete(connection: sqlite3.Connection) -> int:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM gabby_documents WHERE source = ? AND generation = ?",
                    (source, generation),
                )
            return cursor.rowcount

        return await self._run(delete)

    async def delete_source(self, source: str) -> int:
        """Delete a source snapshot and its full-text and metadata index entries."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        def delete(connection: sqlite3.Connection) -> int:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM gabby_documents WHERE source = ?", (source,)
                )
            return cursor.rowcount

        return await self._run(delete)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return FTS5 BM25 matches with exact JSON-valued metadata filters."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if limit == 0:
            return []
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        if generations is not None and not isinstance(generations, Mapping):
            raise TypeError("generations must map sources to active generation IDs")
        if generations is not None and any(
            not isinstance(source, str) or not isinstance(generation, str) or not generation
            for source, generation in generations.items()
        ):
            raise TypeError("generations must map source strings to non-empty generation IDs")
        if generations is not None and not generations:
            return []
        tokens = self._token_pattern.findall(query.casefold())
        if not tokens:
            return []
        # Quote each lexical token so user punctuation cannot inject FTS5 operators.
        match_query = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        clauses = ["gabby_documents_fts MATCH ?"]
        parameters: list[Any] = [match_query]
        if filters:
            for key, value in filters.items():
                clauses.append(
                    """EXISTS (SELECT 1 FROM gabby_document_metadata AS m
                       WHERE m.document_id = d.id AND m.key = ? AND m.value_json = ?)"""
                )
                parameters.extend((key, self._canonical_json(value)))
        if generations is not None:
            clauses.append(
                "(" + " OR ".join("(d.source = ? AND d.generation = ?)" for _ in generations) + ")"
            )
            for source, generation in generations.items():
                parameters.extend((source, generation))
        parameters.append(limit)
        sql = f"""SELECT COALESCE(d.logical_id, d.external_id), d.text, d.source,
                         d.metadata_json, d.generation
                  FROM gabby_documents_fts
                  JOIN gabby_documents AS d ON d.id = gabby_documents_fts.rowid
                  WHERE {" AND ".join(clauses)}
                  ORDER BY bm25(gabby_documents_fts), d.external_id
                  LIMIT ?"""

        def search(connection: sqlite3.Connection) -> list[Document]:
            rows = connection.execute(sql, parameters).fetchall()
            return [
                Document(
                    text=str(row[1]),
                    source=str(row[2]),
                    metadata=json.loads(row[3]),
                    id=str(row[0]),
                    generation=str(row[4]) if row[4] is not None else None,
                )
                for row in rows
            ]

        return await self._run(search)

    @classmethod
    def _prepare(cls, document: Document) -> tuple[str, Document, str, list[tuple[str, str]]]:
        if not isinstance(document, Document):
            raise TypeError("documents must contain Document instances")
        if not isinstance(document.text, str) or not document.text:
            raise ValueError("document text must be a non-empty string")
        if not isinstance(document.source, str):
            raise TypeError("Document source must be a string")
        if document.generation is not None and (
            not isinstance(document.generation, str) or not document.generation.strip()
        ):
            raise ValueError("Document generation must be a non-empty string when provided")
        if not isinstance(document.metadata, dict):
            raise TypeError("Document metadata must be a mapping")
        if any(not isinstance(key, str) for key in document.metadata):
            raise TypeError("Document metadata keys must be strings")
        metadata_json = cls._canonical_json(document.metadata)
        metadata_values = [
            (key, cls._canonical_json(value)) for key, value in document.metadata.items()
        ]
        if document.id is not None:
            if not isinstance(document.id, str) or not document.id.strip():
                raise ValueError("Document id must be a non-empty string when provided")
            external_id = document.id
        else:
            identity = "\0".join((document.source, document.text, metadata_json))
            external_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return external_id, deepcopy(document), metadata_json, metadata_values

    @staticmethod
    def _stored_id(logical_id: str, generation: str | None) -> str:
        if generation is None:
            return logical_id
        return hashlib.sha256(f"{logical_id}\0{generation}".encode()).hexdigest()

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("knowledge metadata must contain finite JSON values") from exc

    async def _run(self, operation: Callable[[sqlite3.Connection], _SQLiteResult]) -> _SQLiteResult:
        """Run one bounded SQLite unit of work away from the event loop."""

        def execute() -> _SQLiteResult:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path, timeout=self.busy_timeout_seconds, isolation_level="DEFERRED"
            )
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                self._ensure_schema(connection)
                return operation(connection)
            finally:
                connection.close()

        try:
            worker = asyncio.create_task(run_sync_callback(execute))
            # Periodic loop wakeups also make completion robust in event loops that do not
            # observe worker-thread wakeups promptly (for example, embedded test/IDE loops).
            while not worker.done():
                await asyncio.wait({worker}, timeout=0.05)
            return cast(_SQLiteResult, await worker)
        except asyncio.CancelledError:
            worker.add_done_callback(self._consume_worker_result)
            raise
        except sqlite3.Error as exc:
            raise KnowledgeStoreError("SQLite FTS5 operation failed") from exc

    @staticmethod
    def _consume_worker_result(worker: asyncio.Task[Any]) -> None:
        """Retrieve late worker failures after the awaiting run has been cancelled."""
        if not worker.cancelled():
            worker.exception()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Create or validate the versioned schema under a process-local lock."""
        with self._schema_lock:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._schema_version:
                raise KnowledgeStoreError(
                    f"Knowledge database schema {version} is newer than supported "
                    f"schema {self._schema_version}"
                )
            if version == self._schema_version:
                return
            connection.execute("PRAGMA journal_mode = WAL")
            if version == 1:
                with connection:
                    connection.execute("ALTER TABLE gabby_documents ADD COLUMN logical_id TEXT")
                    connection.execute("ALTER TABLE gabby_documents ADD COLUMN generation TEXT")
                    connection.execute(
                        "UPDATE gabby_documents SET logical_id = external_id "
                        "WHERE logical_id IS NULL"
                    )
                    connection.execute(
                        "CREATE INDEX IF NOT EXISTS gabby_documents_generation_idx "
                        "ON gabby_documents(source, generation)"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS gabby_source_fences "
                        "(source TEXT PRIMARY KEY, fencing_token INTEGER NOT NULL)"
                    )
                    connection.execute(f"PRAGMA user_version = {self._schema_version}")
                return
            if version == 2:
                with connection:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS gabby_source_fences "
                        "(source TEXT PRIMARY KEY, fencing_token INTEGER NOT NULL)"
                    )
                    connection.execute(f"PRAGMA user_version = {self._schema_version}")
                return
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS gabby_documents (
                       id INTEGER PRIMARY KEY,
                       external_id TEXT NOT NULL UNIQUE,
                       logical_id TEXT NOT NULL,
                       generation TEXT,
                       text TEXT NOT NULL,
                       source TEXT NOT NULL,
                       metadata_json TEXT NOT NULL
                   );
                   CREATE TABLE IF NOT EXISTS gabby_document_metadata (
                       document_id INTEGER NOT NULL
                           REFERENCES gabby_documents(id) ON DELETE CASCADE,
                       key TEXT NOT NULL,
                       value_json TEXT NOT NULL,
                       PRIMARY KEY (document_id, key)
                   );
                   CREATE VIRTUAL TABLE IF NOT EXISTS gabby_documents_fts USING fts5(
                       text,
                       content='gabby_documents',
                       content_rowid='id',
                       tokenize='unicode61 remove_diacritics 2'
                   );
                   CREATE TRIGGER IF NOT EXISTS gabby_documents_ai AFTER INSERT ON gabby_documents
                   BEGIN
                       INSERT INTO gabby_documents_fts(rowid, text) VALUES (new.id, new.text);
                   END;
                   CREATE TRIGGER IF NOT EXISTS gabby_documents_ad AFTER DELETE ON gabby_documents
                   BEGIN
                       INSERT INTO gabby_documents_fts(gabby_documents_fts, rowid, text)
                       VALUES ('delete', old.id, old.text);
                   END;
                   CREATE TRIGGER IF NOT EXISTS gabby_documents_au AFTER UPDATE OF text
                       ON gabby_documents
                   BEGIN
                       INSERT INTO gabby_documents_fts(gabby_documents_fts, rowid, text)
                       VALUES ('delete', old.id, old.text);
                       INSERT INTO gabby_documents_fts(rowid, text) VALUES (new.id, new.text);
                   END;
                   CREATE INDEX IF NOT EXISTS gabby_documents_source_idx
                       ON gabby_documents(source);
                   CREATE INDEX IF NOT EXISTS gabby_documents_generation_idx
                       ON gabby_documents(source, generation);
                   CREATE TABLE IF NOT EXISTS gabby_source_fences (
                       source TEXT PRIMARY KEY,
                       fencing_token INTEGER NOT NULL
                   );
                   PRAGMA user_version = 3;"""
            )

    @staticmethod
    def _validate_fencing_token(fencing_token: int) -> None:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if fencing_token < 0:
            raise ValueError("fencing_token must be a non-negative integer")

    @classmethod
    def _advance_source_fence(
        cls, connection: sqlite3.Connection, source: str, fencing_token: int
    ) -> None:
        cls._validate_fencing_token(fencing_token)
        row = connection.execute(
            "SELECT fencing_token FROM gabby_source_fences WHERE source = ?", (source,)
        ).fetchone()
        if row is not None and fencing_token < int(row[0]):
            raise KnowledgeStoreError("Stale fencing token rejected by lexical store")
        connection.execute(
            "INSERT INTO gabby_source_fences(source, fencing_token) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET fencing_token = excluded.fencing_token",
            (source, fencing_token),
        )


class InMemoryBM25Retriever:
    """Small-corpus lexical retriever using BM25 ranking.

    This reference implementation is useful for local agents and fixtures. It keeps
    documents in process and scans the indexed terms per query; persistent and
    large-scale stores should implement :class:`Retriever` instead.
    """

    _token_pattern = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1):
            raise ValueError("k1 must be a finite positive number")
        if k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b):
            raise ValueError("b must be a finite number between 0 and 1")
        if not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        if any(not isinstance(document, Document) for document in documents):
            raise TypeError("documents must contain Document instances")
        if any(not isinstance(document.metadata, dict) for document in documents):
            raise TypeError("Document metadata must be a mapping")
        if any(not isinstance(document.source, str) for document in documents):
            raise TypeError("Document source must be a string")

        self._documents = tuple(deepcopy(document) for document in documents)
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = tuple(self._tokenize(document.text) for document in self._documents)
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        )

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("Document text must be a string")
        return cls._token_pattern.findall(text.casefold())

    async def retrieve(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """Return the highest scoring lexical matches with exact metadata filters."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if limit == 0 or not self._documents:
            return []
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        query_terms = Counter(self._tokenize(query))
        if not query_terms:
            return []

        total_documents = len(self._documents)
        matches: list[tuple[float, int, Document]] = []
        for index, document in enumerate(self._documents):
            if filters and any(
                document.metadata.get(key) != value for key, value in filters.items()
            ):
                continue
            frequencies = self._term_frequencies[index]
            document_length = len(self._tokens[index])
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_frequency = math.log(
                    1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / (self._average_length or 1.0)
                )
                score += (
                    inverse_frequency * frequency * (self.k1 + 1) / normalization * query_frequency
                )
            if score > 0:
                matches.append((score, index, document))

        matches.sort(key=lambda match: (-match[0], match[1]))
        return [deepcopy(document) for _, _, document in matches[:limit]]
