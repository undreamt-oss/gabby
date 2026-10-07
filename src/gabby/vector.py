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
"""Persistent, model-independent vector storage backed by exact SQLite cosine search."""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import math
import sqlite3
import struct
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar, cast

from ._sync import run_sync_callback
from .knowledge import Document, GenerationAwareVectorStore, KnowledgeStoreError

_Result = TypeVar("_Result")
_MAX_VECTOR_DIMENSIONS = 65_536


class SQLiteVectorStore(GenerationAwareVectorStore):
    """Persist document vectors and rank them by exact cosine similarity.

    Search scans vectors in SQLite and keeps only the best ``limit`` results in memory. This
    implementation favors a dependency-free local backend and small or moderate corpora; use a
    specialized vector database through :class:`VectorStore` for approximate or distributed
    search. The first non-empty write fixes the database's embedding dimension. Recreate the index
    to change embedding models, even if the replacement model happens to use the same dimension.
    """

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 10.0) -> None:
        if isinstance(path, str) and path == ":memory:":
            raise ValueError("SQLiteVectorStore requires a persistent database file, not :memory:")
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

    async def upsert(
        self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert or update documents and their normalized vectors in one transaction."""
        prepared = self._prepare_batch(documents, embeddings)
        if not prepared:
            return 0

        def write(connection: sqlite3.Connection) -> int:
            with connection:
                self._check_dimension(connection, len(prepared[0][2]))
                for external_id, document, vector, metadata_json in prepared:
                    connection.execute(
                        """INSERT INTO gabby_vectors
                           (external_id, logical_id, generation, text, source, metadata_json,
                            embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(external_id) DO UPDATE SET
                             logical_id=excluded.logical_id, generation=excluded.generation,
                             text=excluded.text, source=excluded.source,
                             metadata_json=excluded.metadata_json, embedding=excluded.embedding""",
                        self._row_values(external_id, document, vector, metadata_json),
                    )
            return len(prepared)

        return await self._run(write)

    async def replace_source(
        self,
        source: str,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Atomically replace every vector document attributed to one source."""
        self._validate_source(source)
        prepared = self._prepare_batch(documents, embeddings)
        if any(document.source != source for _, document, _, _ in prepared):
            raise ValueError("every replacement document must have the requested source")

        def replace_rows(connection: sqlite3.Connection) -> int:
            with connection:
                if prepared:
                    self._check_dimension(connection, len(prepared[0][2]))
                connection.execute("DELETE FROM gabby_vectors WHERE source = ?", (source,))
                for external_id, document, vector, metadata_json in prepared:
                    connection.execute(
                        """INSERT INTO gabby_vectors
                           (external_id, logical_id, generation, text, source, metadata_json,
                            embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        self._row_values(external_id, document, vector, metadata_json),
                    )
            return len(prepared)

        return await self._run(replace_rows)

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Stage one complete source generation and reject stale writers transactionally."""
        self._validate_source(source)
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        self._validate_fencing_token(fencing_token)
        if any(
            not isinstance(document, Document) or document.source != source
            for document in documents
        ):
            raise ValueError("every staged document must be a Document for the requested source")
        staged_documents = [replace(document, generation=generation) for document in documents]
        prepared = self._prepare_batch(staged_documents, embeddings)

        def stage(connection: sqlite3.Connection) -> int:
            with connection:
                if prepared:
                    self._check_dimension(connection, len(prepared[0][2]))
                self._advance_fence(connection, source, fencing_token)
                connection.execute(
                    "DELETE FROM gabby_vectors WHERE source = ? AND generation = ?",
                    (source, generation),
                )
                for external_id, document, vector, metadata_json in prepared:
                    connection.execute(
                        """INSERT INTO gabby_vectors
                           (external_id, logical_id, generation, text, source, metadata_json,
                            embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        self._row_values(external_id, document, vector, metadata_json),
                    )
            return len(prepared)

        return await self._run(stage)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Advance a writer fence and atomically delete one incomplete generation."""
        self._validate_source(source)
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        self._validate_fencing_token(fencing_token)

        def discard(connection: sqlite3.Connection) -> int:
            with connection:
                self._advance_fence(connection, source, fencing_token)
                cursor = connection.execute(
                    "DELETE FROM gabby_vectors WHERE source = ? AND generation = ?",
                    (source, generation),
                )
            return cursor.rowcount

        return await self._run(discard)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Persist a newer fencing token without changing staged vector data."""
        self._validate_source(source)
        self._validate_fencing_token(fencing_token)

        def advance(connection: sqlite3.Connection) -> None:
            with connection:
                self._advance_fence(connection, source, fencing_token)

        await self._run(advance)

    async def delete_generation(self, source: str, generation: str) -> int:
        """Delete staged or retired generation rows idempotently."""
        self._validate_source(source)
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

        def delete(connection: sqlite3.Connection) -> int:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM gabby_vectors WHERE source = ? AND generation = ?",
                    (source, generation),
                )
            return cursor.rowcount

        return await self._run(delete)

    async def delete_source(self, source: str) -> int:
        """Delete all stored vector generations for one source."""
        self._validate_source(source)

        def delete(connection: sqlite3.Connection) -> int:
            with connection:
                cursor = connection.execute("DELETE FROM gabby_vectors WHERE source = ?", (source,))
            return cursor.rowcount

        return await self._run(delete)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return nearest documents first, optionally restricted to an active generation map."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        normalized_filters = self._validate_filters(filters)
        if generations is not None:
            if not isinstance(generations, Mapping):
                raise TypeError("generations must map source strings to generation IDs")
            if any(
                not isinstance(source, str) or not isinstance(generation, str) or not generation
                for source, generation in generations.items()
            ):
                raise TypeError("generations must map source strings to non-empty generation IDs")
            if not generations:
                return []
        if limit == 0:
            return []
        query_vector = self._normalize_vector(embedding)

        def search_rows(connection: sqlite3.Connection) -> list[Document]:
            stored_config = connection.execute(
                "SELECT dimension FROM gabby_vector_config WHERE singleton = 1"
            ).fetchone()
            if stored_config is not None and int(stored_config[0]) != len(query_vector):
                raise ValueError("embedding dimension does not match this vector index")
            clauses: list[str] = []
            parameters: list[Any] = []
            if generations is not None:
                clauses.append(
                    "(" + " OR ".join("(source = ? AND generation = ?)" for _ in generations) + ")"
                )
                for source, generation in generations.items():
                    parameters.extend((source, generation))
            sql = (
                "SELECT logical_id, generation, text, source, metadata_json, embedding "
                "FROM gabby_vectors"
            )
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY external_id"
            cursor = connection.execute(sql, parameters)
            query_format = f"<{len(query_vector)}d"

            def candidates() -> Iterator[tuple[float, str, Document]]:
                for row in cursor:
                    try:
                        metadata = json.loads(str(row[4]))
                    except (TypeError, ValueError) as exc:
                        raise KnowledgeStoreError("SQLite vector metadata is invalid") from exc
                    if not isinstance(metadata, dict) or any(
                        not isinstance(key, str) for key in metadata
                    ):
                        raise KnowledgeStoreError("SQLite vector metadata is invalid")
                    if normalized_filters and any(
                        self._canonical_json(metadata.get(key)) != self._canonical_json(value)
                        for key, value in normalized_filters.items()
                    ):
                        continue
                    raw_vector = bytes(row[5])
                    if len(raw_vector) != len(query_vector) * 8:
                        raise KnowledgeStoreError("SQLite vector record has an invalid dimension")
                    stored_vector = struct.unpack(query_format, raw_vector)
                    similarity = max(
                        -1.0,
                        min(
                            1.0,
                            math.fsum(
                                left * right
                                for left, right in zip(query_vector, stored_vector, strict=True)
                            ),
                        ),
                    )
                    logical_id = str(row[0])
                    generation = str(row[1]) if row[1] is not None else None
                    document = Document(
                        text=str(row[2]),
                        source=str(row[3]),
                        metadata=metadata,
                        id=logical_id,
                        generation=generation,
                    )
                    yield (similarity, logical_id, document)

            try:
                ranked = heapq.nsmallest(limit, candidates(), key=lambda item: (-item[0], item[1]))
                return [deepcopy(item[2]) for item in ranked]
            finally:
                cursor.close()

        return await self._run(search_rows)

    @classmethod
    def _prepare_batch(
        cls,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> list[tuple[str, Document, tuple[float, ...], str]]:
        if not isinstance(documents, Sequence) or not isinstance(embeddings, Sequence):
            raise TypeError("documents and embeddings must be sequences")
        if len(documents) != len(embeddings):
            raise ValueError("embedding count must match document count")
        prepared = []
        dimensions: int | None = None
        for document, embedding in zip(documents, embeddings, strict=True):
            normalized = cls._normalize_vector(embedding)
            if dimensions is None:
                dimensions = len(normalized)
            elif len(normalized) != dimensions:
                raise ValueError("embedding vectors must have a consistent dimension")
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
            if not isinstance(document.metadata, dict) or any(
                not isinstance(key, str) for key in document.metadata
            ):
                raise TypeError("Document metadata must be a mapping with string keys")
            metadata_json = cls._canonical_json(document.metadata)
            if document.id is None:
                logical_id = hashlib.sha256(
                    "\0".join((document.source, document.text, metadata_json)).encode("utf-8")
                ).hexdigest()
            elif not isinstance(document.id, str) or not document.id.strip():
                raise ValueError("Document id must be a non-empty string when provided")
            else:
                logical_id = document.id
            prepared.append((logical_id, deepcopy(document), normalized, metadata_json))
        return prepared

    @classmethod
    def _normalize_vector(cls, embedding: Sequence[float]) -> tuple[float, ...]:
        if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
            raise ValueError("embeddings must be non-empty numeric sequences")
        if not embedding or len(embedding) > _MAX_VECTOR_DIMENSIONS:
            raise ValueError(f"embedding dimension must be from 1 through {_MAX_VECTOR_DIMENSIONS}")
        values = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("embedding values must be finite numbers")
            try:
                converted = float(value)
            except OverflowError as exc:
                raise ValueError("embedding values must be finite numbers") from exc
            if not math.isfinite(converted):
                raise ValueError("embedding values must be finite numbers")
            values.append(converted)
        norm = math.hypot(*values)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("embedding vectors must have a finite non-zero norm")
        return tuple(value / norm for value in values)

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

    @classmethod
    def _row_values(
        cls,
        logical_id: str,
        document: Document,
        vector: tuple[float, ...],
        metadata_json: str,
    ) -> tuple[str, str, str | None, str, str, str, bytes]:
        external_id = (
            logical_id
            if document.generation is None
            else hashlib.sha256(f"{logical_id}\0{document.generation}".encode()).hexdigest()
        )
        return (
            external_id,
            logical_id,
            document.generation,
            document.text,
            document.source,
            metadata_json,
            struct.pack(f"<{len(vector)}d", *vector),
        )

    @staticmethod
    def _validate_source(source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

    @classmethod
    def _validate_fencing_token(cls, token: int) -> None:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if token < 0:
            raise ValueError("fencing_token must be a non-negative integer")

    @classmethod
    def _advance_fence(cls, connection: sqlite3.Connection, source: str, token: int) -> None:
        cls._validate_fencing_token(token)
        row = connection.execute(
            "SELECT fencing_token FROM gabby_vector_fences WHERE source = ?", (source,)
        ).fetchone()
        if row is not None and token < int(row[0]):
            raise KnowledgeStoreError("Stale fencing token rejected by vector store")
        connection.execute(
            "INSERT INTO gabby_vector_fences(source, fencing_token) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET fencing_token=excluded.fencing_token",
            (source, token),
        )

    @staticmethod
    def _validate_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        return filters or {}

    def _check_dimension(self, connection: sqlite3.Connection, dimension: int) -> None:
        row = connection.execute(
            "SELECT dimension FROM gabby_vector_config WHERE singleton = 1"
        ).fetchone()
        if row is not None and int(row[0]) != dimension:
            raise ValueError("embedding dimension does not match this vector index")
        connection.execute(
            "INSERT INTO gabby_vector_config(singleton, dimension) VALUES (1, ?) "
            "ON CONFLICT(singleton) DO NOTHING",
            (dimension,),
        )

    async def _run(self, operation: Callable[[sqlite3.Connection], _Result]) -> _Result:
        """Run one SQLite transaction on the bounded sync callback pool."""

        def execute() -> _Result:
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
            return cast(_Result, await run_sync_callback(execute))
        except asyncio.CancelledError:
            raise
        except sqlite3.Error as exc:
            raise KnowledgeStoreError("SQLite vector operation failed") from exc

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        with self._schema_lock:
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS gabby_vector_schema (
                       singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                       version INTEGER NOT NULL
                   );
                   INSERT OR IGNORE INTO gabby_vector_schema(singleton, version) VALUES (1, 1);
                   CREATE TABLE IF NOT EXISTS gabby_vector_config (
                       singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                       dimension INTEGER NOT NULL CHECK (dimension > 0)
                   );
                   CREATE TABLE IF NOT EXISTS gabby_vectors (
                       id INTEGER PRIMARY KEY,
                       external_id TEXT NOT NULL UNIQUE,
                       logical_id TEXT NOT NULL,
                       generation TEXT,
                       text TEXT NOT NULL,
                       source TEXT NOT NULL,
                       metadata_json TEXT NOT NULL,
                       embedding BLOB NOT NULL
                   );
                   CREATE INDEX IF NOT EXISTS gabby_vectors_source_generation_idx
                       ON gabby_vectors(source, generation);
                   CREATE TABLE IF NOT EXISTS gabby_vector_fences (
                       source TEXT PRIMARY KEY,
                       fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0)
                   );"""
            )
            version = connection.execute(
                "SELECT version FROM gabby_vector_schema WHERE singleton = 1"
            ).fetchone()
            if version is None or int(version[0]) != 1:
                raise KnowledgeStoreError("Unsupported SQLite vector schema version")
            connection.execute("PRAGMA journal_mode = WAL")
