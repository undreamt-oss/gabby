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
"""Host-pooled PostgreSQL vector storage using pgvector cosine search."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, NoReturn

from .knowledge import Document, GenerationAwareVectorStore, KnowledgeStoreError
from .postgres_indexing import AsyncPostgresPool


class PostgresVectorStore(GenerationAwareVectorStore):
    """Shared approximate vector search backed by PostgreSQL and the pgvector extension.

    Apply ``sql/postgres_vector.sql`` using the host migration system. ``dimensions`` must match
    the embedding provider and is durably fixed for this store schema on first use. Gabby does not
    create schema at runtime or close the injected pool. The migration creates a cosine HNSW index;
    its pgvector index limit requires 1–2,000 dimensions.
    """

    _max_dimensions = 2_000

    def __init__(
        self,
        pool: AsyncPostgresPool,
        *,
        dimensions: int,
        ef_search: int = 40,
        max_scan_tuples: int = 20_000,
    ) -> None:
        if not callable(getattr(pool, "acquire", None)):
            raise TypeError("pool must provide async acquire() for a PostgreSQL connection")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")
        if not 1 <= dimensions <= self._max_dimensions:
            raise ValueError("dimensions must be from 1 through 2,000 for the HNSW index")
        if isinstance(ef_search, bool) or not isinstance(ef_search, int):
            raise TypeError("ef_search must be an integer")
        if not 1 <= ef_search <= 1_000:
            raise ValueError("ef_search must be from 1 through 1,000")
        if isinstance(max_scan_tuples, bool) or not isinstance(max_scan_tuples, int):
            raise TypeError("max_scan_tuples must be an integer")
        if not 1 <= max_scan_tuples <= 1_000_000:
            raise ValueError("max_scan_tuples must be from 1 through 1,000,000")
        self._pool = pool
        self.dimensions = dimensions
        self.ef_search = ef_search
        self.max_scan_tuples = max_scan_tuples

    async def upsert(
        self, documents: Sequence[Document], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Insert or update stable document IDs and vectors in one transaction."""
        prepared = self._prepare_batch(documents, embeddings)
        if not prepared:
            return 0
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._ensure_dimensions(connection)
                for source in sorted({document.source for _, _, document, _ in prepared}):
                    await self._lock_source(connection, source)
                for storage_id, document_id, document, vector in prepared:
                    await connection.execute(
                        """INSERT INTO gabby_vector_documents
                           (storage_id, document_id, text, source, metadata, generation, embedding)
                           VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::text::vector)
                           ON CONFLICT (storage_id) DO UPDATE SET
                             document_id = EXCLUDED.document_id,
                             text = EXCLUDED.text,
                             source = EXCLUDED.source,
                             metadata = EXCLUDED.metadata,
                             generation = EXCLUDED.generation,
                             embedding = EXCLUDED.embedding,
                             updated_at = clock_timestamp()""",
                        storage_id,
                        document_id,
                        document.text,
                        document.source,
                        self._canonical_json(document.metadata),
                        document.generation,
                        self._vector_literal(vector),
                    )
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def replace_source(
        self,
        source: str,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Atomically replace one source's complete vector snapshot."""
        self._validate_source(source)
        prepared = self._prepare_batch(documents, embeddings)
        if any(document.source != source for _, _, document, _ in prepared):
            raise ValueError("every replacement document must have the requested source")
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                if prepared:
                    await self._ensure_dimensions(connection)
                await self._lock_source(connection, source)
                await connection.execute(
                    "DELETE FROM gabby_vector_documents WHERE source = $1", source
                )
                for storage_id, document_id, document, vector in prepared:
                    await self._insert(connection, storage_id, document_id, document, vector)
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def delete_source(self, source: str) -> int:
        """Delete all stored generations for a source."""
        self._validate_source(source)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                status = await connection.execute(
                    "DELETE FROM gabby_vector_documents WHERE source = $1", source
                )
            return self._affected_rows(status)
        except Exception as exc:
            self._raise_store_error(exc)

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Stage a full source generation and reject stale writers transactionally."""
        self._validate_source_generation(source, generation)
        self._validate_fencing_token(fencing_token)
        staged_documents = [
            self._with_generation(document, source, generation) for document in documents
        ]
        prepared = self._prepare_batch(staged_documents, embeddings)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                if prepared:
                    await self._ensure_dimensions(connection)
                await self._lock_source(connection, source)
                await self._advance_source_fence(connection, source, fencing_token)
                await connection.execute(
                    """DELETE FROM gabby_vector_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
                for storage_id, document_id, document, vector in prepared:
                    await self._insert(connection, storage_id, document_id, document, vector)
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Advance the source fence and delete one incomplete generation atomically."""
        self._validate_source_generation(source, generation)
        self._validate_fencing_token(fencing_token)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                await self._advance_source_fence(connection, source, fencing_token)
                status = await connection.execute(
                    """DELETE FROM gabby_vector_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
            return self._affected_rows(status)
        except Exception as exc:
            self._raise_store_error(exc)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Persist a higher source fencing token without modifying vector data."""
        self._validate_source(source)
        self._validate_fencing_token(fencing_token)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                await self._advance_source_fence(connection, source, fencing_token)
        except Exception as exc:
            self._raise_store_error(exc)

    async def delete_generation(self, source: str, generation: str) -> int:
        """Idempotently delete one staged or retired generation."""
        self._validate_source_generation(source, generation)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                status = await connection.execute(
                    """DELETE FROM gabby_vector_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
            return self._affected_rows(status)
        except Exception as exc:
            self._raise_store_error(exc)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return nearest cosine matches with optional exact filters and generation bounds."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        self._validate_filters(filters)
        if generations is not None:
            if not isinstance(generations, Mapping):
                raise TypeError("generations must map source strings to generation IDs")
            if any(
                not isinstance(source, str)
                or not source
                or not isinstance(generation, str)
                or not generation
                for source, generation in generations.items()
            ):
                raise TypeError("generations must map source strings to non-empty generation IDs")
            if not generations:
                return []
        vector = self._validate_vector(embedding)
        if limit == 0:
            return []
        parameters: list[Any] = [self._vector_literal(vector)]
        where: list[str] = []
        for key, value in (filters or {}).items():
            parameters.extend((key, self._canonical_json(value)))
            where.append(f"metadata -> ${len(parameters) - 1}::text = ${len(parameters)}::jsonb")
        if generations is not None:
            generation_clauses: list[str] = []
            for source, generation in generations.items():
                parameters.extend((source, generation))
                generation_clauses.append(
                    f"(source = ${len(parameters) - 1}::text "
                    f"AND generation = ${len(parameters)}::text)"
                )
            where.append("(" + " OR ".join(generation_clauses) + ")")
        parameters.append(limit)
        where_sql = "" if not where else "WHERE " + " AND ".join(where)
        limit_parameter = len(parameters)
        query = f"""SELECT document_id, text, source, metadata::text AS metadata_json, generation
                    FROM gabby_vector_documents
                    {where_sql}
                    ORDER BY embedding <=> $1::text::vector, storage_id
                    LIMIT ${limit_parameter}"""
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._ensure_dimensions(connection)
                await self._set_search_settings(connection, limit=limit)
                rows = await connection.fetch(query, *parameters)
            return [
                Document(
                    text=str(row["text"]),
                    source=str(row["source"]),
                    metadata=json.loads(str(row["metadata_json"])),
                    id=str(row["document_id"]),
                    generation=str(row["generation"]) if row["generation"] is not None else None,
                )
                for row in rows
            ]
        except Exception as exc:
            self._raise_store_error(exc)

    async def _insert(
        self,
        connection: Any,
        storage_id: str,
        document_id: str,
        document: Document,
        vector: tuple[float, ...],
    ) -> None:
        await connection.execute(
            """INSERT INTO gabby_vector_documents
               (storage_id, document_id, text, source, metadata, generation, embedding)
               VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::text::vector)""",
            storage_id,
            document_id,
            document.text,
            document.source,
            self._canonical_json(document.metadata),
            document.generation,
            self._vector_literal(vector),
        )

    async def _ensure_dimensions(self, connection: Any) -> None:
        row = await connection.fetchrow(
            "SELECT dimensions FROM gabby_vector_store_config WHERE singleton = TRUE"
        )
        if row is None:
            await connection.execute(
                """INSERT INTO gabby_vector_store_config (singleton, dimensions)
                   VALUES (TRUE, $1) ON CONFLICT (singleton) DO NOTHING""",
                self.dimensions,
            )
            row = await connection.fetchrow(
                "SELECT dimensions FROM gabby_vector_store_config WHERE singleton = TRUE"
            )
        if row is None or int(row["dimensions"]) != self.dimensions:
            raise KnowledgeStoreError("PostgreSQL vector dimension does not match this index")

    async def _set_search_settings(self, connection: Any, *, limit: int) -> None:
        """Scope HNSW controls to one transaction, including filtered iterative scans."""
        await connection.execute(
            "SELECT set_config('hnsw.iterative_scan', $1, TRUE)", "strict_order"
        )
        await connection.execute(
            "SELECT set_config('hnsw.ef_search', $1, TRUE), "
            "set_config('hnsw.max_scan_tuples', $2, TRUE)",
            str(max(self.ef_search, limit)),
            str(self.max_scan_tuples),
        )

    def _prepare_batch(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> list[tuple[str, str, Document, tuple[float, ...]]]:
        if len(documents) != len(embeddings):
            raise ValueError("one embedding is required for each document")
        prepared = []
        for document, embedding in zip(documents, embeddings, strict=True):
            storage_id, snapshot, _ = self._prepare_document(document)
            vector = self._validate_vector(embedding)
            prepared.append((storage_id, str(snapshot.id), snapshot, vector))
        return prepared

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        if isinstance(vector, (str, bytes, bytearray)) or not isinstance(vector, Sequence):
            raise TypeError("embedding must be a finite numeric sequence")
        if len(vector) != self.dimensions:
            raise ValueError("embedding dimension does not match this vector index")
        values: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("embedding values must be finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding values must be finite numbers")
            values.append(number)
        if not any(value != 0 for value in values):
            raise ValueError("zero vectors cannot be ranked by cosine distance")
        return tuple(values)

    @classmethod
    def _prepare_document(cls, document: Document) -> tuple[str, Document, str]:
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
        metadata = cls._canonical_json(document.metadata)
        if document.id is not None:
            if not isinstance(document.id, str) or not document.id.strip():
                raise ValueError("Document id must be a non-empty string when provided")
            logical_id = document.id
        else:
            fingerprint = "\0".join((document.source, document.text, metadata))
            logical_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        snapshot = deepcopy(document)
        snapshot.id = logical_id
        storage_id = (
            logical_id
            if snapshot.generation is None
            else hashlib.sha256(f"{logical_id}\0{snapshot.generation}".encode()).hexdigest()
        )
        return storage_id, snapshot, metadata

    @staticmethod
    def _with_generation(document: Document, source: str, generation: str) -> Document:
        if not isinstance(document, Document) or document.source != source:
            raise ValueError("every staged document must be a Document for the requested source")
        snapshot = deepcopy(document)
        snapshot.generation = generation
        return snapshot

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("vector metadata must contain finite JSON values") from exc

    @classmethod
    def _validate_filters(cls, filters: dict[str, Any] | None) -> None:
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        for value in (filters or {}).values():
            cls._canonical_json(value)

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(repr(value) for value in vector) + "]"

    @staticmethod
    def _validate_source(source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

    @classmethod
    def _validate_source_generation(cls, source: str, generation: str) -> None:
        cls._validate_source(source)
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

    @staticmethod
    def _validate_fencing_token(fencing_token: int) -> None:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if fencing_token < 0:
            raise ValueError("fencing_token must be a non-negative integer")

    @staticmethod
    async def _lock_source(connection: Any, source: str) -> None:
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", source
        )

    @staticmethod
    async def _advance_source_fence(connection: Any, source: str, fencing_token: int) -> None:
        row = await connection.fetchrow(
            """INSERT INTO gabby_vector_source_fences (source, fencing_token)
               VALUES ($1, $2)
               ON CONFLICT (source) DO UPDATE SET
                 fencing_token = EXCLUDED.fencing_token,
                 updated_at = clock_timestamp()
               WHERE gabby_vector_source_fences.fencing_token <= EXCLUDED.fencing_token
               RETURNING fencing_token""",
            source,
            fencing_token,
        )
        if row is None:
            raise KnowledgeStoreError("Stale fencing token rejected by PostgreSQL vector store")

    @staticmethod
    def _affected_rows(status: str) -> int:
        try:
            command, count = status.rsplit(" ", 1)
            if command != "DELETE":
                raise ValueError
            return int(count)
        except (AttributeError, ValueError) as exc:
            raise KnowledgeStoreError("PostgreSQL returned an invalid command status") from exc

    @staticmethod
    def _raise_store_error(exc: Exception) -> NoReturn:
        if isinstance(exc, KnowledgeStoreError):
            raise exc
        raise KnowledgeStoreError("PostgreSQL vector operation failed") from None
