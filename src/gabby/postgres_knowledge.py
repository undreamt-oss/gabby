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
"""Host-pooled PostgreSQL lexical knowledge storage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, NoReturn

from .knowledge import Document, GenerationAwareRetriever, KnowledgeStoreError
from .postgres_indexing import AsyncPostgresPool


class PostgresKnowledgeStore(GenerationAwareRetriever):
    """Shared lexical knowledge store backed by a host-owned PostgreSQL pool.

    Apply ``sql/postgres_knowledge.sql`` through the host's migration system before use.
    Gabby neither creates schema at runtime nor closes the injected pool. Search uses
    PostgreSQL's built-in ``simple`` text-search configuration and JSONB exact-value filters.
    Generation staging, retrieval filters, and fencing support coordinated hybrid indexing.
    """

    _token_pattern = re.compile(r"\w+", re.UNICODE)

    def __init__(self, pool: AsyncPostgresPool) -> None:
        if not callable(getattr(pool, "acquire", None)):
            raise TypeError("pool must provide async acquire() for a PostgreSQL connection")
        self._pool = pool

    async def ingest(self, documents: Sequence[Document]) -> int:
        """Insert or update documents by stable ID in a single PostgreSQL transaction."""
        prepared = [self._prepare(document) for document in documents]
        if not prepared:
            return 0
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                for source in sorted({document.source for _, document, _ in prepared}):
                    await self._lock_source(connection, source)
                for storage_id, document, metadata_json in prepared:
                    await connection.execute(
                        """INSERT INTO gabby_knowledge_documents
                           (storage_id, document_id, text, source, metadata, generation)
                           VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                           ON CONFLICT (storage_id) DO UPDATE SET
                             document_id = EXCLUDED.document_id,
                             text = EXCLUDED.text,
                             source = EXCLUDED.source,
                             metadata = EXCLUDED.metadata,
                             generation = EXCLUDED.generation,
                             updated_at = clock_timestamp()""",
                        storage_id,
                        document.id,
                        document.text,
                        document.source,
                        metadata_json,
                        document.generation,
                    )
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def replace_source(self, source: str, documents: Sequence[Document]) -> int:
        """Atomically replace every document attributed to one source."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        prepared = [self._prepare(document) for document in documents]
        if any(document.source != source for _, document, _ in prepared):
            raise ValueError("every replacement document must have the requested source")
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                await connection.execute(
                    "DELETE FROM gabby_knowledge_documents WHERE source = $1", source
                )
                for storage_id, document, metadata_json in prepared:
                    await connection.execute(
                        """INSERT INTO gabby_knowledge_documents
                           (storage_id, document_id, text, source, metadata, generation)
                           VALUES ($1, $2, $3, $4, $5::jsonb, $6)""",
                        storage_id,
                        document.id,
                        document.text,
                        document.source,
                        metadata_json,
                        document.generation,
                    )
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def delete_source(self, source: str) -> int:
        """Delete a source and return its document count."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                status = await connection.execute(
                    "DELETE FROM gabby_knowledge_documents WHERE source = $1", source
                )
            return int(status.rsplit(" ", 1)[-1])
        except Exception as exc:
            self._raise_store_error(exc)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        """Return bounded lexical matches with exact JSON-valued metadata filters."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a mapping of metadata fields to exact values")
        if filters and any(not isinstance(key, str) for key in filters):
            raise TypeError("metadata filter keys must be strings")
        if generations is not None and not isinstance(generations, Mapping):
            raise TypeError("generations must map sources to active generation IDs")
        if generations is not None and any(
            not isinstance(source, str)
            or not source
            or not isinstance(generation, str)
            or not generation
            for source, generation in generations.items()
        ):
            raise TypeError("generations must map source strings to non-empty generation IDs")
        if limit == 0:
            return []
        if generations is not None and not generations:
            return []
        tokens = self._token_pattern.findall(query.casefold())
        if not tokens:
            return []
        parameters: list[Any] = [tokens]
        filter_clauses: list[str] = []
        for key, value in (filters or {}).items():
            parameters.extend((key, self._canonical_json(value)))
            filter_clauses.append(
                f"metadata -> ${len(parameters) - 1}::text = ${len(parameters)}::jsonb"
            )
        if generations is not None:
            generation_clauses: list[str] = []
            for source, generation in generations.items():
                parameters.extend((source, generation))
                generation_clauses.append(
                    f"(source = ${len(parameters) - 1}::text "
                    f"AND generation = ${len(parameters)}::text)"
                )
            filter_clauses.append("(" + " OR ".join(generation_clauses) + ")")
        parameters.append(limit)
        where_filters = "".join(f" AND {clause}" for clause in filter_clauses)
        sql = f"""WITH terms AS (
                     SELECT plainto_tsquery('simple', terms.term) AS tsq
                     FROM unnest($1::text[]) AS terms(term)
                 )
                 SELECT document_id, text, source, metadata::text AS metadata_json, generation,
                        SUM(ts_rank_cd(search_vector, terms.tsq, 32)) AS rank
                 FROM gabby_knowledge_documents CROSS JOIN terms
                 WHERE search_vector @@ terms.tsq{where_filters}
                 GROUP BY storage_id, document_id, text, source, metadata, generation
                 ORDER BY rank DESC, document_id
                 LIMIT ${len(parameters)}"""
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(sql, *parameters)
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

    @classmethod
    def _prepare(cls, document: Document) -> tuple[str, Document, str]:
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
        if document.id is not None:
            if not isinstance(document.id, str) or not document.id.strip():
                raise ValueError("Document id must be a non-empty string when provided")
            document_id = document.id
        else:
            fingerprint = "\0".join((document.source, document.text, metadata_json))
            document_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        snapshot = deepcopy(document)
        snapshot.id = document_id
        return cls._stored_id(document_id, snapshot.generation), snapshot, metadata_json

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
    ) -> int:
        """Stage one generation atomically and reject writers with stale fencing tokens."""
        self._validate_source_generation(source, generation)
        self._validate_fencing_token(fencing_token)
        snapshots: list[Document] = []
        for document in documents:
            if not isinstance(document, Document) or document.source != source:
                raise ValueError(
                    "every staged document must be a Document for the requested source"
                )
            snapshot = deepcopy(document)
            snapshot.generation = generation
            snapshots.append(snapshot)
        prepared = [self._prepare(document) for document in snapshots]
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                await self._advance_source_fence(connection, source, fencing_token)
                await connection.execute(
                    """DELETE FROM gabby_knowledge_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
                for storage_id, document, metadata_json in prepared:
                    await connection.execute(
                        """INSERT INTO gabby_knowledge_documents
                           (storage_id, document_id, text, source, metadata, generation)
                           VALUES ($1, $2, $3, $4, $5::jsonb, $6)""",
                        storage_id,
                        document.id,
                        document.text,
                        document.source,
                        metadata_json,
                        generation,
                    )
            return len(prepared)
        except Exception as exc:
            self._raise_store_error(exc)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        """Advance a source fence and atomically remove one incomplete generation."""
        self._validate_source_generation(source, generation)
        self._validate_fencing_token(fencing_token)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await self._lock_source(connection, source)
                await self._advance_source_fence(connection, source, fencing_token)
                status = await connection.execute(
                    """DELETE FROM gabby_knowledge_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
            return int(status.rsplit(" ", 1)[-1])
        except Exception as exc:
            self._raise_store_error(exc)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        """Persist a source fence without changing staged documents."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
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
                    """DELETE FROM gabby_knowledge_documents
                       WHERE source = $1 AND generation = $2""",
                    source,
                    generation,
                )
            return int(status.rsplit(" ", 1)[-1])
        except Exception as exc:
            self._raise_store_error(exc)

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("knowledge metadata must contain finite JSON values") from exc

    @staticmethod
    def _stored_id(document_id: str, generation: str | None) -> str:
        if generation is None:
            return document_id
        return hashlib.sha256(f"{document_id}\0{generation}".encode()).hexdigest()

    @staticmethod
    def _validate_source_generation(source: str, generation: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

    @staticmethod
    def _validate_fencing_token(fencing_token: int) -> None:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if fencing_token < 0:
            raise ValueError("fencing_token must be a non-negative integer")

    @staticmethod
    async def _advance_source_fence(connection: Any, source: str, fencing_token: int) -> None:
        row = await connection.fetchrow(
            """INSERT INTO gabby_knowledge_source_fences (source, fencing_token)
               VALUES ($1, $2)
               ON CONFLICT (source) DO UPDATE SET
                 fencing_token = EXCLUDED.fencing_token,
                 updated_at = clock_timestamp()
               WHERE gabby_knowledge_source_fences.fencing_token <= EXCLUDED.fencing_token
               RETURNING fencing_token""",
            source,
            fencing_token,
        )
        if row is None:
            raise KnowledgeStoreError("Stale fencing token rejected by PostgreSQL knowledge store")

    @staticmethod
    async def _lock_source(connection: Any, source: str) -> None:
        """Serialize mutations for a source across all adapter instances."""
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))", source
        )

    @staticmethod
    def _raise_store_error(exc: Exception) -> NoReturn:
        if isinstance(exc, KnowledgeStoreError):
            raise exc
        raise KnowledgeStoreError("PostgreSQL knowledge operation failed") from None
