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
"""Durable generation coordination for hybrid lexical and vector indexes."""

from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from ._sync import run_sync_callback
from .hybrid import HybridRetriever, _validated_vectors
from .knowledge import (
    ActiveGenerationReader,
    Document,
    EmbeddingProvider,
    GenerationAwareRetriever,
    GenerationAwareVectorStore,
    KnowledgeStoreError,
    SourceKnowledgeWriter,
)


@dataclass(frozen=True)
class PendingIndexGeneration:
    """Durable progress record for one source generation that is not yet active."""

    source: str
    generation: str
    document_count: int
    lexical_ready: bool
    vector_ready: bool
    owner_id: str | None = None
    fencing_token: int = 0
    lease_expires_at: float = 0.0
    error_type: str | None = None


class GenerationBusyError(KnowledgeStoreError):
    """Another live writer currently owns the source generation lease."""


@dataclass(frozen=True)
class GenerationLease:
    """Ownership proof required to stage, recover, or activate one source generation."""

    source: str
    generation: str
    owner_id: str
    fencing_token: int
    expires_at: float


class GenerationManifestStore(ActiveGenerationReader, Protocol):
    """Persistent commit record shared by hybrid indexing and retrieval."""

    async def begin_source(
        self,
        source: str,
        generation: str,
        document_count: int,
        *,
        owner_id: str,
        lease_ttl_seconds: float,
    ) -> GenerationLease:
        """Create a pending generation and acquire its per-source writer lease."""
        ...

    async def renew_lease(
        self,
        source: str,
        generation: str,
        owner_id: str,
        fencing_token: int,
        lease_ttl_seconds: float,
    ) -> bool:
        """Extend a still-valid lease, returning false when ownership was lost."""
        ...

    async def claim_expired(
        self,
        source: str,
        generation: str,
        *,
        owner_id: str,
        lease_ttl_seconds: float,
    ) -> GenerationLease | None:
        """Take over an expired lease with a higher source fencing token."""
        ...

    async def release_lease(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Expire a failed writer's lease after its backend work has stopped."""
        ...

    async def mark_ready(
        self,
        source: str,
        generation: str,
        backend: Literal["lexical", "vector"],
        *,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        """Record that one backend durably staged the complete generation."""
        ...

    async def mark_error(
        self,
        source: str,
        generation: str,
        error_type: str,
        *,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        """Persist a bounded error type for an incomplete generation."""
        ...

    async def activate_source(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Atomically make a fully staged generation active and retire the old one."""
        ...

    async def pending_generations(self) -> list[PendingIndexGeneration]:
        """List generations whose staging or activation may need recovery."""
        ...

    async def retired_generations(self) -> list[tuple[str, str]]:
        """List inactive generations retained for safe later cleanup."""
        ...

    async def abandon_pending(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Mark an incomplete generation retired after its staged data is removed."""
        ...

    async def remove_retired(self, source: str, generation: str) -> None:
        """Forget a retired generation after both index backends removed it."""
        ...

    async def active_document_count(self, source: str) -> int:
        """Return the committed document count for one source, or zero if not indexed."""
        ...


_SQLiteResult = TypeVar("_SQLiteResult")


class SQLiteGenerationManifestStore:
    """Pluggable-contract implementation backed by a separate SQLite database file."""

    _schema_version = 2
    _default_lease_ttl_seconds = 30.0

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 10.0) -> None:
        if isinstance(path, str) and path == ":memory:":
            raise ValueError("SQLiteGenerationManifestStore requires a persistent database file")
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

    async def begin_source(
        self,
        source: str,
        generation: str,
        document_count: int,
        *,
        owner_id: str,
        lease_ttl_seconds: float = _default_lease_ttl_seconds,
    ) -> GenerationLease:
        """Create a pending generation and acquire its per-source writer lease."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        self._validate_owner(owner_id)
        self._validate_lease_ttl(lease_ttl_seconds)
        if isinstance(document_count, bool) or not isinstance(document_count, int):
            raise TypeError("document_count must be a non-negative integer")
        if document_count < 0:
            raise ValueError("document_count must be a non-negative integer")

        def begin(connection: sqlite3.Connection) -> GenerationLease:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                pending = connection.execute(
                    "SELECT 1 FROM gabby_index_generations "
                    "WHERE source = ? AND state = 'pending' LIMIT 1",
                    (source,),
                ).fetchone()
                if pending is not None:
                    raise GenerationBusyError(
                        f"Source {source!r} has a pending generation; reconcile before reindexing"
                    )
                current = connection.execute(
                    "SELECT fencing_token FROM gabby_index_source_fences WHERE source = ?",
                    (source,),
                ).fetchone()
                fencing_token = int(current[0]) + 1 if current is not None else 1
                connection.execute(
                    "INSERT INTO gabby_index_source_fences(source, fencing_token) VALUES (?, ?) "
                    "ON CONFLICT(source) DO UPDATE SET fencing_token = excluded.fencing_token",
                    (source, fencing_token),
                )
                expires_at = time.time() + lease_ttl_seconds
                connection.execute(
                    """INSERT INTO gabby_index_generations
                       (source, generation, state, document_count, lexical_ready, vector_ready,
                        created_at, owner_id, fencing_token, lease_expires_at)
                       VALUES (?, ?, 'pending', ?, 0, 0, ?, ?, ?, ?)""",
                    (
                        source,
                        generation,
                        document_count,
                        time.time(),
                        owner_id,
                        fencing_token,
                        expires_at,
                    ),
                )
                return GenerationLease(source, generation, owner_id, fencing_token, expires_at)

        return await self._run(begin)

    async def renew_lease(
        self,
        source: str,
        generation: str,
        owner_id: str,
        fencing_token: int,
        lease_ttl_seconds: float,
    ) -> bool:
        """Extend a still-valid lease, returning false when ownership was lost."""
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)
        self._validate_lease_ttl(lease_ttl_seconds)
        now = time.time()

        def renew(connection: sqlite3.Connection) -> bool:
            with connection:
                cursor = connection.execute(
                    "UPDATE gabby_index_generations SET lease_expires_at = ? "
                    "WHERE source = ? AND generation = ? AND state = 'pending' "
                    "AND owner_id = ? AND fencing_token = ? AND lease_expires_at > ?",
                    (
                        now + lease_ttl_seconds,
                        source,
                        generation,
                        owner_id,
                        fencing_token,
                        now,
                    ),
                )
            return cursor.rowcount == 1

        return await self._run(renew)

    async def claim_expired(
        self,
        source: str,
        generation: str,
        *,
        owner_id: str,
        lease_ttl_seconds: float,
    ) -> GenerationLease | None:
        """Take over an expired generation lease using a higher fencing token."""
        self._validate_owner(owner_id)
        self._validate_lease_ttl(lease_ttl_seconds)
        now = time.time()

        def claim(connection: sqlite3.Connection) -> GenerationLease | None:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                row = connection.execute(
                    "SELECT lease_expires_at FROM gabby_index_generations "
                    "WHERE source = ? AND generation = ? AND state = 'pending'",
                    (source, generation),
                ).fetchone()
                if row is None or float(row[0]) > now:
                    return None
                current = connection.execute(
                    "SELECT fencing_token FROM gabby_index_source_fences WHERE source = ?",
                    (source,),
                ).fetchone()
                fencing_token = int(current[0]) + 1 if current is not None else 1
                expires_at = now + lease_ttl_seconds
                connection.execute(
                    "INSERT INTO gabby_index_source_fences(source, fencing_token) VALUES (?, ?) "
                    "ON CONFLICT(source) DO UPDATE SET fencing_token = excluded.fencing_token",
                    (source, fencing_token),
                )
                cursor = connection.execute(
                    "UPDATE gabby_index_generations SET owner_id = ?, fencing_token = ?, "
                    "lease_expires_at = ? WHERE source = ? AND generation = ? "
                    "AND state = 'pending' AND lease_expires_at <= ?",
                    (owner_id, fencing_token, expires_at, source, generation, now),
                )
                if cursor.rowcount != 1:
                    return None
                return GenerationLease(source, generation, owner_id, fencing_token, expires_at)

        return await self._run(claim)

    async def mark_ready(
        self,
        source: str,
        generation: str,
        backend: Literal["lexical", "vector"],
        *,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        """Mark one staged backend ready for activation."""
        if backend not in ("lexical", "vector"):
            raise ValueError("backend must be 'lexical' or 'vector'")
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)
        column = "lexical_ready" if backend == "lexical" else "vector_ready"
        now = time.time()

        def mark(connection: sqlite3.Connection) -> None:
            with connection:
                cursor = connection.execute(
                    f"UPDATE gabby_index_generations SET {column} = 1 "
                    "WHERE source = ? AND generation = ? AND state = 'pending' "
                    "AND owner_id = ? AND fencing_token = ? AND lease_expires_at > ?",
                    (source, generation, owner_id, fencing_token, now),
                )
                if cursor.rowcount != 1:
                    raise KnowledgeStoreError("Cannot mark a missing index generation ready")

        await self._run(mark)

    async def activate_source(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Atomically activate a complete generation and retire the previous one."""
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)
        now = time.time()

        def activate(connection: sqlite3.Connection) -> None:
            with connection:
                row = connection.execute(
                    "SELECT state, lexical_ready, vector_ready, owner_id, fencing_token, "
                    "lease_expires_at FROM gabby_index_generations "
                    "WHERE source = ? AND generation = ?",
                    (source, generation),
                ).fetchone()
                if row is None:
                    raise KnowledgeStoreError("Cannot activate a missing index generation")
                if row[0] == "active":
                    return
                if (
                    row[0] != "pending"
                    or not row[1]
                    or not row[2]
                    or row[3] != owner_id
                    or row[4] != fencing_token
                    or row[5] is None
                    or float(row[5]) <= now
                ):
                    raise KnowledgeStoreError("Cannot activate an incompletely staged generation")
                connection.execute(
                    "UPDATE gabby_index_generations SET state = 'retired' "
                    "WHERE source = ? AND state = 'active'",
                    (source,),
                )
                connection.execute(
                    "UPDATE gabby_index_generations SET state = 'active', error_type = NULL, "
                    "owner_id = NULL, lease_expires_at = NULL "
                    "WHERE source = ? AND generation = ?",
                    (source, generation),
                )

        await self._run(activate)

    async def mark_error(
        self,
        source: str,
        generation: str,
        error_type: str,
        *,
        owner_id: str,
        fencing_token: int,
    ) -> None:
        """Record the exception type observed while staging a generation."""
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)
        safe_type = (
            error_type if error_type.isascii() and len(error_type) <= 128 else "ExtensionError"
        )

        def mark(connection: sqlite3.Connection) -> None:
            with connection:
                connection.execute(
                    "UPDATE gabby_index_generations SET error_type = ? "
                    "WHERE source = ? AND generation = ? AND state = 'pending' "
                    "AND owner_id = ? AND fencing_token = ?",
                    (safe_type, source, generation, owner_id, fencing_token),
                )

        await self._run(mark)

    async def pending_generations(self) -> list[PendingIndexGeneration]:
        """Return every generation that may need crash recovery."""

        def read(connection: sqlite3.Connection) -> list[PendingIndexGeneration]:
            rows = connection.execute(
                "SELECT source, generation, document_count, lexical_ready, vector_ready, "
                "owner_id, fencing_token, lease_expires_at, error_type "
                "FROM gabby_index_generations WHERE state = 'pending' "
                "ORDER BY source, created_at, generation"
            ).fetchall()
            return [
                PendingIndexGeneration(
                    source=str(row[0]),
                    generation=str(row[1]),
                    document_count=int(row[2]),
                    lexical_ready=bool(row[3]),
                    vector_ready=bool(row[4]),
                    owner_id=str(row[5]) if row[5] is not None else None,
                    fencing_token=int(row[6]),
                    lease_expires_at=float(row[7]) if row[7] is not None else 0.0,
                    error_type=str(row[8]) if row[8] is not None else None,
                )
                for row in rows
            ]

        return await self._run(read)

    async def retired_generations(self) -> list[tuple[str, str]]:
        """Return inactive generations waiting for backend cleanup."""

        def read(connection: sqlite3.Connection) -> list[tuple[str, str]]:
            rows = connection.execute(
                "SELECT source, generation FROM gabby_index_generations "
                "WHERE state = 'retired' ORDER BY source, created_at, generation"
            ).fetchall()
            return [(str(row[0]), str(row[1])) for row in rows]

        return await self._run(read)

    async def abandon_pending(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Mark a cleaned incomplete generation as retired."""
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)

        def abandon(connection: sqlite3.Connection) -> None:
            with connection:
                cursor = connection.execute(
                    "UPDATE gabby_index_generations SET state = 'retired', owner_id = NULL, "
                    "lease_expires_at = NULL WHERE source = ? AND generation = ? "
                    "AND state = 'pending' AND owner_id = ? AND fencing_token = ?",
                    (source, generation, owner_id, fencing_token),
                )
                if cursor.rowcount != 1:
                    raise KnowledgeStoreError("Cannot abandon a generation without its lease")

        await self._run(abandon)

    async def release_lease(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Expire a failed writer's lease after its backend work has stopped."""
        self._validate_owner(owner_id)
        self._validate_fencing_token(fencing_token)

        def release(connection: sqlite3.Connection) -> None:
            with connection:
                connection.execute(
                    "UPDATE gabby_index_generations SET lease_expires_at = 0 "
                    "WHERE source = ? AND generation = ? AND state = 'pending' "
                    "AND owner_id = ? AND fencing_token = ?",
                    (source, generation, owner_id, fencing_token),
                )

        await self._run(release)

    async def remove_retired(self, source: str, generation: str) -> None:
        """Remove the manifest row after both backends delete the generation."""

        def remove(connection: sqlite3.Connection) -> None:
            with connection:
                connection.execute(
                    "DELETE FROM gabby_index_generations "
                    "WHERE source = ? AND generation = ? AND state = 'retired'",
                    (source, generation),
                )

        await self._run(remove)

    async def active_generations(self) -> dict[str, str]:
        """Return the active generation selected for each indexed source."""

        def read(connection: sqlite3.Connection) -> dict[str, str]:
            rows = connection.execute(
                "SELECT source, generation FROM gabby_index_generations "
                "WHERE state = 'active' ORDER BY source"
            ).fetchall()
            return {str(source): str(generation) for source, generation in rows}

        return await self._run(read)

    async def active_document_count(self, source: str) -> int:
        """Return the committed document count, or zero for an unknown source."""

        def read(connection: sqlite3.Connection) -> int:
            row = connection.execute(
                "SELECT document_count FROM gabby_index_generations "
                "WHERE source = ? AND state = 'active'",
                (source,),
            ).fetchone()
            return int(row[0]) if row is not None else 0

        return await self._run(read)

    async def _run(self, operation: Callable[[sqlite3.Connection], _SQLiteResult]) -> _SQLiteResult:
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

        worker = asyncio.create_task(run_sync_callback(execute))
        try:
            while not worker.done():
                await asyncio.wait({worker}, timeout=0.05)
            return cast(_SQLiteResult, await worker)
        except asyncio.CancelledError:
            worker.add_done_callback(self._consume_worker_result)
            raise
        except sqlite3.Error as exc:
            raise KnowledgeStoreError("SQLite generation manifest operation failed") from exc

    @staticmethod
    def _consume_worker_result(worker: asyncio.Task[Any]) -> None:
        if not worker.cancelled():
            worker.exception()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        with self._schema_lock:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._schema_version:
                raise KnowledgeStoreError(
                    f"Generation manifest schema {version} is newer than supported "
                    f"schema {self._schema_version}"
                )
            if version == self._schema_version:
                return
            connection.execute("PRAGMA journal_mode = WAL")
            if version == 1:
                with connection:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS gabby_index_source_fences "
                        "(source TEXT PRIMARY KEY, fencing_token INTEGER NOT NULL)"
                    )
                    connection.execute(
                        "ALTER TABLE gabby_index_generations ADD COLUMN owner_id TEXT"
                    )
                    connection.execute(
                        "ALTER TABLE gabby_index_generations ADD COLUMN fencing_token "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                    connection.execute(
                        "ALTER TABLE gabby_index_generations ADD COLUMN lease_expires_at REAL"
                    )
                    connection.execute(
                        "UPDATE gabby_index_generations SET lease_expires_at = 0 "
                        "WHERE state = 'pending'"
                    )
                    connection.execute(
                        "UPDATE gabby_index_generations SET lease_expires_at = NULL "
                        "WHERE state != 'pending'"
                    )
                    connection.execute(
                        "INSERT INTO gabby_index_source_fences(source, fencing_token) "
                        "SELECT source, MAX(fencing_token) FROM gabby_index_generations "
                        "GROUP BY source"
                    )
                    connection.execute(f"PRAGMA user_version = {self._schema_version}")
                return
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS gabby_index_generations (
                       source TEXT NOT NULL,
                       generation TEXT NOT NULL,
                       state TEXT NOT NULL CHECK(state IN ('pending', 'active', 'retired')),
                       document_count INTEGER NOT NULL CHECK(document_count >= 0),
                       lexical_ready INTEGER NOT NULL CHECK(lexical_ready IN (0, 1)),
                       vector_ready INTEGER NOT NULL CHECK(vector_ready IN (0, 1)),
                       error_type TEXT,
                       created_at REAL NOT NULL,
                       owner_id TEXT,
                       fencing_token INTEGER NOT NULL DEFAULT 0,
                       lease_expires_at REAL,
                       PRIMARY KEY(source, generation)
                   );
                   CREATE TABLE IF NOT EXISTS gabby_index_source_fences (
                       source TEXT PRIMARY KEY,
                       fencing_token INTEGER NOT NULL
                   );
                   CREATE UNIQUE INDEX IF NOT EXISTS gabby_one_pending_generation
                       ON gabby_index_generations(source) WHERE state = 'pending';
                   CREATE UNIQUE INDEX IF NOT EXISTS gabby_one_active_generation
                       ON gabby_index_generations(source) WHERE state = 'active';
                   CREATE INDEX IF NOT EXISTS gabby_retired_generations
                       ON gabby_index_generations(state, created_at);
                   PRAGMA user_version = 2;"""
            )

    @staticmethod
    def _validate_owner(owner_id: str) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")

    @staticmethod
    def _validate_fencing_token(fencing_token: int) -> None:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise TypeError("fencing_token must be a non-negative integer")
        if fencing_token < 0:
            raise ValueError("fencing_token must be a non-negative integer")

    @staticmethod
    def _validate_lease_ttl(lease_ttl_seconds: float) -> None:
        if (
            isinstance(lease_ttl_seconds, bool)
            or not isinstance(lease_ttl_seconds, (int, float))
            or not math.isfinite(lease_ttl_seconds)
            or lease_ttl_seconds <= 0
        ):
            raise ValueError("lease_ttl_seconds must be a finite positive number")


class HybridIndexCoordinator(SourceKnowledgeWriter):
    """Stage lexical and vector generations, then atomically activate their manifest entry.

    The first retrieval or write reconciles incomplete generations left by process crashes. Retired
    data is retained to protect in-flight readers; ``prune_retired`` requires an explicit quiescent
    window before it removes old backend generations.
    """

    def __init__(
        self,
        lexical: GenerationAwareRetriever,
        embeddings: EmbeddingProvider,
        vectors: GenerationAwareVectorStore,
        manifest: GenerationManifestStore,
        *,
        candidate_limit: int = 20,
        rrf_constant: float = 60.0,
        lease_ttl_seconds: float = 30.0,
    ) -> None:
        SQLiteGenerationManifestStore._validate_lease_ttl(lease_ttl_seconds)
        self.lexical = lexical
        self.embeddings = embeddings
        self.vectors = vectors
        self.manifest = manifest
        self._retriever = HybridRetriever(
            lexical,
            embeddings,
            vectors,
            candidate_limit=candidate_limit,
            rrf_constant=rrf_constant,
            manifest=manifest,
        )
        self._reconcile_lock = asyncio.Lock()
        self._reconciled = False
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self._owner_id = uuid.uuid4().hex

    async def retrieve(
        self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """Reconcile pending writes once, then search only committed source generations."""
        await self._ensure_reconciled()
        return await self._retriever.retrieve(query, limit=limit, filters=filters)

    async def replace_source(self, source: str, documents: Sequence[Document]) -> int:
        """Stage both backend snapshots and activate them with one manifest transaction."""
        await self._ensure_reconciled()
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if any(
            not isinstance(document, Document) or document.source != source
            for document in documents
        ):
            raise ValueError(
                "every replacement document must be a Document for the requested source"
            )
        if any(document.generation is not None for document in documents):
            raise ValueError("replacement documents must not set a generation")

        generation = uuid.uuid4().hex
        document_snapshot = [
            Document(
                text=document.text,
                source=document.source,
                metadata=document.metadata.copy(),
                id=document.id,
            )
            for document in documents
        ]
        try:
            lease = await self.manifest.begin_source(
                source,
                generation,
                len(document_snapshot),
                owner_id=self._owner_id,
                lease_ttl_seconds=self.lease_ttl_seconds,
            )
        except GenerationBusyError:
            self._reconciled = False
            await self.reconcile()
            lease = await self.manifest.begin_source(
                source,
                generation,
                len(document_snapshot),
                owner_id=self._owner_id,
                lease_ttl_seconds=self.lease_ttl_seconds,
            )
        lease_lost = asyncio.Event()
        try:
            async with self._maintain_lease(lease, lease_lost):
                if document_snapshot:
                    embed_documents = getattr(self.embeddings, "embed_documents", None)
                    document_texts = [document.text for document in document_snapshot]
                    embedded = (
                        await embed_documents(document_texts)
                        if callable(embed_documents)
                        else await self.embeddings.embed(document_texts)
                    )
                    vectors = _validated_vectors(embedded, expected_count=len(document_snapshot))
                else:
                    vectors = []
                async with asyncio.TaskGroup() as group:
                    lexical_task = group.create_task(
                        self.lexical.stage_source(
                            source, generation, lease.fencing_token, document_snapshot
                        )
                    )
                    vector_task = group.create_task(
                        self.vectors.stage_source(
                            source,
                            generation,
                            lease.fencing_token,
                            document_snapshot,
                            vectors,
                        )
                    )
                if lexical_task.result() != len(document_snapshot):
                    raise KnowledgeStoreError("Lexical backend staged an unexpected document count")
                if vector_task.result() != len(document_snapshot):
                    raise KnowledgeStoreError("Vector backend staged an unexpected document count")
                await self.manifest.mark_ready(
                    source,
                    generation,
                    "lexical",
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                )
                await self.manifest.mark_ready(
                    source,
                    generation,
                    "vector",
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                )
                await self.manifest.activate_source(
                    source,
                    generation,
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                )
        except asyncio.CancelledError as exc:
            self._reconciled = False
            if lease_lost.is_set():
                raise KnowledgeStoreError("Hybrid source writer lease was lost") from exc
            raise
        except Exception as exc:
            self._reconciled = False
            with suppress(Exception):
                await self.manifest.mark_error(
                    source,
                    generation,
                    type(exc).__name__,
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                )
            with suppress(Exception):
                await self.manifest.release_lease(
                    source,
                    generation,
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                )
            raise KnowledgeStoreError(
                "Hybrid source generation failed; reconcile before retrying"
            ) from exc
        return len(document_snapshot)

    async def delete_source(self, source: str) -> int:
        """Activate an empty generation for a source while retaining old data for readers."""
        await self._ensure_reconciled()
        previous_count = await self.manifest.active_document_count(source)
        await self.replace_source(source, [])
        return previous_count

    async def reconcile(self) -> None:
        """Recover expired leases without interfering with a live index writer."""
        self._reconciled = False
        async with self._reconcile_lock:
            pending = await self.manifest.pending_generations()
            for generation in pending:
                lease = await self.manifest.claim_expired(
                    generation.source,
                    generation.generation,
                    owner_id=self._owner_id,
                    lease_ttl_seconds=self.lease_ttl_seconds,
                )
                if lease is None:
                    continue
                try:
                    if generation.lexical_ready and generation.vector_ready:
                        lost = asyncio.Event()
                        async with self._maintain_lease(lease, lost):
                            async with asyncio.TaskGroup() as group:
                                group.create_task(
                                    self.lexical.advance_fence(
                                        generation.source, lease.fencing_token
                                    )
                                )
                                group.create_task(
                                    self.vectors.advance_fence(
                                        generation.source, lease.fencing_token
                                    )
                                )
                            await self.manifest.activate_source(
                                generation.source,
                                generation.generation,
                                owner_id=lease.owner_id,
                                fencing_token=lease.fencing_token,
                            )
                        continue
                    lost = asyncio.Event()
                    async with self._maintain_lease(lease, lost):
                        async with asyncio.TaskGroup() as group:
                            group.create_task(
                                self.lexical.discard_generation(
                                    generation.source,
                                    generation.generation,
                                    lease.fencing_token,
                                )
                            )
                            group.create_task(
                                self.vectors.discard_generation(
                                    generation.source,
                                    generation.generation,
                                    lease.fencing_token,
                                )
                            )
                        await self.manifest.abandon_pending(
                            generation.source,
                            generation.generation,
                            owner_id=lease.owner_id,
                            fencing_token=lease.fencing_token,
                        )
                except Exception:
                    with suppress(Exception):
                        await self.manifest.release_lease(
                            generation.source,
                            generation.generation,
                            owner_id=lease.owner_id,
                            fencing_token=lease.fencing_token,
                        )
                    raise
            self._reconciled = True

    async def prune_retired(self, *, readers_quiescent: bool = False) -> int:
        """Remove retired backend generations during an operator-confirmed quiescent window."""
        if readers_quiescent is not True:
            raise ValueError("pruning retired generations requires readers_quiescent=True")
        await self._ensure_reconciled()
        retired = await self.manifest.retired_generations()
        for source, generation in retired:
            async with asyncio.TaskGroup() as group:
                group.create_task(self.lexical.delete_generation(source, generation))
                group.create_task(self.vectors.delete_generation(source, generation))
            await self.manifest.remove_retired(source, generation)
        return len(retired)

    async def _ensure_reconciled(self) -> None:
        if self._reconciled:
            return
        await self.reconcile()

    @asynccontextmanager
    async def _maintain_lease(
        self, lease: GenerationLease, lost: asyncio.Event
    ) -> AsyncIterator[None]:
        owner_task = asyncio.current_task()
        stop = asyncio.Event()
        interval = self.lease_ttl_seconds / 3

        async def renew() -> None:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    pass
                try:
                    renewed = await self.manifest.renew_lease(
                        lease.source,
                        lease.generation,
                        lease.owner_id,
                        lease.fencing_token,
                        self.lease_ttl_seconds,
                    )
                except Exception:
                    renewed = False
                if not renewed:
                    lost.set()
                    if owner_task is not None:
                        owner_task.cancel()
                    return

        heartbeat = asyncio.create_task(renew())
        try:
            yield
        finally:
            stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
