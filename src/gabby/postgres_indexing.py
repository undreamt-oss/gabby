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
"""Shared PostgreSQL generation manifests over a host-owned async connection pool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypeVar

from .indexing import (
    GenerationBusyError,
    GenerationLease,
    PendingIndexGeneration,
    SQLiteGenerationManifestStore,
)
from .knowledge import KnowledgeStoreError


class AsyncPostgresPool(Protocol):
    """Minimal asyncpg-compatible pool contract used by the shared manifest adapter."""

    def acquire(self) -> Any:
        """Return an async context manager yielding a connection with transactions and queries."""
        ...


_T = TypeVar("_T")


class PostgresGenerationManifestStore:
    """Coordinate hybrid-index generations through a host-owned PostgreSQL pool.

    The host installs ``sql/postgres_generation_manifest.sql`` through its migration system and
    owns pool credentials, TLS, routing, lifecycle, backups, and consistency. Gabby neither creates
    schema nor closes the pool. Lease timestamps use the PostgreSQL server clock; every state change
    is transactionally committed before returning.
    """

    def __init__(self, pool: AsyncPostgresPool) -> None:
        if not callable(getattr(pool, "acquire", None)):
            raise TypeError("pool must provide async acquire() for a PostgreSQL connection")
        self._pool = pool

    async def begin_source(
        self,
        source: str,
        generation: str,
        document_count: int,
        *,
        owner_id: str,
        lease_ttl_seconds: float = 30.0,
    ) -> GenerationLease:
        """Create a pending generation and acquire its per-source writer lease."""
        self._validate_source_generation(source, generation)
        SQLiteGenerationManifestStore._validate_owner(owner_id)
        SQLiteGenerationManifestStore._validate_lease_ttl(lease_ttl_seconds)
        if isinstance(document_count, bool) or not isinstance(document_count, int):
            raise TypeError("document_count must be a non-negative integer")
        if document_count < 0:
            raise ValueError("document_count must be a non-negative integer")

        async def begin(connection: Any) -> GenerationLease:
            await connection.execute(
                "INSERT INTO gabby_index_sources(source) VALUES ($1) "
                "ON CONFLICT(source) DO NOTHING",
                source,
            )
            row = await connection.fetchrow(
                "SELECT fencing_token, active_generation, pending_generation "
                "FROM gabby_index_sources WHERE source = $1 FOR UPDATE",
                source,
            )
            if row is None:
                raise KnowledgeStoreError("Could not lock the PostgreSQL source manifest")
            if row["pending_generation"] is not None:
                raise GenerationBusyError(
                    f"Source {source!r} has a pending generation; reconcile before reindexing"
                )
            if row["active_generation"] == generation:
                raise ValueError("generation must differ from the active generation")
            retired = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM gabby_index_retired_generations "
                "WHERE source = $1 AND generation = $2)",
                source,
                generation,
            )
            if retired:
                raise ValueError("generation must not reuse a retained retired generation")
            fencing_token = int(row["fencing_token"]) + 1
            lease_row = await connection.fetchrow(
                "UPDATE gabby_index_sources SET fencing_token = $2, "
                "pending_generation = $3, pending_document_count = $4, "
                "pending_lexical_ready = FALSE, pending_vector_ready = FALSE, "
                "pending_created_at = clock_timestamp(), pending_owner_id = $5, "
                "pending_lease_expires_at = clock_timestamp() + "
                "($6::double precision * INTERVAL '1 second'), pending_error_type = NULL "
                "WHERE source = $1 RETURNING fencing_token, "
                "extract(epoch FROM pending_lease_expires_at)::double precision AS expires_at",
                source,
                fencing_token,
                generation,
                document_count,
                owner_id,
                float(lease_ttl_seconds),
            )
            if lease_row is None:
                raise KnowledgeStoreError("Could not create the PostgreSQL source lease")
            return GenerationLease(
                source,
                generation,
                owner_id,
                int(lease_row["fencing_token"]),
                float(lease_row["expires_at"]),
            )

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
        self._validate_lease_identity(owner_id, fencing_token)
        SQLiteGenerationManifestStore._validate_lease_ttl(lease_ttl_seconds)

        async def renew(connection: Any) -> bool:
            row = await connection.fetchrow(
                "UPDATE gabby_index_sources SET pending_lease_expires_at = clock_timestamp() + "
                "($5::double precision * INTERVAL '1 second') "
                "WHERE source = $1 AND pending_generation = $2 AND pending_owner_id = $3 "
                "AND fencing_token = $4 AND pending_lease_expires_at > clock_timestamp() "
                "RETURNING TRUE AS renewed",
                source,
                generation,
                owner_id,
                fencing_token,
                float(lease_ttl_seconds),
            )
            return row is not None

        return await self._run(renew)

    async def claim_expired(
        self,
        source: str,
        generation: str,
        *,
        owner_id: str,
        lease_ttl_seconds: float,
    ) -> GenerationLease | None:
        """Take over an expired source lease with a strictly higher fencing token."""
        SQLiteGenerationManifestStore._validate_owner(owner_id)
        SQLiteGenerationManifestStore._validate_lease_ttl(lease_ttl_seconds)

        async def claim(connection: Any) -> GenerationLease | None:
            row = await connection.fetchrow(
                "UPDATE gabby_index_sources SET fencing_token = fencing_token + 1, "
                "pending_owner_id = $3, pending_lease_expires_at = clock_timestamp() + "
                "($4::double precision * INTERVAL '1 second') "
                "WHERE source = $1 AND pending_generation = $2 "
                "AND pending_lease_expires_at <= clock_timestamp() "
                "RETURNING fencing_token, "
                "extract(epoch FROM pending_lease_expires_at)::double precision AS expires_at",
                source,
                generation,
                owner_id,
                float(lease_ttl_seconds),
            )
            if row is None:
                return None
            return GenerationLease(
                source,
                generation,
                owner_id,
                int(row["fencing_token"]),
                float(row["expires_at"]),
            )

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
        self._validate_lease_identity(owner_id, fencing_token)
        column = "pending_lexical_ready" if backend == "lexical" else "pending_vector_ready"

        async def mark(connection: Any) -> None:
            row = await connection.fetchrow(
                f"UPDATE gabby_index_sources SET {column} = TRUE "
                "WHERE source = $1 AND pending_generation = $2 AND pending_owner_id = $3 "
                "AND fencing_token = $4 AND pending_lease_expires_at > clock_timestamp() "
                "RETURNING TRUE AS marked",
                source,
                generation,
                owner_id,
                fencing_token,
            )
            if row is None:
                raise KnowledgeStoreError("Cannot mark a missing index generation ready")

        await self._run(mark)

    async def activate_source(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Atomically activate the complete generation and retire its former active snapshot."""
        self._validate_lease_identity(owner_id, fencing_token)

        async def activate(connection: Any) -> None:
            row = await connection.fetchrow(
                "SELECT active_generation, pending_generation, pending_document_count, "
                "pending_lexical_ready, pending_vector_ready, pending_owner_id, fencing_token, "
                "pending_lease_expires_at > clock_timestamp() AS lease_valid "
                "FROM gabby_index_sources WHERE source = $1 FOR UPDATE",
                source,
            )
            if row is None:
                raise KnowledgeStoreError("Cannot activate a missing index generation")
            if row["active_generation"] == generation:
                return
            if (
                row["pending_generation"] != generation
                or not row["pending_lexical_ready"]
                or not row["pending_vector_ready"]
                or row["pending_owner_id"] != owner_id
                or int(row["fencing_token"]) != fencing_token
                or not row["lease_valid"]
            ):
                raise KnowledgeStoreError("Cannot activate an incompletely staged generation")
            active_generation = row["active_generation"]
            if active_generation is not None:
                await connection.execute(
                    "INSERT INTO gabby_index_retired_generations(source, generation) "
                    "VALUES ($1, $2)",
                    source,
                    active_generation,
                )
            await connection.execute(
                "UPDATE gabby_index_sources SET active_generation = pending_generation, "
                "active_document_count = pending_document_count, pending_generation = NULL, "
                "pending_document_count = NULL, pending_lexical_ready = FALSE, "
                "pending_vector_ready = FALSE, pending_created_at = NULL, pending_owner_id = NULL, "
                "pending_lease_expires_at = NULL, pending_error_type = NULL "
                "WHERE source = $1",
                source,
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
        """Persist a bounded error type for an incomplete generation."""
        self._validate_lease_identity(owner_id, fencing_token)
        safe_type = (
            error_type
            if isinstance(error_type, str) and error_type.isascii() and len(error_type) <= 128
            else "ExtensionError"
        )

        async def mark(connection: Any) -> None:
            await connection.execute(
                "UPDATE gabby_index_sources SET pending_error_type = $5 "
                "WHERE source = $1 AND pending_generation = $2 AND pending_owner_id = $3 "
                "AND fencing_token = $4",
                source,
                generation,
                owner_id,
                fencing_token,
                safe_type,
            )

        await self._run(mark)

    async def pending_generations(self) -> list[PendingIndexGeneration]:
        """Return pending generations that may need live-lease observation or recovery."""

        async def read(connection: Any) -> list[PendingIndexGeneration]:
            rows = await connection.fetch(
                "SELECT source, pending_generation AS generation, pending_document_count, "
                "pending_lexical_ready, pending_vector_ready, pending_owner_id, fencing_token, "
                "extract(epoch FROM pending_lease_expires_at)::double precision "
                "AS lease_expires_at, "
                "pending_error_type FROM gabby_index_sources "
                "WHERE pending_generation IS NOT NULL ORDER BY source, pending_created_at, "
                "pending_generation"
            )
            return [
                PendingIndexGeneration(
                    source=str(row["source"]),
                    generation=str(row["generation"]),
                    document_count=int(row["pending_document_count"]),
                    lexical_ready=bool(row["pending_lexical_ready"]),
                    vector_ready=bool(row["pending_vector_ready"]),
                    owner_id=(
                        str(row["pending_owner_id"])
                        if row["pending_owner_id"] is not None
                        else None
                    ),
                    fencing_token=int(row["fencing_token"]),
                    lease_expires_at=float(row["lease_expires_at"] or 0.0),
                    error_type=(
                        str(row["pending_error_type"])
                        if row["pending_error_type"] is not None
                        else None
                    ),
                )
                for row in rows
            ]

        return await self._run(read)

    async def retired_generations(self) -> list[tuple[str, str]]:
        """List inactive generations retained for safe later backend cleanup."""

        async def read(connection: Any) -> list[tuple[str, str]]:
            rows = await connection.fetch(
                "SELECT source, generation FROM gabby_index_retired_generations "
                "ORDER BY source, retired_at, generation"
            )
            return [(str(row["source"]), str(row["generation"])) for row in rows]

        return await self._run(read)

    async def abandon_pending(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Mark a cleaned incomplete generation retired while retaining its fence history."""
        self._validate_lease_identity(owner_id, fencing_token)

        async def abandon(connection: Any) -> None:
            row = await connection.fetchrow(
                "SELECT pending_generation FROM gabby_index_sources "
                "WHERE source = $1 AND pending_generation = $2 AND pending_owner_id = $3 "
                "AND fencing_token = $4 FOR UPDATE",
                source,
                generation,
                owner_id,
                fencing_token,
            )
            if row is None:
                raise KnowledgeStoreError("Cannot abandon a generation without its lease")
            await connection.execute(
                "INSERT INTO gabby_index_retired_generations(source, generation) VALUES ($1, $2)",
                source,
                generation,
            )
            await self._clear_pending(connection, source)

        await self._run(abandon)

    async def release_lease(
        self, source: str, generation: str, *, owner_id: str, fencing_token: int
    ) -> None:
        """Expire a failed writer's lease after its backend work has stopped."""
        self._validate_lease_identity(owner_id, fencing_token)

        async def release(connection: Any) -> None:
            await connection.execute(
                "UPDATE gabby_index_sources SET pending_lease_expires_at = clock_timestamp() "
                "WHERE source = $1 AND pending_generation = $2 AND pending_owner_id = $3 "
                "AND fencing_token = $4",
                source,
                generation,
                owner_id,
                fencing_token,
            )

        await self._run(release)

    async def remove_retired(self, source: str, generation: str) -> None:
        """Forget a retired generation after both index backends have removed it."""

        async def remove(connection: Any) -> None:
            await connection.execute(
                "DELETE FROM gabby_index_retired_generations WHERE source = $1 AND generation = $2",
                source,
                generation,
            )

        await self._run(remove)

    async def active_generations(self) -> dict[str, str]:
        """Return the active generation selected for each indexed source."""

        async def read(connection: Any) -> dict[str, str]:
            rows = await connection.fetch(
                "SELECT source, active_generation FROM gabby_index_sources "
                "WHERE active_generation IS NOT NULL ORDER BY source"
            )
            return {str(row["source"]): str(row["active_generation"]) for row in rows}

        return await self._run(read)

    async def active_document_count(self, source: str) -> int:
        """Return committed document count for one source, or zero if not indexed."""

        async def read(connection: Any) -> int:
            value = await connection.fetchval(
                "SELECT active_document_count FROM gabby_index_sources "
                "WHERE source = $1 AND active_generation IS NOT NULL",
                source,
            )
            return int(value) if value is not None else 0

        return await self._run(read)

    async def _run(self, operation: Callable[[Any], Awaitable[_T]]) -> _T:
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                return await operation(connection)
        except KnowledgeStoreError:
            raise
        except Exception:
            raise KnowledgeStoreError("PostgreSQL generation manifest operation failed") from None

    @staticmethod
    async def _clear_pending(connection: Any, source: str) -> None:
        await connection.execute(
            "UPDATE gabby_index_sources SET pending_generation = NULL, "
            "pending_document_count = NULL, pending_lexical_ready = FALSE, "
            "pending_vector_ready = FALSE, pending_created_at = NULL, pending_owner_id = NULL, "
            "pending_lease_expires_at = NULL, pending_error_type = NULL WHERE source = $1",
            source,
        )

    @staticmethod
    def _validate_source_generation(source: str, generation: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")

    @staticmethod
    def _validate_lease_identity(owner_id: str, fencing_token: int) -> None:
        SQLiteGenerationManifestStore._validate_owner(owner_id)
        SQLiteGenerationManifestStore._validate_fencing_token(fencing_token)
