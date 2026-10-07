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
"""Opt-in contract acceptance against a real PostgreSQL database."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gabby import PostgresGenerationManifestStore
from gabby.indexing import GenerationBusyError
from gabby.knowledge import KnowledgeStoreError

pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_manifest() -> AsyncIterator[PostgresGenerationManifestStore]:
    dsn = os.environ.get("GABBY_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set GABBY_POSTGRES_DSN to run PostgreSQL generation-manifest acceptance")
    asyncpg = pytest.importorskip("asyncpg", reason="install gabby-agent-runtime[postgres]")
    schema = f"gabby_test_{uuid.uuid4().hex}"
    admin = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    pool = None
    try:
        async with admin.acquire() as connection:
            await connection.execute(f'CREATE SCHEMA "{schema}"')
        pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=8,
            server_settings={"search_path": schema},
        )
        migration = (
            Path(__file__).resolve().parents[2] / "sql" / "postgres_generation_manifest.sql"
        ).read_text(encoding="utf-8")
        async with pool.acquire() as connection:
            await connection.execute(migration)
        yield PostgresGenerationManifestStore(pool)
    finally:
        if pool is not None:
            await pool.close()
        async with admin.acquire() as connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
async def test_postgres_manifest_exclusive_leases_fencing_and_activation(
    postgres_manifest: PostgresGenerationManifestStore,
) -> None:
    manifest = postgres_manifest
    first, collision = await asyncio.gather(
        manifest.begin_source("source", "generation-1", 2, owner_id="writer-1"),
        manifest.begin_source("source", "generation-race", 1, owner_id="writer-2"),
        return_exceptions=True,
    )
    lease = first if not isinstance(first, BaseException) else collision
    busy = collision if lease is first else first
    assert not isinstance(lease, BaseException)
    assert isinstance(busy, GenerationBusyError)

    assert await manifest.renew_lease(
        lease.source,
        lease.generation,
        lease.owner_id,
        lease.fencing_token,
        60,
    )
    await manifest.release_lease(
        lease.source,
        lease.generation,
        owner_id=lease.owner_id,
        fencing_token=lease.fencing_token,
    )
    recovery = await manifest.claim_expired(
        lease.source,
        lease.generation,
        owner_id="recovery",
        lease_ttl_seconds=60,
    )
    assert recovery is not None
    assert recovery.fencing_token > lease.fencing_token
    with pytest.raises(KnowledgeStoreError, match="ready"):
        await manifest.mark_ready(
            lease.source,
            lease.generation,
            "lexical",
            owner_id=lease.owner_id,
            fencing_token=lease.fencing_token,
        )

    await manifest.mark_ready(
        recovery.source,
        recovery.generation,
        "lexical",
        owner_id=recovery.owner_id,
        fencing_token=recovery.fencing_token,
    )
    await manifest.mark_ready(
        recovery.source,
        recovery.generation,
        "vector",
        owner_id=recovery.owner_id,
        fencing_token=recovery.fencing_token,
    )
    await manifest.activate_source(
        recovery.source,
        recovery.generation,
        owner_id=recovery.owner_id,
        fencing_token=recovery.fencing_token,
    )
    assert await manifest.active_generations() == {"source": "generation-1"}
    assert await manifest.active_document_count("source") == 2

    second = await manifest.begin_source("source", "generation-2", 3, owner_id="writer-3")
    await manifest.mark_ready(
        second.source,
        second.generation,
        "lexical",
        owner_id=second.owner_id,
        fencing_token=second.fencing_token,
    )
    await manifest.mark_ready(
        second.source,
        second.generation,
        "vector",
        owner_id=second.owner_id,
        fencing_token=second.fencing_token,
    )
    await manifest.activate_source(
        second.source,
        second.generation,
        owner_id=second.owner_id,
        fencing_token=second.fencing_token,
    )
    assert await manifest.retired_generations() == [("source", "generation-1")]
    assert await manifest.active_document_count("source") == 3
    await manifest.remove_retired("source", "generation-1")
    assert await manifest.retired_generations() == []


@pytest.mark.asyncio
async def test_postgres_manifest_abandons_pending_and_preserves_active_snapshot(
    postgres_manifest: PostgresGenerationManifestStore,
) -> None:
    manifest = postgres_manifest
    active = await manifest.begin_source("source", "active", 1, owner_id="writer-1")
    for backend in ("lexical", "vector"):
        await manifest.mark_ready(
            active.source,
            active.generation,
            backend,
            owner_id=active.owner_id,
            fencing_token=active.fencing_token,
        )
    await manifest.activate_source(
        active.source,
        active.generation,
        owner_id=active.owner_id,
        fencing_token=active.fencing_token,
    )
    incomplete = await manifest.begin_source("source", "incomplete", 2, owner_id="writer-2")
    await manifest.mark_error(
        incomplete.source,
        incomplete.generation,
        "ValueError",
        owner_id=incomplete.owner_id,
        fencing_token=incomplete.fencing_token,
    )
    pending = await manifest.pending_generations()
    assert len(pending) == 1
    assert pending[0].error_type == "ValueError"
    await manifest.release_lease(
        incomplete.source,
        incomplete.generation,
        owner_id=incomplete.owner_id,
        fencing_token=incomplete.fencing_token,
    )
    recovery = await manifest.claim_expired(
        incomplete.source,
        incomplete.generation,
        owner_id="recovery",
        lease_ttl_seconds=30,
    )
    assert recovery is not None
    await manifest.abandon_pending(
        recovery.source,
        recovery.generation,
        owner_id=recovery.owner_id,
        fencing_token=recovery.fencing_token,
    )
    assert await manifest.active_generations() == {"source": "active"}
    assert await manifest.active_document_count("source") == 1
    assert await manifest.pending_generations() == []
    assert await manifest.retired_generations() == [("source", "incomplete")]
