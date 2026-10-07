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
"""Opt-in acceptance against PostgreSQL with the pgvector extension installed."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gabby import Document, KnowledgeStoreError, PostgresVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_vectors() -> AsyncIterator[PostgresVectorStore]:
    dsn = os.environ.get("GABBY_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set GABBY_POSTGRES_DSN to run PostgreSQL vector-store acceptance")
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
            server_settings={"search_path": f"{schema},public"},
        )
        migration = (Path(__file__).resolve().parents[2] / "sql" / "postgres_vector.sql").read_text(
            encoding="utf-8"
        )
        async with pool.acquire() as connection:
            await connection.execute(migration)
        yield PostgresVectorStore(pool, dimensions=3)
    finally:
        if pool is not None:
            await pool.close()
        async with admin.acquire() as connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
async def test_postgres_vector_cosine_metadata_and_source_replacement(
    postgres_vectors: PostgresVectorStore,
) -> None:
    store = postgres_vectors
    documents = [
        Document(
            text="authentication details",
            source="guide.md",
            metadata={"team": "runtime", "revision": 1},
            id="auth",
        ),
        Document(
            text="database details",
            source="guide.md",
            metadata={"team": "data", "revision": 1},
            id="database",
        ),
    ]
    assert await store.upsert(documents, [[1, 0, 0], [0, 1, 0]]) == 2
    assert [document.id for document in await store.search([1, 0, 0], limit=2)] == [
        "auth",
        "database",
    ]
    assert [
        document.id
        for document in await store.search([1, 0, 0], limit=2, filters={"team": "runtime"})
    ] == ["auth"]
    assert await store.search([1, 0, 0], limit=2, filters={"revision": "1"}) == []

    replacement = Document(
        text="current policy details",
        source="guide.md",
        metadata={"team": "runtime"},
        id="policy",
    )
    assert await store.replace_source("guide.md", [replacement], [[0, 0, 1]]) == 1
    assert [document.id for document in await store.search([0, 0, 1], limit=5)] == ["policy"]
    assert await store.delete_source("guide.md") == 1
    assert await store.delete_source("guide.md") == 0


@pytest.mark.asyncio
async def test_postgres_vector_generations_fencing_and_dimension_contract(
    postgres_vectors: PostgresVectorStore,
) -> None:
    store = postgres_vectors
    assert (
        await store.stage_source(
            "guide.md",
            "generation-1",
            4,
            [Document(text="old policy", source="guide.md", id="policy")],
            [[1, 0, 0]],
        )
        == 1
    )
    assert (
        await store.stage_source(
            "guide.md",
            "generation-2",
            5,
            [Document(text="current policy", source="guide.md", id="policy")],
            [[0, 1, 0]],
        )
        == 1
    )
    current = await store.search([0, 1, 0], limit=5, generations={"guide.md": "generation-2"})
    assert [(document.id, document.generation) for document in current] == [
        ("policy", "generation-2")
    ]
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source(
            "guide.md",
            "stale",
            4,
            [Document(text="stale policy", source="guide.md", id="policy")],
            [[0, 0, 1]],
        )
    assert await store.discard_generation("guide.md", "generation-1", 6) == 1
    assert await store.delete_generation("guide.md", "generation-1") == 0
