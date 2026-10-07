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
"""Opt-in acceptance against a real PostgreSQL knowledge store."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gabby import Document, KnowledgeStoreError, PostgresKnowledgeStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_knowledge() -> AsyncIterator[PostgresKnowledgeStore]:
    dsn = os.environ.get("GABBY_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set GABBY_POSTGRES_DSN to run PostgreSQL knowledge-store acceptance")
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
            Path(__file__).resolve().parents[2] / "sql" / "postgres_knowledge.sql"
        ).read_text(encoding="utf-8")
        async with pool.acquire() as connection:
            await connection.execute(migration)
        yield PostgresKnowledgeStore(pool)
    finally:
        if pool is not None:
            await pool.close()
        async with admin.acquire() as connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
async def test_postgres_knowledge_ingest_filter_replace_and_delete(
    postgres_knowledge: PostgresKnowledgeStore,
) -> None:
    store = postgres_knowledge
    first = Document(
        text="Authentication broke after the dependency upgrade.",
        source="guide.md",
        metadata={"collection": "support", "revision": 1},
        id="auth-issue",
    )
    assert await store.ingest([first]) == 1
    matches = await store.retrieve(
        "authentication dependency",
        filters={"collection": "support", "revision": 1},
    )
    assert [doc.id for doc in matches] == ["auth-issue"]
    assert await store.retrieve("authentication", filters={"revision": "1"}) == []

    second = Document(
        text="Authentication now works after the rollback.",
        source="guide.md",
        metadata={"collection": "support", "revision": 2},
        id="auth-fixed",
    )
    assert await store.replace_source("guide.md", [second]) == 1
    assert await store.retrieve("dependency", filters={"collection": "support"}) == []
    assert [doc.id for doc in await store.retrieve("rollback")] == ["auth-fixed"]
    assert await store.delete_source("guide.md") == 1
    assert await store.delete_source("guide.md") == 0


@pytest.mark.asyncio
async def test_postgres_knowledge_generation_fencing_and_active_filter(
    postgres_knowledge: PostgresKnowledgeStore,
) -> None:
    store = postgres_knowledge
    assert (
        await store.stage_source(
            "guide.md",
            "generation-1",
            3,
            [Document(text="old policy instructions", source="guide.md", id="policy")],
        )
        == 1
    )
    assert (
        await store.stage_source(
            "guide.md",
            "generation-2",
            4,
            [Document(text="current policy instructions", source="guide.md", id="policy")],
        )
        == 1
    )
    current = await store.retrieve("policy", generations={"guide.md": "generation-2"})
    assert [(document.id, document.generation) for document in current] == [
        ("policy", "generation-2")
    ]
    assert await store.retrieve("old", generations={"guide.md": "generation-2"}) == []

    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source(
            "guide.md",
            "generation-stale",
            3,
            [Document(text="stale policy instructions", source="guide.md", id="policy")],
        )
    await store.advance_fence("guide.md", 5)
    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await store.stage_source(
            "guide.md",
            "generation-3",
            4,
            [Document(text="late policy instructions", source="guide.md", id="policy")],
        )

    assert await store.delete_generation("guide.md", "generation-1") == 1
    assert await store.delete_generation("guide.md", "generation-1") == 0
    assert [doc.generation for doc in await store.retrieve("current")] == ["generation-2"]
