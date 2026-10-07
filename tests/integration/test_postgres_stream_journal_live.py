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
"""Opt-in acceptance of stream replay against a real PostgreSQL service."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from importlib.resources import files
from pathlib import Path

import pytest

from gabby import PostgresStreamJournal

pytestmark = pytest.mark.integration


@pytest.fixture
async def postgres_journal() -> AsyncIterator[PostgresStreamJournal]:
    dsn = os.environ.get("GABBY_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set GABBY_POSTGRES_DSN to run PostgreSQL stream-journal acceptance")
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
        source_migration = (
            Path(__file__).resolve().parents[2] / "sql" / "postgres_stream_journal.sql"
        ).read_text(encoding="utf-8")
        packaged_migration = (
            files("gabby").joinpath("sql/postgres_stream_journal.sql").read_text(encoding="utf-8")
        )
        assert packaged_migration == source_migration
        async with pool.acquire() as connection:
            await connection.execute(packaged_migration)
        yield PostgresStreamJournal(pool)
    finally:
        if pool is not None:
            await pool.close()
        async with admin.acquire() as connection:
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()


@pytest.mark.asyncio
async def test_postgres_journal_replays_and_binds_session_identity(
    postgres_journal: PostgresStreamJournal,
) -> None:
    journal = postgres_journal
    key = uuid.uuid4().hex
    snapshot = await journal.get_or_create(
        key,
        "request-fingerprint",
        "principal-hash",
        last_event_id=0,
        max_sessions=4,
        max_response_bytes=1024,
        ttl_seconds=60,
        active_ttl_seconds=60,
    )
    assert snapshot.created and snapshot.frames == () and not snapshot.done

    assert await journal.append(key, b"event-one", max_bytes=64)
    assert await journal.append(key, b"event-two", max_bytes=64)
    assert await journal.append(key, b"x" * 100, max_bytes=64) is False
    assert await journal.read_after(key, 0) == ([b"event-one", b"event-two"], False)
    await journal.finish(key)
    await journal.finish(key)
    resumed = await journal.get_or_create(
        key,
        "request-fingerprint",
        "principal-hash",
        last_event_id=1,
        max_sessions=4,
        max_response_bytes=1024,
        ttl_seconds=60,
        active_ttl_seconds=60,
    )
    assert not resumed.created and resumed.done and resumed.event_count == 2
    assert resumed.frames == (b"event-one", b"event-two")
    with pytest.raises(ValueError, match="different request"):
        await journal.get_or_create(
            key,
            "other-fingerprint",
            "principal-hash",
            last_event_id=0,
            max_sessions=4,
            max_response_bytes=1024,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )
    with pytest.raises(LookupError):
        await journal.get_or_create(
            key,
            "request-fingerprint",
            "other-principal",
            last_event_id=0,
            max_sessions=4,
            max_response_bytes=1024,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )
    await journal.remove(key)


@pytest.mark.asyncio
async def test_postgres_journal_serializes_capacity_and_recovers_abandoned_runs(
    postgres_journal: PostgresStreamJournal,
) -> None:
    journal = postgres_journal
    keys = [uuid.uuid4().hex, uuid.uuid4().hex]
    first, duplicate = await asyncio.gather(
        journal.get_or_create(
            keys[0],
            "fingerprint",
            "principal",
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=2048,
            ttl_seconds=60,
            active_ttl_seconds=60,
        ),
        journal.get_or_create(
            keys[1],
            "fingerprint",
            "principal",
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=2048,
            ttl_seconds=60,
            active_ttl_seconds=60,
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(item, BaseException) for item in (first, duplicate)) == 1
    assert sum(isinstance(item, OverflowError) for item in (first, duplicate)) == 1
    session_key = keys[0] if not isinstance(first, BaseException) else keys[1]
    await journal.reap(ttl_seconds=60, active_ttl_seconds=0.001)
    frames, done = await journal.read_after(session_key, 0)
    assert done and len(frames) == 1
    assert b"StreamRecoveryError" in frames[0]
