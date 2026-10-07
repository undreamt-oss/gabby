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
"""Contract tests for the host-pool PostgreSQL stream journal adapter."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from gabby import PostgresStreamJournal


class _Connection:
    def __init__(
        self,
        *,
        fetch: list[list[dict[str, Any]]] | None = None,
        fetchrow: list[dict[str, Any] | None] | None = None,
        fetchval: list[Any] | None = None,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fetchrow_results = deque(fetchrow or [])
        self.fetchval_results = deque(fetchval or [])
        self.executions: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        del query, args
        return self.fetch_results.popleft() if self.fetch_results else []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        del query, args
        return self.fetchrow_results.popleft() if self.fetchrow_results else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        del query, args
        return self.fetchval_results.popleft() if self.fetchval_results else 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Connection]:
        yield self.connection


def _journal(connection: _Connection) -> PostgresStreamJournal:
    return PostgresStreamJournal(_Pool(connection))


def test_constructor_and_limit_validation() -> None:
    with pytest.raises(TypeError, match="acquire"):
        PostgresStreamJournal(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="last_event_id"):
        PostgresStreamJournal._validate_limits(
            last_event_id=True,
            max_sessions=1,
            max_response_bytes=1,
            ttl_seconds=1,
            active_ttl_seconds=1,
        )
    with pytest.raises(ValueError, match="max_sessions"):
        PostgresStreamJournal._validate_limits(
            last_event_id=0,
            max_sessions=0,
            max_response_bytes=1,
            ttl_seconds=1,
            active_ttl_seconds=1,
        )
    with pytest.raises(ValueError, match="max_response_bytes"):
        PostgresStreamJournal._validate_limits(
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=0,
            ttl_seconds=1,
            active_ttl_seconds=1,
        )
    for name in ("ttl_seconds", "active_ttl_seconds"):
        limits: dict[str, Any] = {
            "last_event_id": 0,
            "max_sessions": 1,
            "max_response_bytes": 1,
            "ttl_seconds": 1,
            "active_ttl_seconds": 1,
        }
        limits[name] = float("nan")
        with pytest.raises(ValueError, match=name):
            PostgresStreamJournal._validate_limits(**limits)


@pytest.mark.asyncio
async def test_get_or_create_new_session_and_missing_replay_cursor() -> None:
    connection = _Connection(fetch=[[]], fetchrow=[None], fetchval=[0])
    journal = _journal(connection)
    snapshot = await journal.get_or_create(
        "key",
        "fingerprint",
        "principal",
        last_event_id=0,
        max_sessions=2,
        max_response_bytes=1000,
        ttl_seconds=60,
        active_ttl_seconds=60,
    )
    assert snapshot.created and snapshot.frames == () and not snapshot.done
    assert any("INSERT INTO gabby_stream_sessions" in query for query, _ in connection.executions)

    missing = _journal(_Connection(fetch=[[]], fetchrow=[None]))
    with pytest.raises(LookupError, match="not found"):
        await missing.get_or_create(
            "key",
            "fingerprint",
            "principal",
            last_event_id=1,
            max_sessions=2,
            max_response_bytes=1000,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )


@pytest.mark.asyncio
async def test_get_or_create_rejects_full_capacity_identity_mismatch_and_bad_frames() -> None:
    full = _journal(_Connection(fetch=[[]], fetchrow=[None], fetchval=[2]))
    with pytest.raises(OverflowError, match="capacity"):
        await full.get_or_create(
            "key",
            "fingerprint",
            "principal",
            last_event_id=0,
            max_sessions=2,
            max_response_bytes=1000,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )

    for record, error, message in (
        (
            {"principal_hash": "other", "fingerprint": "f", "event_count": 0, "finished_at": None},
            LookupError,
            "not found",
        ),
        (
            {"principal_hash": "p", "fingerprint": "other", "event_count": 0, "finished_at": None},
            ValueError,
            "different request",
        ),
    ):
        journal = _journal(_Connection(fetch=[[]], fetchrow=[record]))
        with pytest.raises(error, match=message):
            await journal.get_or_create(
                "key",
                "f",
                "p",
                last_event_id=0,
                max_sessions=2,
                max_response_bytes=1000,
                ttl_seconds=60,
                active_ttl_seconds=60,
            )

    inconsistent = _journal(
        _Connection(
            fetch=[[], [{"event_id": 2, "frame": b"gap"}]],
            fetchrow=[
                {"principal_hash": "p", "fingerprint": "f", "event_count": 1, "finished_at": None}
            ],
        )
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        await inconsistent.get_or_create(
            "key",
            "f",
            "p",
            last_event_id=0,
            max_sessions=2,
            max_response_bytes=1000,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )


@pytest.mark.asyncio
async def test_get_or_create_resume_and_cursor_ahead() -> None:
    record = {"principal_hash": "p", "fingerprint": "f", "event_count": 1, "finished_at": True}
    connection = _Connection(
        fetch=[[], [{"event_id": 1, "frame": bytearray(b"saved")}]], fetchrow=[record]
    )
    journal = _journal(connection)
    snapshot = await journal.get_or_create(
        "key",
        "f",
        "p",
        last_event_id=1,
        max_sessions=2,
        max_response_bytes=1000,
        ttl_seconds=60,
        active_ttl_seconds=60,
    )
    assert not snapshot.created and snapshot.done and snapshot.frames == (b"saved",)

    ahead = _journal(
        _Connection(
            fetch=[[], [{"event_id": 1, "frame": b"saved"}]],
            fetchrow=[record],
        )
    )
    with pytest.raises(ValueError, match="ahead"):
        await ahead.get_or_create(
            "key",
            "f",
            "p",
            last_event_id=2,
            max_sessions=2,
            max_response_bytes=1000,
            ttl_seconds=60,
            active_ttl_seconds=60,
        )

    for value in ("", None):
        with pytest.raises(ValueError, match="identifiers"):
            await journal.get_or_create(
                value,  # type: ignore[arg-type]
                "f",
                "p",
                last_event_id=0,
                max_sessions=2,
                max_response_bytes=1000,
                ttl_seconds=60,
                active_ttl_seconds=60,
            )


@pytest.mark.asyncio
async def test_append_validates_and_bounds_events() -> None:
    for key, frame, max_bytes in (("", b"x", 2), ("key", bytearray(b"x"), 2), ("key", b"x", 0)):
        journal = _journal(_Connection())
        with pytest.raises(ValueError):
            await journal.append(key, frame, max_bytes=max_bytes)  # type: ignore[arg-type]

    for row in (
        None,
        {"event_count": 0, "total_bytes": 0, "finished_at": True},
        {"event_count": 0, "total_bytes": 5, "finished_at": None},
    ):
        assert not await _journal(_Connection(fetchrow=[row])).append("key", b"event", max_bytes=5)

    connection = _Connection(fetchrow=[{"event_count": 1, "total_bytes": 2, "finished_at": None}])
    assert await _journal(connection).append("key", b"event", max_bytes=20)
    assert any("INSERT INTO gabby_stream_events" in query for query, _ in connection.executions)


@pytest.mark.asyncio
async def test_finish_read_reap_remove_and_recover_expired_sessions() -> None:
    connection = _Connection(
        fetch=[
            [{"event_id": 2, "frame": b"two"}],
            [
                {
                    "session_key": "old",
                    "event_count": 0,
                    "total_bytes": 0,
                    "max_response_bytes": 1000,
                },
                {
                    "session_key": "too-small",
                    "event_count": 0,
                    "total_bytes": 999,
                    "max_response_bytes": 1000,
                },
            ],
        ],
        fetchrow=[None, {"finished_at": True}],
    )
    journal = _journal(connection)
    await journal.finish("key")
    assert await journal.read_after("missing", 0) == ([], True)
    assert await journal.read_after("done", 0) == ([b"two"], True)
    with pytest.raises(ValueError, match="cursor"):
        await journal.read_after("key", True)
    await journal.reap(60, 60)
    await journal.remove("key")
    inserted = [
        query for query, _ in connection.executions if "INSERT INTO gabby_stream_events" in query
    ]
    assert len(inserted) == 1


@pytest.mark.asyncio
async def test_invalid_session_identifiers_and_reap_limits() -> None:
    journal = _journal(_Connection())
    for args in ((float("inf"), 1), (1, 0)):
        with pytest.raises(ValueError):
            await journal.reap(*args)
    with pytest.raises(ValueError, match="identifiers"):
        await journal.get_or_create(
            "key",
            "",
            "principal",
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=1000,
            ttl_seconds=1,
            active_ttl_seconds=1,
        )
