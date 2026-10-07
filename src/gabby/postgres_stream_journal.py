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
"""Shared PostgreSQL storage for resumable SSE execution events."""

from __future__ import annotations

import math
from typing import Any

from .postgres_indexing import AsyncPostgresPool
from .server import StreamJournalSnapshot, _sse_frame


class PostgresStreamJournal:
    """Persist bounded SSE sessions in a host-managed PostgreSQL database.

    Apply ``sql/postgres_stream_journal.sql`` using the host application's migration system.
    The host owns the pool, schema lifecycle, credentials, TLS, backups, and availability policy;
    Gabby does not close the pool. PostgreSQL server time is used for session expiration.
    """

    def __init__(self, pool: AsyncPostgresPool) -> None:
        if not callable(getattr(pool, "acquire", None)):
            raise TypeError("pool must provide async acquire() for a PostgreSQL connection")
        self._pool = pool

    @staticmethod
    def _validate_limits(
        *,
        last_event_id: int,
        max_sessions: int,
        max_response_bytes: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
    ) -> None:
        for name, value, minimum in (
            ("last_event_id", last_event_id, 0),
            ("max_sessions", max_sessions, 1),
            ("max_response_bytes", max_response_bytes, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is outside the supported range")
        for name, duration in (
            ("ttl_seconds", ttl_seconds),
            ("active_ttl_seconds", active_ttl_seconds),
        ):
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")

    async def _recover_expired(self, connection: Any, active_ttl_seconds: float) -> None:
        stale = await connection.fetch(
            "SELECT session_key, event_count, total_bytes, max_response_bytes "
            "FROM gabby_stream_sessions "
            "WHERE finished_at IS NULL AND created_at <= "
            "clock_timestamp() - ($1::double precision * INTERVAL '1 second') "
            "FOR UPDATE SKIP LOCKED",
            float(active_ttl_seconds),
        )
        for row in stale:
            event_id = int(row["event_count"]) + 1
            max_bytes = int(row["max_response_bytes"])
            recovery = _sse_frame(
                "error",
                {
                    "error": "Stream execution ended before completion",
                    "error_type": "StreamRecoveryError",
                },
                max_bytes=max_bytes,
                event_id=event_id,
            )
            count = int(row["event_count"])
            total_bytes = int(row["total_bytes"])
            if total_bytes + len(recovery) <= max_bytes:
                await connection.execute(
                    "INSERT INTO gabby_stream_events(session_key, event_id, frame) "
                    "VALUES ($1, $2, $3)",
                    row["session_key"],
                    event_id,
                    recovery,
                )
                count += 1
                total_bytes += len(recovery)
            await connection.execute(
                "UPDATE gabby_stream_sessions "
                "SET event_count = $2, total_bytes = $3, finished_at = clock_timestamp() "
                "WHERE session_key = $1 AND finished_at IS NULL",
                row["session_key"],
                count,
                total_bytes,
            )

    async def get_or_create(
        self,
        key: str,
        fingerprint: str,
        principal_hash: str,
        *,
        last_event_id: int,
        max_sessions: int,
        max_response_bytes: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
    ) -> StreamJournalSnapshot:
        """Atomically resume a matching session or reserve a bounded new session."""
        self._validate_limits(
            last_event_id=last_event_id,
            max_sessions=max_sessions,
            max_response_bytes=max_response_bytes,
            ttl_seconds=ttl_seconds,
            active_ttl_seconds=active_ttl_seconds,
        )
        if not all(
            isinstance(value, str) and value for value in (key, fingerprint, principal_hash)
        ):
            raise ValueError("stream session identifiers must be non-empty strings")

        async with self._pool.acquire() as connection, connection.transaction():
            # Serialize capacity checks across application replicas.
            await connection.execute("SELECT pg_advisory_xact_lock(741128309221)")
            await self._recover_expired(connection, active_ttl_seconds)
            await connection.execute(
                "DELETE FROM gabby_stream_sessions WHERE finished_at IS NOT NULL "
                "AND finished_at <= clock_timestamp() - "
                "($1::double precision * INTERVAL '1 second')",
                float(ttl_seconds),
            )
            row = await connection.fetchrow(
                "SELECT fingerprint, principal_hash, event_count, finished_at "
                "FROM gabby_stream_sessions WHERE session_key = $1 FOR UPDATE",
                key,
            )
            created = row is None
            if row is None:
                if last_event_id:
                    raise LookupError("stream session not found")
                count = int(await connection.fetchval("SELECT count(*) FROM gabby_stream_sessions"))
                if count >= max_sessions:
                    raise OverflowError("resumable stream capacity is full")
                await connection.execute(
                    "INSERT INTO gabby_stream_sessions "
                    "(session_key, fingerprint, principal_hash, max_response_bytes) "
                    "VALUES ($1, $2, $3, $4)",
                    key,
                    fingerprint,
                    principal_hash,
                    max_response_bytes,
                )
                frames: tuple[bytes, ...] = ()
                done = False
                event_count = 0
            else:
                if row["principal_hash"] != principal_hash:
                    raise LookupError("stream session not found")
                if row["fingerprint"] != fingerprint:
                    raise ValueError("idempotency key was already used with a different request")
                event_count = int(row["event_count"])
                done = row["finished_at"] is not None
                frame_rows = await connection.fetch(
                    "SELECT event_id, frame FROM gabby_stream_events "
                    "WHERE session_key = $1 ORDER BY event_id",
                    key,
                )
                if len(frame_rows) != event_count or any(
                    int(frame["event_id"]) != index
                    for index, frame in enumerate(frame_rows, start=1)
                ):
                    raise RuntimeError("stream journal is inconsistent")
                frames = tuple(bytes(frame["frame"]) for frame in frame_rows)
            if last_event_id > event_count:
                raise ValueError("Last-Event-ID is ahead of this stream")
            return StreamJournalSnapshot(created, frames, done, event_count)

    async def append(self, key: str, frame: bytes, *, max_bytes: int) -> bool:
        """Append one ordered event if the session is active and remains within its byte cap."""
        if not isinstance(key, str) or not key or not isinstance(frame, bytes):
            raise ValueError("stream key and frame are invalid")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        async with self._pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT event_count, total_bytes, finished_at FROM gabby_stream_sessions "
                "WHERE session_key = $1 FOR UPDATE",
                key,
            )
            if (
                row is None
                or row["finished_at"] is not None
                or int(row["total_bytes"]) + len(frame) > max_bytes
            ):
                return False
            event_id = int(row["event_count"]) + 1
            await connection.execute(
                "INSERT INTO gabby_stream_events(session_key, event_id, frame) VALUES ($1, $2, $3)",
                key,
                event_id,
                frame,
            )
            await connection.execute(
                "UPDATE gabby_stream_sessions SET event_count = $2, total_bytes = $3 "
                "WHERE session_key = $1",
                key,
                event_id,
                int(row["total_bytes"]) + len(frame),
            )
            return True

    async def finish(self, key: str) -> None:
        """Mark a session finished; repeated calls are safe."""
        async with self._pool.acquire() as connection:
            await connection.execute(
                "UPDATE gabby_stream_sessions SET finished_at = clock_timestamp() "
                "WHERE session_key = $1 AND finished_at IS NULL",
                key,
            )

    async def read_after(self, key: str, cursor: int) -> tuple[list[bytes], bool]:
        """Read ordered frames after a cursor and return whether the session has finished."""
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        async with self._pool.acquire() as connection, connection.transaction():
            status = await connection.fetchrow(
                "SELECT finished_at FROM gabby_stream_sessions WHERE session_key = $1",
                key,
            )
            if status is None:
                return [], True
            frames = await connection.fetch(
                "SELECT event_id, frame FROM gabby_stream_events "
                "WHERE session_key = $1 AND event_id > $2 ORDER BY event_id",
                key,
                cursor,
            )
            return [bytes(row["frame"]) for row in frames], status["finished_at"] is not None

    async def reap(self, ttl_seconds: float, active_ttl_seconds: float) -> None:
        """Finish abandoned runs and remove completed sessions beyond the retention window."""
        self._validate_limits(
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=1,
            ttl_seconds=ttl_seconds,
            active_ttl_seconds=active_ttl_seconds,
        )
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(741128309221)")
            await self._recover_expired(connection, active_ttl_seconds)
            await connection.execute(
                "DELETE FROM gabby_stream_sessions WHERE finished_at IS NOT NULL "
                "AND finished_at <= clock_timestamp() - "
                "($1::double precision * INTERVAL '1 second')",
                float(ttl_seconds),
            )

    async def remove(self, key: str) -> None:
        """Remove a newly reserved session and its events."""
        async with self._pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM gabby_stream_sessions WHERE session_key = $1",
                key,
            )
