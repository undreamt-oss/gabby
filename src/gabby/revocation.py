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
"""Durable local JWT revocation storage."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from contextlib import closing
from os import PathLike, fspath
from pathlib import Path
from typing import cast

from ._sync import run_sync_callback
from .auth import TokenRevocationChecker


class SQLiteTokenRevocationStore(TokenRevocationChecker):
    """A durable SQLite implementation of ``TokenRevocationChecker``.

    Operations use a short-lived connection in a worker thread, so SQLite file I/O does not block
    the event loop. Multiple store instances and processes may share the same database file. The
    database path and its filesystem permissions are owned by the host application.
    """

    def __init__(
        self,
        database: str | PathLike[str],
        *,
        busy_timeout_seconds: float = 1.0,
    ) -> None:
        if not isinstance(database, (str, PathLike)):
            raise TypeError("database must be a filesystem path")
        path = fspath(database)
        if not isinstance(path, str) or not path.strip() or path == ":memory:":
            raise ValueError("database must be a non-empty file path")
        if (
            isinstance(busy_timeout_seconds, bool)
            or not isinstance(busy_timeout_seconds, (int, float))
            or not math.isfinite(busy_timeout_seconds)
            or not 0 < busy_timeout_seconds <= 30
        ):
            raise ValueError("busy_timeout_seconds must be greater than 0 and at most 30")
        self._database = str(Path(path))
        self._busy_timeout_seconds = float(busy_timeout_seconds)
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    async def is_revoked(self, *, issuer: str, token_id: str, expires_at: float) -> bool:
        """Return whether this issuer and token ID have an unexpired revocation record."""
        _validate_token_key(issuer, token_id)
        _validate_timestamp(expires_at, "expires_at")
        return cast(bool, await run_sync_callback(self._is_revoked, issuer, token_id))

    async def revoke(self, *, issuer: str, token_id: str, expires_at: float) -> None:
        """Persist a revocation until the token's expiration; repeating the call is safe."""
        _validate_token_key(issuer, token_id)
        _validate_timestamp(expires_at, "expires_at")
        await run_sync_callback(self._revoke, issuer, token_id, float(expires_at))

    async def cleanup_expired(self, *, now: float | None = None) -> int:
        """Delete expired records and return the number removed."""
        cutoff = time.time() if now is None else _validate_timestamp(now, "now")
        return cast(int, await run_sync_callback(self._cleanup_expired, float(cutoff)))

    def _connect(self) -> sqlite3.Connection:
        try:
            file_descriptor = os.open(
                self._database,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(file_descriptor)
        connection = sqlite3.connect(
            self._database,
            timeout=self._busy_timeout_seconds,
            isolation_level="IMMEDIATE",
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_seconds * 1000)}")
            if not self._schema_ready:
                with self._schema_lock:
                    if not self._schema_ready:
                        connection.execute(
                            """CREATE TABLE IF NOT EXISTS revoked_tokens (
                                issuer TEXT NOT NULL,
                                token_id TEXT NOT NULL,
                                expires_at REAL NOT NULL,
                                PRIMARY KEY (issuer, token_id)
                            )"""
                        )
                        connection.execute(
                            "CREATE INDEX IF NOT EXISTS revoked_tokens_expiry "
                            "ON revoked_tokens(expires_at)"
                        )
                        connection.commit()
                        self._schema_ready = True
        except Exception:
            connection.close()
            raise
        return connection

    def _is_revoked(self, issuer: str, token_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT expires_at FROM revoked_tokens WHERE issuer = ? AND token_id = ?",
                (issuer, token_id),
            ).fetchone()
        if row is None:
            return False
        stored_expiry = row[0]
        return isinstance(stored_expiry, (int, float)) and stored_expiry > time.time()

    def _revoke(self, issuer: str, token_id: str, expires_at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO revoked_tokens (issuer, token_id, expires_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT (issuer, token_id) DO UPDATE SET
                       expires_at = MAX(revoked_tokens.expires_at, excluded.expires_at)""",
                (issuer, token_id, expires_at),
            )

    def _cleanup_expired(self, now: float) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute("DELETE FROM revoked_tokens WHERE expires_at <= ?", (now,))
            return cursor.rowcount


def _validate_token_key(issuer: str, token_id: str) -> None:
    if not isinstance(issuer, str) or not issuer or len(issuer) > 2048:
        raise ValueError("issuer must be a non-empty string of at most 2048 characters")
    if (
        not isinstance(token_id, str)
        or not token_id
        or not token_id.isascii()
        or len(token_id) > 512
        or any(ord(char) < 33 or ord(char) > 126 for char in token_id)
    ):
        raise ValueError("token_id must contain 1 to 512 visible ASCII characters")


def _validate_timestamp(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite timestamp")
    return float(value)
