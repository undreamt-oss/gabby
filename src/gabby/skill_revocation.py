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
"""Durable local revocation storage for skill publisher keys."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from contextlib import closing
from os import PathLike, fspath
from pathlib import Path
from typing import cast

from ._sync import run_sync_callback
from .skill_trust import SkillRevocationStore, SkillRevokedError

_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_MAX_CHECK_KEY_IDS = 1024


class SQLiteSkillRevocationStore(SkillRevocationStore):
    """SQLite signer revocations checked on every execution for trusted skills.

    Instances and processes may share the database. Use a local filesystem that provides SQLite
    locking semantics; distributed propagation and backup remain host responsibilities.
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

    async def check_not_revoked(self, key_ids: frozenset[str]) -> None:
        """Raise `SkillRevokedError` if any requested publisher ID is currently revoked."""
        keys = _validate_key_ids(key_ids)
        if not keys:
            return
        revoked = await run_sync_callback(self._find_revoked, keys)
        if revoked:
            raise SkillRevokedError("A skill publisher has been revoked")

    async def revoke(self, key_id: str) -> None:
        """Persist a publisher revocation immediately for subsequent run checks."""
        _validate_key_id(key_id)
        await run_sync_callback(self._revoke, key_id)

    async def reinstate(self, key_id: str) -> bool:
        """Remove a publisher revocation and return whether a record was removed."""
        _validate_key_id(key_id)
        return cast(bool, await run_sync_callback(self._reinstate, key_id))

    async def revoked_key_ids(self) -> frozenset[str]:
        """Return the complete current revocation set for operator inspection."""
        return cast(frozenset[str], await run_sync_callback(self._list_revoked))

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
                            """CREATE TABLE IF NOT EXISTS revoked_skill_signers (
                                key_id TEXT PRIMARY KEY,
                                revoked_at REAL NOT NULL
                            )"""
                        )
                        connection.commit()
                        self._schema_ready = True
        except Exception:
            connection.close()
            raise
        return connection

    def _find_revoked(self, key_ids: frozenset[str]) -> frozenset[str]:
        placeholders = ",".join("?" for _ in key_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT key_id FROM revoked_skill_signers WHERE key_id IN ({placeholders})",
                tuple(sorted(key_ids)),
            ).fetchall()
        return frozenset(row[0] for row in rows if isinstance(row[0], str))

    def _revoke(self, key_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO revoked_skill_signers (key_id, revoked_at)
                   VALUES (?, ?) ON CONFLICT (key_id) DO NOTHING""",
                (key_id, time.time()),
            )

    def _reinstate(self, key_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM revoked_skill_signers WHERE key_id = ?", (key_id,)
            )
            return cursor.rowcount > 0

    def _list_revoked(self) -> frozenset[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT key_id FROM revoked_skill_signers").fetchall()
        return frozenset(row[0] for row in rows if isinstance(row[0], str))


def _validate_key_ids(value: Iterable[str]) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise ValueError("skill signer checks must provide a collection of key IDs")
    try:
        key_ids = frozenset(value)
    except TypeError:
        raise ValueError("skill signer checks must provide an iterable of key IDs") from None
    if len(key_ids) > _MAX_CHECK_KEY_IDS:
        raise ValueError("skill signer checks exceed the key ID limit")
    for key_id in key_ids:
        _validate_key_id(key_id)
    return key_ids


def _validate_key_id(value: object) -> None:
    if not isinstance(value, str) or _KEY_ID.fullmatch(value) is None:
        raise ValueError("skill signer key IDs use an invalid format")
