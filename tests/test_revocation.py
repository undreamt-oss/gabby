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
"""Durable SQLite token-revocation contract tests."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from gabby import SQLiteTokenRevocationStore


@pytest.mark.asyncio
async def test_revocations_survive_new_store_instances_and_are_issuer_scoped(
    tmp_path: Path,
) -> None:
    database = tmp_path / "revocations.sqlite3"
    first = SQLiteTokenRevocationStore(database)
    second = SQLiteTokenRevocationStore(database)
    expires_at = 2_000_000_000.0

    assert (
        await first.is_revoked(
            issuer="https://issuer.example/one", token_id="token-1", expires_at=expires_at
        )
        is False
    )
    await first.revoke(
        issuer="https://issuer.example/one", token_id="token-1", expires_at=expires_at
    )

    assert (
        await second.is_revoked(
            issuer="https://issuer.example/one", token_id="token-1", expires_at=expires_at
        )
        is True
    )
    assert (
        await second.is_revoked(
            issuer="https://issuer.example/two", token_id="token-1", expires_at=expires_at
        )
        is False
    )


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX file modes")
async def test_new_database_file_is_created_with_private_permissions(tmp_path: Path) -> None:
    database = tmp_path / "revocations.sqlite3"
    store = SQLiteTokenRevocationStore(database)
    assert not await store.is_revoked(issuer="issuer", token_id="id", expires_at=2_000_000_000.0)

    assert stat.S_IMODE(database.stat().st_mode) & 0o077 == 0


@pytest.mark.asyncio
async def test_revoke_is_idempotent_and_keeps_the_later_expiry(tmp_path: Path) -> None:
    store = SQLiteTokenRevocationStore(tmp_path / "revocations.sqlite3")
    now = time.time()
    await store.revoke(issuer="issuer", token_id="id", expires_at=now + 200.0)
    await store.revoke(issuer="issuer", token_id="id", expires_at=now + 300.0)

    assert await store.is_revoked(issuer="issuer", token_id="id", expires_at=now + 300.0)
    assert await store.cleanup_expired(now=now + 250.0) == 0
    assert await store.cleanup_expired(now=now + 300.0) == 1
    assert not await store.is_revoked(issuer="issuer", token_id="id", expires_at=now + 300.0)


@pytest.mark.asyncio
async def test_concurrent_store_instances_can_revoke_and_check(tmp_path: Path) -> None:
    database = tmp_path / "revocations.sqlite3"
    stores = [SQLiteTokenRevocationStore(database) for _ in range(4)]
    expiry = 2_000_000_000.0

    await asyncio.gather(
        *(
            store.revoke(issuer="issuer", token_id=f"token-{index}", expires_at=expiry)
            for index, store in enumerate(stores)
        )
    )
    assert await asyncio.gather(
        *(
            store.is_revoked(issuer="issuer", token_id=f"token-{index}", expires_at=expiry)
            for index, store in enumerate(stores)
        )
    ) == [True] * len(stores)


@pytest.mark.parametrize(
    ("database", "kwargs", "error"),
    [
        ("", {}, ValueError),
        (":memory:", {}, ValueError),
        ("state.db", {"busy_timeout_seconds": 0}, ValueError),
        ("state.db", {"busy_timeout_seconds": float("inf")}, ValueError),
        (3, {}, TypeError),
    ],
)
def test_store_validates_configuration(
    database: str | Path | int, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        SQLiteTokenRevocationStore(database, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("issuer", "token_id", "expires_at"),
    [
        ("", "id", 100.0),
        ("issuer", "", 100.0),
        ("issuer", "two words", 100.0),
        ("issuer", "x" * 513, 100.0),
        ("issuer", "id", True),
        ("issuer", "id", float("nan")),
    ],
)
async def test_revoke_rejects_invalid_token_identifiers_and_expiry(
    tmp_path: Path, issuer: str, token_id: str, expires_at: float | bool
) -> None:
    store = SQLiteTokenRevocationStore(tmp_path / "revocations.sqlite3")
    with pytest.raises(ValueError):
        await store.revoke(issuer=issuer, token_id=token_id, expires_at=expires_at)
