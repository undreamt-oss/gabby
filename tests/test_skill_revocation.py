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
"""Durable publisher signer revocation contract tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from gabby import SkillRevocationStore, SkillRevokedError, SQLiteSkillRevocationStore


@pytest.mark.asyncio
async def test_sqlite_skill_revocation_is_durable_and_reinstatable(tmp_path: Path) -> None:
    database = tmp_path / "trust" / "revocations.sqlite3"
    database.parent.mkdir()
    store = SQLiteSkillRevocationStore(database)
    await store.check_not_revoked(frozenset())
    await store.check_not_revoked(frozenset({"current-publisher"}))

    await store.revoke("old-publisher")
    assert await store.revoked_key_ids() == frozenset({"old-publisher"})

    restarted = SQLiteSkillRevocationStore(database)
    assert isinstance(restarted, SkillRevocationStore)
    with pytest.raises(SkillRevokedError):
        await restarted.check_not_revoked(frozenset({"current-publisher", "old-publisher"}))

    assert await restarted.reinstate("old-publisher") is True
    assert await restarted.reinstate("old-publisher") is False
    await restarted.check_not_revoked(frozenset({"old-publisher"}))


@pytest.mark.asyncio
async def test_sqlite_skill_revocation_coordinates_concurrent_processes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shared-trust" / "revocations.sqlite3"
    database.parent.mkdir()
    stores = [SQLiteSkillRevocationStore(database) for _ in range(4)]
    writer = (
        "import asyncio,sys; from gabby import SQLiteSkillRevocationStore; "
        "asyncio.run(SQLiteSkillRevocationStore(sys.argv[1], busy_timeout_seconds=5).revoke("
        "'shared-publisher'))"
    )
    processes: list[subprocess.Popen[str]] = []
    try:
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", writer, str(database)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in stores
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, f"concurrent writer failed: {stdout} {stderr}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    assert await asyncio.gather(*(store.revoked_key_ids() for store in stores)) == [
        frozenset({"shared-publisher"})
    ] * len(stores)

    reinstated = await asyncio.gather(*(store.reinstate("shared-publisher") for store in stores))
    assert sum(reinstated) == 1
    await asyncio.gather(
        *(store.check_not_revoked(frozenset({"shared-publisher"})) for store in stores)
    )


@pytest.mark.parametrize(
    "key_id",
    ["", "contains space", "../publisher", "a" * 129],
)
@pytest.mark.asyncio
async def test_sqlite_skill_revocation_rejects_invalid_key_ids(tmp_path: Path, key_id: str) -> None:
    store = SQLiteSkillRevocationStore(tmp_path / "revocations.sqlite3")
    with pytest.raises(ValueError, match="invalid format"):
        await store.revoke(key_id)


def test_sqlite_skill_revocation_validates_store_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty file path"):
        SQLiteSkillRevocationStore(":memory:")
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        SQLiteSkillRevocationStore(tmp_path / "revocations.sqlite3", busy_timeout_seconds=0)
