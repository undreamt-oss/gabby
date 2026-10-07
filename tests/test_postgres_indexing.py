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
"""Protocol contract tests for the host-pool PostgreSQL generation manifest."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from gabby import (
    GenerationBusyError,
    KnowledgeStoreError,
    PostgresGenerationManifestStore,
)


class _Connection:
    def __init__(
        self,
        *,
        fetch: list[list[dict[str, Any]]] | None = None,
        fetchrow: list[dict[str, Any] | None] | None = None,
        fetchval: list[Any] | None = None,
        fail: bool = False,
    ) -> None:
        self.fetch_results = deque(fetch or [])
        self.fetchrow_results = deque(fetchrow or [])
        self.fetchval_results = deque(fetchval or [])
        self.fail = fail
        self.executions: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args))
        if self.fail:
            raise RuntimeError("private database details")
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        del query, args
        return self.fetch_results.popleft() if self.fetch_results else []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        del query, args
        return self.fetchrow_results.popleft() if self.fetchrow_results else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        del query, args
        return self.fetchval_results.popleft() if self.fetchval_results else None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_Connection]:
        yield self.connection


def _store(connection: _Connection) -> PostgresGenerationManifestStore:
    return PostgresGenerationManifestStore(_Pool(connection))


def _manifest_row(**changes: Any) -> dict[str, Any]:
    row = {
        "fencing_token": 4,
        "active_generation": None,
        "pending_generation": None,
        "pending_document_count": 3,
        "pending_lexical_ready": True,
        "pending_vector_ready": True,
        "pending_owner_id": "worker",
        "pending_lease_expires_at": 99.0,
        "expires_at": 99.0,
        "pending_created_at": None,
        "pending_error_type": None,
        "lease_expires_at": 99.0,
        "lease_valid": True,
        "renewed": True,
        "marked": True,
    }
    row.update(changes)
    return row


@pytest.mark.asyncio
async def test_begin_source_validation_and_creation() -> None:
    with pytest.raises(TypeError, match="acquire"):
        PostgresGenerationManifestStore(object())  # type: ignore[arg-type]
    for source, generation in (("", "g"), ("s", " ")):
        with pytest.raises(ValueError):
            await _store(_Connection()).begin_source(source, generation, 0, owner_id="worker")
    with pytest.raises(TypeError, match="document_count"):
        await _store(_Connection()).begin_source("s", "g", True, owner_id="worker")
    with pytest.raises(ValueError, match="document_count"):
        await _store(_Connection()).begin_source("s", "g", -1, owner_id="worker")

    connection = _Connection(
        fetchrow=[
            _manifest_row(fencing_token=4),
            _manifest_row(fencing_token=5, expires_at=42.0),
        ],
        fetchval=[False],
    )
    lease = await _store(connection).begin_source("source", "generation", 7, owner_id="worker")
    assert (lease.source, lease.generation, lease.fencing_token, lease.expires_at) == (
        "source",
        "generation",
        5,
        42.0,
    )


@pytest.mark.asyncio
async def test_begin_source_rejects_pending_active_retired_and_missing_rows() -> None:
    with pytest.raises(GenerationBusyError):
        await _store(
            _Connection(fetchrow=[_manifest_row(pending_generation="older")])
        ).begin_source("s", "g", 1, owner_id="worker")
    with pytest.raises(KnowledgeStoreError, match="operation failed"):
        await _store(_Connection(fetchrow=[_manifest_row(active_generation="g")])).begin_source(
            "s", "g", 1, owner_id="worker"
        )
    with pytest.raises(KnowledgeStoreError, match="operation failed"):
        await _store(_Connection(fetchrow=[_manifest_row()], fetchval=[True])).begin_source(
            "s", "g", 1, owner_id="worker"
        )
    with pytest.raises(KnowledgeStoreError, match="lock"):
        await _store(_Connection(fetchrow=[None])).begin_source("s", "g", 1, owner_id="worker")
    with pytest.raises(KnowledgeStoreError, match="lease"):
        await _store(_Connection(fetchrow=[_manifest_row(), None], fetchval=[False])).begin_source(
            "s", "g", 1, owner_id="worker"
        )


@pytest.mark.asyncio
async def test_lease_renewal_claim_and_readiness_transitions() -> None:
    assert await _store(_Connection(fetchrow=[_manifest_row()])).renew_lease(
        "s", "g", "worker", 4, 3
    )
    assert not await _store(_Connection(fetchrow=[None])).renew_lease("s", "g", "worker", 4, 3)
    claimed = await _store(
        _Connection(fetchrow=[_manifest_row(fencing_token=5, expires_at=42)])
    ).claim_expired("s", "g", owner_id="next", lease_ttl_seconds=5)
    assert claimed is not None and claimed.fencing_token == 5 and claimed.expires_at == 42
    assert (
        await _store(_Connection(fetchrow=[None])).claim_expired(
            "s", "g", owner_id="next", lease_ttl_seconds=5
        )
        is None
    )
    with pytest.raises(ValueError, match="backend"):
        await _store(_Connection()).mark_ready("s", "g", "other", owner_id="w", fencing_token=1)  # type: ignore[arg-type]
    await _store(_Connection(fetchrow=[_manifest_row()])).mark_ready(
        "s", "g", "lexical", owner_id="worker", fencing_token=4
    )
    with pytest.raises(KnowledgeStoreError, match="ready"):
        await _store(_Connection(fetchrow=[None])).mark_ready(
            "s", "g", "vector", owner_id="worker", fencing_token=4
        )


@pytest.mark.asyncio
async def test_activation_validates_staged_state_and_retires_previous_generation() -> None:
    with pytest.raises(KnowledgeStoreError, match="missing"):
        await _store(_Connection(fetchrow=[None])).activate_source(
            "s", "g", owner_id="worker", fencing_token=4
        )
    await _store(_Connection(fetchrow=[_manifest_row(active_generation="g")])).activate_source(
        "s", "g", owner_id="worker", fencing_token=4
    )
    incomplete = _manifest_row(pending_generation="g", pending_vector_ready=False)
    with pytest.raises(KnowledgeStoreError, match="incompletely"):
        await _store(_Connection(fetchrow=[incomplete])).activate_source(
            "s", "g", owner_id="worker", fencing_token=4
        )
    prior = _manifest_row(active_generation="old", pending_generation="g")
    connection = _Connection(fetchrow=[prior])
    await _store(connection).activate_source("s", "g", owner_id="worker", fencing_token=4)
    assert any(
        "INSERT INTO gabby_index_retired_generations" in query for query, _ in connection.executions
    )
    assert any(
        "active_generation = pending_generation" in query for query, _ in connection.executions
    )
    no_prior = _manifest_row(pending_generation="g")
    no_prior_connection = _Connection(fetchrow=[no_prior])
    await _store(no_prior_connection).activate_source("s", "g", owner_id="worker", fencing_token=4)
    assert not any(
        "INSERT INTO gabby_index_retired_generations" in query
        for query, _ in no_prior_connection.executions
    )


@pytest.mark.asyncio
async def test_mark_error_and_pending_retired_active_reads() -> None:
    connection = _Connection(
        fetch=[
            [
                {
                    "source": "s",
                    "generation": "g",
                    "pending_document_count": 2,
                    "pending_lexical_ready": 1,
                    "pending_vector_ready": 0,
                    "pending_owner_id": None,
                    "fencing_token": 8,
                    "lease_expires_at": None,
                    "pending_error_type": "ValueError",
                }
            ],
            [{"source": "s", "generation": "old"}],
            [{"source": "s", "active_generation": "g"}],
        ],
        fetchval=[17],
    )
    store = _store(connection)
    await store.mark_error("s", "g", "privaté", owner_id="worker", fencing_token=4)
    assert connection.executions[-1][1][-1] == "ExtensionError"
    pending = await store.pending_generations()
    assert (
        pending[0].owner_id is None
        and pending[0].lease_expires_at == 0
        and pending[0].error_type == "ValueError"
    )
    assert await store.retired_generations() == [("s", "old")]
    assert await store.active_generations() == {"s": "g"}
    assert await store.active_document_count("s") == 17
    assert await _store(_Connection(fetchval=[None])).active_document_count("none") == 0


@pytest.mark.asyncio
async def test_abandon_release_retired_cleanup_and_error_redaction() -> None:
    with pytest.raises(KnowledgeStoreError, match="lease"):
        await _store(_Connection(fetchrow=[None])).abandon_pending(
            "s", "g", owner_id="worker", fencing_token=4
        )
    connection = _Connection(fetchrow=[_manifest_row()])
    await _store(connection).abandon_pending("s", "g", owner_id="worker", fencing_token=4)
    assert any("pending_generation = NULL" in query for query, _ in connection.executions)

    cleanup = _Connection()
    store = _store(cleanup)
    await store.release_lease("s", "g", owner_id="worker", fencing_token=4)
    await store.remove_retired("s", "g")
    assert any(
        "pending_lease_expires_at = clock_timestamp()" in query for query, _ in cleanup.executions
    )
    with pytest.raises(KnowledgeStoreError, match="operation failed") as captured:
        await _store(_Connection(fail=True)).remove_retired("s", "g")
    assert "private database details" not in str(captured.value)
