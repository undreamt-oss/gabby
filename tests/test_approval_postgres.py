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
"""Contract tests for the host-client PostgreSQL approval audit sink."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from gabby import ApprovalRequest, Principal
from gabby.approval_postgres import PostgresApprovalAudit
from gabby.approval_sqlite import ApprovalAuditError


class FakePostgresClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.failure: Exception | None = None

    async def execute(self, query: str, *args: object) -> str:
        if self.failure is not None:
            raise self.failure
        self.calls.append((query, args))
        return "INSERT 0 1"


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        agent_name="reports",
        run_id="run-1",
        tool_name="send_report",
        call_id="call-1",
        arguments={"recipient": "private@example.test"},
        principal=Principal(subject="reviewed-user"),
    )


@pytest.mark.asyncio
async def test_postgres_sink_inserts_digest_without_raw_arguments() -> None:
    client = FakePostgresClient()
    sink = PostgresApprovalAudit(client)
    request = _request()

    await sink.record(request, approved=True)

    assert len(client.calls) == 1
    query, args = client.calls[0]
    canonical = json.dumps(
        request.arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert "VALUES ($1, $2, $3, $4, $5, $6, $7)" in query
    assert args == (
        "run-1",
        "call-1",
        "reports",
        "send_report",
        None,
        hashlib.sha256(canonical).hexdigest(),
        True,
    )
    assert "private@example.test" not in repr(args)
    assert "reviewed-user" not in repr(args)


@pytest.mark.asyncio
async def test_postgres_sink_subject_storage_is_opt_in() -> None:
    client = FakePostgresClient()
    sink = PostgresApprovalAudit(client, include_principal_subject=True)

    await sink.record(_request(), approved=False)

    assert client.calls[0][1][4] == "reviewed-user"
    assert client.calls[0][1][-1] is False


@pytest.mark.asyncio
async def test_postgres_sink_sanitizes_backend_failures() -> None:
    client = FakePostgresClient()
    client.failure = RuntimeError("password and DSN must not escape")
    sink = PostgresApprovalAudit(client)

    with pytest.raises(ApprovalAuditError, match="Could not write approval audit record") as error:
        await sink.record(_request(), approved=True)

    assert error.value.__cause__ is None
    assert "password" not in str(error.value)


@pytest.mark.asyncio
async def test_postgres_sink_rejects_oversized_arguments_before_database_call() -> None:
    client = FakePostgresClient()
    sink = PostgresApprovalAudit(client, max_argument_bytes=8)

    with pytest.raises(ApprovalAuditError, match="audit size limit"):
        await sink.record(_request(), approved=True)

    assert not client.calls


@pytest.mark.asyncio
async def test_postgres_sink_validates_records_and_redacts_invalid_arguments() -> None:
    client = FakePostgresClient()
    sink = PostgresApprovalAudit(client)
    with pytest.raises(TypeError, match="ApprovalRequest"):
        await sink.record(object(), approved=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="approved"):
        await sink.record(_request(), approved=1)  # type: ignore[arg-type]
    with pytest.raises(ApprovalAuditError, match="run_id"):
        await sink.record(replace(_request(), run_id=""), approved=True)
    with pytest.raises(ApprovalAuditError, match="agent_name"):
        await sink.record(replace(_request(), agent_name="\ud800"), approved=True)
    with pytest.raises(ApprovalAuditError, match="safely audited"):
        await sink.record(replace(_request(), arguments={"number": float("nan")}), approved=True)
    assert not client.calls


@pytest.mark.asyncio
async def test_postgres_sink_bounds_and_validates_opt_in_principal_subject() -> None:
    client = FakePostgresClient()
    sink = PostgresApprovalAudit(client, include_principal_subject=True)
    with pytest.raises(ApprovalAuditError, match="exceeds its size limit"):
        await sink.record(
            replace(_request(), principal=Principal(subject="x" * 4097)), approved=True
        )
    with pytest.raises(ApprovalAuditError, match="subject is invalid"):
        await sink.record(replace(_request(), principal=Principal(subject="\ud800")), approved=True)
    assert not client.calls


def test_postgres_sink_validates_configuration_and_executor() -> None:
    with pytest.raises(TypeError, match="async execute"):
        PostgresApprovalAudit(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_principal_subject"):
        PostgresApprovalAudit(FakePostgresClient(), include_principal_subject=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_argument_bytes"):
        PostgresApprovalAudit(FakePostgresClient(), max_argument_bytes=0)
