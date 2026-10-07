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
"""Contract tests for the optional content-minimizing SQLite approval audit."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from gabby import ApprovalDecision, ApprovalRequest, Principal
from gabby.approval_sqlite import (
    ApprovalAuditError,
    AuditedApprovalHandler,
    SQLiteApprovalAudit,
)


def _request(*, subject: str = "user-1") -> ApprovalRequest:
    return ApprovalRequest(
        agent_name="reports",
        run_id="run-1",
        tool_name="send_report",
        call_id="call-1",
        arguments={"recipient": "private@example.test"},
        principal=Principal(subject=subject),
    )


@pytest.mark.asyncio
async def test_audit_omits_raw_arguments_and_principal_by_default(tmp_path: Path) -> None:
    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3")
    request = _request()
    canonical = json.dumps(
        request.arguments,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    await audit.record(request, approved=True)

    records = await audit.list_records()
    assert len(records) == 1
    record = records[0]
    assert record.arguments_sha256 == hashlib.sha256(canonical).hexdigest()
    assert record.approved is True
    assert record.principal_subject is None
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        table_data = connection.execute("SELECT * FROM gabby_tool_approval_audit").fetchone()
    assert b"private@example.test" not in repr(table_data).encode()


@pytest.mark.asyncio
async def test_audit_can_opt_in_to_principal_subject_and_bounds_listing(tmp_path: Path) -> None:
    audit = SQLiteApprovalAudit(
        tmp_path / "audit.sqlite3",
        include_principal_subject=True,
    )
    await audit.record(_request(), approved=False)
    await audit.record(
        ApprovalRequest(
            agent_name="reports",
            run_id="run-2",
            tool_name="send_report",
            call_id="call-2",
            arguments={},
        ),
        approved=True,
    )

    records = await audit.list_records(limit=1)
    assert len(records) == 1
    assert records[0].run_id == "run-2"
    assert records[0].principal_subject is None
    with pytest.raises(ValueError, match="limit"):
        await audit.list_records(limit=1001)


@pytest.mark.asyncio
async def test_audited_handler_records_decision_before_returning(tmp_path: Path) -> None:
    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3")
    seen: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    handler = AuditedApprovalHandler(approve, audit)
    decision = await handler.approve(_request())

    assert decision == ApprovalDecision(approved=True)
    assert len(seen) == 1
    assert (await audit.list_records())[0].approved is True


@pytest.mark.asyncio
async def test_audited_handler_accepts_host_owned_async_audit_sink() -> None:
    class SharedAuditSink:
        def __init__(self) -> None:
            self.entries: list[tuple[ApprovalRequest, bool]] = []

        async def record(self, request: ApprovalRequest, *, approved: bool) -> None:
            self.entries.append((request, approved))

    sink = SharedAuditSink()

    async def approve(_: ApprovalRequest) -> bool:
        return True

    handler = AuditedApprovalHandler(approve, sink)
    request = _request()
    assert await handler.approve(request) == ApprovalDecision(approved=True)
    assert sink.entries == [(request, True)]


@pytest.mark.asyncio
async def test_audited_handler_fails_closed_when_host_sink_fails() -> None:
    class UnavailableAuditSink:
        async def record(self, _: ApprovalRequest, *, approved: bool) -> None:
            raise RuntimeError("private backend details")

    async def approve(_: ApprovalRequest) -> bool:
        return True

    handler = AuditedApprovalHandler(approve, UnavailableAuditSink())
    with pytest.raises(RuntimeError, match="private backend details"):
        await handler.approve(_request())


@pytest.mark.asyncio
async def test_duplicate_invocation_ids_and_invalid_review_results_fail_closed(
    tmp_path: Path,
) -> None:
    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3")
    request = _request()
    await audit.record(request, approved=True)
    with pytest.raises(ApprovalAuditError, match="Could not write"):
        await audit.record(request, approved=True)

    async def invalid_review(_request: ApprovalRequest) -> bool:
        return "yes"  # type: ignore[return-value]

    handler = AuditedApprovalHandler(invalid_review, audit)
    with pytest.raises(TypeError, match="must return bool"):
        await handler.approve(
            ApprovalRequest(
                agent_name="reports",
                run_id="run-2",
                tool_name="send_report",
                call_id="call-2",
                arguments={},
            )
        )


@pytest.mark.asyncio
async def test_audit_failure_fails_closed_and_does_not_leak_database_error(tmp_path: Path) -> None:
    database_directory = tmp_path / "directory.sqlite3"
    database_directory.mkdir()
    audit = SQLiteApprovalAudit(database_directory)

    async def review(_request: ApprovalRequest) -> bool:
        return True

    handler = AuditedApprovalHandler(review, audit)
    with pytest.raises(ApprovalAuditError, match="Could not write approval audit record"):
        await handler.approve(_request())


@pytest.mark.asyncio
async def test_oversized_or_non_json_arguments_are_not_recorded(tmp_path: Path) -> None:
    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3", max_argument_bytes=8)
    with pytest.raises(ApprovalAuditError, match="size limit"):
        await audit.record(
            ApprovalRequest(
                agent_name="reports",
                run_id="run-1",
                tool_name="send_report",
                call_id="call-1",
                arguments={"value": "too large"},
            ),
            approved=True,
        )
    assert await audit.list_records() == []

    with pytest.raises(ValueError, match="max_argument_bytes"):
        SQLiteApprovalAudit(tmp_path / "invalid.sqlite3", max_argument_bytes=0)

    with pytest.raises(ApprovalAuditError, match="cannot be safely audited"):
        await SQLiteApprovalAudit(tmp_path / "nan.sqlite3").record(
            ApprovalRequest(
                agent_name="reports",
                run_id="run-nan",
                tool_name="send_report",
                call_id="call-nan",
                arguments={"value": float("nan")},
            ),
            approved=True,
        )


@pytest.mark.asyncio
async def test_audit_configuration_and_record_arguments_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty path"):
        SQLiteApprovalAudit("")
    with pytest.raises(ValueError, match="must be a boolean"):
        SQLiteApprovalAudit(
            tmp_path / "invalid.sqlite3",
            include_principal_subject=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        SQLiteApprovalAudit(tmp_path / "invalid.sqlite3", busy_timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        SQLiteApprovalAudit(tmp_path / "invalid.sqlite3", busy_timeout_seconds=31)
    with pytest.raises(ValueError, match="max_argument_bytes"):
        SQLiteApprovalAudit(tmp_path / "invalid.sqlite3", max_argument_bytes=True)

    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3")
    with pytest.raises(TypeError, match="ApprovalRequest"):
        await audit.record(None, approved=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="approved must be a boolean"):
        await audit.record(_request(), approved=1)  # type: ignore[arg-type]
    with pytest.raises(ApprovalAuditError, match="run_id is invalid"):
        await audit.record(
            ApprovalRequest(
                agent_name="reports",
                run_id="r" * 513,
                tool_name="send_report",
                call_id="call-oversized",
                arguments={},
            ),
            approved=True,
        )


@pytest.mark.asyncio
async def test_audit_list_error_is_sanitized_and_handler_contract_is_checked(
    tmp_path: Path,
) -> None:
    directory_path = tmp_path / "database-directory"
    directory_path.mkdir()
    audit = SQLiteApprovalAudit(directory_path)
    with pytest.raises(ApprovalAuditError, match="Could not read approval audit records"):
        await audit.list_records()

    async def review(_request: ApprovalRequest) -> bool:
        return True

    with pytest.raises(TypeError, match="review must be callable"):
        AuditedApprovalHandler(None, audit)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="audit must provide an async record"):
        AuditedApprovalHandler(review, object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cancelled_audit_wait_consumes_late_worker_failure(tmp_path: Path) -> None:
    audit = SQLiteApprovalAudit(tmp_path / "audit.sqlite3")
    started = threading.Event()
    release = threading.Event()

    def delayed_failure() -> None:
        started.set()
        release.wait(timeout=2)
        raise RuntimeError("private audit storage detail")

    task = asyncio.create_task(audit._run_sync(delayed_failure))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0.1)
