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
"""Optional SQLite audit support for host-owned human approval flows."""

from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ._approval_audit import (
    DEFAULT_MAX_AUDIT_ARGUMENT_BYTES,
    MAX_AUDIT_ID_BYTES,
    MAX_AUDIT_NAME_BYTES,
    MAX_AUDIT_QUERY_LIMIT,
    MAX_AUDIT_SUBJECT_BYTES,
    ArgumentSizeError,
    canonical_argument_digest,
)
from ._approval_audit import (
    ApprovalAuditError as ApprovalAuditError,
)
from ._sync import run_sync_callback
from .approval import ApprovalAuditSink, ApprovalDecision, ApprovalRequest

ReviewCallback = Callable[[ApprovalRequest], Awaitable[bool]]


@dataclass(frozen=True)
class ApprovalAuditRecord:
    """Content-minimizing record for one host approval decision."""

    run_id: str
    call_id: str
    agent_name: str
    tool_name: str
    arguments_sha256: str
    approved: bool
    created_at: float
    principal_subject: str | None = None


class SQLiteApprovalAudit:
    """Store bounded approval metadata in an application-selected SQLite database.

    Raw tool arguments are never stored. The host owns the database path, retention, backups, and
    access control. Principal subjects are omitted unless explicitly enabled.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_principal_subject: bool = False,
        busy_timeout_seconds: float = 5.0,
        max_argument_bytes: int = DEFAULT_MAX_AUDIT_ARGUMENT_BYTES,
    ) -> None:
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("approval audit path must be a non-empty path")
        if not isinstance(include_principal_subject, bool):
            raise ValueError("include_principal_subject must be a boolean")
        if (
            isinstance(busy_timeout_seconds, bool)
            or not isinstance(busy_timeout_seconds, (int, float))
            or not math.isfinite(busy_timeout_seconds)
            or not 0 < busy_timeout_seconds <= 30
        ):
            raise ValueError("busy_timeout_seconds must be greater than 0 and at most 30")
        if (
            isinstance(max_argument_bytes, bool)
            or not isinstance(max_argument_bytes, int)
            or not 1 <= max_argument_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("max_argument_bytes must be from 1 through 16777216")
        self._path = str(path)
        self._include_principal_subject = include_principal_subject
        self._busy_timeout_seconds = float(busy_timeout_seconds)
        self._max_argument_bytes = max_argument_bytes

    async def record(self, request: ApprovalRequest, *, approved: bool) -> None:
        """Persist a decision without blocking the caller's event loop."""
        if not isinstance(request, ApprovalRequest):
            raise TypeError("request must be an ApprovalRequest")
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean")
        try:
            await self._run_sync(self._record_sync, request, approved)
        except ApprovalAuditError:
            raise
        except Exception:
            raise ApprovalAuditError("Could not write approval audit record") from None

    async def list_records(self, *, limit: int = 100) -> list[ApprovalAuditRecord]:
        """Return recent records, newest first, with a caller-bounded result count."""
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_AUDIT_QUERY_LIMIT
        ):
            raise ValueError(f"limit must be from 1 through {MAX_AUDIT_QUERY_LIMIT}")
        try:
            return cast(list[ApprovalAuditRecord], await self._run_sync(self._list_sync, limit))
        except ApprovalAuditError:
            raise
        except Exception:
            raise ApprovalAuditError("Could not read approval audit records") from None

    async def _run_sync(self, callback: Callable[..., Any], *args: Any) -> Any:
        """Run bounded SQLite work off-loop and poll completion for embedded-loop robustness."""
        worker = asyncio.create_task(run_sync_callback(callback, *args))
        try:
            while not worker.done():
                await asyncio.wait({worker}, timeout=0.05)
            return await worker
        except asyncio.CancelledError:
            worker.add_done_callback(self._consume_worker_result)
            raise

    @staticmethod
    def _consume_worker_result(worker: asyncio.Task[Any]) -> None:
        """Retrieve late worker failures after the awaiting approval run is cancelled."""
        if not worker.cancelled():
            worker.exception()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_seconds,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_seconds * 1000)}")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS gabby_tool_approval_audit (
                    run_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    principal_subject TEXT,
                    arguments_sha256 TEXT NOT NULL,
                    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, call_id)
                )
                """
            )
        except BaseException:
            connection.close()
            raise
        return connection

    def _arguments_digest(self, arguments: Any) -> str:
        try:
            return canonical_argument_digest(arguments, max_bytes=self._max_argument_bytes)
        except ArgumentSizeError as exc:
            raise ApprovalAuditError(str(exc)) from None
        except ValueError:
            raise ApprovalAuditError("Tool arguments cannot be safely audited") from None

    def _record_sync(self, request: ApprovalRequest, approved: bool) -> None:
        for label, value, maximum in (
            ("run_id", request.run_id, MAX_AUDIT_ID_BYTES),
            ("call_id", request.call_id, MAX_AUDIT_ID_BYTES),
            ("agent_name", request.agent_name, MAX_AUDIT_NAME_BYTES),
            ("tool_name", request.tool_name, MAX_AUDIT_NAME_BYTES),
        ):
            try:
                valid = (
                    isinstance(value, str) and bool(value) and len(value.encode("utf-8")) <= maximum
                )
            except UnicodeEncodeError:
                valid = False
            if not valid:
                raise ApprovalAuditError(f"Approval audit {label} is invalid or oversized")
        arguments_digest = self._arguments_digest(request.arguments)
        principal_subject = (
            request.principal.subject
            if self._include_principal_subject and request.principal is not None
            else None
        )
        if (
            principal_subject is not None
            and len(principal_subject.encode("utf-8")) > MAX_AUDIT_SUBJECT_BYTES
        ):
            raise ApprovalAuditError("Approval audit principal subject exceeds its size limit")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO gabby_tool_approval_audit (
                        run_id, call_id, agent_name, tool_name, principal_subject,
                        arguments_sha256, approved, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.run_id,
                        request.call_id,
                        request.agent_name,
                        request.tool_name,
                        principal_subject,
                        arguments_digest,
                        int(approved),
                        time.time(),
                    ),
                )
        except sqlite3.Error:
            raise ApprovalAuditError("Could not write approval audit record") from None

    def _list_sync(self, limit: int) -> list[ApprovalAuditRecord]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT run_id, call_id, agent_name, tool_name, principal_subject,
                           arguments_sha256, approved, created_at
                    FROM gabby_tool_approval_audit
                    ORDER BY created_at DESC, run_id DESC, call_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error:
            raise ApprovalAuditError("Could not read approval audit records") from None
        return [
            ApprovalAuditRecord(
                run_id=row["run_id"],
                call_id=row["call_id"],
                agent_name=row["agent_name"],
                tool_name=row["tool_name"],
                principal_subject=row["principal_subject"],
                arguments_sha256=row["arguments_sha256"],
                approved=bool(row["approved"]),
                created_at=float(row["created_at"]),
            )
            for row in rows
        ]


class AuditedApprovalHandler:
    """Connect a host-owned approval callback to an awaited audit sink."""

    def __init__(self, review: ReviewCallback, audit: ApprovalAuditSink) -> None:
        if not callable(review):
            raise TypeError("review must be callable")
        if not callable(getattr(audit, "record", None)):
            raise TypeError("audit must provide an async record(request, *, approved) method")
        self._review = review
        self._audit = audit

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask the host for a decision and durably record it before returning."""
        approved = await self._review(request)
        if not isinstance(approved, bool):
            raise TypeError("approval callback must return bool")
        await self._audit.record(request, approved=approved)
        return ApprovalDecision(approved=approved)
