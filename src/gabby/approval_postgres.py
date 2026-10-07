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
"""Optional PostgreSQL-backed approval audit sink over a host-owned async client."""

from __future__ import annotations

from typing import Protocol

from ._approval_audit import (
    DEFAULT_MAX_AUDIT_ARGUMENT_BYTES,
    MAX_AUDIT_ID_BYTES,
    MAX_AUDIT_NAME_BYTES,
    MAX_AUDIT_SUBJECT_BYTES,
    ApprovalAuditError,
    ArgumentSizeError,
    canonical_argument_digest,
)
from .approval import ApprovalAuditSink, ApprovalRequest


class AsyncPostgresExecutor(Protocol):
    """Minimal asyncpg-compatible surface for a host-owned PostgreSQL client or pool."""

    async def execute(self, query: str, *args: object) -> str:
        """Execute one parameterized statement and return the backend status string."""
        ...


_INSERT_APPROVAL = """
INSERT INTO gabby_tool_approval_audit (
    run_id, call_id, agent_name, tool_name, principal_subject,
    arguments_sha256, approved
) VALUES ($1, $2, $3, $4, $5, $6, $7)
"""


class PostgresApprovalAudit(ApprovalAuditSink):
    """Persist bounded approval metadata through a host-owned async PostgreSQL executor.

    The adapter does not create tables, construct the client, own its credentials, or close its
    pool. The host applies the schema documented in ``docs/APPROVALS.md`` and owns database access,
    durability, backups, and retention. Raw tool arguments are never stored.
    """

    def __init__(
        self,
        client: AsyncPostgresExecutor,
        *,
        include_principal_subject: bool = False,
        max_argument_bytes: int = DEFAULT_MAX_AUDIT_ARGUMENT_BYTES,
    ) -> None:
        if not callable(getattr(client, "execute", None)):
            raise TypeError("client must provide async execute(query, *args)")
        if not isinstance(include_principal_subject, bool):
            raise ValueError("include_principal_subject must be a boolean")
        if (
            isinstance(max_argument_bytes, bool)
            or not isinstance(max_argument_bytes, int)
            or not 1 <= max_argument_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("max_argument_bytes must be from 1 through 16777216")
        self._client = client
        self._include_principal_subject = include_principal_subject
        self._max_argument_bytes = max_argument_bytes

    async def record(self, request: ApprovalRequest, *, approved: bool) -> None:
        """Insert one decision; duplicate invocation IDs fail closed at the primary key."""
        if not isinstance(request, ApprovalRequest):
            raise TypeError("request must be an ApprovalRequest")
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean")
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
        try:
            arguments_sha256 = canonical_argument_digest(
                request.arguments,
                max_bytes=self._max_argument_bytes,
            )
        except ArgumentSizeError as exc:
            raise ApprovalAuditError(str(exc)) from None
        except ValueError:
            raise ApprovalAuditError("Tool arguments cannot be safely audited") from None
        principal_subject = (
            request.principal.subject
            if self._include_principal_subject and request.principal is not None
            else None
        )
        try:
            if (
                principal_subject is not None
                and len(principal_subject.encode("utf-8")) > MAX_AUDIT_SUBJECT_BYTES
            ):
                raise ApprovalAuditError("Approval audit principal subject exceeds its size limit")
        except UnicodeEncodeError:
            raise ApprovalAuditError("Approval audit principal subject is invalid") from None
        try:
            await self._client.execute(
                _INSERT_APPROVAL,
                request.run_id,
                request.call_id,
                request.agent_name,
                request.tool_name,
                principal_subject,
                arguments_sha256,
                approved,
            )
        except ApprovalAuditError:
            raise
        except Exception:
            raise ApprovalAuditError("Could not write approval audit record") from None
