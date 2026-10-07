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
"""Host-owned approval UX backed by Gabby's content-minimizing SQLite audit helper."""

from __future__ import annotations

import asyncio

from gabby import (
    ApprovalRequest,
    AuditedApprovalHandler,
    SQLiteApprovalAudit,
)

__all__ = ["AuditedApprovalHandler", "SQLiteApprovalAudit"]


async def example_review(request: ApprovalRequest) -> bool:
    """Stand in for an application UI; production reviewers must be authenticated."""
    print(
        f"Approval needed: agent={request.agent_name} tool={request.tool_name} "
        f"call={request.call_id} subject="
        f"{request.principal.subject if request.principal is not None else 'embedded-caller'}"
    )
    return False


async def main() -> None:
    audit = SQLiteApprovalAudit("approval-audit.sqlite3")
    handler = AuditedApprovalHandler(example_review, audit)
    request = ApprovalRequest(
        agent_name="report-agent",
        run_id="demo-run",
        tool_name="send_report",
        call_id="demo-call",
        arguments={"recipient": "finance@example.test"},
    )
    decision = await handler.approve(request)
    print(f"Recorded sample decision: approved={decision.approved}")
    print("Inject this handler into Agent(..., approval_handler=handler) in your application.")


if __name__ == "__main__":
    asyncio.run(main())
