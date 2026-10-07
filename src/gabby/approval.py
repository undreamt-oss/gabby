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
"""Host-injected human approval contracts for sensitive tool calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .auth import Principal


@dataclass(frozen=True)
class ApprovalRequest:
    """Transient request for a host application to approve one tool invocation."""

    agent_name: str
    run_id: str
    tool_name: str
    call_id: str
    arguments: Mapping[str, Any]
    principal: Principal | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """Host decision; Gabby consumes only the approval bit and ignores free-form detail."""

    approved: bool


class ApprovalHandler(Protocol):
    """Host-owned approval UX and decision boundary."""

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """Return an explicit decision before the tool is allowed to execute."""
        ...


class ApprovalAuditSink(Protocol):
    """Host-owned durable sink for approval decisions.

    Implementations should return only after the decision is durably recorded. Gabby supplies the
    validated invocation and decision, but the host chooses storage, retention, access controls,
    and any data minimization beyond the request bounds enforced by Gabby.
    """

    async def record(self, request: ApprovalRequest, *, approved: bool) -> None:
        """Persist one approval outcome before the handler returns it to the runtime."""
        ...
