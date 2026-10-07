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
"""Execution trace data structures."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

_PARENT_TRACE_ID: ContextVar[str | None] = ContextVar("gabby_parent_trace_id", default=None)


@dataclass
class TraceEvent:
    """One timestamped runtime event with structured details."""

    kind: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Transient trace for one agent execution."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, kind: str, *, duration_ms: float | None = None, **details: Any) -> TraceEvent:
        """Append one event, keeping its duration in the typed duration field."""
        event = TraceEvent(kind=kind, duration_ms=duration_ms, details=details)
        self.events.append(event)
        return event

    def as_dict(self) -> dict[str, Any]:
        """Return the trace as a JSON-compatible mapping."""
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "events": [asdict(event) for event in self.events],
            "metadata": self.metadata,
        }


class Tracer(Protocol):
    """Host-supplied async exporter; runs are ordered internally but may interleave."""

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Export one event; callbacks can overlap across runs and should avoid blocking."""
        ...
