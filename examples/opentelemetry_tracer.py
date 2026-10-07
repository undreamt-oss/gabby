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
"""Optional OpenTelemetry event exporter for Gabby execution traces.

This example exports one short span per Gabby event and intentionally excludes prompts, model
output, tool arguments, retrieved text, verifier evidence, and arbitrary trace details.
"""

from __future__ import annotations

import math
from typing import Any

from gabby import TraceEvent

_SAFE_DETAIL_KEYS = frozenset(
    {
        "attempt",
        "delay_ms",
        "document_count",
        "input_chars",
        "output_chars",
        "passed",
        "purpose",
        "step",
        "tool_call_count",
    }
)
_ERROR_EVENT_KINDS = frozenset({"model_error", "retrieval_error", "tool_error", "tracer_error"})


def _safe_attribute(value: Any) -> str | bool | int | float | None:
    if isinstance(value, bool | str | int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


class OpenTelemetryEventTracer:
    """Export Gabby events as spans, with a conservative attribute allowlist.

    Install ``gabby-agent-runtime[observability]`` to use the global OpenTelemetry tracer, or
    inject a tracer from the host's configured SDK. The host owns SDK setup and shutdown.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        if tracer is None:
            from opentelemetry import trace

            tracer = trace.get_tracer("gabby")
        self._tracer = tracer

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Export one event without forwarding arbitrary or content-bearing details."""
        attributes: dict[str, str | bool | int | float] = {
            "gabby.trace_id": trace_id,
            "gabby.agent.name": agent_name,
            "gabby.event.kind": event.kind,
        }
        for key in _SAFE_DETAIL_KEYS:
            value = _safe_attribute(event.details.get(key))
            if value is not None:
                attributes[f"gabby.{key}"] = value
        if event.duration_ms is not None and math.isfinite(event.duration_ms):
            attributes["gabby.event.duration_ms"] = event.duration_ms

        end_time = int(event.timestamp * 1_000_000_000)
        start_time = end_time
        if event.duration_ms is not None and math.isfinite(event.duration_ms):
            start_time -= int(max(0.0, event.duration_ms) * 1_000_000)
        span = self._tracer.start_span(
            f"gabby.{event.kind}",
            attributes=attributes,
            start_time=start_time,
        )
        try:
            if event.kind in _ERROR_EVENT_KINDS:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, event.kind))
        finally:
            span.end(end_time=end_time)
