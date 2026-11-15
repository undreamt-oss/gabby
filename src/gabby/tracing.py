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
#
# Copyright 2026-present Gabby Contributors.

"""Execution trace data structures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

_TRACE_IO_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gabby-trace-io")

_PARENT_TRACE_ID: ContextVar[str | None] = ContextVar("gabby_parent_trace_id", default=None)
_CURRENT_TRACE_ID: ContextVar[str | None] = ContextVar("gabby_current_trace_id", default=None)


def _safe_tool_name(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
        else "invalid-tool-name"
    )


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


class CompositeTracer:
    """Fan trace events out to multiple independent exporters."""

    def __init__(self, *tracers: Tracer) -> None:
        if not tracers or any(not callable(getattr(item, "on_event", None)) for item in tracers):
            raise ValueError("CompositeTracer requires one or more Tracer implementations")
        self._tracers = tuple(tracers)
        self.errors: dict[str, int] = {}

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Export concurrently while retaining failures as sanitized exception type names."""
        results = await asyncio.gather(
            *(
                tracer.on_event(trace_id=trace_id, agent_name=agent_name, event=event)
                for tracer in self._tracers
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                error_type = type(result).__name__
                self.errors[error_type] = self.errors.get(error_type, 0) + 1


class OpenTelemetryTracer:
    """Export Gabby trace events as OpenTelemetry spans.

    Install ``gabby-ai[opentelemetry]`` and pass an SDK tracer. Only event
    names, durations, and caller-selected safe attributes are exported; event payloads
    are intentionally omitted because they can contain user data or secrets.
    """

    def __init__(self, tracer: Any = None) -> None:
        if tracer is None:
            try:
                from opentelemetry import trace
            except ImportError as exc:
                raise RuntimeError(
                    "OpenTelemetry support requires gabby-ai[opentelemetry]"
                ) from exc
            tracer = trace.get_tracer("gabby")
        if not callable(getattr(tracer, "start_as_current_span", None)):
            raise TypeError("tracer must provide start_as_current_span()")
        self._tracer = tracer

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Record an event span without exporting its potentially sensitive details."""
        span_context: Any = None
        try:
            from opentelemetry import trace

            trace_number = int(trace_id.replace("-", ""), 16)
            if trace_number == 0 or trace_number >= 2**128:
                raise ValueError
            span_context = trace.set_span_in_context(
                trace.NonRecordingSpan(
                    trace.SpanContext(
                        trace_id=trace_number,
                        span_id=uuid.uuid4().int & ((1 << 64) - 1) or 1,
                        is_remote=True,
                        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
                    )
                )
            )
        except (ImportError, ValueError):
            # Injected test/host tracers remain usable without the optional OTel API installed.
            span_context = None
        span_manager = (
            self._tracer.start_as_current_span(f"gabby.{event.kind}", context=span_context)
            if span_context is not None
            else self._tracer.start_as_current_span(f"gabby.{event.kind}")
        )
        with span_manager as span:
            span.set_attribute("gabby.trace_id", trace_id)
            span.set_attribute("gabby.agent", agent_name)
            span.set_attribute("gabby.event", event.kind)
            if event.duration_ms is not None:
                span.set_attribute("gabby.duration_ms", event.duration_ms)


def create_otlp_tracer(
    *,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    service_name: str = "gabby",
) -> tuple[OpenTelemetryTracer, Any]:
    """Create an OTLP/HTTP span exporter and return the tracer plus its owned provider."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("OTLP export requires gabby-ai[opentelemetry]") from exc
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    return OpenTelemetryTracer(provider.get_tracer("gabby")), provider


class SignedAuditExporter:
    """Write hash-chained, Ed25519-signed audit evidence for policy/persona decisions.

    The JSONL format is Gabby's audit format, not a claim of TRACE conformance. The key
    must be protected and chain heads should be anchored outside this process for resistance
    to whole-file replacement. One exporter owns a path at a time; close it after the host
    stops exporting so another worker can acquire the path lock.
    """

    def __init__(self, path: str | Path, private_key: bytes) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError as exc:
            raise RuntimeError("Signed audit export requires gabby-ai[skill-signing]") from exc
        if not isinstance(private_key, bytes) or len(private_key) != 32:
            raise ValueError("private_key must be a 32-byte Ed25519 private key")
        self._key = Ed25519PrivateKey.from_private_bytes(private_key)
        self._path = Path(path).expanduser().resolve()
        self._previous_hash = "0" * 64
        self._write_lock = threading.Lock()
        self._pid = os.getpid()
        self._closed = False
        self._lock_path = self._path.with_name(self._path.name + ".lock")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = self._acquire_file_lock()
        try:
            if self._path.exists() and self._path.stat().st_size:
                self._previous_hash = self._last_hash()
        except BaseException:
            self.close()
            raise

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Append an integrity-protected record; omit prompts, arguments, and model output."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _TRACE_IO_EXECUTOR, self._append_event, trace_id, agent_name, event
        )

    def _append_event(self, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Serialize signing and synchronous filesystem operations off the event loop."""
        if os.getpid() != self._pid:
            raise RuntimeError("Construct SignedAuditExporter after starting worker processes")
        if not isinstance(trace_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id) is None:
            raise ValueError("trace_id must contain only letters, digits, underscores, or hyphens")
        with self._write_lock:
            if self._closed:
                raise RuntimeError("SignedAuditExporter is closed")
            self._append_record(trace_id, agent_name, event)

    def _append_record(self, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        proof: dict[str, Any] = {
            "trace_id": trace_id,
            "agent": agent_name[:128],
            "timestamp": event.timestamp,
            "event": event.kind,
            "previous_hash": self._previous_hash,
        }
        safe_tool_name = _safe_tool_name(event.details.get("name"))
        if event.kind == "tool_call":
            proof["tool"] = safe_tool_name
            proof["policy_decision"] = "allowed"
            proof["tool_outcome"] = "completed"
        elif event.kind == "tool_error":
            proof["tool"] = safe_tool_name
            denied = event.details.get("error_code") == "policy_denied"
            proof["policy_decision"] = "denied" if denied else "unknown"
            proof["tool_outcome"] = "failed"
        elif event.kind == "persona_verification":
            result = event.details.get("result")
            if isinstance(result, dict):
                passed = result.get("passed")
                proof["persona_check"] = (
                    {"passed": passed} if isinstance(passed, bool) else {"status": "invalid-result"}
                )
            else:
                status = (
                    result
                    if isinstance(result, str) and result in {"started", "error", "invalid_result"}
                    else "unknown"
                )
                proof["persona_check"] = {"status": status}
        elif event.kind == "run_failed":
            proof["run_status"] = "failed"
        canonical = json.dumps(proof, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record = {
            **proof,
            "record_hash": digest,
            "signature": self._key.sign(digest.encode()).hex(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._previous_hash = digest

    def _acquire_file_lock(self) -> int:
        """Acquire a cross-process exclusive sidecar lock for this audit path."""
        descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                native_locking = cast(Any, msvcrt)
                native_locking.locking(descriptor, native_locking.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as exc:
            os.close(descriptor)
            raise RuntimeError(
                "Signed audit path already has an active writer or cannot be locked"
            ) from exc
        return descriptor

    def close(self) -> None:
        """Release the path lock after the host has stopped exporting events."""
        with self._write_lock:
            if self._closed:
                return
            self._closed = True
            descriptor = self._lock_fd
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    native_locking = cast(Any, msvcrt)
                    native_locking.locking(descriptor, native_locking.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _last_hash(self) -> str:
        last: dict[str, Any] | None = None
        expected_previous = "0" * 64
        with self._path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                current = json.loads(line)
                signature = current.pop("signature", None)
                digest = current.pop("record_hash", None)
                canonical = json.dumps(
                    current, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if current.get("previous_hash") != expected_previous or digest != computed:
                    raise ValueError("audit file hash chain is invalid")
                try:
                    self._key.public_key().verify(bytes.fromhex(signature), digest.encode())
                except (TypeError, ValueError) as exc:
                    raise ValueError("audit file signature is invalid") from exc
                except Exception as exc:
                    raise ValueError("audit file signature is invalid") from exc
                current["record_hash"] = digest
                last = current
                expected_previous = digest
        if last is None or not isinstance(last.get("record_hash"), str):
            raise ValueError("audit file has no valid chain head")
        return cast(str, last["record_hash"])


class LFTraceExporter:
    """Emit signed LF TRACE v0.2 Trust Records plus Gabby persona/policy evidence.

    ``claims`` must contain deployment-owned TRACE fields such as ``subject``, ``model``,
    ``runtime``, ``policy``, ``data_class``, ``build_provenance``, and ``appraisal``. Gabby
    derives the tool-transcript commitment from policy outcomes. Persona-verifier outcomes are
    written to a separate signed Gabby audit chain because TRACE v0.2 has no persona claim.
    """

    _RESERVED = frozenset({"eat_profile", "iat", "tool_transcript", "signature"})

    def __init__(
        self,
        directory: str | Path,
        *,
        claims: Mapping[str, Any],
        private_key: Any,
        audit_exporter: SignedAuditExporter | None = None,
        max_pending_traces: int = 256,
        max_tool_entries_per_trace: int = 4096,
        pending_trace_ttl_seconds: float = 3600.0,
    ) -> None:
        if not isinstance(claims, Mapping) or self._RESERVED.intersection(claims):
            raise ValueError("claims must be a mapping without TRACE-generated fields")
        self._directory = Path(directory)
        self._claims = deepcopy(dict(claims))
        self._private_key = private_key
        self._audit_exporter = audit_exporter
        if (
            isinstance(max_pending_traces, bool)
            or not isinstance(max_pending_traces, int)
            or not 1 <= max_pending_traces <= 4096
        ):
            raise ValueError("max_pending_traces must be from 1 through 4096")
        if (
            isinstance(max_tool_entries_per_trace, bool)
            or not isinstance(max_tool_entries_per_trace, int)
            or not 1 <= max_tool_entries_per_trace <= 65536
        ):
            raise ValueError("max_tool_entries_per_trace must be from 1 through 65536")
        if (
            isinstance(pending_trace_ttl_seconds, bool)
            or not isinstance(pending_trace_ttl_seconds, (int, float))
            or not math.isfinite(pending_trace_ttl_seconds)
            or not 1 <= pending_trace_ttl_seconds <= 86400
        ):
            raise ValueError("pending_trace_ttl_seconds must be from 1 through 86400")
        self._max_pending_traces = max_pending_traces
        self._max_tool_entries_per_trace = max_tool_entries_per_trace
        self._pending_trace_ttl_seconds = float(pending_trace_ttl_seconds)
        self._tool_entries: dict[str, tuple[float, list[dict[str, str]], bool]] = {}
        self._dropped_traces: set[str] = set()
        self._reject_untracked_traces = False
        self.dropped_trace_count = 0

    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        """Accumulate tool decisions and sign a Trust Record at successful run completion."""
        if not isinstance(trace_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id) is None:
            raise ValueError("trace_id must contain only letters, digits, underscores, or hyphens")
        if self._audit_exporter is not None:
            await self._audit_exporter.on_event(
                trace_id=trace_id, agent_name=agent_name, event=event
            )
        now = time.monotonic()
        expired = [
            pending_id
            for pending_id, (updated_at, _, _) in self._tool_entries.items()
            if now - updated_at > self._pending_trace_ttl_seconds
        ]
        for pending_id in expired:
            self._tool_entries.pop(pending_id, None)
            self._remember_dropped_trace(pending_id)
            self.dropped_trace_count += 1
        terminal = event.kind in {"run_failed", "run_cancelled", "run_timed_out", "run_completed"}
        if trace_id in self._dropped_traces:
            if terminal:
                self._dropped_traces.discard(trace_id)
            return
        if self._reject_untracked_traces and trace_id not in self._tool_entries:
            return
        if event.kind in {"run_failed", "run_cancelled", "run_timed_out"}:
            self._tool_entries.pop(trace_id, None)
            return
        if event.kind == "run_completed":
            state = self._tool_entries.pop(trace_id, None)
            if state is None:
                return
            if state[2]:
                self.dropped_trace_count += 1
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_TRACE_IO_EXECUTOR, self._write_record, trace_id, state[1])
            return

        state = self._tool_entries.get(trace_id)
        if state is None:
            if len(self._tool_entries) >= self._max_pending_traces:
                self._remember_dropped_trace(trace_id)
                self.dropped_trace_count += 1
                return
            state = (now, [], False)
        _, entries, overflowed = state
        if event.kind in {"tool_call", "tool_error"}:
            if len(entries) >= self._max_tool_entries_per_trace:
                overflowed = True
            elif not overflowed:
                tool_entry = {"tool_name": _safe_tool_name(event.details.get("name"))}
                if event.kind == "tool_call":
                    tool_entry["decision"] = "allow"
                else:
                    tool_entry["decision"] = (
                        "deny" if event.details.get("error_code") == "policy_denied" else "unknown"
                    )
                entries.append(tool_entry)
        self._tool_entries[trace_id] = (now, entries, overflowed)

    def _remember_dropped_trace(self, trace_id: str) -> None:
        """Retain bounded tombstones so an omitted run cannot later become a partial record."""
        if trace_id in self._dropped_traces or self._reject_untracked_traces:
            return
        if len(self._dropped_traces) >= self._max_pending_traces:
            # At extreme overload, fail closed for unknown traces rather than risk signing an
            # incomplete transcript. This latch is intentionally process-lifetime state.
            self._reject_untracked_traces = True
            return
        self._dropped_traces.add(trace_id)

    def _write_record(self, trace_id: str, entries: list[dict[str, str]]) -> None:
        try:
            import rfc8785
            from agentrust_trace import sign_record
        except ImportError as exc:
            raise RuntimeError("LF TRACE export requires gabby-ai[trace]") from exc
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id) is None:
            raise ValueError("trace_id must contain only letters, digits, underscores, or hyphens")
        transcript_hash = hashlib.sha256(rfc8785.dumps(entries)).hexdigest()
        record = {
            **deepcopy(self._claims),
            "eat_profile": "tag:agentrust-io.com,2026:trace-v0.2",
            "iat": int(time.time()),
            "tool_transcript": {
                "hash": f"sha256:{transcript_hash}",
                "call_count": len(entries),
            },
        }
        signed = sign_record(record, self._private_key)
        if not isinstance(signed, Mapping):
            raise TypeError("agentrust-trace sign_record() returned an invalid record")
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / f"{trace_id}.trace.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{trace_id}.", suffix=".trace.json.tmp", dir=self._directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(signed, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
