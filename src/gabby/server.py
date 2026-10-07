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
"""FastAPI transport for stateless agent execution."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import sqlite3
import stat
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .agent import Agent
from .auth import Authenticator, BearerTokenAuthenticator, Principal
from .runtime import MAX_RUN_INPUT_CHARS, AgentRuntimeError, StreamEventType
from .skill_trust import SkillIntegrityError, SkillRevocationUnavailable, SkillRevokedError

_bearer_security = HTTPBearer(auto_error=False)
_SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0
DEFAULT_MAX_HTTP_REQUEST_BYTES = 1_000_000
DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS = 30.0
DEFAULT_AUTHENTICATOR_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_HTTP_RESPONSE_BYTES = 4 * 1024 * 1024
_MIN_HTTP_RESPONSE_BYTES = 256
_RESPONSE_TOO_LARGE = {
    "error": "HTTP response size limit exceeded",
    "error_type": "ResponseSizeLimitError",
}
_JournalResult = TypeVar("_JournalResult")


def _is_positive_finite_number(value: object) -> bool:
    """Return whether a value is a finite positive int or float, including huge integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _validate_required_scopes(value: object, *, name: str) -> frozenset[str]:
    """Validate one route's immutable required-scope configuration."""
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of scope tokens")
    try:
        scopes = frozenset(value)
        Principal("scope-validation", scopes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid scope tokens") from exc
    if len(scopes) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return scopes


class RunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1, max_length=MAX_RUN_INPUT_CHARS)
    context: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    include_trace: bool = True


class TraceEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    timestamp: float
    duration_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionTracePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    started_at: float
    events: list[TraceEventPayload]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str
    trace: ExecutionTracePayload | None = None


class AgentErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    error_type: str


class AgentErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: AgentErrorDetail


class RequestBodyErrorResponse(BaseModel):
    """Error returned before routing when the request body exceeds its bounds."""

    model_config = ConfigDict(extra="forbid")

    detail: str


class StreamEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: StreamEventType
    data: dict[str, Any]


class ResponseSizeLimitError(ValueError):
    """A serialized HTTP response would exceed the configured body limit."""


def _public_execution_error_type(error: Exception) -> str:
    """Return a stable Gabby error type without exposing extension class names."""
    if isinstance(error, SkillIntegrityError):
        return "SkillIntegrityError"
    if isinstance(error, SkillRevokedError):
        return "SkillRevokedError"
    if isinstance(error, SkillRevocationUnavailable):
        return "SkillRevocationUnavailable"
    if isinstance(error, AgentRuntimeError):
        return "AgentRuntimeError"
    return "AgentExecutionError"


def _public_execution_error_message(error: Exception) -> str:
    if isinstance(error, SkillIntegrityError):
        return "agent skill integrity check failed"
    if isinstance(error, SkillRevokedError):
        return "agent skill publisher is revoked"
    if isinstance(error, SkillRevocationUnavailable):
        return "skill revocation check is unavailable"
    return "agent execution failed"


def _bounded_json_bytes(value: Any, *, max_bytes: int, default: Any = str) -> bytes:
    """Serialize JSON while retaining no more than the configured byte limit."""
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, default=default, separators=(",", ":")
    )
    chunks: list[bytes] = []
    byte_count = 0
    for chunk in encoder.iterencode(value):
        remaining = max_bytes - byte_count
        if len(chunk) > remaining:
            raise ResponseSizeLimitError
        encoded = chunk.encode("utf-8")
        if len(encoded) > remaining:
            raise ResponseSizeLimitError
        chunks.append(encoded)
        byte_count += len(encoded)
    return b"".join(chunks)


def _sse_frame(
    event_type: StreamEventType,
    data: dict[str, Any],
    *,
    max_bytes: int,
    event_id: int | None = None,
) -> bytes:
    """Serialize and bound one complete SSE event frame."""
    envelope = StreamEventPayload(type=event_type, data=data)
    prefix = f"id: {event_id}\n".encode("ascii") if event_id is not None else b""
    prefix += f"event: {event_type}\ndata: ".encode("ascii")
    suffix = b"\n\n"
    payload = _bounded_json_bytes(
        envelope.model_dump(), max_bytes=max_bytes - len(prefix) - len(suffix)
    )
    return prefix + payload + suffix


class BodySizeLimitMiddleware:
    """Admit bounded requests before buffering, then enforce body size and read time."""

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        timeout_seconds: float = DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS,
        run_capacity: _RunCapacity | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.run_capacity = run_capacity if run_capacity is not None else _RunCapacity(8)
        self.resumable_request_capacity = _RunCapacity(self.run_capacity.limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") == "/health" and scope.get("method") == "GET":
            await self.app(scope, receive, send)
            return

        is_resumable_request = (
            scope.get("method") == "POST"
            and str(scope.get("path", "")).endswith("/stream")
            and any(name.lower() == b"idempotency-key" for name, _ in scope.get("headers", []))
        )
        if is_resumable_request:
            # Bound request buffering and authentication separately from the execution
            # slots held by resumable runs, so reattachment can work at full run capacity.
            request_lease = self.resumable_request_capacity.try_acquire()
            if request_lease is None:
                body = json.dumps(
                    {
                        "detail": {
                            "error": "request capacity exhausted",
                            "error_type": "CapacityLimitError",
                        }
                    },
                    separators=(",", ":"),
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"retry-after", b"1"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return

            async def release_request_slot_on_start(message: Message) -> None:
                if message["type"] == "http.response.start":
                    request_lease.release()
                await send(message)

            try:
                await self._buffer_and_dispatch(scope, receive, release_request_slot_on_start)
            finally:
                request_lease.release()
            return

        lease = self.run_capacity.try_acquire()
        if lease is None:
            body = json.dumps(
                {
                    "detail": {
                        "error": "execution capacity exhausted",
                        "error_type": "CapacityLimitError",
                    }
                },
                separators=(",", ":"),
            ).encode()
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", b"1"),
            ]
            if scope.get("http_version") in (None, "1.0", "1.1"):
                headers.append((b"connection", b"close"))
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": headers,
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        state = scope.setdefault("state", {})
        state["_gabby_run_capacity_lease"] = lease
        try:
            await self._buffer_and_dispatch(scope, receive, send)
        finally:
            state.pop("_gabby_run_capacity_lease", None)
            if not state.pop("_gabby_run_capacity_lease_transferred", False):
                lease.release()

    async def _buffer_and_dispatch(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Read and replay one bounded request body before entering FastAPI."""

        async def reject_oversized() -> None:
            body = json.dumps({"detail": "request body too large"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        async def reject_timeout() -> None:
            body = json.dumps({"detail": "request body timed out"}).encode()
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]
            if scope.get("http_version") in (None, "1.0", "1.1"):
                headers.append((b"connection", b"close"))
            await send(
                {
                    "type": "http.response.start",
                    "status": 408,
                    "headers": headers,
                }
            )
            await send({"type": "http.response.body", "body": body})

        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                declared_bytes = int(value)
            except ValueError:
                continue
            if declared_bytes > self.max_bytes:
                await reject_oversized()
                return

        buffered_body = bytearray()
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                await reject_timeout()
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError:
                await reject_timeout()
                return
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if len(buffered_body) + len(chunk) > self.max_bytes:
                    await reject_oversized()
                    return
                buffered_body.extend(chunk)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        replayed_body = False

        async def replay() -> Message:
            nonlocal replayed_body
            if not replayed_body:
                replayed_body = True
                return {
                    "type": "http.request",
                    "body": bytes(buffered_body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


class ResponseSizeLimitMiddleware:
    """Bound ordinary responses and guard streaming bodies at the ASGI boundary."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_start: Message | None = None
        buffered_body = bytearray()
        streamed_bytes = 0
        streaming = False
        oversized = False
        stream_finished = False

        async def bounded_send(message: Message) -> None:
            nonlocal response_start, streamed_bytes, streaming, oversized, stream_finished
            if message["type"] == "http.response.start":
                response_start = message
                streaming = any(
                    name.lower() == b"content-type"
                    and value.lower().startswith(b"text/event-stream")
                    for name, value in message.get("headers", [])
                )
                if streaming:
                    await send(message)
                return
            if message["type"] != "http.response.body":
                await send(message)
                return

            chunk = message.get("body", b"")
            if streaming:
                if stream_finished:
                    return
                if streamed_bytes + len(chunk) > self.max_bytes:
                    stream_finished = True
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return
                streamed_bytes += len(chunk)
                await send(message)
                return

            if not oversized:
                if len(buffered_body) + len(chunk) > self.max_bytes:
                    oversized = True
                    buffered_body.clear()
                else:
                    buffered_body.extend(chunk)
            if message.get("more_body", False):
                return

            if response_start is None:
                await send(message)
                return
            if oversized:
                body = _bounded_json_bytes(
                    {"detail": _RESPONSE_TOO_LARGE}, max_bytes=self.max_bytes
                )
                headers = [
                    (name, value)
                    for name, value in response_start.get("headers", [])
                    if name.lower()
                    not in {
                        b"content-length",
                        b"content-type",
                        b"content-encoding",
                        b"content-range",
                        b"transfer-encoding",
                    }
                ]
                headers.extend(
                    [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ]
                )
                await send({"type": "http.response.start", "status": 500, "headers": headers})
                await send({"type": "http.response.body", "body": body, "more_body": False})
                return

            await send(response_start)
            await send(
                {
                    "type": "http.response.body",
                    "body": bytes(buffered_body),
                    "more_body": False,
                }
            )

        await self.app(scope, receive, bounded_send)


class _RunCapacityLease:
    """One execution slot owned by the service instance."""

    def __init__(self, capacity: _RunCapacity) -> None:
        self._capacity = capacity
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._capacity.active -= 1


class _RunCapacity:
    """Non-blocking per-process admission control, shared by run and stream routes."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> _RunCapacityLease | None:
        if self.active >= self.limit:
            return None
        self.active += 1
        return _RunCapacityLease(self)


@dataclass(frozen=True, slots=True)
class StreamJournalSnapshot:
    """Atomic journal lookup result returned by a resumable-stream backend."""

    created: bool
    frames: tuple[bytes, ...]
    done: bool
    event_count: int

    def __post_init__(self) -> None:
        if type(self.created) is not bool or type(self.done) is not bool:
            raise ValueError("stream journal snapshot flags must be booleans")
        if not isinstance(self.frames, tuple) or any(
            not isinstance(frame, bytes) for frame in self.frames
        ):
            raise ValueError("stream journal snapshot frames must be a tuple of bytes")
        if (
            isinstance(self.event_count, bool)
            or not isinstance(self.event_count, int)
            or self.event_count != len(self.frames)
        ):
            raise ValueError("stream journal snapshot event_count must match its frames")
        if self.created and (self.frames or self.done):
            raise ValueError("new stream journal snapshots cannot contain prior events")


@runtime_checkable
class StreamJournal(Protocol):
    """Storage contract for bounded, resumable SSE execution journals.

    Implementations own backend lifecycle. ``get_or_create`` must atomically bind a key to the
    request fingerprint and principal hash, enforce the shared session cap, and reject cursors
    beyond the stored event count. Raise ``LookupError`` for a missing resumed session,
    ``ValueError`` for request/cursor conflicts, and ``OverflowError`` at capacity. Event IDs are
    contiguous starting at one. ``append`` must
    atomically enforce the byte cap and reject writes after ``finish``. ``read_after`` returns
    ordered frames strictly after the cursor and whether the session is finished.
    """

    async def get_or_create(
        self,
        key: str,
        fingerprint: str,
        principal_hash: str,
        *,
        last_event_id: int,
        max_sessions: int,
        max_response_bytes: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
    ) -> StreamJournalSnapshot:
        """Atomically resume a matching session or reserve a new session key."""
        ...

    async def append(self, key: str, frame: bytes, *, max_bytes: int) -> bool:
        """Atomically append the next ordered frame unless the session is finished or full."""
        ...

    async def finish(self, key: str) -> None:
        """Mark an existing session finished; repeated calls must be safe."""
        ...

    async def read_after(self, key: str, cursor: int) -> tuple[list[bytes], bool]:
        """Return ordered frames after ``cursor`` and whether the session is finished."""
        ...

    async def reap(self, ttl_seconds: float, active_ttl_seconds: float) -> None:
        """Expire completed sessions and finish abandoned sessions with recovery events."""
        ...

    async def remove(self, key: str) -> None:
        """Remove a newly reserved session when request admission cannot proceed."""
        ...


@dataclass
class _StreamSession:
    """One bounded stream event journal, optionally backed by shared SQLite storage."""

    fingerprint: str
    subject: str
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    frames: list[bytes] = field(default_factory=list)
    total_bytes: int = 0
    done: bool = False
    finished_at: float | None = None
    task: asyncio.Task[None] | None = None
    journal: StreamJournal | None = None
    key: str | None = None

    async def append(self, frame: bytes, *, max_bytes: int) -> bool:
        """Append one event if the session journal remains within its configured bound."""
        async with self.condition:
            if self.done or self.total_bytes + len(frame) > max_bytes:
                return False
            if (
                self.journal is not None
                and self.key is not None
                and not await self.journal.append(self.key, frame, max_bytes=max_bytes)
            ):
                return False
            self.frames.append(frame)
            self.total_bytes += len(frame)
            self.condition.notify_all()
            return True

    async def finish(self) -> None:
        async with self.condition:
            if self.done:
                return
            if self.journal is not None and self.key is not None:
                await self.journal.finish(self.key)
            self.done = True
            self.finished_at = asyncio.get_running_loop().time()
            self.condition.notify_all()

    async def replay(self, last_event_id: int) -> AsyncIterator[bytes]:
        """Yield journal frames after a validated numeric event cursor."""
        cursor = last_event_id
        while True:
            if self.journal is not None and self.key is not None:
                try:
                    frames, done = await self.journal.read_after(self.key, cursor)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    yield _sse_frame(
                        "error",
                        {
                            "error": "resumable stream journal unavailable",
                            "error_type": "StreamJournalUnavailable",
                        },
                        max_bytes=4096,
                    )
                    return
                if frames:
                    for frame in frames:
                        cursor += 1
                        yield frame
                    continue
                if done:
                    return
                await asyncio.sleep(0.1)
                continue
            async with self.condition:
                while cursor >= len(self.frames) and not self.done:
                    try:
                        await asyncio.wait_for(
                            self.condition.wait(), timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS
                        )
                    except TimeoutError:
                        break
                if cursor < len(self.frames):
                    frame = self.frames[cursor]
                    cursor += 1
                elif self.done:
                    return
                else:
                    frame = b": keepalive\n\n"
            yield frame


class _SQLiteStreamJournal:
    """Durable bounded event rows shared by API workers using one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gabby-sse-journal")

    async def _call(
        self, operation: Callable[..., _JournalResult], /, *args: Any, **kwargs: Any
    ) -> _JournalResult:
        loop = asyncio.get_running_loop()
        result = loop.run_in_executor(self._executor, partial(operation, *args, **kwargs))
        while not result.done():
            await asyncio.wait({result}, timeout=0.1)
        return result.result()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("stream journal path must be a regular file")
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS gabby_stream_sessions (
                    session_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    subject_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    total_bytes INTEGER NOT NULL DEFAULT 0,
                    finished_at REAL,
                    max_response_bytes INTEGER NOT NULL DEFAULT 4194304
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS gabby_stream_events (
                    session_key TEXT NOT NULL REFERENCES gabby_stream_sessions(session_key)
                        ON DELETE CASCADE,
                    event_id INTEGER NOT NULL,
                    frame BLOB NOT NULL,
                    PRIMARY KEY (session_key, event_id)
                )"""
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(gabby_stream_sessions)")
            }
            if "created_at" not in columns:
                connection.execute(
                    "ALTER TABLE gabby_stream_sessions "
                    "ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
                )
            if "max_response_bytes" not in columns:
                connection.execute(
                    "ALTER TABLE gabby_stream_sessions "
                    "ADD COLUMN max_response_bytes INTEGER NOT NULL DEFAULT 4194304"
                )

    @staticmethod
    def _recover_expired_sync(
        connection: sqlite3.Connection, *, now: float, active_ttl_seconds: float
    ) -> None:
        """Finish abandoned sessions with a terminal error rather than allowing duplicate runs."""
        cutoff = now - active_ttl_seconds
        stale = connection.execute(
            "SELECT session_key, frame_count, total_bytes, max_response_bytes "
            "FROM gabby_stream_sessions "
            "WHERE finished_at IS NULL AND created_at<=?",
            (cutoff,),
        ).fetchall()
        for key, frame_count, total_bytes, max_response_bytes in stale:
            terminal = _sse_frame(
                "error",
                {
                    "error": "Stream execution ended before completion",
                    "error_type": "StreamRecoveryError",
                },
                max_bytes=int(max_response_bytes),
                event_id=int(frame_count) + 1,
            )
            next_count = int(frame_count)
            next_bytes = int(total_bytes)
            if next_bytes + len(terminal) <= int(max_response_bytes):
                next_count += 1
                next_bytes += len(terminal)
                connection.execute(
                    "INSERT INTO gabby_stream_events(session_key, event_id, frame) "
                    "VALUES (?, ?, ?)",
                    (key, next_count, terminal),
                )
            connection.execute(
                "UPDATE gabby_stream_sessions SET frame_count=?, total_bytes=?, finished_at=? "
                "WHERE session_key=? AND finished_at IS NULL",
                (next_count, next_bytes, now, key),
            )

    async def initialize(self) -> None:
        await self._call(self._initialize_sync)

    def _get_or_create_sync(
        self,
        key: str,
        fingerprint: str,
        subject: str,
        *,
        last_event_id: int,
        max_sessions: int,
        max_response_bytes: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
    ) -> StreamJournalSnapshot:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_sync(connection, now=now, active_ttl_seconds=active_ttl_seconds)
            connection.execute(
                "DELETE FROM gabby_stream_sessions "
                "WHERE finished_at IS NOT NULL AND finished_at<=?",
                (now - ttl_seconds,),
            )
            row = connection.execute(
                "SELECT fingerprint, subject_hash, frame_count, finished_at "
                "FROM gabby_stream_sessions WHERE session_key=?",
                (key,),
            ).fetchone()
            created = row is None
            if row is None:
                if last_event_id:
                    raise LookupError("stream session not found")
                count = connection.execute("SELECT COUNT(*) FROM gabby_stream_sessions").fetchone()[
                    0
                ]
                if count >= max_sessions:
                    raise OverflowError("resumable stream capacity is full")
                connection.execute(
                    "INSERT INTO gabby_stream_sessions "
                    "(session_key, fingerprint, subject_hash, created_at, max_response_bytes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, fingerprint, subject, now, max_response_bytes),
                )
                frames: tuple[bytes, ...] = ()
                done = False
            else:
                if row[1] != subject:
                    raise LookupError("stream session not found")
                if row[0] != fingerprint:
                    raise ValueError("idempotency key was already used with a different request")
                count = int(row[2])
                done = row[3] is not None
                frames = tuple(
                    bytes(item[0])
                    for item in connection.execute(
                        "SELECT frame FROM gabby_stream_events "
                        "WHERE session_key=? ORDER BY event_id",
                        (key,),
                    )
                )
                if len(frames) != count:
                    raise RuntimeError("stream journal is inconsistent")
            if last_event_id > count:
                raise ValueError("Last-Event-ID is ahead of this stream")
            connection.commit()
            return StreamJournalSnapshot(created, frames, done, count)

    async def get_or_create(
        self,
        key: str,
        fingerprint: str,
        subject: str,
        *,
        last_event_id: int,
        max_sessions: int,
        max_response_bytes: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
    ) -> StreamJournalSnapshot:
        return await self._call(
            self._get_or_create_sync,
            key,
            fingerprint,
            subject,
            last_event_id=last_event_id,
            max_sessions=max_sessions,
            max_response_bytes=max_response_bytes,
            ttl_seconds=ttl_seconds,
            active_ttl_seconds=active_ttl_seconds,
        )

    def _append_sync(self, key: str, frame: bytes, max_bytes: int) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT frame_count, total_bytes, finished_at "
                "FROM gabby_stream_sessions WHERE session_key=?",
                (key,),
            ).fetchone()
            if row is None or row[2] is not None or int(row[1]) + len(frame) > max_bytes:
                connection.rollback()
                return False
            event_id = int(row[0]) + 1
            connection.execute(
                "INSERT INTO gabby_stream_events(session_key, event_id, frame) VALUES (?, ?, ?)",
                (key, event_id, frame),
            )
            connection.execute(
                "UPDATE gabby_stream_sessions SET frame_count=?, total_bytes=? WHERE session_key=?",
                (event_id, int(row[1]) + len(frame), key),
            )
            connection.commit()
            return True

    async def append(self, key: str, frame: bytes, *, max_bytes: int) -> bool:
        return await self._call(self._append_sync, key, frame, max_bytes)

    def _finish_sync(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE gabby_stream_sessions SET finished_at=? "
                "WHERE session_key=? AND finished_at IS NULL",
                (time.time(), key),
            )

    async def finish(self, key: str) -> None:
        await self._call(self._finish_sync, key)

    def _read_after_sync(self, key: str, cursor: int) -> tuple[list[bytes], bool]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT frame FROM gabby_stream_events "
                "WHERE session_key=? AND event_id>? ORDER BY event_id",
                (key, cursor),
            ).fetchall()
            status = connection.execute(
                "SELECT finished_at FROM gabby_stream_sessions WHERE session_key=?", (key,)
            ).fetchone()
            return [bytes(row[0]) for row in rows], status is None or status[0] is not None

    async def read_after(self, key: str, cursor: int) -> tuple[list[bytes], bool]:
        return await self._call(self._read_after_sync, key, cursor)

    def _reap_sync(self, ttl_seconds: float, active_ttl_seconds: float) -> None:
        with self._connect() as connection:
            now = time.time()
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired_sync(connection, now=now, active_ttl_seconds=active_ttl_seconds)
            connection.execute(
                "DELETE FROM gabby_stream_sessions "
                "WHERE finished_at IS NOT NULL AND finished_at<=?",
                (now - ttl_seconds,),
            )

    async def reap(self, ttl_seconds: float, active_ttl_seconds: float) -> None:
        await self._call(self._reap_sync, ttl_seconds, active_ttl_seconds)

    def _remove_sync(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM gabby_stream_sessions WHERE session_key=?", (key,))

    async def remove(self, key: str) -> None:
        await self._call(self._remove_sync, key)

    def close(self) -> None:
        """Drain and stop the journal's bounded worker pool after stream producers finish."""
        self._executor.shutdown(wait=True, cancel_futures=True)


class _StreamSessionStore:
    """Bounded session manager with in-memory or shared SQLite event persistence."""

    def __init__(
        self,
        *,
        max_sessions: int,
        ttl_seconds: float,
        active_ttl_seconds: float,
        max_response_bytes: int,
        journal: StreamJournal | None = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.active_ttl_seconds = active_ttl_seconds
        self.max_response_bytes = max_response_bytes
        self.journal = journal
        self._sessions: dict[str, _StreamSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        key: str,
        fingerprint: str,
        subject: str,
        *,
        last_event_id: int = 0,
    ) -> tuple[_StreamSession, bool]:
        """Return an existing matching run or reserve one new bounded session."""
        await self.reap_expired()
        if self.journal is not None:
            async with self._lock:
                snapshot = await self.journal.get_or_create(
                    key,
                    fingerprint,
                    hashlib.sha256(subject.encode("utf-8")).hexdigest(),
                    last_event_id=last_event_id,
                    max_sessions=self.max_sessions,
                    max_response_bytes=self.max_response_bytes,
                    ttl_seconds=self.ttl_seconds,
                    active_ttl_seconds=self.active_ttl_seconds,
                )
                session = self._sessions.get(key)
                if session is None:
                    session = _StreamSession(
                        fingerprint,
                        subject,
                        frames=list(snapshot.frames),
                        total_bytes=sum(map(len, snapshot.frames)),
                        done=snapshot.done,
                        journal=self.journal,
                        key=key,
                    )
                    self._sessions[key] = session
                elif snapshot.created:
                    raise RuntimeError("stream journal returned a duplicate creation")
                else:
                    async with session.condition:
                        session.frames = list(snapshot.frames)
                        session.total_bytes = sum(map(len, snapshot.frames))
                        session.done = snapshot.done
                        if snapshot.done and session.finished_at is None:
                            session.finished_at = asyncio.get_running_loop().time()
                return session, snapshot.created
        async with self._lock:
            existing = self._sessions.get(key)
            if existing is not None:
                if existing.subject != subject:
                    raise LookupError("stream session not found")
                if existing.fingerprint != fingerprint:
                    raise ValueError("idempotency key was already used with a different request")
                return existing, False
            if last_event_id:
                raise LookupError("stream session not found")
            if len(self._sessions) >= self.max_sessions:
                raise OverflowError("resumable stream capacity is full")
            session = _StreamSession(fingerprint, subject)
            self._sessions[key] = session
            return session, True

    async def reap_expired(self) -> None:
        """Drop completed event journals after their retention window."""
        now = asyncio.get_running_loop().time()
        if self.journal is not None:
            await self.journal.reap(self.ttl_seconds, self.active_ttl_seconds)
        async with self._lock:
            expired = [
                key
                for key, session in self._sessions.items()
                if session.finished_at is not None and now - session.finished_at >= self.ttl_seconds
            ]
            for key in expired:
                del self._sessions[key]

    async def aclose(self) -> None:
        """Cancel and drain producer tasks before the owning agent is closed."""
        tasks = [session.task for session in self._sessions.values() if session.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def remove(self, key: str, session: _StreamSession) -> None:
        """Discard a newly reserved session when execution admission fails."""
        async with self._lock:
            if self._sessions.get(key) is session:
                del self._sessions[key]
        if self.journal is not None:
            await self.journal.remove(key)


async def _wait_for_disconnect(request: Request) -> None:
    """Wait for the ASGI server to report that a non-streaming client disconnected."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


class _CapacityLimitedStreamingResponse(StreamingResponse):
    """Release a reserved run slot even if a streaming response never starts sending."""

    def __init__(self, *args: Any, lease: _RunCapacityLease, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lease = lease

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                close = getattr(self.body_iterator, "aclose", None)
                if callable(close):
                    with suppress(Exception):
                        await close()
            finally:
                self._lease.release()


def create_app(
    agent: Agent,
    *,
    max_request_bytes: int = DEFAULT_MAX_HTTP_REQUEST_BYTES,
    request_body_timeout_seconds: float = DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS,
    authenticator_timeout_seconds: float = DEFAULT_AUTHENTICATOR_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_HTTP_RESPONSE_BYTES,
    max_concurrent_runs: int = 8,
    max_resumable_streams: int | None = None,
    stream_session_ttl_seconds: float = 600.0,
    stream_journal_path: str | Path | None = None,
    stream_journal: StreamJournal | None = None,
    authenticator: Authenticator | None = None,
    run_scopes: tuple[str, ...] = (),
    stream_scopes: tuple[str, ...] = (),
    allow_unauthenticated: bool = False,
) -> FastAPI:
    """Create the HTTP API around an already constructed agent.

    Request bodies are bounded by ``max_request_bytes`` and
    ``request_body_timeout_seconds`` before authentication or execution begins.
    ``authenticator_timeout_seconds`` bounds each custom authentication call.
    ``run_scopes`` and ``stream_scopes`` require every configured principal scope on
    their respective routes; both default to no additional scope requirement. Supply either
    ``stream_journal_path`` for Gabby's owned same-host SQLite journal or ``stream_journal`` for
    a host-owned backend implementing :class:`StreamJournal`. Gabby manages the SQLite journal's
    lifecycle; injected journal lifecycle remains with the host.
    """
    if (
        isinstance(max_request_bytes, bool)
        or not isinstance(max_request_bytes, int)
        or max_request_bytes < 1
    ):
        raise ValueError("max_request_bytes must be a positive integer")
    if not _is_positive_finite_number(request_body_timeout_seconds):
        raise ValueError("request_body_timeout_seconds must be a finite positive number")
    if not _is_positive_finite_number(authenticator_timeout_seconds):
        raise ValueError("authenticator_timeout_seconds must be a finite positive number")
    if (
        isinstance(max_concurrent_runs, bool)
        or not isinstance(max_concurrent_runs, int)
        or max_concurrent_runs < 1
    ):
        raise ValueError("max_concurrent_runs must be a positive integer")
    if max_resumable_streams is None:
        max_resumable_streams = max_concurrent_runs * 4
    if (
        isinstance(max_resumable_streams, bool)
        or not isinstance(max_resumable_streams, int)
        or max_resumable_streams < 1
    ):
        raise ValueError("max_resumable_streams must be a positive integer")
    if not _is_positive_finite_number(stream_session_ttl_seconds):
        raise ValueError("stream_session_ttl_seconds must be a finite positive number")
    if stream_journal_path is not None and (
        not isinstance(stream_journal_path, (str, Path)) or not str(stream_journal_path).strip()
    ):
        raise ValueError("stream_journal_path must be a non-empty filesystem path")
    if stream_journal_path is not None and stream_journal is not None:
        raise ValueError("stream_journal_path and stream_journal cannot both be configured")
    if stream_journal is not None and not isinstance(stream_journal, StreamJournal):
        raise ValueError("stream_journal must implement the StreamJournal protocol")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes < _MIN_HTTP_RESPONSE_BYTES
    ):
        raise ValueError(
            f"max_response_bytes must be an integer of at least {_MIN_HTTP_RESPONSE_BYTES}"
        )
    if authenticator is None and not allow_unauthenticated:
        raise ValueError("Provide an authenticator or explicitly set allow_unauthenticated=True")
    if authenticator is not None and allow_unauthenticated:
        raise ValueError("allow_unauthenticated cannot be combined with an authenticator")
    required_run_scopes = _validate_required_scopes(run_scopes, name="run_scopes")
    required_stream_scopes = _validate_required_scopes(stream_scopes, name="stream_scopes")
    if (required_run_scopes or required_stream_scopes) and authenticator is None:
        raise ValueError("run_scopes and stream_scopes require an authenticator")

    run_timeout = agent.definition.policies.get("timeout_seconds", 120.0)
    active_session_ttl = max(float(stream_session_ttl_seconds), float(run_timeout) * 2 + 30)
    owned_stream_journal = (
        _SQLiteStreamJournal(Path(stream_journal_path)) if stream_journal_path is not None else None
    )
    selected_stream_journal = stream_journal if stream_journal is not None else owned_stream_journal
    stream_sessions = _StreamSessionStore(
        max_sessions=max_resumable_streams,
        ttl_seconds=float(stream_session_ttl_seconds),
        active_ttl_seconds=active_session_ttl,
        max_response_bytes=max_response_bytes,
        journal=selected_stream_journal,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async def reap_sessions() -> None:
            interval = min(60.0, max(1.0, float(stream_session_ttl_seconds) / 4))
            while True:
                await asyncio.sleep(interval)
                await stream_sessions.reap_expired()

        reaper: asyncio.Task[None] | None = None
        try:
            if owned_stream_journal is not None:
                await owned_stream_journal.initialize()
            reaper = asyncio.create_task(reap_sessions())
            yield
        finally:
            if reaper is not None:
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
            try:
                await stream_sessions.aclose()
                await agent.aclose()
            finally:
                if owned_stream_journal is not None:
                    owned_stream_journal.close()

    app = FastAPI(title="Gabby Agent API", version="1.0.0", lifespan=lifespan)
    run_capacity = _RunCapacity(max_concurrent_runs)
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=max_request_bytes,
        timeout_seconds=float(request_body_timeout_seconds),
        run_capacity=run_capacity,
    )
    app.add_middleware(ResponseSizeLimitMiddleware, max_bytes=max_response_bytes)

    async def authenticate_request(
        request: Request, required_scopes: frozenset[str]
    ) -> Principal | None:
        if authenticator is None:
            return None
        try:
            principal = await asyncio.wait_for(
                authenticator.authenticate(request), timeout=float(authenticator_timeout_seconds)
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="authentication service unavailable",
            ) from exc
        except Exception as exc:
            # Identity provider details may contain credentials or provider metadata.
            raise HTTPException(
                status_code=503,
                detail="authentication service unavailable",
            ) from exc
        if not isinstance(principal, Principal):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not required_scopes.issubset(principal.scopes):
            raise HTTPException(status_code=403, detail="required scope is missing")
        request.state.gabby_principal = principal
        return principal

    def authentication_dependencies(required_scopes: frozenset[str]) -> list[Any]:
        async def require_authentication(request: Request) -> Principal | None:
            return await authenticate_request(request, required_scopes)

        dependencies: list[Any] = [Depends(require_authentication)]
        if authenticator is not None:
            dependencies.insert(0, Security(_bearer_security))
        return dependencies

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/v1/agents/{agent_name}/run",
        response_model=RunResponse,
        responses={
            408: {"model": RequestBodyErrorResponse},
            413: {"model": RequestBodyErrorResponse},
            429: {"model": AgentErrorResponse},
            500: {"model": AgentErrorResponse},
            403: {"description": "Required scope is missing or a skill publisher is revoked"},
            503: {"description": "Authentication or skill revocation service is unavailable"},
        },
        dependencies=authentication_dependencies(required_run_scopes),
    )
    async def run_agent(
        agent_name: str,
        payload: RunPayload,
        http_request: Request,
    ) -> Response:
        if agent_name != agent.definition.name:
            raise HTTPException(status_code=404, detail="agent not found")
        lease = http_request.scope.get("state", {}).get("_gabby_run_capacity_lease")
        owns_lease = lease is None
        if owns_lease:
            lease = run_capacity.try_acquire()
        if lease is None:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "execution capacity exhausted",
                    "error_type": "CapacityLimitError",
                },
            )
        execution: asyncio.Task[Any] | None = None
        disconnected: asyncio.Task[None] | None = None
        principal = getattr(http_request.state, "gabby_principal", None)
        try:
            execution = asyncio.create_task(
                agent.arun(
                    payload.input,
                    context=payload.context,
                    memory=payload.memory,
                    metadata=payload.metadata,
                    principal=principal,
                )
            )
            disconnected = asyncio.create_task(_wait_for_disconnect(http_request))
            done, _ = await asyncio.wait(
                {execution, disconnected}, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done and not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                raise asyncio.CancelledError("HTTP client disconnected during agent execution")
            disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)
            result = execution.result()
            response = RunResponse.model_validate(
                result.as_dict(include_trace=payload.include_trace)
            )
            body = _bounded_json_bytes(
                response.model_dump(mode="json"), max_bytes=max_response_bytes
            )
            return Response(content=body, media_type="application/json")
        except ResponseSizeLimitError:
            body = _bounded_json_bytes(
                {"detail": _RESPONSE_TOO_LARGE}, max_bytes=max_response_bytes
            )
            return Response(content=body, status_code=500, media_type="application/json")
        except Exception as exc:
            # Do not send provider/tool exception contents to remote callers.
            status_code = (
                403
                if isinstance(exc, (SkillIntegrityError, SkillRevokedError))
                else 503
                if isinstance(exc, SkillRevocationUnavailable)
                else 500
            )
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": _public_execution_error_message(exc),
                    "error_type": _public_execution_error_type(exc),
                },
            ) from exc
        finally:
            tasks = [task for task in (execution, disconnected) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # The body middleware owns leases acquired before buffering and holds
            # them until the response has been sent. Direct endpoint invocations
            # (for example, framework integrations) own their fallback lease here.
            if owns_lease:
                lease.release()

    @app.post(
        "/v1/agents/{agent_name}/stream",
        response_class=StreamingResponse,
        responses={
            400: {"description": "Invalid idempotency or event cursor header"},
            408: {"model": RequestBodyErrorResponse},
            413: {"model": RequestBodyErrorResponse},
            429: {"model": AgentErrorResponse},
            404: {"description": "Agent or resumable stream session not found"},
            409: {"description": "Idempotency key conflict or event cursor ahead of stream"},
            403: {"description": "Required scope is missing or a skill publisher is revoked"},
            503: {"description": "Authentication or skill revocation service is unavailable"},
            200: {
                "description": (
                    "Server-Sent Events. Each data field is a JSON StreamEventPayload envelope."
                ),
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
        },
        dependencies=authentication_dependencies(required_stream_scopes),
    )
    async def stream_agent(
        agent_name: str,
        request: RunPayload,
        http_request: Request,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        if agent_name != agent.definition.name:
            raise HTTPException(status_code=404, detail="agent not found")
        capacity_state = http_request.scope.get("state", {})
        lease = capacity_state.get("_gabby_run_capacity_lease")
        resumable_request = idempotency_key is not None or last_event_id is not None
        owns_lease = lease is None
        if lease is None and not resumable_request:
            lease = run_capacity.try_acquire()
        if lease is None and not resumable_request:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "execution capacity exhausted",
                    "error_type": "CapacityLimitError",
                },
            )
        principal = getattr(http_request.state, "gabby_principal", None)

        if idempotency_key is not None or last_event_id is not None:
            if idempotency_key is None:
                raise HTTPException(status_code=400, detail="Idempotency-Key is required to resume")
            if (
                not idempotency_key
                or len(idempotency_key) > 128
                or any(not 0x21 <= ord(char) <= 0x7E for char in idempotency_key)
            ):
                raise HTTPException(status_code=400, detail="Idempotency-Key is invalid")
            cursor = 0
            if last_event_id is not None:
                if (
                    len(last_event_id) > 20
                    or not last_event_id.isascii()
                    or not last_event_id.isdecimal()
                ):
                    raise HTTPException(status_code=400, detail="Last-Event-ID is invalid")
                cursor = int(last_event_id)
            subject = principal.subject if isinstance(principal, Principal) else "anonymous"
            session_key = hashlib.sha256(
                f"{agent.definition.name}\0{subject}\0{idempotency_key}".encode()
            ).hexdigest()
            try:
                fingerprint = hashlib.sha256(
                    _bounded_json_bytes(
                        request.model_dump(mode="json"),
                        max_bytes=max_request_bytes,
                    )
                ).hexdigest()
            except ResponseSizeLimitError as exc:
                raise HTTPException(status_code=413, detail="request body too large") from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="request body is not valid JSON data"
                ) from exc
            try:
                session, created = await stream_sessions.get_or_create(
                    session_key,
                    fingerprint,
                    subject,
                    last_event_id=cursor,
                )
            except ResponseSizeLimitError as exc:
                raise HTTPException(status_code=413, detail="request body too large") from exc
            except LookupError as exc:
                raise HTTPException(status_code=404, detail="stream session not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except OverflowError as exc:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "resumable stream capacity exhausted",
                        "error_type": "CapacityLimitError",
                    },
                    headers={"Retry-After": "1"},
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "resumable stream journal unavailable",
                        "error_type": "StreamJournalUnavailable",
                    },
                ) from exc
            if cursor > len(session.frames):
                raise HTTPException(status_code=409, detail="Last-Event-ID is ahead of this stream")

            if created:
                if lease is None:
                    lease = run_capacity.try_acquire()
                    if lease is None:
                        await stream_sessions.remove(session_key, session)
                        raise HTTPException(
                            status_code=429,
                            detail={
                                "error": "execution capacity exhausted",
                                "error_type": "CapacityLimitError",
                            },
                            headers={"Retry-After": "1"},
                        )
                    owns_lease = True
                session_lease = lease

                async def produce_session() -> None:
                    events = agent.astream(
                        request.input,
                        context=request.context,
                        memory=request.memory,
                        metadata=request.metadata,
                        principal=principal,
                    )
                    try:
                        event_id = 0
                        error_reserve = len(
                            _sse_frame(
                                "error",
                                _RESPONSE_TOO_LARGE,
                                max_bytes=max_response_bytes,
                                event_id=10**20 - 1,
                            )
                        )
                        async for event in events:
                            event_id += 1
                            event_data = event.data
                            if event.type == "completed" and not request.include_trace:
                                event_data = dict(event.data)
                                result = event_data.get("result")
                                if isinstance(result, dict):
                                    event_data["result"] = {**result, "trace": None}
                            try:
                                frame = _sse_frame(
                                    event.type,
                                    event_data,
                                    max_bytes=max_response_bytes,
                                    event_id=event_id,
                                )
                            except ResponseSizeLimitError:
                                frame = None
                            terminal_event = event.type in {"completed", "error"}
                            event_budget = (
                                max_response_bytes
                                if terminal_event
                                else max_response_bytes - error_reserve
                            )
                            if frame is None or not await session.append(
                                frame, max_bytes=event_budget
                            ):
                                terminal = _sse_frame(
                                    "error",
                                    _RESPONSE_TOO_LARGE,
                                    max_bytes=max_response_bytes,
                                    event_id=event_id,
                                )
                                await session.append(terminal, max_bytes=max_response_bytes)
                                break
                            if event.type in {"completed", "error"}:
                                break
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        event_id = len(session.frames) + 1
                        try:
                            terminal = _sse_frame(
                                "error",
                                {
                                    "error": _public_execution_error_message(exc),
                                    "error_type": _public_execution_error_type(exc),
                                },
                                max_bytes=max_response_bytes,
                                event_id=event_id,
                            )
                        except ResponseSizeLimitError:
                            terminal = b""
                        if terminal:
                            with suppress(Exception):
                                await session.append(terminal, max_bytes=max_response_bytes)
                    finally:
                        try:
                            await events.aclose()
                        finally:
                            try:
                                await session.finish()
                            finally:
                                session_lease.release()

                try:
                    session.task = asyncio.create_task(produce_session())
                except BaseException:
                    await session.finish()
                    if owns_lease:
                        lease.release()
                    raise
                if capacity_state.get("_gabby_run_capacity_lease") is session_lease:
                    capacity_state["_gabby_run_capacity_lease_transferred"] = True

            async def replay_session() -> AsyncIterator[bytes]:
                sent_bytes = 0
                async for frame in session.replay(cursor):
                    if sent_bytes + len(frame) > max_response_bytes:
                        return
                    sent_bytes += len(frame)
                    yield frame

            if lease is not None and owns_lease and not created:
                return _CapacityLimitedStreamingResponse(
                    replay_session(),
                    lease=lease,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return StreamingResponse(
                replay_session(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        async def event_stream() -> AsyncIterator[bytes]:
            events = agent.astream(
                request.input,
                context=request.context,
                memory=request.memory,
                metadata=request.metadata,
                principal=principal,
            )
            iterator = events.__aiter__()
            pending: asyncio.Future[Any] | None = None
            response_bytes = 0
            error_frame = _sse_frame("error", _RESPONSE_TOO_LARGE, max_bytes=max_response_bytes)

            def fit_frame(
                frame: bytes | None, *, reserve_error: bool = True
            ) -> tuple[bytes | None, bool]:
                nonlocal response_bytes
                # Keep room for a typed terminal error if the final result or a later
                # progress event cannot fit. Completion itself may use the full budget.
                reserve = len(error_frame) if reserve_error else 0
                if (
                    frame is not None
                    and response_bytes + len(frame) + reserve <= max_response_bytes
                ):
                    response_bytes += len(frame)
                    return frame, False
                if response_bytes + len(error_frame) <= max_response_bytes:
                    response_bytes += len(error_frame)
                    return error_frame, True
                return None, True

            def event_frame(
                event_type: StreamEventType, data: dict[str, Any]
            ) -> tuple[bytes | None, bool]:
                try:
                    frame = _sse_frame(event_type, data, max_bytes=max_response_bytes)
                except ResponseSizeLimitError:
                    frame = None
                return fit_frame(frame, reserve_error=event_type not in {"completed", "error"})

            try:
                while True:
                    pending = asyncio.ensure_future(iterator.__anext__())
                    done, _ = await asyncio.wait({pending}, timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS)
                    if not done:
                        frame, stop = fit_frame(b": keepalive\n\n")
                        if frame is not None:
                            yield frame
                        if stop:
                            break
                        while not pending.done():
                            done, _ = await asyncio.wait(
                                {pending}, timeout=_SSE_KEEPALIVE_INTERVAL_SECONDS
                            )
                            if not done:
                                frame, stop = fit_frame(b": keepalive\n\n")
                                if frame is not None:
                                    yield frame
                                if stop:
                                    break
                        if stop:
                            break
                    try:
                        event = pending.result()
                    except StopAsyncIteration:
                        break
                    event_data = event.data
                    if event.type == "completed" and not request.include_trace:
                        event_data = dict(event.data)
                        result = event_data.get("result")
                        if isinstance(result, dict):
                            event_data["result"] = {**result, "trace": None}
                    frame, stop = event_frame(event.type, event_data)
                    if frame is not None:
                        yield frame
                    if stop:
                        break
                    pending = None
            except Exception as exc:
                frame, _ = event_frame(
                    "error",
                    {
                        "error": _public_execution_error_message(exc),
                        "error_type": _public_execution_error_type(exc),
                    },
                )
                if frame is not None:
                    yield frame
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                try:
                    await events.aclose()
                finally:
                    if owns_lease:
                        lease.release()

        try:
            return _CapacityLimitedStreamingResponse(
                event_stream(),
                lease=lease,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except BaseException:
            if owns_lease:
                lease.release()
            raise

    return app


def serve(
    agent: Agent,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    max_request_bytes: int = DEFAULT_MAX_HTTP_REQUEST_BYTES,
    request_body_timeout_seconds: float = DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS,
    authenticator_timeout_seconds: float = DEFAULT_AUTHENTICATOR_TIMEOUT_SECONDS,
    max_response_bytes: int = DEFAULT_MAX_HTTP_RESPONSE_BYTES,
    max_concurrent_runs: int = 8,
    max_resumable_streams: int | None = None,
    stream_session_ttl_seconds: float = 600.0,
    stream_journal_path: str | Path | None = None,
    stream_journal: StreamJournal | None = None,
    authenticator: Authenticator | None = None,
    bearer_token: str | None = None,
    bearer_scopes: frozenset[str] = frozenset(),
    run_scopes: tuple[str, ...] = (),
    stream_scopes: tuple[str, ...] = (),
) -> None:
    import uvicorn

    if authenticator is not None and bearer_token is not None:
        raise ValueError("Provide authenticator or bearer_token, not both")
    if bearer_scopes and (authenticator is not None or bearer_token is None):
        raise ValueError("bearer_scopes require bearer_token and cannot be used with authenticator")
    if bearer_token is not None:
        authenticator = BearerTokenAuthenticator(bearer_token, scopes=bearer_scopes)
    local_host = host.casefold() == "localhost"
    with suppress(ValueError):
        local_host = local_host or ipaddress.ip_address(host).is_loopback
    if authenticator is None and not local_host:
        raise ValueError("Non-loopback serving requires an authenticator or bearer token")
    app = create_app(
        agent,
        max_request_bytes=max_request_bytes,
        request_body_timeout_seconds=request_body_timeout_seconds,
        authenticator_timeout_seconds=authenticator_timeout_seconds,
        max_response_bytes=max_response_bytes,
        max_concurrent_runs=max_concurrent_runs,
        max_resumable_streams=max_resumable_streams,
        stream_session_ttl_seconds=stream_session_ttl_seconds,
        stream_journal_path=stream_journal_path,
        stream_journal=stream_journal,
        authenticator=authenticator,
        run_scopes=run_scopes,
        stream_scopes=stream_scopes,
        allow_unauthenticated=local_host and authenticator is None,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
