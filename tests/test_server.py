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
"""FastAPI transport contract coverage."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from gabby.agent import Agent
from gabby.approval import ApprovalDecision, ApprovalRequest
from gabby.auth import BearerTokenAuthenticator, Principal
from gabby.config import AgentDefinition
from gabby.models import ModelResponse, ModelStreamDelta
from gabby.runtime import AgentRuntimeError
from gabby.server import (
    BodySizeLimitMiddleware,
    ResponseSizeLimitError,
    ResponseSizeLimitMiddleware,
    RunPayload,
    StreamJournalSnapshot,
    _bounded_json_bytes,
    _RunCapacity,
    _SQLiteStreamJournal,
    _StreamSession,
    _StreamSessionStore,
    create_app,
    serve,
)
from gabby.skill_trust import SkillIntegrityError, SkillRevocationUnavailable, SkillRevokedError
from gabby.tools import Tool, ToolRegistry

_ANY_JSON_SCHEMA = {"type": ["object", "array", "string", "number", "boolean", "null"]}


@pytest.mark.parametrize(
    ("created", "frames", "done", "event_count"),
    [
        (1, (), False, 0),
        (False, [], False, 0),
        (False, ("text",), False, 1),
        (False, (b"event",), False, True),
        (False, (b"event",), False, 0),
        (True, (b"event",), False, 1),
        (True, (), True, 0),
    ],
)
def test_stream_journal_snapshot_rejects_invalid_backend_results(
    created: Any, frames: Any, done: Any, event_count: Any
) -> None:
    with pytest.raises(ValueError):
        StreamJournalSnapshot(created, frames, done, event_count)


def test_stream_journal_snapshot_accepts_new_and_resumed_shapes() -> None:
    assert StreamJournalSnapshot(True, (), False, 0).created
    assert StreamJournalSnapshot(False, (b"id: 1\n\n",), True, 1).done


@pytest.mark.asyncio
async def test_in_memory_stream_session_enforces_append_and_finish_contract() -> None:
    session = _StreamSession("fingerprint", "subject")
    assert await session.append(b"a", max_bytes=1)
    assert not await session.append(b"b", max_bytes=1)
    await session.finish()
    await session.finish()
    assert not await session.append(b"c", max_bytes=100)
    assert [frame async for frame in session.replay(0)] == [b"a"]


@pytest.mark.asyncio
async def test_sqlite_stream_journal_remove_releases_a_reserved_key(tmp_path: Path) -> None:
    journal = _SQLiteStreamJournal(tmp_path / "stream-journal.sqlite3")
    key = "remove-reservation-contract-01"
    try:
        await journal.initialize()
        created = await journal.get_or_create(
            key,
            "fingerprint",
            "principal-hash",
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=1024,
            ttl_seconds=60,
            active_ttl_seconds=10,
        )
        assert created.created
        await journal.remove(key)
        await journal.remove(key)
        assert await journal.read_after(key, 0) == ([], True)
        recreated = await journal.get_or_create(
            key,
            "fingerprint",
            "principal-hash",
            last_event_id=0,
            max_sessions=1,
            max_response_bytes=1024,
            ttl_seconds=60,
            active_ttl_seconds=10,
        )
        assert recreated.created
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_in_memory_stream_store_enforces_identity_capacity_and_removal() -> None:
    store = _StreamSessionStore(
        max_sessions=1,
        ttl_seconds=60,
        active_ttl_seconds=10,
        max_response_bytes=1024,
    )
    session, created = await store.get_or_create("key", "fingerprint", "alice")
    assert created
    existing, created_again = await store.get_or_create("key", "fingerprint", "alice")
    assert existing is session and not created_again

    with pytest.raises(LookupError):
        await store.get_or_create("key", "fingerprint", "bob")
    with pytest.raises(ValueError):
        await store.get_or_create("key", "different-fingerprint", "alice")
    with pytest.raises(LookupError):
        await store.get_or_create("missing", "fingerprint", "alice", last_event_id=1)
    with pytest.raises(OverflowError):
        await store.get_or_create("other", "fingerprint", "alice")

    await store.remove("key", session)
    await store.remove("other", session)
    replacement, replacement_created = await store.get_or_create("other", "fingerprint", "alice")
    assert replacement_created and replacement is not session
    replacement.finished_at = asyncio.get_running_loop().time() - 120
    expired_replacement, expired_created = await store.get_or_create(
        "other", "fingerprint", "alice"
    )
    assert expired_created and expired_replacement is not replacement
    await store.aclose()


class FakeModel:
    name = "fake"

    def __init__(self, output: str = "ok") -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []
        self.stream_calls = 0

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return ModelResponse(content=self.output)

    async def stream(self, **_: Any) -> Any:
        self.stream_calls += 1
        yield ModelStreamDelta(content_delta="o")
        yield ModelStreamDelta(content_delta="k")


def _request_with_principal(principal: Principal | None = None) -> Request:
    request = Request({"type": "http", "headers": []})
    if principal is not None:
        request.state.gabby_principal = principal
    return request


def make_agent(model: FakeModel | None = None) -> Agent:
    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    return Agent(definition, model=model if model is not None else FakeModel())


@pytest.mark.asyncio
async def test_fastapi_stateless_run_and_health_routes() -> None:
    app = create_app(make_agent(), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).json() == {"status": "ok"}
        response = await client.post(
            "/v1/agents/test-agent/run", json={"input": "hello", "context": {"x": 1}}
        )
        unversioned = await client.post("/agents/test-agent/run", json={"input": "hello"})

    assert response.status_code == 200
    assert unversioned.status_code == 404
    payload = response.json()
    assert payload["output"] == "ok"
    assert payload["trace_id"]
    assert "trace" in payload
    assert payload["trace"] is not None


@pytest.mark.asyncio
async def test_idempotent_stream_replays_after_last_event_without_reexecuting() -> None:
    model = FakeModel()
    app = create_app(make_agent(model), allow_unauthenticated=True)
    request = {"input": "hello"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = await client.post(
            "/v1/agents/test-agent/stream",
            json=request,
            headers={"Idempotency-Key": "stream-acceptance-0001"},
        )
        event_ids = [
            line.removeprefix("id: ") for line in first.text.splitlines() if line.startswith("id: ")
        ]
        assert event_ids
        resumed = await client.post(
            "/v1/agents/test-agent/stream",
            json=request,
            headers={
                "Idempotency-Key": "stream-acceptance-0001",
                "Last-Event-ID": event_ids[0],
            },
        )
        mismatch = await client.post(
            "/v1/agents/test-agent/stream",
            json={"input": "different"},
            headers={"Idempotency-Key": "stream-acceptance-0001"},
        )

    resumed_ids = [
        line.removeprefix("id: ") for line in resumed.text.splitlines() if line.startswith("id: ")
    ]
    assert first.status_code == resumed.status_code == 200
    assert resumed_ids == event_ids[1:]
    assert model.stream_calls == 1
    assert mismatch.status_code == 409


@pytest.mark.asyncio
async def test_sqlite_stream_journal_replays_between_app_instances(tmp_path: Path) -> None:
    journal_path = tmp_path / "streams" / "journal.sqlite3"
    first_model = FakeModel()
    first_app = create_app(
        make_agent(first_model),
        allow_unauthenticated=True,
        stream_journal_path=journal_path,
    )
    headers = {"Idempotency-Key": "shared-journal-acceptance-0001"}
    async with (
        first_app.router.lifespan_context(first_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first_app), base_url="http://first-worker"
        ) as first_client,
    ):
        first = await first_client.post(
            "/v1/agents/test-agent/stream", json={"input": "hello"}, headers=headers
        )

    second_model = FakeModel()
    second_app = create_app(
        make_agent(second_model),
        allow_unauthenticated=True,
        stream_journal_path=journal_path,
    )
    async with (
        second_app.router.lifespan_context(second_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second_app), base_url="http://second-worker"
        ) as second_client,
    ):
        replay = await second_client.post(
            "/v1/agents/test-agent/stream",
            json={"input": "hello"},
            headers={**headers, "Last-Event-ID": "0"},
        )

    assert first.status_code == replay.status_code == 200
    assert replay.content == first.content
    assert first_model.stream_calls == 1
    assert second_model.stream_calls == 0


@pytest.mark.asyncio
async def test_sqlite_stream_journal_serializes_concurrent_process_appends(tmp_path: Path) -> None:
    journal_path = tmp_path / "shared" / "streams.sqlite3"
    key = "cross-process-append-0001"
    journal = _SQLiteStreamJournal(journal_path)
    await journal.initialize()
    await journal.get_or_create(
        key,
        "request-fingerprint",
        "principal-hash",
        last_event_id=0,
        max_sessions=1,
        max_response_bytes=1024 * 1024,
        ttl_seconds=600,
        active_ttl_seconds=60,
    )
    journal.close()

    script = """
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from gabby.server import _SQLiteStreamJournal

async def main():
    journal = _SQLiteStreamJournal(Path(sys.argv[1]))
    try:
        for index in range(20):
            frame = f"{sys.argv[2]}:{index}".encode()
            if not await journal.append("cross-process-append-0001", frame, max_bytes=1048576):
                raise AssertionError("an append was rejected")
    finally:
        journal.close()

asyncio.run(main())
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(journal_path), f"worker-{worker}"],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in range(4)
    ]
    try:
        outputs = [process.communicate(timeout=20) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert [
        (process.returncode, stderr)
        for process, (_, stderr) in zip(processes, outputs, strict=True)
    ] == [(0, "") for _ in processes]

    reader = _SQLiteStreamJournal(journal_path)
    try:
        frames, done = await reader.read_after(key, 0)
    finally:
        reader.close()
    assert not done
    assert len(frames) == 80
    expected_frames = {
        f"worker-{worker}:{index}".encode() for worker in range(4) for index in range(20)
    }
    assert set(frames) == expected_frames


@pytest.mark.asyncio
async def test_sqlite_stream_journal_recovers_an_abandoned_process_run(tmp_path: Path) -> None:
    import sqlite3
    import time

    journal_path = tmp_path / "crash" / "streams.sqlite3"
    key = "cross-process-crash-0001"
    script = """
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from gabby.server import _SQLiteStreamJournal

async def main():
    journal = _SQLiteStreamJournal(Path(sys.argv[1]))
    await journal.initialize()
    snapshot = await journal.get_or_create(
        "cross-process-crash-0001", "request-fingerprint", "principal-hash",
        last_event_id=0, max_sessions=1, max_response_bytes=4096,
        ttl_seconds=600, active_ttl_seconds=1,
    )
    assert snapshot.created
    assert await journal.append(
        "cross-process-crash-0001", b"id: 1\\ndata: partial\\n\\n", max_bytes=4096
    )
    journal.close()
    os._exit(0)

asyncio.run(main())
"""
    subprocess.run(
        [sys.executable, "-c", script, str(journal_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )

    with sqlite3.connect(journal_path) as connection:
        connection.execute(
            "UPDATE gabby_stream_sessions SET created_at=? WHERE session_key=?",
            (time.time() - 60, key),
        )

    recovery = _SQLiteStreamJournal(journal_path)
    try:
        await recovery.reap(ttl_seconds=600, active_ttl_seconds=1)
        frames, done = await recovery.read_after(key, 0)
    finally:
        recovery.close()
    assert done
    assert len(frames) == 2
    assert frames[0] == b"id: 1\ndata: partial\n\n"
    assert "StreamRecoveryError" in frames[1].decode("utf-8")


@pytest.mark.asyncio
async def test_injected_stream_journal_lifecycle_remains_host_owned(tmp_path: Path) -> None:
    journal = _SQLiteStreamJournal(tmp_path / "host-owned.sqlite3")
    await journal.initialize()
    request_headers = {"Idempotency-Key": "host-journal-lifecycle-0001"}

    try:
        owner_model = FakeModel()
        owner_app = create_app(
            make_agent(owner_model),
            allow_unauthenticated=True,
            stream_journal=journal,
        )
        async with (
            owner_app.router.lifespan_context(owner_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=owner_app), base_url="http://host-owner"
            ) as client,
        ):
            first = await client.post(
                "/v1/agents/test-agent/stream",
                json={"input": "hello"},
                headers=request_headers,
            )

        replay_model = FakeModel()
        replay_app = create_app(
            make_agent(replay_model),
            allow_unauthenticated=True,
            stream_journal=journal,
        )
        async with (
            replay_app.router.lifespan_context(replay_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=replay_app), base_url="http://host-replay"
            ) as client,
        ):
            replay = await client.post(
                "/v1/agents/test-agent/stream",
                json={"input": "hello"},
                headers={**request_headers, "Last-Event-ID": "0"},
            )
    finally:
        journal.close()

    assert first.status_code == replay.status_code == 200
    assert replay.content == first.content
    assert replay_model.stream_calls == 0


@pytest.mark.asyncio
async def test_injected_stream_journal_lookup_failure_is_sanitized(tmp_path: Path) -> None:
    class UnavailableJournal(_SQLiteStreamJournal):
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
            raise OSError("/private/path/to/shared/journal")

    journal = UnavailableJournal(tmp_path / "unavailable.sqlite3")
    await journal.initialize()
    app = create_app(
        make_agent(),
        allow_unauthenticated=True,
        stream_journal=journal,
    )
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://journal-error"
            ) as client,
        ):
            response = await client.post(
                "/v1/agents/test-agent/stream",
                json={"input": "hello"},
                headers={"Idempotency-Key": "broken-host-journal-0001"},
            )
    finally:
        journal.close()

    assert response.status_code == 503
    assert response.json()["detail"]["error_type"] == "StreamJournalUnavailable"
    assert "/private/path" not in response.text


@pytest.mark.asyncio
async def test_sqlite_stream_journal_attaches_to_live_run_from_another_app(
    tmp_path: Path,
) -> None:
    class PausingModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, **_: Any) -> Any:
            self.stream_calls += 1
            self.started.set()
            yield ModelStreamDelta(content_delta="first")
            await self.release.wait()
            yield ModelStreamDelta(content_delta="last")

    journal_path = tmp_path / "streams.sqlite3"
    owner_model = PausingModel()
    owner_app = create_app(
        make_agent(owner_model),
        allow_unauthenticated=True,
        stream_journal_path=journal_path,
    )
    other_model = FakeModel()
    other_app = create_app(
        make_agent(other_model),
        allow_unauthenticated=True,
        stream_journal_path=journal_path,
    )
    headers = {"Idempotency-Key": "shared-journal-live-0001"}
    async with (
        owner_app.router.lifespan_context(owner_app),
        other_app.router.lifespan_context(other_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=owner_app), base_url="http://owner-worker"
        ) as owner_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=other_app), base_url="http://other-worker"
        ) as other_client,
    ):
        owner_request = asyncio.create_task(
            owner_client.post(
                "/v1/agents/test-agent/stream", json={"input": "hello"}, headers=headers
            )
        )
        await asyncio.wait_for(owner_model.started.wait(), timeout=2)
        other_request = asyncio.create_task(
            other_client.post(
                "/v1/agents/test-agent/stream", json={"input": "hello"}, headers=headers
            )
        )
        await asyncio.sleep(0.15)
        owner_model.release.set()
        owner_response, other_response = await asyncio.wait_for(
            asyncio.gather(owner_request, other_request), timeout=5
        )

    assert owner_response.status_code == other_response.status_code == 200
    assert "first" in other_response.text and "last" in other_response.text
    assert other_model.stream_calls == 0


@pytest.mark.asyncio
async def test_idempotent_stream_survives_transport_send_failure() -> None:
    class RecoverableModel:
        name = "recoverable-stream"

        def __init__(self) -> None:
            self.stream_calls = 0
            self.finished = asyncio.Event()

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="recovered")

        async def stream(self, **_: Any) -> Any:
            self.stream_calls += 1
            try:
                yield ModelStreamDelta(content_delta="first")
                yield ModelStreamDelta(content_delta="second")
            finally:
                self.finished.set()

    model = RecoverableModel()
    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    app = create_app(Agent(definition, model=model), allow_unauthenticated=True)
    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/stream"
    )
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/agents/test-agent/stream",
            "raw_path": b"/v1/agents/test-agent/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        },
    )

    async def receive() -> Message:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async with app.router.lifespan_context(app):
        response = await route.endpoint(
            "test-agent",
            RunPayload(input="hello"),
            _request_with_principal(),
            "recoverable-run-0001",
            None,
        )
        first_response_ids: list[str] = []

        async def fail_after_first_event(message: Message) -> None:
            if message["type"] != "http.response.body":
                return
            body = message.get("body", b"")
            for line in body.decode().splitlines():
                if line.startswith("id: "):
                    first_response_ids.append(line.removeprefix("id: "))
                    raise OSError("simulated client disconnect")

        with pytest.raises(OSError, match="simulated client disconnect"):
            await response(scope, receive, fail_after_first_event)
        await asyncio.wait_for(model.finished.wait(), timeout=1)

        resumed = await route.endpoint(
            "test-agent",
            RunPayload(input="hello"),
            _request_with_principal(),
            "recoverable-run-0001",
            first_response_ids[-1],
        )
        resumed_bodies: list[bytes] = []

        async def collect(message: Message) -> None:
            if message["type"] == "http.response.body":
                resumed_bodies.append(message.get("body", b""))

        await resumed(scope, receive, collect)

    replayed = b"".join(resumed_bodies).decode()
    replayed_ids = [
        line.removeprefix("id: ") for line in replayed.splitlines() if line.startswith("id: ")
    ]
    assert first_response_ids
    assert replayed_ids
    assert all(int(event_id) > int(first_response_ids[-1]) for event_id in replayed_ids)
    assert "second" in replayed
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_idempotent_stream_can_reattach_at_full_execution_capacity() -> None:
    class SlowModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, **_: Any) -> Any:
            self.stream_calls += 1
            self.started.set()
            yield ModelStreamDelta(content_delta="first")
            await self.release.wait()
            yield ModelStreamDelta(content_delta="last")

    model = SlowModel()
    app = create_app(
        make_agent(model),
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    headers = {"Idempotency-Key": "single-capacity-run-0001"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first_request = asyncio.create_task(
            client.post("/v1/agents/test-agent/stream", json={"input": "hello"}, headers=headers)
        )
        await asyncio.wait_for(model.started.wait(), timeout=1)
        reattach_request = asyncio.create_task(
            client.post("/v1/agents/test-agent/stream", json={"input": "hello"}, headers=headers)
        )
        await asyncio.sleep(0.01)
        model.release.set()
        first, reattached = await asyncio.gather(first_request, reattach_request)

    assert first.status_code == reattached.status_code == 200
    assert '"text":"last"' in reattached.text
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_fastapi_run_exposes_validated_structured_output_metadata() -> None:
    class StructuredModel:
        name = "structured-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content='{"category":"account"}')

    definition = AgentDefinition(
        name="classifier",
        model={"provider": "fake", "model": "test-model"},
        output_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    )
    app = create_app(Agent(definition, model=StructuredModel()), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/agents/classifier/run",
            json={"input": "Classify the request"},
        )

    assert response.status_code == 200
    assert response.json()["output"] == '{"category":"account"}'
    assert response.json()["metadata"]["structured_output"] == {"category": "account"}


@pytest.mark.parametrize(
    ("error", "status_code", "error_type", "public_message"),
    [
        (
            SkillIntegrityError("private filesystem path"),
            403,
            "SkillIntegrityError",
            "agent skill integrity check failed",
        ),
        (
            SkillRevokedError("private publisher ID"),
            403,
            "SkillRevokedError",
            "agent skill publisher is revoked",
        ),
        (
            SkillRevocationUnavailable("database path secret"),
            503,
            "SkillRevocationUnavailable",
            "skill revocation check is unavailable",
        ),
    ],
)
@pytest.mark.asyncio
async def test_skill_policy_failures_are_sanitized_for_run_and_stream(
    error: Exception,
    status_code: int,
    error_type: str,
    public_message: str,
) -> None:
    agent = make_agent()

    async def fail_closed(*, deadline: float) -> float | None:
        raise error

    agent._check_skill_revocations = fail_closed  # type: ignore[method-assign]
    app = create_app(agent, allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        run = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
        stream = await client.post("/v1/agents/test-agent/stream", json={"input": "hello"})

    assert run.status_code == status_code
    assert run.json()["detail"] == {"error": public_message, "error_type": error_type}
    assert stream.status_code == 200
    assert f'"error_type":"{error_type}"' in stream.text
    assert public_message in stream.text
    assert "private publisher ID" not in stream.text
    assert "database path secret" not in stream.text


@pytest.mark.asyncio
async def test_stream_reports_revocation_that_occurs_during_model_call() -> None:
    class BlockingModel:
        name = "blocking-revocation-model"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def complete(self, **_: Any) -> ModelResponse:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ModelResponse(content="unexpected")

    class RevocationChecker:
        def __init__(self) -> None:
            self.revoked = False
            self.check_count = 0

        async def check_not_revoked(self, _key_ids: frozenset[str]) -> None:
            self.check_count += 1
            if self.revoked:
                raise SkillRevokedError("private publisher identifier")
            return None

    model = BlockingModel()
    checker = RevocationChecker()
    agent = make_agent(cast(FakeModel, model))
    agent._trusted_skill_signers = frozenset({"publisher-key"})
    agent.skill_revocation_checker = checker
    agent.skill_revocation_poll_interval_seconds = 0.1
    app = create_app(agent, allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response_task = asyncio.create_task(
            client.post("/v1/agents/test-agent/stream", json={"input": "inspect"})
        )
        await asyncio.wait_for(model.started.wait(), timeout=1)
        checker.revoked = True
        response = await asyncio.wait_for(response_task, timeout=1)

    assert response.status_code == 200
    assert '"error_type":"SkillRevokedError"' in response.text
    assert "agent skill publisher is revoked" in response.text
    assert "private publisher identifier" not in response.text
    assert model.cancelled.is_set()
    assert checker.check_count == 2


@pytest.mark.asyncio
async def test_fastapi_run_can_omit_trace_while_preserving_trace_id() -> None:
    class RecordingTracer:
        calls = 0

        async def on_event(self, **_: Any) -> None:
            self.calls += 1

    tracer = RecordingTracer()
    agent = make_agent()
    agent.tracer = tracer
    app = create_app(agent, allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/agents/test-agent/run", json={"input": "hello", "include_trace": False}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"] == "ok"
    assert payload["trace_id"]
    assert payload["trace"] is None
    assert tracer.calls > 0


@pytest.mark.asyncio
async def test_fastapi_sse_stream_emits_typed_events_and_completion() -> None:
    app = create_app(make_agent(), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/stream", json={"input": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Cache-Control" in response.headers and response.headers["cache-control"] == "no-cache"
    assert "event: run_started\n" in response.text
    assert 'event: text_delta\ndata: {"type":"text_delta","data":{"text":"o"}}' in response.text
    assert "event: completed\n" in response.text
    assert '"output":"ok"' in response.text


@pytest.mark.asyncio
async def test_fastapi_sse_can_omit_trace_from_completion() -> None:
    app = create_app(make_agent(), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/agents/test-agent/stream", json={"input": "hello", "include_trace": False}
        )

    completed = next(
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ") and '"type":"completed"' in line
    )
    result = json.loads(completed)["data"]["result"]
    assert result["trace_id"]
    assert result["trace"] is None


@pytest.mark.asyncio
async def test_fastapi_sse_stream_sends_sanitized_error_event() -> None:
    class ProviderCredentialParseFailure(RuntimeError):
        pass

    class FailingStreamModel:
        name = "failing-stream"

        async def complete(self, **_: Any) -> ModelResponse:
            raise ProviderCredentialParseFailure("private model response")

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="partial provider output")
            raise ProviderCredentialParseFailure("private model response")

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    app = create_app(Agent(definition, model=FailingStreamModel()), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/stream", json={"input": "hello"})

    assert response.status_code == 200
    assert "event: error\n" in response.text
    assert "partial provider output" in response.text
    assert '"error":"agent execution failed"' in response.text
    assert '"error_type":"AgentExecutionError"' in response.text
    assert "ProviderCredentialParseFailure" not in response.text
    assert "private model response" not in response.text


@pytest.mark.asyncio
async def test_sse_transport_send_failure_cancels_run_and_releases_capacity() -> None:
    provider_cancelled = asyncio.Event()

    class BlockingStreamModel:
        name = "transport-failure-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="follow-up accepted")

        async def stream(self, **_: Any) -> Any:
            try:
                yield ModelStreamDelta(content_delta="transport-failure marker")
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 30},
    )
    app = create_app(
        Agent(definition, model=BlockingStreamModel()),
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    stream_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/stream"
    )

    async def receive() -> Message:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body" and b"transport-failure marker" in message.get(
            "body", b""
        ):
            raise OSError("client transport reset")

    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/agents/test-agent/stream",
            "raw_path": b"/v1/agents/test-agent/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        },
    )

    async with app.router.lifespan_context(app):
        response = await stream_route.endpoint(
            "test-agent", RunPayload(input="hello"), _request_with_principal()
        )
        with pytest.raises(OSError, match="client transport reset"):
            await response(scope, receive, send)
        await asyncio.wait_for(provider_cancelled.wait(), timeout=1)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            follow_up = await client.post("/v1/agents/test-agent/run", json={"input": "retry"})
        assert follow_up.status_code == 200
        assert follow_up.json()["output"] == "follow-up accepted"


@pytest.mark.asyncio
async def test_sse_run_deadline_emits_error_and_releases_capacity() -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class SlowStreamModel:
        name = "slow-stream-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="follow-up accepted")

        async def stream(self, **_: Any) -> Any:
            provider_started.set()
            try:
                yield ModelStreamDelta(content_delta="partial before timeout")
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 0.25},
    )
    app = create_app(
        Agent(definition, model=SlowStreamModel()),
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/stream", json={"input": "slow"})
        assert response.status_code == 200
        assert "partial before timeout" in response.text
        assert '"error_type":"AgentRuntimeError"' in response.text
        assert "event: completed\n" not in response.text
        assert provider_started.is_set()
        await asyncio.wait_for(provider_cancelled.wait(), timeout=1)

        follow_up = await client.post("/v1/agents/test-agent/run", json={"input": "retry"})
        assert follow_up.status_code == 200
        assert follow_up.json()["output"] == "follow-up accepted"


@pytest.mark.asyncio
async def test_sse_retry_starts_a_fresh_stateless_execution() -> None:
    class CountingStreamModel:
        name = "counting-stream-test"

        def __init__(self) -> None:
            self.runs = 0

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider path should be used")

        async def stream(self, **_: Any) -> Any:
            self.runs += 1
            yield ModelStreamDelta(content_delta=f"fresh run {self.runs}")

    model = CountingStreamModel()
    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    app = create_app(Agent(definition, model=model), allow_unauthenticated=True)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        responses = [
            await client.post("/v1/agents/test-agent/stream", json={"input": "fresh task"})
            for _ in range(2)
        ]

    assert all(response.status_code == 200 for response in responses)
    completed_events = [
        payload
        for response in responses
        for frame in response.text.split("\n\n")
        if frame.startswith("event: completed\n")
        for data_line in frame.splitlines()
        if data_line.startswith("data: ")
        for payload in [json.loads(data_line.removeprefix("data: "))]
    ]
    results = [payload["data"]["result"] for payload in completed_events]
    assert [result["output"] for result in results] == ["fresh run 1", "fresh run 2"]
    assert len({result["trace"]["trace_id"] for result in results}) == 2
    assert model.runs == 2


@pytest.mark.asyncio
async def test_fastapi_sse_keepalive_and_disconnect_cancel_provider_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("gabby.server._SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class BlockingStreamModel:
        name = "blocking-stream"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AssertionError("streaming provider path should be used")

        async def stream(self, **_: Any) -> Any:
            provider_started.set()
            try:
                yield ModelStreamDelta(content_delta="first chunk")
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 30},
    )
    app = create_app(Agent(definition, model=BlockingStreamModel()), allow_unauthenticated=True)
    stream_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/stream"
    )
    async with app.router.lifespan_context(app):
        response = await stream_route.endpoint(
            "test-agent", RunPayload(input="hello"), _request_with_principal()
        )
        body = response.body_iterator
        assert b"event: run_started" in await anext(body)
        chunk = (await anext(body)).decode("utf-8")
        assert "first chunk" in chunk
        assert provider_started.is_set()
        assert await anext(body) == b": keepalive\n\n"
        await body.aclose()
        await asyncio.wait_for(provider_cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_sse_asgi_disconnect_cancels_run_and_releases_capacity() -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class BlockingStreamModel:
        name = "disconnect-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="follow-up")

        async def stream(self, **_: Any) -> Any:
            provider_started.set()
            try:
                yield ModelStreamDelta(content_delta="disconnect marker")
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 30},
    )
    app = create_app(
        Agent(definition, model=BlockingStreamModel()),
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    stream_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/stream"
    )
    disconnect = asyncio.Event()
    sent: list[Message] = []

    async def receive() -> Message:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and b"disconnect marker" in message["body"]:
            disconnect.set()

    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/agents/test-agent/stream",
            "raw_path": b"/v1/agents/test-agent/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        },
    )

    async with app.router.lifespan_context(app):
        response = await stream_route.endpoint(
            "test-agent", RunPayload(input="hello"), _request_with_principal()
        )
        await asyncio.wait_for(response(scope, receive, send), timeout=2)
        await asyncio.wait_for(provider_cancelled.wait(), timeout=1)
        assert provider_started.is_set()

        # A leaked slot would reject this follow-up run immediately.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            follow_up = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
        assert follow_up.status_code == 200
        assert follow_up.json()["output"] == "follow-up"

    assert any(message["type"] == "http.response.start" for message in sent)


@pytest.mark.asyncio
async def test_run_disconnect_cancels_provider_and_releases_capacity() -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    call_count = 0

    class BlockingModel:
        name = "run-disconnect-test"

        async def complete(self, **_: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                provider_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    provider_cancelled.set()
            return ModelResponse(content="follow-up accepted")

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 30},
    )
    app = create_app(
        Agent(definition, model=BlockingModel()),
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    run_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/run"
    )
    disconnect = asyncio.Event()

    async def receive() -> Message:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/agents/test-agent/run",
            "raw_path": b"/v1/agents/test-agent/run",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        },
    )

    async with app.router.lifespan_context(app):
        request = Request(scope, receive)
        active_run = asyncio.create_task(
            run_route.endpoint("test-agent", RunPayload(input="cancel this"), request)
        )
        await provider_started.wait()
        disconnect.set()
        with pytest.raises(asyncio.CancelledError):
            await active_run
        await asyncio.wait_for(provider_cancelled.wait(), timeout=1)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            follow_up = await client.post("/v1/agents/test-agent/run", json={"input": "retry"})
        assert follow_up.status_code == 200
        assert follow_up.json()["output"] == "follow-up accepted"


@pytest.mark.asyncio
async def test_fastapi_rejects_unknown_agent_and_extra_request_fields() -> None:
    app = create_app(make_agent(), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        missing = await client.post("/v1/agents/other/run", json={"input": "hello"})
        extra = await client.post(
            "/v1/agents/test-agent/run", json={"input": "hello", "history": []}
        )

    assert missing.status_code == 404
    assert extra.status_code == 422


@pytest.mark.asyncio
async def test_fastapi_rejects_oversized_run_and_stream_without_consuming_capacity() -> None:
    model = FakeModel("accepted")
    app = create_app(
        make_agent(model),
        max_request_bytes=128,
        max_concurrent_runs=1,
        allow_unauthenticated=True,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        oversized_run = await client.post("/v1/agents/test-agent/run", json={"input": "x" * 1_000})
        oversized_stream = await client.post(
            "/v1/agents/test-agent/stream", json={"input": "x" * 1_000}
        )
        accepted = await client.post("/v1/agents/test-agent/run", json={"input": "small"})

    assert oversized_run.status_code == 413
    assert oversized_stream.status_code == 413
    assert accepted.status_code == 200
    assert accepted.json()["output"] == "accepted"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_fastapi_bounds_oversized_run_and_stream_results() -> None:
    class LargeModel(FakeModel):
        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="x" * 1000)

    app = create_app(
        make_agent(LargeModel("x" * 1000)),
        max_response_bytes=256,
        allow_unauthenticated=True,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        run = await client.post("/v1/agents/test-agent/run", json={"input": "large"})
        stream = await client.post("/v1/agents/test-agent/stream", json={"input": "large"})

    assert run.status_code == 500
    assert len(run.content) <= 256
    assert run.json()["detail"]["error_type"] == "ResponseSizeLimitError"
    assert stream.status_code == 200
    assert len(stream.content) <= 256
    assert '"error_type":"ResponseSizeLimitError"' in stream.text
    assert "event: completed" not in stream.text


@pytest.mark.asyncio
async def test_response_middleware_bounds_non_agent_responses() -> None:
    app = create_app(make_agent(), max_response_bytes=256, allow_unauthenticated=True)

    async def oversized_json() -> dict[str, str]:
        return {"value": "x" * 512}

    async def oversized_stream() -> StreamingResponse:
        async def body() -> Any:
            yield b"data: " + b"x" * 512 + b"\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    app.add_api_route("/oversized-json", oversized_json, methods=["GET"])
    app.add_api_route("/oversized-stream", oversized_stream, methods=["GET"])
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        json_response = await client.get("/oversized-json")
        stream_response = await client.get("/oversized-stream")

    assert json_response.status_code == 500
    assert len(json_response.content) <= 256
    assert json_response.json()["detail"]["error_type"] == "ResponseSizeLimitError"
    assert stream_response.status_code == 200
    assert len(stream_response.content) <= 256


def test_response_size_helpers_count_utf8_bytes_and_reject_nonfinite_json() -> None:
    encoded = _bounded_json_bytes({"value": "é"}, max_bytes=64)
    assert len(encoded) == len(encoded.decode("utf-8").encode("utf-8"))
    with pytest.raises(ResponseSizeLimitError):
        _bounded_json_bytes({"value": "é"}, max_bytes=len(encoded) - 1)
    with pytest.raises(ValueError):
        _bounded_json_bytes({"value": float("nan")}, max_bytes=64)


@pytest.mark.asyncio
async def test_fastapi_hides_agent_exception_details_from_callers() -> None:
    class ProviderCredentialParseFailure(RuntimeError):
        pass

    class FailingModel:
        name = "failing"

        async def complete(self, **_: Any) -> ModelResponse:
            raise ProviderCredentialParseFailure("private provider response")

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    app = create_app(Agent(definition, model=FailingModel()), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "AgentExecutionError"
    assert "ProviderCredentialParseFailure" not in response.text
    assert "private provider response" not in response.text


@pytest.mark.asyncio
async def test_fastapi_preserves_gabby_execution_error_type() -> None:
    class FailingModel:
        name = "failing-runtime"

        async def complete(self, **_: Any) -> ModelResponse:
            raise AgentRuntimeError("private runtime detail")

    definition = AgentDefinition(
        name="test-agent",
        model={"provider": "fake", "model": "test-model"},
        policies={"max_steps": 1, "timeout_seconds": 2},
    )
    app = create_app(Agent(definition, model=FailingModel()), allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})

    assert response.status_code == 500
    assert response.json()["detail"]["error_type"] == "AgentRuntimeError"
    assert "private runtime detail" not in response.text


def test_create_app_requires_authentication_or_explicit_local_mode() -> None:
    with pytest.raises(ValueError, match="authenticator or explicitly"):
        create_app(make_agent())
    with pytest.raises(ValueError, match="cannot be combined"):
        create_app(
            make_agent(),
            authenticator=BearerTokenAuthenticator("secret"),
            allow_unauthenticated=True,
        )
    local_app = create_app(make_agent(), allow_unauthenticated=True)
    openapi = local_app.openapi()
    assert openapi["info"]["version"] == "1.0.0"
    assert "/agents/{agent_name}/run" not in openapi["paths"]
    operation = openapi["paths"]["/v1/agents/{agent_name}/run"]["post"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"] == "#/components/schemas/RunResponse"
    assert operation["responses"]["408"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/RequestBodyErrorResponse"
    )
    assert operation["responses"]["413"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/RequestBodyErrorResponse"
    )
    assert (
        operation["responses"]["500"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/AgentErrorResponse"
    )
    assert "security" not in operation
    stream_operation = openapi["paths"]["/v1/agents/{agent_name}/stream"]["post"]
    assert stream_operation["responses"]["200"]["content"]["text/event-stream"]
    stream_headers = {parameter["name"] for parameter in stream_operation["parameters"]}
    assert {"Idempotency-Key", "Last-Event-ID"}.issubset(stream_headers)
    assert {"400", "404", "409"}.issubset(stream_operation["responses"])
    assert (
        stream_operation["responses"]["408"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/RequestBodyErrorResponse"
    )
    assert (
        stream_operation["responses"]["413"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/RequestBodyErrorResponse"
    )
    assert "security" not in stream_operation


@pytest.mark.parametrize("max_request_bytes", [0, -1, True, 1.5])
def test_create_app_rejects_invalid_request_body_limits(max_request_bytes: object) -> None:
    with pytest.raises(ValueError, match="max_request_bytes"):
        create_app(
            make_agent(),
            max_request_bytes=max_request_bytes,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), 10**1000])
def test_create_app_rejects_invalid_request_body_timeouts(timeout: object) -> None:
    with pytest.raises(ValueError, match="request_body_timeout_seconds"):
        create_app(
            make_agent(),
            request_body_timeout_seconds=timeout,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), 10**1000])
def test_create_app_rejects_invalid_authenticator_timeouts(timeout: object) -> None:
    with pytest.raises(ValueError, match="authenticator_timeout_seconds"):
        create_app(
            make_agent(),
            authenticator_timeout_seconds=timeout,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.parametrize("max_concurrent_runs", [0, -1, True, 1.5])
def test_create_app_rejects_invalid_run_capacity(max_concurrent_runs: object) -> None:
    with pytest.raises(ValueError, match="max_concurrent_runs"):
        create_app(
            make_agent(),
            max_concurrent_runs=max_concurrent_runs,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.parametrize("max_resumable_streams", [0, -1, True, 1.5])
def test_create_app_rejects_invalid_resumable_stream_capacity(
    max_resumable_streams: object,
) -> None:
    with pytest.raises(ValueError, match="max_resumable_streams"):
        create_app(
            make_agent(),
            max_resumable_streams=max_resumable_streams,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.parametrize("ttl", [0, -1, True, float("inf"), float("nan")])
def test_create_app_rejects_invalid_stream_session_ttl(ttl: object) -> None:
    with pytest.raises(ValueError, match="stream_session_ttl_seconds"):
        create_app(
            make_agent(),
            stream_session_ttl_seconds=ttl,  # type: ignore[arg-type]
            allow_unauthenticated=True,
        )


@pytest.mark.asyncio
async def test_run_and_stream_share_configured_capacity_limit() -> None:
    app = create_app(make_agent(), max_concurrent_runs=1, allow_unauthenticated=True)
    stream_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/v1/agents/{agent_name}/stream"
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        stream_response = await stream_route.endpoint(
            "test-agent",
            RunPayload(input="reserve one execution slot"),
            _request_with_principal(),
        )
        overloaded = await client.post("/v1/agents/test-agent/run", json={"input": "busy"})
        overloaded_stream = await client.post(
            "/v1/agents/test-agent/stream", json={"input": "also busy"}
        )
        health = await client.get("/health")
        assert overloaded.status_code == 429
        assert overloaded.json()["detail"]["error_type"] == "CapacityLimitError"
        assert overloaded_stream.status_code == 429
        assert health.status_code == 200

        body = stream_response.body_iterator
        assert b"event: run_started" in await anext(body)
        await body.aclose()
        accepted = await client.post("/v1/agents/test-agent/run", json={"input": "available"})

    assert accepted.status_code == 200


def test_bearer_token_validation_and_environment_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    for invalid in ("", "contains space", "line\nbreak", "tøken"):
        with pytest.raises(ValueError):
            BearerTokenAuthenticator(invalid)
    monkeypatch.delenv("GABBY_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="environment variable is not set"):
        BearerTokenAuthenticator.from_env()
    monkeypatch.setenv("GABBY_API_TOKEN", "loaded-secret")
    authenticator = BearerTokenAuthenticator.from_env()
    assert isinstance(authenticator, BearerTokenAuthenticator)
    with pytest.raises(ValueError, match="valid identifier"):
        BearerTokenAuthenticator.from_env("BAD-NAME")
    with pytest.raises(ValueError, match="principal subject"):
        Principal(" ")
    with pytest.raises(ValueError, match="scope tokens"):
        Principal("caller", frozenset({"bad scope"}))


@pytest.mark.asyncio
async def test_bearer_authenticator_attaches_configured_scopes() -> None:
    authenticator = BearerTokenAuthenticator(
        "token", scopes=frozenset({"agent:run", "knowledge:read"})
    )

    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer token")]})
    assert await authenticator.authenticate(request) == Principal(
        "bearer-token", frozenset({"agent:run", "knowledge:read"})
    )


@pytest.mark.asyncio
async def test_bearer_auth_protects_run_and_leaves_health_public() -> None:
    app = create_app(make_agent(), authenticator=BearerTokenAuthenticator("test-secret"))
    schema = app.openapi()["paths"]
    assert schema["/v1/agents/{agent_name}/run"]["post"]["security"] == [{"HTTPBearer": []}]
    assert schema["/v1/agents/{agent_name}/stream"]["post"]["security"] == [{"HTTPBearer": []}]
    assert "security" not in schema["/health"]["get"]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).status_code == 200
        missing = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
        stream_missing = await client.post("/v1/agents/test-agent/stream", json={"input": "hello"})
        wrong = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer wrong"},
        )
        malformed = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Basic test-secret"},
        )
        allowed = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "bEaReR test-secret"},
        )
    assert missing.status_code == wrong.status_code == malformed.status_code == 401
    assert stream_missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert allowed.status_code == 200


@pytest.mark.asyncio
async def test_api_enforces_configured_required_scopes() -> None:
    app = create_app(
        make_agent(),
        authenticator=BearerTokenAuthenticator(
            "scoped-token", scopes=frozenset({"agent:run", "agent:stream"})
        ),
        run_scopes=("agent:run",),
        stream_scopes=("agent:stream",),
    )
    schema = app.openapi()["paths"]
    assert "403" in schema["/v1/agents/{agent_name}/run"]["post"]["responses"]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        allowed = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer scoped-token"},
        )
        stream = await client.post(
            "/v1/agents/test-agent/stream",
            json={"input": "hello"},
            headers={"Authorization": "Bearer scoped-token"},
        )
    assert allowed.status_code == stream.status_code == 200

    run_only_app = create_app(
        make_agent(),
        authenticator=BearerTokenAuthenticator("run-only", scopes=frozenset({"agent:run"})),
        run_scopes=("agent:run",),
        stream_scopes=("agent:stream",),
    )
    async with (
        run_only_app.router.lifespan_context(run_only_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=run_only_app), base_url="http://test"
        ) as client,
    ):
        run_allowed = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer run-only"},
        )
        stream_denied = await client.post(
            "/v1/agents/test-agent/stream",
            json={"input": "hello"},
            headers={"Authorization": "Bearer run-only"},
        )
    assert run_allowed.status_code == 200
    assert stream_denied.status_code == 403

    denied_model = FakeModel()
    missing_scope_app = create_app(
        make_agent(denied_model),
        authenticator=BearerTokenAuthenticator("unscoped-token"),
        run_scopes=("agent:run",),
    )
    async with (
        missing_scope_app.router.lifespan_context(missing_scope_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=missing_scope_app), base_url="http://test"
        ) as client,
    ):
        denied = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer unscoped-token"},
        )
    assert denied.status_code == 403
    assert denied_model.calls == []


def test_api_rejects_invalid_required_scope_configuration() -> None:
    authenticator = BearerTokenAuthenticator("token")
    for scopes in (("bad scope",), ("agent:run", "agent:run"), ["agent:run"]):
        with pytest.raises(ValueError, match="run_scopes"):
            create_app(
                make_agent(),
                authenticator=authenticator,
                run_scopes=scopes,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="require an authenticator"):
        create_app(make_agent(), allow_unauthenticated=True, run_scopes=("agent:run",))


@pytest.mark.asyncio
async def test_pluggable_authenticator_and_provider_failure_handling() -> None:
    seen: list[str] = []

    class CustomAuthenticator:
        async def authenticate(self, request: Any) -> Principal | None:
            seen.append(request.url.path)
            return Principal("custom-user")

    app = create_app(make_agent(), authenticator=CustomAuthenticator())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
    assert response.status_code == 200
    assert seen == ["/v1/agents/test-agent/run"]

    class BrokenAuthenticator:
        async def authenticate(self, request: Any) -> Principal | None:
            raise RuntimeError("private identity provider detail")

    app = create_app(make_agent(), authenticator=BrokenAuthenticator())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
    assert response.status_code == 503
    assert "private identity provider detail" not in response.text


@pytest.mark.asyncio
async def test_authenticator_timeout_is_sanitized_and_releases_run_capacity() -> None:
    class SlowOnceAuthenticator:
        calls = 0

        async def authenticate(self, request: Request) -> Principal:
            self.calls += 1
            if self.calls == 1:
                await asyncio.Event().wait()
            return Principal("custom-user")

    authenticator = SlowOnceAuthenticator()
    app = create_app(
        make_agent(),
        authenticator=authenticator,
        authenticator_timeout_seconds=0.01,
        max_concurrent_runs=1,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        timed_out = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})
        succeeded = await client.post("/v1/agents/test-agent/run", json={"input": "hello"})

    assert timed_out.status_code == 503
    assert timed_out.json() == {"detail": "authentication service unavailable"}
    assert succeeded.status_code == 200
    assert authenticator.calls == 2


@pytest.mark.asyncio
async def test_authenticated_principal_reaches_tool_approval_outside_model_context() -> None:
    approval_requests: list[ApprovalRequest] = []
    executed: list[bool] = []

    def publish() -> dict[str, bool]:
        executed.append(True)
        return {"published": True}

    class Authenticator:
        async def authenticate(self, request: Request) -> Principal:
            assert request.headers.get("authorization") == "Bearer valid"
            return Principal("authenticated-caller")

    class Approval:
        async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
            approval_requests.append(request)
            return ApprovalDecision(approved=True)

    class Model:
        name = "approval-api-test"

        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    tool_calls=[
                        {
                            "id": "approved-call",
                            "type": "function",
                            "function": {"name": "publish", "arguments": "{}"},
                        }
                    ]
                ),
                ModelResponse(content="published"),
            ]
            self.calls: list[dict[str, Any]] = []

        async def complete(self, **kwargs: Any) -> ModelResponse:
            self.calls.append(kwargs)
            return self.responses.pop(0)

    tools = ToolRegistry()
    tools.register(
        Tool(
            output_schema=_ANY_JSON_SCHEMA,
            name="publish",
            description="Publish the prepared item.",
            parameters={"type": "object", "properties": {}},
            handler=publish,
            requires_approval=True,
        )
    )
    model = Model()
    definition = AgentDefinition(
        name="approval-agent",
        model={"provider": "test", "model": "approval-model"},
        tools=["publish"],
        policies={"max_steps": 2, "timeout_seconds": 2, "allowed_tools": ["publish"]},
    )
    app = create_app(
        Agent(definition, model=model, tools=tools, approval_handler=Approval()),
        authenticator=Authenticator(),
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/agents/approval-agent/run",
            headers={"Authorization": "Bearer valid"},
            json={"input": "publish", "metadata": {"principal": "untrusted-metadata"}},
        )

    assert response.status_code == 200
    assert executed == [True]
    assert len(approval_requests) == 1
    assert approval_requests[0].principal == Principal("authenticated-caller")
    assert "authenticated-caller" not in str(model.calls[0]["messages"])


@pytest.mark.asyncio
async def test_body_limit_middleware_passes_non_http_and_replays_buffered_body() -> None:
    observed: list[Message] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        observed.append(await receive())
        observed.append(await receive())

    middleware = BodySizeLimitMiddleware(downstream, max_bytes=100)
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    web_socket_scope = cast(Scope, {"type": "websocket"})

    async def unused_receive() -> Message:
        return {"type": "http.disconnect"}

    await middleware(web_socket_scope, unused_receive, send)
    assert observed == [{"type": "http.disconnect"}, {"type": "http.disconnect"}]

    observed.clear()
    incoming = iter(
        [
            {"type": "http.request", "body": b"first", "more_body": False},
            {"type": "http.request", "body": b"second", "more_body": False},
        ]
    )

    async def receive() -> Message:
        return next(incoming)

    http_scope = cast(Scope, {"type": "http"})
    await middleware(http_scope, receive, send)

    assert observed == [
        {"type": "http.request", "body": b"first", "more_body": False},
        {"type": "http.request", "body": b"second", "more_body": False},
    ]
    assert sent == []


@pytest.mark.asyncio
async def test_body_limit_middleware_stops_on_disconnect_and_serve_forwards_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = BodySizeLimitMiddleware(
        lambda *_: None,  # type: ignore[arg-type]
        max_bytes=100,
    )
    messages: list[Message] = []

    async def receive_disconnect() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)

    await middleware(cast(Scope, {"type": "http"}), receive_disconnect, send)
    assert messages == []

    forwarded: dict[str, Any] = {}

    def run(app: Any, *, host: str, port: int, log_level: str) -> None:
        forwarded.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr("uvicorn.run", run)
    agent = make_agent()
    with pytest.raises(ValueError, match="requires an authenticator"):
        serve(agent, host="0.0.0.0", port=9000)
    with pytest.raises(ValueError, match="not both"):
        serve(
            agent,
            host="0.0.0.0",
            port=9000,
            authenticator=BearerTokenAuthenticator("server-token"),
            bearer_token="other-token",
        )
    serve(
        agent,
        host="0.0.0.0",
        port=9000,
        bearer_token="server-token",
    )

    assert forwarded["host"] == "0.0.0.0"
    assert forwarded["port"] == 9000
    assert forwarded["log_level"] == "info"


def test_serve_forwards_custom_authenticator_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_app(agent: Agent, **kwargs: Any) -> object:
        captured.update(agent=agent, **kwargs)
        return object()

    def fake_run(app: object, *, host: str, port: int, log_level: str) -> None:
        captured.update(app=app, host=host, port=port, log_level=log_level)

    class CustomAuthenticator:
        async def authenticate(self, request: Request) -> Principal:
            return Principal("caller")

    monkeypatch.setattr("gabby.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_run)
    authenticator = CustomAuthenticator()

    serve(
        make_agent(),
        host="0.0.0.0",
        authenticator=authenticator,
        authenticator_timeout_seconds=2.5,
    )

    assert captured["authenticator"] is authenticator
    assert captured["authenticator_timeout_seconds"] == 2.5


@pytest.mark.asyncio
async def test_body_limit_middleware_returns_bounded_timeout_for_slow_request() -> None:
    downstream_called = False
    sent: list[Message] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    await BodySizeLimitMiddleware(downstream, max_bytes=100, timeout_seconds=0.01)(
        cast(Scope, {"type": "http", "headers": []}), receive, send
    )

    assert downstream_called is False
    assert sent[0]["status"] == 408
    assert (b"connection", b"close") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == {"detail": "request body timed out"}


@pytest.mark.asyncio
async def test_body_capacity_rejects_before_reading_when_run_slots_are_full() -> None:
    capacity = _RunCapacity(1)
    occupied = capacity.try_acquire()
    assert occupied is not None
    received = False
    sent: list[Message] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        pytest.fail("capacity-rejected requests must not reach FastAPI")

    async def receive() -> Message:
        nonlocal received
        received = True
        return {"type": "http.request", "body": b"payload", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = BodySizeLimitMiddleware(
        downstream,
        max_bytes=100,
        run_capacity=capacity,
    )
    await middleware(
        cast(
            Scope,
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": "/v1/agents/test-agent/run",
                "headers": [(b"content-length", b"7")],
            },
        ),
        receive,
        send,
    )

    assert sent[0]["status"] == 429
    assert received is False
    occupied.release()


@pytest.mark.asyncio
async def test_resumable_request_capacity_is_bounded_separately_from_execution_capacity() -> None:
    capacity = _RunCapacity(1)
    downstream_started = asyncio.Event()
    finish_downstream = asyncio.Event()
    sent_for_second: list[Message] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("path") == "/v1/agents/test-agent/stream":
            downstream_started.set()
            await finish_downstream.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def discard_send(_message: Message) -> None:
        return None

    async def capture_second(message: Message) -> None:
        sent_for_second.append(message)

    middleware = BodySizeLimitMiddleware(downstream, max_bytes=100, run_capacity=capacity)
    scope = cast(
        Scope,
        {
            "type": "http",
            "http_version": "2",
            "method": "POST",
            "path": "/v1/agents/test-agent/stream",
            "headers": [(b"idempotency-key", b"stream-capacity-key")],
        },
    )
    first_request = asyncio.create_task(middleware(scope, receive, discard_send))
    await asyncio.wait_for(downstream_started.wait(), timeout=1)

    await middleware(dict(scope), receive, capture_second)
    assert sent_for_second[0]["status"] == 429
    assert json.loads(sent_for_second[1]["body"])["detail"]["error"] == (
        "request capacity exhausted"
    )

    finish_downstream.set()
    await first_request
    assert capacity.active == 0


@pytest.mark.asyncio
async def test_health_check_bypasses_a_full_execution_capacity() -> None:
    capacity = _RunCapacity(1)
    occupied = capacity.try_acquire()
    assert occupied is not None
    observed: list[Message] = []

    async def downstream(_scope: Scope, receive: Receive, _send: Send) -> None:
        observed.append(await receive())

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message: Message) -> None:
        return None

    await BodySizeLimitMiddleware(downstream, max_bytes=100, run_capacity=capacity)(
        cast(
            Scope,
            {"type": "http", "method": "GET", "path": "/health", "headers": []},
        ),
        receive,
        send,
    )
    assert observed == [{"type": "http.request", "body": b"", "more_body": False}]
    assert capacity.active == 1
    occupied.release()


@pytest.mark.asyncio
async def test_body_capacity_is_held_until_downstream_response_finishes() -> None:
    capacity = _RunCapacity(1)
    response_complete = False

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        assert capacity.active == 1
        assert scope["state"]["_gabby_run_capacity_lease"] is not None
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message: Message) -> None:
        nonlocal response_complete
        if message["type"] == "http.response.body":
            assert capacity.active == 1
            response_complete = True

    await BodySizeLimitMiddleware(
        downstream,
        max_bytes=100,
        run_capacity=capacity,
    )(
        cast(
            Scope,
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": "/v1/agents/test-agent/run",
                "headers": [(b"content-length", b"2")],
            },
        ),
        receive,
        send,
    )

    assert response_complete is True
    assert capacity.active == 0


@pytest.mark.asyncio
async def test_body_timeout_does_not_emit_connection_header_for_http2() -> None:
    sent: list[Message] = []

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    await BodySizeLimitMiddleware(
        lambda *_: None,  # type: ignore[arg-type]
        max_bytes=100,
        timeout_seconds=0.01,
    )(cast(Scope, {"type": "http", "http_version": "2", "headers": []}), receive, send)

    assert sent[0]["status"] == 408
    assert all(name != b"connection" for name, _ in sent[0]["headers"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "chunks"),
    [
        ([(b"content-length", b"5")], []),
        (
            [(b"content-length", b"not-a-number")],
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"45", "more_body": False},
            ],
        ),
    ],
)
async def test_body_limit_middleware_rejects_declared_and_chunked_oversize(
    headers: list[tuple[bytes, bytes]], chunks: list[Message]
) -> None:
    called_downstream = False
    sent: list[Message] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal called_downstream
        called_downstream = True

    async def receive() -> Message:
        return chunks.pop(0)

    async def send(message: Message) -> None:
        sent.append(message)

    await BodySizeLimitMiddleware(downstream, max_bytes=4)(
        cast(Scope, {"type": "http", "headers": headers}), receive, send
    )

    assert called_downstream is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"detail": "request body too large"}


@pytest.mark.asyncio
async def test_response_limit_middleware_forwards_protocol_events_and_stops_oversized_sse() -> None:
    forwarded: list[Message] = []

    async def send(message: Message) -> None:
        forwarded.append(message)

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def non_http_app(scope: Scope, inner_receive: Receive, inner_send: Send) -> None:
        await inner_send({"type": "http.response.body", "body": b"passthrough"})

    middleware = ResponseSizeLimitMiddleware(non_http_app, max_bytes=4)
    await middleware(cast(Scope, {"type": "websocket"}), receive, send)
    assert forwarded == [{"type": "http.response.body", "body": b"passthrough"}]

    forwarded.clear()

    async def streaming_app(scope: Scope, inner_receive: Receive, inner_send: Send) -> None:
        await inner_send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await inner_send({"type": "http.response.body", "body": b"1234", "more_body": True})
        await inner_send({"type": "http.response.body", "body": b"5", "more_body": True})
        await inner_send({"type": "http.response.body", "body": b"ignored", "more_body": False})

    await ResponseSizeLimitMiddleware(streaming_app, max_bytes=4)(
        cast(Scope, {"type": "http"}), receive, send
    )
    assert forwarded == [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        },
        {"type": "http.response.body", "body": b"1234", "more_body": True},
        {"type": "http.response.body", "body": b"", "more_body": False},
    ]


@pytest.mark.asyncio
async def test_response_limit_middleware_forwards_non_body_messages_without_a_start() -> None:
    sent: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.debug", "info": "trace"})
        await send({"type": "http.response.body", "body": b"body"})

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await ResponseSizeLimitMiddleware(app, max_bytes=10)(
        cast(Scope, {"type": "http"}), receive, send
    )
    assert sent == [
        {"type": "http.response.debug", "info": "trace"},
        {"type": "http.response.body", "body": b"body"},
    ]


def test_serve_defaults_to_loopback_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded: dict[str, Any] = {}

    def run(app: Any, *, host: str, port: int, log_level: str) -> None:
        forwarded.update(host=host, port=port, app=app)

    monkeypatch.setattr("uvicorn.run", run)
    serve(make_agent())
    assert forwarded["host"] == "127.0.0.1"
    assert forwarded["port"] == 8787


@pytest.mark.asyncio
async def test_serve_applies_bearer_capabilities_to_route_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(app: Any, *, host: str, port: int, log_level: str) -> None:
        captured["app"] = app

    monkeypatch.setattr("uvicorn.run", run)
    serve(
        make_agent(),
        bearer_token="scoped-token",
        bearer_scopes=frozenset({"agent:run"}),
        run_scopes=("agent:run",),
        stream_scopes=("agent:stream",),
    )
    app = captured["app"]

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        run_response = await client.post(
            "/v1/agents/test-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer scoped-token"},
        )
        stream_response = await client.post(
            "/v1/agents/test-agent/stream",
            json={"input": "hello"},
            headers={"Authorization": "Bearer scoped-token"},
        )

    assert run_response.status_code == 200
    assert stream_response.status_code == 403
