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
"""Acceptance checks for runnable, user-facing examples."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from examples.agent_composition import run_example as run_composition_example
from examples.approval_audit import AuditedApprovalHandler, SQLiteApprovalAudit
from examples.hosted_service import create_app
from examples.opentelemetry_tracer import OpenTelemetryEventTracer
from examples.sqlite_data_agent import run_example as run_sqlite_data_example
from examples.stateless_sse_client import stream_agent
from gabby import Agent, AgentDefinition, ApprovalRequest, Principal, TraceEvent
from gabby.models import ModelResponse, ModelStreamDelta
from gabby.server import create_app as create_gabby_app

ROOT = Path(__file__).resolve().parents[1]


class _SSEByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _ExampleStreamingModel:
    name = "example-streaming-model"

    async def complete(self, **_kwargs: Any) -> ModelResponse:
        return ModelResponse(content="Tree cover grew.")

    async def stream(self, **_kwargs: Any) -> AsyncIterator[ModelStreamDelta]:
        yield ModelStreamDelta(content_delta="Tree ")
        yield ModelStreamDelta(content_delta="cover grew.")


class _FakeSpan:
    def __init__(self) -> None:
        self.end_time: int | None = None

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def end(self, *, end_time: int) -> None:
        self.end_time = end_time


class _FakeOpenTelemetryTracer:
    def __init__(self) -> None:
        self.name: str | None = None
        self.attributes: dict[str, Any] | None = None
        self.start_time: int | None = None
        self.span = _FakeSpan()

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any],
        start_time: int,
    ) -> _FakeSpan:
        self.name = name
        self.attributes = attributes
        self.start_time = start_time
        return self.span


@pytest.mark.asyncio
async def test_offline_agent_composition_example_runs_and_links_traces() -> None:
    (
        output,
        parent_trace_id,
        child_trace_id,
        linked_parent_trace_id,
    ) = await run_composition_example()

    assert output.startswith("Coordinator summary:")
    assert parent_trace_id
    assert child_trace_id
    assert linked_parent_trace_id == parent_trace_id


@pytest.mark.asyncio
async def test_read_only_sqlite_data_agent_example_runs() -> None:
    output = await run_sqlite_data_example()
    assert "north" in output
    assert "40.0" in output


@pytest.mark.asyncio
async def test_approval_audit_example_records_decision_without_raw_arguments(
    tmp_path: Path,
) -> None:
    database = tmp_path / "approval.sqlite3"
    audit = SQLiteApprovalAudit(database, include_principal_subject=True)
    received: list[ApprovalRequest] = []

    async def review(request: ApprovalRequest) -> bool:
        received.append(request)
        return True

    request = ApprovalRequest(
        agent_name="report-agent",
        run_id="run-1",
        tool_name="send_report",
        call_id="call-1",
        arguments={"recipient": "private@example.test"},
        principal=Principal("reviewer-7", frozenset({"reports:approve"})),
    )
    decision = await AuditedApprovalHandler(review, audit).approve(request)

    assert decision.approved is True
    assert received == [request]
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT run_id, call_id, tool_name, principal_subject, "
            "arguments_sha256, approved FROM gabby_tool_approval_audit"
        ).fetchone()
    assert row is not None
    assert row[:4] == ("run-1", "call-1", "send_report", "reviewer-7")
    assert len(row[4]) == 64
    assert row[5] == 1
    assert "private@example.test" not in repr(row)


@pytest.mark.asyncio
async def test_stateless_sse_client_sends_caller_state_and_decodes_chunked_events() -> None:
    request_data: dict[str, Any] = {}
    frames: list[dict[str, Any]] = [
        {"type": "run_started", "data": {}},
        {"type": "text_delta", "data": {"text": "héllo"}},
        {"type": "completed", "data": {"result": {"trace_id": "trace-1"}}},
    ]
    wire = b"".join(
        f"event: {frame['type']}\r\ndata: ".encode()
        + json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\r\n\r\n"
        for frame in frames
    )
    split = wire.index(b"\xc3") + 1
    chunks = [wire[:split], wire[split : split + 3], wire[split + 3 :]]

    async def respond(request: httpx.Request) -> httpx.Response:
        request_data["path"] = request.url.path
        request_data["authorization"] = request.headers.get("authorization")
        request_data["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=_SSEByteStream(chunks),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        events = [
            event
            async for event in stream_agent(
                client,
                base_url="https://gabby.example",
                agent_name="research-synthesizer",
                api_token="host-secret",
                task="Summarize the supplied records",
                context={"records": ["source text"]},
                memory={"application": "state"},
            )
        ]

    assert request_data == {
        "path": "/v1/agents/research-synthesizer/stream",
        "authorization": "Bearer host-secret",
        "payload": {
            "input": "Summarize the supplied records",
            "context": {"records": ["source text"]},
            "memory": {"application": "state"},
            "include_trace": False,
        },
    }
    assert [event["type"] for event in events] == ["run_started", "text_delta", "completed"]
    assert events[1]["data"]["text"] == "héllo"
    assert events[2]["data"]["result"]["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_stateless_sse_client_interoperates_with_embedded_gabby_service() -> None:
    definition = AgentDefinition(
        name="research-synthesizer",
        model={"provider": "example", "model": "streaming"},
        policies={"max_steps": 1, "timeout_seconds": 5},
    )
    agent = Agent(definition, model=_ExampleStreamingModel())
    app = create_gabby_app(agent, allow_unauthenticated=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8787"
        ) as client,
    ):
        events = [
            event
            async for event in stream_agent(
                client,
                base_url="http://127.0.0.1:8787",
                agent_name="research-synthesizer",
                api_token="local-example-token",
                task="Summarize the current context",
                context={"source": "application data"},
                memory={"conversation": "application owned"},
            )
        ]

    text = "".join(event["data"]["text"] for event in events if event["type"] == "text_delta")
    completed = next(event for event in events if event["type"] == "completed")
    assert text == "Tree cover grew."
    assert completed["data"]["result"]["output"] == text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://gabby.example",
        "https://user:secret@gabby.example",
        "https://gabby.example?token=secret",
    ],
)
async def test_stateless_sse_client_rejects_insecure_or_credentialed_base_urls(
    base_url: str,
) -> None:
    async def fail_if_requested(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid base URL must be rejected before making a request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_if_requested)) as client:
        with pytest.raises(ValueError, match="base URL"):
            _ = [
                event
                async for event in stream_agent(
                    client,
                    base_url=base_url,
                    agent_name="research-synthesizer",
                    api_token="host-secret",
                    task="Summarize",
                )
            ]


@pytest.mark.asyncio
async def test_opentelemetry_example_exports_allowlisted_trace_attributes() -> None:
    upstream = _FakeOpenTelemetryTracer()
    exporter = OpenTelemetryEventTracer(upstream)
    event = TraceEvent(
        kind="model_response",
        timestamp=1_750_000_000,
        duration_ms=12.5,
        details={
            "step": 2,
            "tool_call_count": 1,
            "content": "private generated output",
            "arguments": {"secret": "private tool input"},
        },
    )

    await exporter.on_event(trace_id="run-123", agent_name="support", event=event)

    assert upstream.name == "gabby.model_response"
    assert upstream.start_time == 1_749_999_999_987_500_000
    assert upstream.span.end_time == 1_750_000_000_000_000_000
    assert upstream.attributes == {
        "gabby.trace_id": "run-123",
        "gabby.agent.name": "support",
        "gabby.event.kind": "model_response",
        "gabby.step": 2,
        "gabby.tool_call_count": 1,
        "gabby.event.duration_ms": 12.5,
    }


@pytest.mark.asyncio
async def test_opentelemetry_example_works_with_optional_sdk() -> None:
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    export_api = pytest.importorskip("opentelemetry.sdk.trace.export")
    memory_export = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    provider = trace_sdk.TracerProvider()
    memory = memory_export.InMemorySpanExporter()
    provider.add_span_processor(export_api.SimpleSpanProcessor(memory))
    exporter = OpenTelemetryEventTracer(provider.get_tracer("gabby-test"))

    await exporter.on_event(
        trace_id="run-456",
        agent_name="research",
        event=TraceEvent(
            kind="tool_error",
            timestamp=1_750_000_000,
            duration_ms=3.0,
            details={"arguments": {"secret": "must not export"}},
        ),
    )

    (span,) = memory.get_finished_spans()
    assert span.name == "gabby.tool_error"
    assert span.status.status_code.name == "ERROR"
    assert span.attributes == {
        "gabby.trace_id": "run-456",
        "gabby.agent.name": "research",
        "gabby.event.kind": "tool_error",
        "gabby.event.duration_ms": 3.0,
    }
    provider.shutdown()


@pytest.mark.asyncio
async def test_hosted_service_factory_loads_config_and_requires_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GABBY_AGENT_CONFIG", str(ROOT / "examples" / "hosted-agent.yaml"))
    monkeypatch.setenv("GABBY_API_TOKEN", "hosted-example-secret")
    monkeypatch.setenv("HF_TOKEN", "model-token-for-construction-only")

    app = create_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).json() == {"status": "ok"}
        response = await client.post(
            "/v1/agents/research-synthesizer/run",
            json={"input": "Summarize the supplied source."},
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
