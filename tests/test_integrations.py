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

from __future__ import annotations

import asyncio
import json
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gabby.agent import Agent
from gabby.client import GabbyClient
from gabby.config import AgentDefinition
from gabby.mcp import _client_type, create_mcp_server, load_mcp_tools
from gabby.orchestrators import crewai_tool, langgraph_node
from gabby.tracing import (
    LFTraceExporter,
    OpenTelemetryTracer,
    SignedAuditExporter,
    TraceEvent,
)


class _InlineExecutor:
    """Keep exporter format tests deterministic; async offloading has separate coverage."""

    def submit(self, function: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


@pytest.mark.asyncio
async def test_mcp_tools_are_allowlisted_and_return_structured_results() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def list_tools(self) -> Any:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="lookup",
                        description="Look up a record",
                        inputSchema={"type": "object", "properties": {"id": {"type": "string"}}},
                    ),
                    SimpleNamespace(
                        name="delete",
                        description="Delete a record",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            calls.append((self.url, name, arguments))
            return SimpleNamespace(isError=False, structuredContent={"name": name})

    registry = await load_mcp_tools(
        {
            "servers": {
                "records": {
                    "url": "https://mcp.example.test",
                    "allow_tools": ["lookup", "delete"],
                }
            }
        },
        client_factory=FakeClient,
    )
    assert registry.names() == ["records_delete", "records_lookup"]
    lookup = registry.get("records_lookup")
    delete = registry.get("records_delete")
    assert lookup.handler is not None and delete.handler is not None
    # Reserved routing arguments from an untrusted model call stay ordinary arguments and cannot
    # change the configured server or remote tool selected by the handler closure.
    assert await lookup.handler(
        id="42", _server_url="https://attacker.invalid", _remote_name="delete"
    ) == {"name": "lookup"}
    assert await delete.handler() == {"name": "delete"}
    assert calls[0][0:2] == ("https://mcp.example.test", "lookup")
    assert calls[1][0:2] == ("https://mcp.example.test", "delete")


@pytest.mark.asyncio
async def test_mcp_default_client_uses_supported_v1_session_transport(monkeypatch: Any) -> None:
    import contextlib
    import types

    initialized: list[bool] = []
    used_urls: list[str] = []

    class ClientSession:
        def __init__(self, read_stream: Any, write_stream: Any) -> None:
            assert read_stream == "read"
            assert write_stream == "write"

        async def __aenter__(self) -> ClientSession:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def initialize(self) -> None:
            initialized.append(True)

    @contextlib.asynccontextmanager
    async def streamable_http_client(url: str) -> Any:
        used_urls.append(url)
        yield "read", "write", None

    mcp_module = types.ModuleType("mcp")
    mcp_module.__path__ = []
    client_module = types.ModuleType("mcp.client")
    client_module.__path__ = []
    session_module = types.ModuleType("mcp.client.session")
    session_module.ClientSession = ClientSession  # type: ignore[attr-defined]
    transport_module = types.ModuleType("mcp.client.streamable_http")
    transport_module.streamable_http_client = streamable_http_client  # type: ignore[attr-defined]
    client_module.session = session_module  # type: ignore[attr-defined]
    client_module.streamable_http = transport_module  # type: ignore[attr-defined]
    mcp_module.client = client_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.session", session_module)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", transport_module)

    async with _client_type()("https://mcp.example.test") as client:
        assert isinstance(client, ClientSession)

    assert initialized == [True]
    assert used_urls == ["https://mcp.example.test"]


def test_mcp_server_can_read_resolved_agent_configuration(monkeypatch: Any) -> None:
    import types

    class Model:
        name = "fixture"

        async def complete(self, **_: Any) -> Any:
            from gabby.models import ModelResponse

            return ModelResponse(content="governed result")

    class FakeFastMCP:
        def __init__(self, name: str, **_: Any) -> None:
            self.name = name
            self.tools: dict[str, Any] = {}

        def tool(self) -> Any:
            def register(function: Any) -> Any:
                self.tools[function.__name__] = function
                return function

            return register

    root = types.ModuleType("mcp")
    root.__path__ = []
    server_package = types.ModuleType("mcp.server")
    server_package.__path__ = []
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", root)
    monkeypatch.setitem(sys.modules, "mcp.server", server_package)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    agent = Agent(
        AgentDefinition(
            name="mcp-agent",
            model={"provider": "fake", "model": "fixture"},
            mcp={"expose": True},
            policies={"max_steps": 1, "timeout_seconds": 2},
        ),
        model=Model(),
    )
    server = create_mcp_server(agent, trusted_transport=True)
    assert server.name == "mcp-agent"
    result = asyncio.run(server.tools["run_agent"]("do the task"))
    assert result["output"] == "governed result"
    assert isinstance(result["trace_id"], str) and result["trace_id"]


@pytest.mark.asyncio
async def test_langgraph_adapter_maps_input_to_result_and_trace() -> None:
    class Client:
        async def arun(self, task: str, *, context: Any = None) -> Any:
            assert task == "Summarize this"
            assert context == {"locale": "en"}
            return SimpleNamespace(output="Summary", trace_id="trace-1")

    node = langgraph_node(cast(GabbyClient, Client()), context_key="context")
    assert await node({"input": "Summarize this", "context": {"locale": "en"}}) == {
        "gabby_output": "Summary",
        "gabby_trace_id": "trace-1",
    }


@pytest.mark.asyncio
async def test_crewai_adapter_uses_requested_name_and_closes_owned_clients(
    monkeypatch: Any,
) -> None:
    import types

    class BaseTool:
        def __init__(self, *, name: str, description: str) -> None:
            self.name = name
            self.description = description

    package = types.ModuleType("crewai")
    package.__path__ = []
    tools_module = types.ModuleType("crewai.tools")
    tools_module.BaseTool = BaseTool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", package)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools_module)

    class Client:
        def __init__(self) -> None:
            self.closed = False
            self.async_closed = False

        def clone(self) -> Client:
            clone = Client()
            clients.append(clone)
            return clone

        def run(self, task: str) -> Any:
            return SimpleNamespace(output=f"sync:{task}")

        async def arun(self, task: str) -> Any:
            return SimpleNamespace(output=f"async:{task}")

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.async_closed = True

    clients: list[Client] = []
    tool = crewai_tool(
        cast(GabbyClient, Client()), name="review_with_gabby", description="Governed review"
    )
    assert tool.name == "review_with_gabby"
    assert tool.description == "Governed review"
    assert tool._run("task") == "sync:task"
    assert await tool._arun("task") == "async:task"
    await tool.aclose()
    assert clients[0].closed
    assert clients[1].async_closed


def test_otel_adapter_exports_safe_attributes_only() -> None:
    class Span:
        attributes: dict[str, Any]

        def __enter__(self) -> Span:
            self.attributes = {}
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

    class Tracer:
        def start_as_current_span(self, _: str) -> Span:
            self.span = Span()
            return self.span

    tracer = Tracer()
    exporter = OpenTelemetryTracer(tracer)

    async def record() -> None:
        await exporter.on_event(
            trace_id="trace-1",
            agent_name="test-agent",
            event=TraceEvent("tool_call", duration_ms=5, details={"secret": "do-not-export"}),
        )

    asyncio.run(record())
    assert tracer.span.attributes == {
        "gabby.trace_id": "trace-1",
        "gabby.agent": "test-agent",
        "gabby.event": "tool_call",
        "gabby.duration_ms": 5,
    }


def test_signed_audit_chain_records_policy_and_persona_without_payloads(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gabby.tracing as tracing_module

    monkeypatch.setattr(tracing_module, "_TRACE_IO_EXECUTOR", _InlineExecutor())
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    key = cryptography.Ed25519PrivateKey.generate()
    raw_key = key.private_bytes_raw()
    path = tmp_path / "audit.jsonl"
    exporter = SignedAuditExporter(path, raw_key)
    with pytest.raises(RuntimeError, match="active writer"):
        SignedAuditExporter(path, raw_key)

    async def record() -> None:
        async def record_with_tick(event: TraceEvent) -> None:
            task = asyncio.create_task(
                exporter.on_event(trace_id="trace-1", agent_name="test-agent", event=event)
            )
            await task

        await record_with_tick(
            TraceEvent("tool_call", details={"name": "search", "arguments": {"secret": "x"}})
        )
        await record_with_tick(
            TraceEvent(
                "persona_verification",
                details={"result": {"passed": True, "method": "rule-check", "details": "private"}},
            )
        )
        await record_with_tick(
            TraceEvent("tool_error", details={"name": "search", "error_code": "execution_failed"})
        )

    asyncio.run(record())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["policy_decision"] == "allowed"
    assert rows[1]["persona_check"] == {"passed": True}
    assert rows[1]["previous_hash"] == rows[0]["record_hash"]
    assert rows[2]["policy_decision"] == "unknown"
    assert "secret" not in path.read_text()
    exporter.close()
    resumed_exporter = SignedAuditExporter(path, raw_key)
    resumed_exporter.close()


def test_signed_audit_filesystem_work_runs_off_the_event_loop(
    tmp_path: Any, monkeypatch: Any
) -> None:
    cryptography = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    exporter = SignedAuditExporter(
        tmp_path / "audit.jsonl", cryptography.Ed25519PrivateKey.generate().private_bytes_raw()
    )
    worker_started = threading.Event()
    release_worker = threading.Event()

    def block_append(*_: Any) -> None:
        worker_started.set()
        assert release_worker.wait(timeout=2)

    monkeypatch.setattr(exporter, "_append_record", block_append)

    async def record_with_heartbeat() -> bool:
        task = asyncio.create_task(
            exporter.on_event(
                trace_id="trace-async",
                agent_name="test-agent",
                event=TraceEvent("tool_call", details={"name": "lookup"}),
            )
        )
        await asyncio.sleep(0.01)
        started = worker_started.is_set()
        responsive = not task.done()
        release_worker.set()
        await asyncio.sleep(0.01)
        await task
        return started and responsive

    assert asyncio.run(record_with_heartbeat())
    exporter.close()


def test_tool_worker_redirects_import_and_handler_stdout(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    from gabby import tool_worker

    request = tmp_path / "request.json"
    request.write_text('{"value":"ok"}', encoding="utf-8")

    def load_handler(_: str) -> Any:
        print("module import diagnostic")

        def handler(value: str) -> dict[str, str]:
            print("handler diagnostic")
            return {"value": value}

        return handler

    monkeypatch.setattr(tool_worker, "_load_handler", load_handler)

    assert tool_worker.main(["fixture:handler", str(request)]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"value":"ok"}\n'
    assert "module import diagnostic" in captured.err
    assert "handler diagnostic" in captured.err


def test_tool_worker_rejects_invalid_invocation_and_non_object_payload(
    tmp_path: Any, capsys: Any
) -> None:
    from gabby import tool_worker

    assert tool_worker.main([]) == 2

    request = tmp_path / "request.json"
    request.write_text("[]", encoding="utf-8")
    assert tool_worker.main(["fixture:handler", str(request)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Sandboxed Python tool failed" in captured.err


def test_tool_worker_reports_invalid_json_and_unavailable_handler(
    tmp_path: Any, capsys: Any
) -> None:
    from gabby import tool_worker

    request = tmp_path / "request.json"
    request.write_text("{invalid", encoding="utf-8")
    assert tool_worker.main(["fixture:handler", str(request)]) == 1

    request.write_text("{}", encoding="utf-8")
    assert tool_worker.main(["missing.module:handler", str(request)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Sandboxed Python tool failed") == 2


def test_tool_worker_load_handler_requires_importable_callable(monkeypatch: Any) -> None:
    from gabby import tool_worker

    with pytest.raises(ValueError, match="module:function"):
        tool_worker._load_handler("missing-separator")

    module_name = "gabby_tool_worker_fixture"
    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace())
    with pytest.raises(AttributeError):
        tool_worker._load_handler(f"{module_name}:missing")

    monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(value=42))
    with pytest.raises(TypeError, match="not callable"):
        tool_worker._load_handler(f"{module_name}:value")


def test_lf_trace_exporter_signs_tool_transcript_and_correlates_audit(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import gabby.tracing as tracing_module

    monkeypatch.setattr(tracing_module, "_TRACE_IO_EXECUTOR", _InlineExecutor())
    signed_records: list[dict[str, Any]] = []

    def sign_record(record: dict[str, Any], key: Any) -> dict[str, Any]:
        signed = {**record, "signature": f"signed-by-{key}"}
        signed_records.append(signed)
        return signed

    canonical_json = SimpleNamespace(
        dumps=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )
    monkeypatch.setitem(sys.modules, "agentrust_trace", SimpleNamespace(sign_record=sign_record))
    monkeypatch.setitem(sys.modules, "rfc8785", canonical_json)
    exporter = LFTraceExporter(
        tmp_path,
        claims={"subject": "spiffe://example/agent/test"},
        private_key="test-key",
    )

    async def record() -> None:
        async def dispatch(event: TraceEvent) -> None:
            task = asyncio.create_task(
                exporter.on_event(trace_id="trace-2", agent_name="test-agent", event=event)
            )
            await task

        await dispatch(TraceEvent("tool_call", details={"name": "search"}))
        await dispatch(
            TraceEvent("tool_error", details={"name": "delete", "error_code": "policy_denied"})
        )
        await dispatch(TraceEvent("run_completed"))

    asyncio.run(record())
    trace_record = json.loads((tmp_path / "trace-2.trace.json").read_text())
    assert trace_record["tool_transcript"]["call_count"] == 2
    assert trace_record["tool_transcript"]["hash"].startswith("sha256:")
    assert trace_record["eat_profile"] == "tag:agentrust-io.com,2026:trace-v0.2"
    assert signed_records[0]["signature"] == "signed-by-test-key"


def test_lf_trace_exporter_cleans_failed_runs_and_drops_overflowed_traces(
    tmp_path: Any, monkeypatch: Any
) -> None:
    def sign_record(record: dict[str, Any], key: Any) -> dict[str, Any]:
        return {**record, "signature": str(key)}

    canonical_json = SimpleNamespace(
        dumps=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )
    monkeypatch.setitem(sys.modules, "agentrust_trace", SimpleNamespace(sign_record=sign_record))
    monkeypatch.setitem(sys.modules, "rfc8785", canonical_json)
    exporter = LFTraceExporter(
        tmp_path,
        claims={"subject": "test-agent"},
        private_key="test-key",
        max_pending_traces=1,
        max_tool_entries_per_trace=1,
    )

    async def record() -> None:
        async def dispatch(trace_id: str, event: TraceEvent) -> None:
            task = asyncio.create_task(
                exporter.on_event(trace_id=trace_id, agent_name="test", event=event)
            )
            await task

        await dispatch("cancelled", TraceEvent("tool_call"))
        await dispatch("cancelled", TraceEvent("run_cancelled"))
        await dispatch("overflow", TraceEvent("tool_call"))
        await dispatch("overflow", TraceEvent("tool_call"))
        await dispatch("overflow", TraceEvent("run_completed"))

    asyncio.run(record())
    assert exporter._tool_entries == {}
    assert exporter.dropped_trace_count == 1
    assert list(tmp_path.glob("*.trace.json")) == []
