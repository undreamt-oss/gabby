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

"""Offline regression tests for the host-orchestrated pipeline example."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest

import examples.host_orchestrated_pipeline.orchestrator as pipeline_module
from examples.host_orchestrated_pipeline.orchestrator import (
    MAX_WORKER_OUTPUT_BYTES,
    AgentEndpoint,
    AuthenticatedCaller,
    HostOrchestratedPipeline,
    WorkerName,
    _bounded_output,
)
from gabby.host_context import HostRAGSnippet


class _FakeTokenIssuer:
    def __init__(self) -> None:
        self.issued: list[tuple[str, str, str, frozenset[str]]] = []

    async def issue(
        self,
        *,
        subject: str,
        tenant_id: str,
        audience: str,
        scopes: frozenset[str],
        expires_in_seconds: int,
    ) -> str:
        assert expires_in_seconds == 60
        self.issued.append((subject, tenant_id, audience, scopes))
        return "test-token"


class _FakeRetriever:
    def __init__(self, snippets: list[HostRAGSnippet] | None = None) -> None:
        self.snippets = snippets or [HostRAGSnippet(text="Evidence", source="policy.md")]
        self.calls: list[tuple[str, str, int]] = []

    async def retrieve(self, *, tenant_id: str, query: str, limit: int) -> list[HostRAGSnippet]:
        self.calls.append((tenant_id, query, limit))
        return self.snippets


class _GatedRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active = False

    async def retrieve(self, *, tenant_id: str, query: str, limit: int) -> list[HostRAGSnippet]:
        self.calls.append((tenant_id, query, limit))
        self.started.set()
        self.active = True
        try:
            await self.release.wait()
        finally:
            self.active = False
        return [HostRAGSnippet(text="Evidence", source="policy.md")]


class _MemoryCheckpoints:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}

    async def load(
        self, *, tenant_id: str, execution_id: str, stage: str, workflow_version: str
    ) -> Mapping[str, Any] | None:
        return self.values.get((tenant_id, execution_id, stage, workflow_version))

    async def save(
        self,
        *,
        tenant_id: str,
        execution_id: str,
        stage: str,
        workflow_version: str,
        value: Mapping[str, Any],
    ) -> None:
        self.values[(tenant_id, execution_id, stage, workflow_version)] = dict(value)


class _FakeGateway:
    def __init__(self) -> None:
        self.token_issuer = _FakeTokenIssuer()
        self.calls: list[str] = []
        self.plans_by_task: dict[str, dict[str, str]] = {}
        self.default_plan = {"rag_query": "find evidence", "analyst_task": ""}
        self.worker_outputs = {"rag": "RAG answer", "analyst": "Analysis"}
        self.failing_workers: set[str] = set()
        self.blocked_worker: str | None = None
        self.worker_started = asyncio.Event()
        self.release_worker = asyncio.Event()
        self.synthesis_output = "Combined answer"
        self.synthesis_chunks: list[str] | None = None
        self.incomplete_synthesis = False
        self.synthesis_output_override: str | None = None

    def client(self, _base_url: str, agent_name: str, **_kwargs: Any) -> _FakeClient:
        return _FakeClient(self, agent_name)


class _FakeClient:
    def __init__(self, gateway: _FakeGateway, worker: str) -> None:
        self.gateway = gateway
        self.worker = worker

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def arun(self, task: str, **_kwargs: Any) -> SimpleNamespace:
        self.gateway.calls.append(self.worker)
        if self.worker == "triage":
            plan = self.gateway.plans_by_task.get(task, self.gateway.default_plan)
            return SimpleNamespace(
                output="",
                trace_id="trace-triage",
                metadata={"structured_output": plan},
            )
        if self.worker == self.gateway.blocked_worker:
            self.gateway.worker_started.set()
            await self.gateway.release_worker.wait()
        if self.worker in self.gateway.failing_workers:
            raise RuntimeError("simulated worker outage")
        return SimpleNamespace(
            output=self.gateway.worker_outputs[self.worker],
            trace_id=f"trace-{self.worker}",
            metadata={},
        )

    async def astream(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.gateway.calls.append(self.worker)
        chunks = self.gateway.synthesis_chunks
        if chunks is None:
            chunks = [self.gateway.synthesis_output]
        for text in chunks:
            yield SimpleNamespace(type="text_delta", data={"text": text})
        if not self.gateway.incomplete_synthesis:
            output = self.gateway.synthesis_output_override or self.gateway.synthesis_output
            yield SimpleNamespace(
                type="completed",
                data={"result": {"output": output, "trace_id": "trace-synthesis"}},
            )


def _make_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gateway: _FakeGateway | None = None,
    retriever: _FakeRetriever | _GatedRetriever | None = None,
    checkpoints: _MemoryCheckpoints | None = None,
    retrieval_timeout_seconds: float = pipeline_module.DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
) -> tuple[
    HostOrchestratedPipeline,
    _FakeGateway,
    _FakeRetriever | _GatedRetriever,
    _MemoryCheckpoints,
]:
    fake_gateway = gateway or _FakeGateway()
    fake_retriever = retriever or _FakeRetriever()
    fake_checkpoints = checkpoints or _MemoryCheckpoints()
    monkeypatch.setattr(pipeline_module, "GabbyClient", fake_gateway.client)
    worker_names = cast(tuple[WorkerName, ...], ("triage", "rag", "analyst", "synthesis"))
    endpoints: dict[WorkerName, AgentEndpoint] = {
        worker: AgentEndpoint(
            base_url=f"https://{worker}.example.test",
            agent_name=worker,
            audience=f"gabby:{worker}",
            revision="revision-1",
        )
        for worker in worker_names
    }
    pipeline = HostOrchestratedPipeline(
        endpoints=endpoints,
        token_issuer=fake_gateway.token_issuer,
        retriever=fake_retriever,
        checkpoints=fake_checkpoints,
        retrieval_timeout_seconds=retrieval_timeout_seconds,
    )
    return pipeline, fake_gateway, fake_retriever, fake_checkpoints


def _caller(*scopes: str) -> AuthenticatedCaller:
    return AuthenticatedCaller("user-1", "tenant-1", frozenset(scopes))


@pytest.mark.asyncio
async def test_pipeline_does_not_synthesize_retrieval_count_without_worker_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "find evidence", "analyst_task": ""}
    gateway.failing_workers.add("rag")
    pipeline, gateway, retriever, _ = _make_pipeline(monkeypatch, gateway=gateway)

    result = await pipeline.run(
        execution_id="no-evidence",
        task="Summarize the policy",
        caller=_caller("pipeline:run", "rag:read"),
    )

    assert result.status == "failed"
    assert result.workers["retrieval"].output == "1 bounded snippets"
    assert "synthesis" not in gateway.calls
    assert retriever.calls == [("tenant-1", "find evidence", 5)]


@pytest.mark.asyncio
async def test_streamed_synthesis_matches_bounded_saved_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": "analyze"}
    synthesis = "A🙂bc" * 4_000
    gateway.synthesis_output = synthesis
    gateway.synthesis_chunks = [
        synthesis[index : index + 37] for index in range(0, len(synthesis), 37)
    ]
    pipeline, gateway, _, _ = _make_pipeline(monkeypatch, gateway=gateway)
    streamed: list[str] = []

    async def collect(text: str) -> None:
        streamed.append(text)

    result = await pipeline.run(
        execution_id="bounded-stream",
        task="Analyze the report",
        caller=_caller("pipeline:run", "sql:finance_readonly"),
        on_synthesis_delta=collect,
    )

    assert "".join(streamed) == result.output == _bounded_output(synthesis)
    assert len(result.output.encode("utf-8")) <= MAX_WORKER_OUTPUT_BYTES
    assert result.output.endswith("[Worker output truncated by host limit.]")


@pytest.mark.asyncio
async def test_unauthorized_workers_are_not_called_or_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "find evidence", "analyst_task": "analyze"}
    pipeline, gateway, retriever, _ = _make_pipeline(monkeypatch, gateway=gateway)

    result = await pipeline.run(
        execution_id="unauthorized",
        task="Analyze restricted data",
        caller=_caller("pipeline:run"),
    )

    assert result.status == "failed"
    assert gateway.calls == ["triage"]
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_worker_tokens_preserve_the_authenticated_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": "analyze"}
    pipeline, _, _, _ = _make_pipeline(monkeypatch, gateway=gateway)

    await pipeline.run(
        execution_id="tenant-token",
        task="Analyze this report",
        caller=_caller("pipeline:run", "sql:finance_readonly"),
    )

    assert all(issued[1] == "tenant-1" for issued in gateway.token_issuer.issued)


@pytest.mark.asyncio
async def test_analyst_runs_while_retrieval_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "find evidence", "analyst_task": "analyze"}
    gateway.blocked_worker = "analyst"
    retriever = _GatedRetriever()
    pipeline, gateway, _, _ = _make_pipeline(
        monkeypatch,
        gateway=gateway,
        retriever=retriever,
        retrieval_timeout_seconds=10,
    )
    run = asyncio.create_task(
        pipeline.run(
            execution_id="parallel-retrieval",
            task="Analyze this report",
            caller=_caller("pipeline:run", "rag:read", "sql:finance_readonly"),
        )
    )
    try:
        await asyncio.wait_for(gateway.worker_started.wait(), timeout=2)
        assert retriever.active
        retriever.release.set()
        gateway.release_worker.set()
        result = await run
    finally:
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert result.workers["analyst"].status == "completed"
    assert result.workers["rag"].status == "completed"


@pytest.mark.asyncio
async def test_retrieval_timeout_keeps_authorized_analyst_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "find evidence", "analyst_task": "analyze"}
    retriever = _GatedRetriever()
    pipeline, gateway, _, _ = _make_pipeline(
        monkeypatch,
        gateway=gateway,
        retriever=retriever,
        retrieval_timeout_seconds=0.01,
    )

    result = await pipeline.run(
        execution_id="retrieval-timeout",
        task="Analyze this report",
        caller=_caller("pipeline:run", "rag:read", "sql:finance_readonly"),
    )

    assert result.status == "partial"
    assert result.workers["retrieval"].status == "unavailable"
    assert result.workers["retrieval"].error_type == "RetrieverTimeout"
    assert result.workers["analyst"].status == "completed"
    assert "analyst" in gateway.calls
    assert "rag" not in gateway.calls


@pytest.mark.asyncio
async def test_compatible_checkpoints_are_reused_and_mismatched_request_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": ""}
    gateway.plans_by_task = {
        "question one": {"rag_query": "", "analyst_task": "calculate one"},
        "question two": {"rag_query": "", "analyst_task": "calculate two"},
    }
    pipeline, gateway, _, _ = _make_pipeline(monkeypatch, gateway=gateway)
    caller = _caller("pipeline:run", "sql:finance_readonly")

    first = await pipeline.run(execution_id="checkpoint", task="question one", caller=caller)
    calls_after_first = Counter(gateway.calls)
    replay = await pipeline.run(execution_id="checkpoint", task="question one", caller=caller)

    assert replay == first
    assert Counter(gateway.calls) == calls_after_first

    changed = await pipeline.run(execution_id="checkpoint", task="question two", caller=caller)

    assert changed.output == first.output
    assert Counter(gateway.calls)["triage"] == 2
    assert Counter(gateway.calls)["analyst"] == 2
    assert Counter(gateway.calls)["synthesis"] == 2


@pytest.mark.asyncio
async def test_cancellation_propagates_without_saving_completed_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": "analyze"}
    gateway.blocked_worker = "analyst"
    pipeline, gateway, _, checkpoints = _make_pipeline(monkeypatch, gateway=gateway)
    task = asyncio.create_task(
        pipeline.run(
            execution_id="cancelled",
            task="Analyze this report",
            caller=_caller("pipeline:run", "sql:finance_readonly"),
        )
    )
    await gateway.worker_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any(key[2] == "completed" for key in checkpoints.values)


@pytest.mark.asyncio
async def test_stream_without_completion_is_not_checkpointed_and_can_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": "analyze"}
    gateway.synthesis_chunks = ["provisional answer"]
    gateway.incomplete_synthesis = True
    pipeline, gateway, _, checkpoints = _make_pipeline(monkeypatch, gateway=gateway)

    with pytest.raises(RuntimeError, match="without a completed event"):
        await pipeline.run(
            execution_id="resume-stream",
            task="Analyze this report",
            caller=_caller("pipeline:run", "sql:finance_readonly"),
        )

    assert not any(key[2] == "completed" for key in checkpoints.values)
    assert Counter(gateway.calls)["analyst"] == 1

    gateway.incomplete_synthesis = False
    gateway.synthesis_chunks = None
    result = await pipeline.run(
        execution_id="resume-stream",
        task="Analyze this report",
        caller=_caller("pipeline:run", "sql:finance_readonly"),
    )

    assert result.output == "Combined answer"
    assert Counter(gateway.calls)["analyst"] == 1
    assert Counter(gateway.calls)["triage"] == 1
    assert Counter(gateway.calls)["synthesis"] == 2


@pytest.mark.asyncio
async def test_stream_completion_must_match_forwarded_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FakeGateway()
    gateway.default_plan = {"rag_query": "", "analyst_task": "analyze"}
    gateway.synthesis_chunks = ["streamed answer"]
    gateway.synthesis_output_override = "different saved answer"
    pipeline, _, _, checkpoints = _make_pipeline(monkeypatch, gateway=gateway)

    with pytest.raises(RuntimeError, match="did not match its completed output"):
        await pipeline.run(
            execution_id="mismatched-stream",
            task="Analyze this report",
            caller=_caller("pipeline:run", "sql:finance_readonly"),
        )

    assert not any(key[2] == "completed" for key in checkpoints.values)
