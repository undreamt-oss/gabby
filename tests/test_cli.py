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
"""CLI tests for signed skill package workflows."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gabby import EmbeddingInputFormat
from gabby import training as training_module
from gabby.cli import _load_trusted_skill_keys, _resolve_trusted_skill_keys, main
from gabby.config import ConfigError, load_agent, load_skill
from gabby.knowledge import Document, SQLiteFTS5Store
from gabby.runtime import ExecutionResult
from gabby.skill_packages import pack_skill
from gabby.tracing import ExecutionTrace


@pytest.mark.parametrize(
    ("template", "environment_type"),
    [
        ("generic", "generic"),
        ("research", "research"),
        ("data-analysis", "data"),
        ("customer-support", "customer_support"),
    ],
)
def test_cli_init_generates_valid_non_overwriting_agent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    template: str,
    environment_type: str,
) -> None:
    output = tmp_path / f"{template}.yaml"

    assert main(["init", str(output), "--template", template]) == 0
    created = capsys.readouterr()
    assert "Created" in created.out
    definition = load_agent(output)
    assert definition.environment["type"] == environment_type
    assert definition.model["api_key_env"] == "OPENAI_API_KEY"
    assert "api_key" not in definition.model
    if template == "customer-support":
        assert "Never invent policy terms" in definition.instructions
        assert "unless a configured tool confirms it" in definition.instructions
    assert main(["validate", str(output)]) == 0
    capsys.readouterr()

    original = output.read_text(encoding="utf-8")
    assert main(["init", str(output), "--template", "research"]) == 2
    capsys.readouterr()
    assert output.read_text(encoding="utf-8") == original


def test_cli_skill_init_creates_packable_non_overwriting_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "skills" / "research"
    source.parent.mkdir()

    assert (
        main(
            [
                "skill",
                "init",
                "research/evidence-synthesis",
                "--output",
                str(source),
                "--description",
                "Synthesize research with traceable claims.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    skill = load_skill(source / "skill.yaml")
    assert skill.name == "research/evidence-synthesis"
    assert skill.version == "0.1.0"
    assert "## Verification" in skill.instructions
    assert skill.examples
    archive = tmp_path / "evidence-synthesis.gabskill"
    assert main(["skill", "pack", str(source), "--output", str(archive)]) == 0
    assert archive.is_file()
    capsys.readouterr()

    files_before = {path.name: path.read_bytes() for path in source.iterdir()}
    assert main(["skill", "init", "research/evidence-synthesis", "--output", str(source)]) == 2
    capsys.readouterr()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == files_before


class _FakeAgent:
    def __init__(self, _definition: object, **_kwargs: object) -> None:
        self.skills = [
            SimpleNamespace(
                name="support-review",
                version="1.0.0",
                description="Review support cases",
                tools=("read_ticket",),
            )
        ]
        self._skill_input_schema_json = {"support-review": '{"type":"object","required":["task"]}'}
        self._skill_output_schema_json = {
            "support-review": '{"type":"object","required":["result"]}'
        }

    def __enter__(self) -> _FakeAgent:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_cli_validate_inspect_and_list_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    definition = SimpleNamespace(
        name="support-agent",
        description="Handles customer support",
        model={"provider": "mock"},
        environment={"type": "support"},
        tools=("read_ticket",),
        policies={"network": "disabled"},
        knowledge={"sources": ["./docs"]},
        verification={"enabled": True},
    )
    monkeypatch.setattr(cli_module, "load_agent", lambda _path: definition)
    monkeypatch.setattr(cli_module, "Agent", _FakeAgent)
    config_path = tmp_path / "agent.yaml"

    assert main(["validate", str(config_path)]) == 0
    assert "is valid" in capsys.readouterr().out

    assert main(["inspect", str(config_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["name"] == "support-agent"
    assert inspected["skills"] == ["support-review"]
    assert inspected["skill_contracts"] == [
        {
            "name": "support-review",
            "input_schema": {"type": "object", "required": ["task"]},
            "output_schema": {"type": "object", "required": ["result"]},
        }
    ]

    assert main(["skills", str(config_path)]) == 0
    assert "support-review@1.0.0" in capsys.readouterr().out


def test_cli_knowledge_ingests_and_searches_persistent_sqlite_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_root = tmp_path / "knowledge"
    source_root.mkdir()
    (source_root / "refund.txt").write_text(
        "Travel refund processing takes five business days.", encoding="utf-8"
    )
    database = tmp_path / "state" / "knowledge.db"

    assert (
        main(
            [
                "knowledge",
                "ingest",
                str(source_root),
                "--database",
                str(database),
                "--metadata",
                '{"collection":"support"}',
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["source_count"] == 1
    assert report["document_count"] == 1
    assert report["sources"] == ["refund.txt"]
    assert database.is_file()

    assert (
        main(
            [
                "knowledge",
                "search",
                "refund processing",
                "--database",
                str(database),
                "--limit",
                "3",
            ]
        )
        == 0
    )
    results = json.loads(capsys.readouterr().out)
    assert results["query"] == "refund processing"
    assert results["results"][0]["source"] == "refund.txt"
    assert results["results"][0]["metadata"]["collection"] == "support"


def test_cli_knowledge_delete_removes_a_stale_indexed_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_root = tmp_path / "knowledge"
    source_root.mkdir()
    source_path = source_root / "retired-guide.txt"
    source_path.write_text("Retired recovery procedure must not be used.", encoding="utf-8")
    database = tmp_path / "state" / "knowledge.db"

    assert main(["knowledge", "ingest", str(source_root), "--database", str(database)]) == 0
    capsys.readouterr()
    source_path.unlink()

    assert (
        main(
            [
                "knowledge",
                "delete",
                str(source_root),
                "retired-guide.txt",
                "--database",
                str(database),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == {"deleted_documents": 1, "source": "retired-guide.txt"}
    assert main(["knowledge", "search", "retired recovery", "--database", str(database)]) == 0
    assert json.loads(capsys.readouterr().out)["results"] == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--metadata", "[]"], "JSON object"),
        (["--metadata", "x" * (64 * 1024 + 1)], "65536 UTF-8 bytes"),
    ],
)
def test_cli_knowledge_ingest_validates_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    source_root = tmp_path / "knowledge"
    source_root.mkdir()
    assert (
        main(
            [
                "knowledge",
                "ingest",
                str(source_root),
                "--database",
                str(tmp_path / "db.sqlite"),
                *arguments,
            ]
        )
        == 2
    )
    assert message in capsys.readouterr().err


def test_cli_knowledge_search_rejects_invalid_result_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "knowledge",
                "search",
                "query",
                "--database",
                str(tmp_path / "db.sqlite"),
                "--limit",
                "101",
            ]
        )
        == 2
    )
    assert "--limit must be from 1 through 100" in capsys.readouterr().err


def test_cli_knowledge_search_reports_missing_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "knowledge",
                "search",
                "query",
                "--database",
                str(tmp_path / "missing.db"),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "does not exist" in error
    assert "gabby knowledge ingest ROOT" in error


def test_cli_knowledge_ingest_reports_missing_roots_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "knowledge",
                "ingest",
                str(tmp_path / "missing"),
                "--database",
                str(tmp_path / "knowledge.db"),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert error.startswith("gabby:")
    assert "Traceback" not in error


def test_cli_run_connects_agent_to_ingested_knowledge_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    source_root = tmp_path / "knowledge"
    source_root.mkdir()
    (source_root / "guide.txt").write_text("Refunds take five days.", encoding="utf-8")
    database = tmp_path / "knowledge.db"
    assert (
        main(
            [
                "knowledge",
                "ingest",
                str(source_root),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    capsys.readouterr()
    definition = SimpleNamespace(knowledge={"sources": ["support-docs"]})
    captured: dict[str, object] = {}

    class RunAgent:
        def __init__(self, _definition: object, *, retriever: object) -> None:
            captured["retriever"] = retriever

        def __enter__(self) -> RunAgent:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, task: str, *, context: dict[str, object]) -> SimpleNamespace:
            captured["task"] = task
            captured["context"] = context
            return SimpleNamespace(output="answered", trace=SimpleNamespace(trace_id="trace"))

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: definition)
    monkeypatch.setattr(cli_module, "Agent", RunAgent)
    assert (
        main(
            [
                "run",
                str(tmp_path / "agent.yaml"),
                "When are refunds processed?",
                "--knowledge-db",
                str(database),
            ]
        )
        == 0
    )
    assert isinstance(captured["retriever"], SQLiteFTS5Store)
    retriever = captured["retriever"]
    assert isinstance(retriever, SQLiteFTS5Store)
    assert retriever.path == database
    indexed = asyncio.run(retriever.retrieve("refunds", limit=1))
    assert indexed[0].source == "guide.txt"
    assert captured["task"] == "When are refunds processed?"
    assert capsys.readouterr().out == "answered\n"


def test_cli_run_requires_a_persistent_store_for_knowledge_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import gabby.cli as cli_module

    monkeypatch.setattr(
        cli_module, "load_agent", lambda _path: SimpleNamespace(knowledge={"sources": ["docs"]})
    )
    assert (
        main(
            [
                "run",
                str(tmp_path / "agent.yaml"),
                "question",
                "--knowledge-db",
                str(tmp_path / "missing.db"),
            ]
        )
        == 2
    )
    assert "gabby knowledge ingest ROOT --database PATH" in capsys.readouterr().err


def test_cli_serve_injects_the_configured_knowledge_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gabby.cli as cli_module

    definition = SimpleNamespace(knowledge={"sources": ["docs"]})
    database = tmp_path / "knowledge.db"
    database.write_bytes(b"initialized elsewhere")
    captured: dict[str, object] = {}

    class ServiceAgent:
        def __init__(self, _definition: object, *, retriever: object) -> None:
            captured["retriever"] = retriever

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: definition)
    monkeypatch.setattr(cli_module, "Agent", ServiceAgent)
    monkeypatch.setattr(cli_module, "serve", lambda agent, **_: captured.update(agent=agent))

    assert main(["serve", str(tmp_path / "agent.yaml"), "--knowledge-db", str(database)]) == 0
    retriever = captured["retriever"]
    assert isinstance(retriever, SQLiteFTS5Store)
    assert retriever.path == database


def test_cli_mcp_starts_stdio_server_with_configured_response_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gabby.cli as cli_module
    import gabby.mcp_adapter as mcp_module

    definition = SimpleNamespace(name="mcp-agent", knowledge={})
    captured: dict[str, object] = {}

    class FakeMCPServer:
        def run(self, *, transport: str) -> None:
            captured["transport"] = transport

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: definition)
    monkeypatch.setattr(cli_module, "Agent", _FakeAgent)
    monkeypatch.setattr(
        mcp_module,
        "create_mcp_server",
        lambda agent, *, max_response_bytes: (
            captured.update(agent=agent, max_response_bytes=max_response_bytes) or FakeMCPServer()
        ),
    )

    assert main(["mcp", str(tmp_path / "agent.yaml"), "--max-response-bytes", "2048"]) == 0
    assert captured["transport"] == "stdio"
    assert captured["max_response_bytes"] == 2048


def test_cli_mcp_rejects_invalid_response_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import gabby.cli as cli_module

    monkeypatch.setattr(
        cli_module, "load_agent", lambda _path: SimpleNamespace(name="mcp-agent", knowledge={})
    )
    assert main(["mcp", str(tmp_path / "agent.yaml"), "--max-response-bytes", "255"]) == 2
    assert "--max-response-bytes must be at least 256" in capsys.readouterr().err


def test_cli_evaluate_emits_json_and_returns_failure_status_for_regressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        '{"id":"pass","input":"match","expected_output":"match"}\n'
        '{"id":"fail","input":"mismatch","expected_output":"expected"}\n',
        encoding="utf-8",
    )

    class EvaluationAgent:
        def __init__(self, _definition: object) -> None:
            pass

        async def __aenter__(self) -> EvaluationAgent:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def arun(self, input: str, **_: Any) -> ExecutionResult:
            return ExecutionResult(input, ExecutionTrace())

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: SimpleNamespace(knowledge={}))
    monkeypatch.setattr(cli_module, "Agent", EvaluationAgent)

    assert main(["evaluate", str(tmp_path / "agent.yaml"), "--dataset", str(dataset)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["case_count"] == 2
    assert result["passed_count"] == 1
    assert result["score"] == 0.5
    assert [case["passed"] for case in result["results"]] == [True, False]


def test_cli_run_passes_input_context_and_reports_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    calls: list[tuple[str, dict[str, Any]]] = []

    class RunAgent:
        def __init__(self, _definition: object) -> None:
            pass

        def __enter__(self) -> RunAgent:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, task: str, *, context: dict[str, Any]) -> SimpleNamespace:
            calls.append((task, context))
            return SimpleNamespace(output="completed", trace=SimpleNamespace(trace_id="trace-1"))

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: object())
    monkeypatch.setattr(cli_module, "Agent", RunAgent)
    assert (
        main(
            [
                "run",
                str(tmp_path / "agent.yaml"),
                "Review this request",
                "--context",
                '{"request_id":"req-1"}',
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == "completed\n"
    assert "trace_id: trace-1" in captured.err
    assert calls == [("Review this request", {"request_id": "req-1"})]


def test_cli_run_reads_stdin_and_rejects_empty_or_non_object_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    calls: list[str] = []

    class RunAgent:
        def __init__(self, _definition: object) -> None:
            pass

        def __enter__(self) -> RunAgent:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, task: str, *, context: dict[str, Any]) -> SimpleNamespace:
            calls.append(task)
            return SimpleNamespace(output="stdin completed", trace=SimpleNamespace(trace_id="t"))

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: object())
    monkeypatch.setattr(cli_module, "Agent", RunAgent)
    monkeypatch.setattr(sys, "stdin", io.StringIO("  task from stdin  \n"))
    assert main(["run", str(tmp_path / "agent.yaml")]) == 0
    assert calls == ["task from stdin"]
    assert capsys.readouterr().out == "stdin completed\n"

    monkeypatch.setattr(sys, "stdin", io.StringIO("  \n"))
    assert main(["run", str(tmp_path / "agent.yaml")]) == 2
    assert "Provide input" in capsys.readouterr().err

    assert main(["run", str(tmp_path / "agent.yaml"), "task", "--context", "[]"]) == 2
    assert "JSON object" in capsys.readouterr().err


def test_cli_serve_forwards_bounded_options_and_token_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gabby.cli as cli_module

    agent = object()
    received: dict[str, Any] = {}

    def capture_serve(value: object, **kwargs: Any) -> None:
        received.update(agent=value, **kwargs)

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: object())
    monkeypatch.setattr(cli_module, "Agent", lambda _definition: agent)
    monkeypatch.setattr(cli_module, "serve", capture_serve)
    monkeypatch.setenv("TEST_GABBY_TOKEN", "secret-token")

    assert (
        main(
            [
                "serve",
                str(tmp_path / "agent.yaml"),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--max-concurrent-runs",
                "4",
                "--max-request-bytes",
                "8192",
                "--request-body-timeout",
                "12.5",
                "--max-response-bytes",
                "4096",
                "--bearer-token-env",
                "TEST_GABBY_TOKEN",
                "--bearer-scope",
                "agent:run",
                "--stream-scope",
                "agent:stream",
                "--stream-journal-db",
                str(tmp_path / "streams.sqlite3"),
            ]
        )
        == 0
    )
    assert received == {
        "agent": agent,
        "host": "0.0.0.0",
        "port": 9000,
        "max_request_bytes": 8192,
        "request_body_timeout_seconds": 12.5,
        "max_response_bytes": 4096,
        "max_concurrent_runs": 4,
        "max_resumable_streams": None,
        "stream_session_ttl_seconds": 600.0,
        "stream_journal_path": tmp_path / "streams.sqlite3",
        "bearer_token": "secret-token",
        "bearer_scopes": frozenset({"agent:run"}),
        "run_scopes": (),
        "stream_scopes": ("agent:stream",),
    }


@pytest.mark.parametrize(
    ("options", "expected_error"),
    [
        (["--port", "0"], "Port must be between"),
        (["--port", "65536"], "Port must be between"),
        (["--bearer-token-env", "NOT-VALID"], "valid environment variable name"),
        (["--max-concurrent-runs", "0"], "positive integer"),
        (["--max-resumable-streams", "0"], "positive integer"),
        (["--stream-session-ttl", "nan"], "finite positive number"),
        (["--max-request-bytes", "0"], "positive integer"),
        (["--request-body-timeout", "nan"], "finite positive number"),
        (["--max-response-bytes", "255"], "at least 256"),
    ],
)
def test_cli_serve_rejects_invalid_runtime_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    expected_error: str,
) -> None:
    import gabby.cli as cli_module

    monkeypatch.setattr(cli_module, "load_agent", lambda _path: object())
    assert main(["serve", str(tmp_path / "agent.yaml"), *options]) == 2
    assert expected_error in capsys.readouterr().err


def test_cli_registry_commands_accept_no_token_and_reject_invalid_token_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    headers_seen: list[dict[str, str]] = []

    class PublicRegistryClient:
        def __init__(
            self,
            _url: str,
            *,
            headers: dict[str, str],
            max_catalog_age_seconds: float | None = None,
        ) -> None:
            assert max_catalog_age_seconds is None
            headers_seen.append(headers)

        async def __aenter__(self) -> PublicRegistryClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def search(self, _query: str) -> list[SimpleNamespace]:
            return []

    monkeypatch.setattr(cli_module, "SkillRegistryClient", PublicRegistryClient)
    args = ["skill", "search", "support", "--registry-url", "https://skills.example.test"]
    assert main(args) == 0
    assert headers_seen == [{}]

    assert main([*args, "--token-env", "INVALID-NAME"]) == 2
    assert "valid environment variable name" in capsys.readouterr().err
    assert main([*args, "--token-env", "MISSING_REGISTRY_TOKEN"]) == 2
    assert "is unset" in capsys.readouterr().err


def test_cli_trusted_key_rejects_path_replacement_between_check_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    replacement = tmp_path / "replacement.pub"
    replacement.write_bytes(b"q" * 32)
    original_open = os.open

    def replace_key_before_open(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == key_path:
            replacement.replace(key_path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_key_before_open)
    with pytest.raises(ConfigError, match="changed while loading"):
        _load_trusted_skill_keys([f"publisher={key_path}"])


def test_cli_trusted_key_rejects_truncation_during_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_path = tmp_path / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    original_open = os.open

    def truncate_open_file(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == key_path:
            key_path.write_bytes(b"q" * 31)
        return descriptor

    monkeypatch.setattr(os, "open", truncate_open_file)
    with pytest.raises(ConfigError, match="must contain 32 raw bytes"):
        _load_trusted_skill_keys([f"publisher={key_path}"])


def test_cli_trusted_key_rejects_key_replaced_by_fifo_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFO creation and nonblocking file opens are unavailable")
    key_path = tmp_path / "publisher.pub"
    key_path.write_bytes(b"p" * 32)
    original_open = os.open

    def replace_key_with_fifo(
        path: str | os.PathLike[str], flags: int, *args: Any, **kwargs: Any
    ) -> int:
        if Path(path) == key_path:
            key_path.unlink()
            os.mkfifo(key_path)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_key_with_fifo)
    with pytest.raises(ConfigError, match="changed while loading"):
        _load_trusted_skill_keys([f"publisher={key_path}"])


def test_cli_sign_verify_and_require_trusted_key_for_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_key = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv("GABBY_SKILL_SIGNING_KEY", base64.b64encode(private_key).decode("ascii"))

    source = tmp_path / "skill"
    source.mkdir()
    (source / "skill.yaml").write_text(
        "name: support-review\nversion: 1.0.0\ndescription: Review support cases\n",
        encoding="utf-8",
    )
    package = tmp_path / "support-review.gabskill"
    pack_skill(source, package)
    assert (
        main(
            [
                "skill",
                "sign",
                str(package),
                "--key-id",
                "release-key",
                "--private-key-env",
                "GABBY_SKILL_SIGNING_KEY",
            ]
        )
        == 0
    )
    assert "Signed" in capsys.readouterr().out

    trust_directory = tmp_path / "trusted-publishers"
    trust_directory.mkdir()
    public_key_path = trust_directory / "release-key.pub"
    public_key_path.write_bytes(public_key)
    assert (
        main(
            [
                "skill",
                "verify",
                str(package),
                "--trusted-key-dir",
                str(trust_directory),
            ]
        )
        == 0
    )
    assert "Verified" in capsys.readouterr().out

    registry = tmp_path / "registry"
    assert (
        main(
            [
                "skill",
                "install",
                str(package),
                "--registry",
                str(registry),
                "--require-signature",
                "--trusted-key-dir",
                str(trust_directory),
            ]
        )
        == 0
    )
    assert "verified key_id:release-key" in capsys.readouterr().out
    assert (registry / "support-review" / "1.0.0" / "skill.yaml").is_file()

    assert (
        main(
            [
                "skill",
                "audit",
                "--registry",
                str(registry),
                "--revoked-key",
                "release-key",
            ]
        )
        == 1
    )
    audit_record = json.loads(capsys.readouterr().out)
    assert audit_record["signing_key_id"] == "release-key"
    assert audit_record["status"] == "revoked"

    assert (
        main(
            [
                "skill",
                "audit",
                "--registry",
                str(registry),
                "--trusted-key-dir",
                str(trust_directory),
            ]
        )
        == 0
    )
    audit_record = json.loads(capsys.readouterr().out)
    assert audit_record["status"] == "verified"


def test_cli_pack_and_unsigned_install(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "local-skill"
    source.mkdir()
    (source / "skill.yaml").write_text(
        "name: local-helper\nversion: 1.0.0\ndescription: A local helper\n",
        encoding="utf-8",
    )
    package = tmp_path / "local-helper.gabskill"
    assert main(["skill", "pack", str(source), "--output", str(package)]) == 0
    assert "Created local-helper@1.0.0" in capsys.readouterr().out
    registry = tmp_path / "registry"
    assert main(["skill", "install", str(package), "--registry", str(registry)]) == 0
    assert "Installed local-helper@1.0.0" in capsys.readouterr().out
    installed = registry / "local-helper" / "1.0.0"
    assert (installed / "skill.yaml").is_file()
    uninstall_args = [
        "skill",
        "uninstall",
        "local-helper",
        "1.0.0",
        "--registry",
        str(registry),
    ]
    assert main(uninstall_args) == 2
    assert "requires --yes" in capsys.readouterr().err
    assert (installed / "skill.yaml").is_file()
    assert main([*uninstall_args, "--yes"]) == 0
    assert not installed.exists()
    assert "Uninstalled local-helper@1.0.0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("env_value", "expected_error"),
    [
        (None, "is unset"),
        ("not base64!", "valid base64"),
        ("a" * 129, "exceeds its size limit"),
        ("YQ==", "exactly 32 raw bytes"),
    ],
)
def test_cli_sign_rejects_invalid_host_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    env_value: str | None,
    expected_error: str,
) -> None:
    package = tmp_path / "not-read.gabskill"
    package.write_bytes(b"x")
    if env_value is None:
        monkeypatch.delenv("GABBY_SIGN_KEY", raising=False)
    else:
        monkeypatch.setenv("GABBY_SIGN_KEY", env_value)
    assert (
        main(
            [
                "skill",
                "sign",
                str(package),
                "--key-id",
                "release",
                "--private-key-env",
                "GABBY_SIGN_KEY",
            ]
        )
        == 2
    )
    assert expected_error in capsys.readouterr().err


def test_cli_trusted_key_file_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="KEY_ID=PATH"):
        _load_trusted_skill_keys(["missing-separator"])
    valid_key = tmp_path / "valid.pub"
    valid_key.write_bytes(b"k" * 32)
    with pytest.raises(ConfigError, match="more than once"):
        _load_trusted_skill_keys([f"publisher={valid_key}", f"publisher={valid_key}"])
    with pytest.raises(ConfigError, match="At least one"):
        _load_trusted_skill_keys([])

    key_file = tmp_path / "key.pub"
    key_file.write_bytes(b"short")
    with pytest.raises(ConfigError, match="32-byte regular file"):
        _load_trusted_skill_keys([f"publisher={key_file}"])

    unavailable = tmp_path / "unavailable.pub"
    with pytest.raises(ConfigError, match="unavailable"):
        _load_trusted_skill_keys([f"publisher={unavailable}"])

    symlink = tmp_path / "link.pub"
    symlink.symlink_to(valid_key)
    with pytest.raises(ConfigError, match="must not be symlinks"):
        _load_trusted_skill_keys([f"publisher={symlink}"])
    assert _load_trusted_skill_keys([f"publisher={valid_key}"]) == {"publisher": b"k" * 32}


def test_cli_resolves_and_rejects_duplicate_trust_sources(tmp_path: Path) -> None:
    directory = tmp_path / "trust"
    directory.mkdir()
    key_file = directory / "publisher.pub"
    key_file.write_bytes(b"p" * 32)

    assert _resolve_trusted_skill_keys([], directory, required=True) == {"publisher": b"p" * 32}
    assert _resolve_trusted_skill_keys([], None, required=False) is None
    with pytest.raises(ConfigError, match="more than once"):
        _resolve_trusted_skill_keys([f"publisher={key_file}"], directory, required=True)
    with pytest.raises(ConfigError, match="Provide at least one"):
        _resolve_trusted_skill_keys([], None, required=True)


def test_cli_rejects_invalid_key_environment_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "package.gabskill"
    package.write_bytes(b"package")
    assert (
        main(
            [
                "skill",
                "sign",
                str(package),
                "--key-id",
                "publisher",
                "--private-key-env",
                "INVALID-NAME",
            ]
        )
        == 2
    )
    assert "valid environment variable name" in capsys.readouterr().err


def test_cli_remote_registry_search_versions_and_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    catalog_ages: list[float | None] = []

    class FakeRegistryClient:
        def __init__(
            self,
            url: str,
            *,
            headers: dict[str, str],
            max_catalog_age_seconds: float | None = None,
        ) -> None:
            assert url == "https://skills.example.test"
            assert headers == {"Authorization": "Bearer test-token"}
            self.max_catalog_age_seconds = max_catalog_age_seconds
            catalog_ages.append(max_catalog_age_seconds)

        async def __aenter__(self) -> FakeRegistryClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def search(self, query: str) -> list[SimpleNamespace]:
            assert query == "support"
            return [
                SimpleNamespace(
                    name="support/triage",
                    versions=("1.2.0", "1.1.0"),
                    description="Support\x1b[31m workflow\nforged entry",
                )
            ]

        async def versions(self, name: str) -> tuple[str, ...]:
            assert name == "support/triage"
            return ("1.2.0",)

        async def install(
            self,
            name: str,
            version: str,
            local_registry: Path,
            *,
            trusted_keys: dict[str, bytes],
        ) -> SimpleNamespace:
            assert (name, version) == ("support/triage", "1.2.0")
            assert local_registry == tmp_path / "local"
            assert trusted_keys == {"publisher": b"k" * 32}
            return SimpleNamespace(
                name=name,
                version=version,
                path=local_registry / name / version,
                signing_key_id="publisher",
            )

    monkeypatch.setattr(cli_module, "SkillRegistryClient", FakeRegistryClient)
    monkeypatch.setenv("GABBY_REGISTRY_TOKEN", "test-token")
    assert (
        main(
            [
                "skill",
                "search",
                "support",
                "--registry-url",
                "https://skills.example.test",
                "--token-env",
                "GABBY_REGISTRY_TOKEN",
                "--max-catalog-age-seconds",
                "604800",
            ]
        )
        == 0
    )
    search_output = capsys.readouterr().out
    assert catalog_ages[0] == 604800
    assert "support/triage\t1.2.0, 1.1.0\tSupport[31m workflowforged entry" in search_output
    assert "\x1b" not in search_output
    assert (
        main(
            [
                "skill",
                "versions",
                "support/triage",
                "--registry-url",
                "https://skills.example.test",
                "--token-env",
                "GABBY_REGISTRY_TOKEN",
            ]
        )
        == 0
    )
    assert "1.2.0" in capsys.readouterr().out

    public_key = tmp_path / "publisher.pub"
    public_key.write_bytes(b"k" * 32)
    assert (
        main(
            [
                "skill",
                "fetch",
                "support/triage",
                "1.2.0",
                "--registry-url",
                "https://skills.example.test",
                "--local-registry",
                str(tmp_path / "local"),
                "--trusted-key",
                f"publisher={public_key}",
                "--token-env",
                "GABBY_REGISTRY_TOKEN",
            ]
        )
        == 0
    )
    assert "verified key_id:publisher" in capsys.readouterr().out


def test_cli_knowledge_evaluate_emits_ranking_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "knowledge.db"
    asyncio.run(
        SQLiteFTS5Store(database).ingest(
            [Document(text="Refunds take five days", source="refund-policy.md")]
        )
    )
    dataset = tmp_path / "retrieval.jsonl"
    dataset.write_text(
        '{"id":"refund","query":"refunds five days","relevant_sources":["refund-policy.md"]}\n',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "knowledge",
                "evaluate",
                "--dataset",
                str(dataset),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["mean_recall"] == 1.0
    assert report["mean_reciprocal_rank"] == 1.0


def test_cli_embedding_evaluate_runs_a_bounded_jsonl_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gabby.cli as cli_module

    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False
            self.received: list[str] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.received.extend(texts)
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

        async def aclose(self) -> None:
            self.closed = True

    provider = FakeProvider()
    monkeypatch.setattr(cli_module, "_create_embedding_provider", lambda _config: provider)
    dataset = tmp_path / "embedding.jsonl"
    dataset.write_text(
        '{"id":"reset","query":"reset query","documents":['
        '{"id":"relevant","text":"reset steps","relevance":5},'
        '{"id":"unrelated","text":"support hours","relevance":0}],"limit":2}\n',
        encoding="utf-8",
    )
    profile = tmp_path / "provider.json"
    profile.write_text(
        '{"version":1,"provider":{"type":"openai_compatible","model":"fixture"},'
        '"input_format":{"query_prefix":"query: ","document_prefix":"passage: "}}',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "embeddings",
                "evaluate",
                "--dataset",
                str(dataset),
                "--provider-config",
                str(profile),
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["mean_ndcg"] == 1.0
    assert provider.received == [
        "query: reset query",
        "passage: reset steps",
        "passage: support hours",
    ]
    assert provider.closed


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"version":1,"version":1,"provider":{"type":"openai_compatible"}}', "repeat keys"),
        ('{"version":1,"provider":{"type":"other"}}', "provider.type must be"),
        (
            '{"version":1,"provider":{"type":"openai_compatible","typo":true}}',
            "unknown option",
        ),
    ],
)
def test_cli_embedding_evaluate_rejects_invalid_provider_profiles(
    tmp_path: Path, contents: str, message: str
) -> None:
    import gabby.cli as cli_module

    profile = tmp_path / "provider.json"
    profile.write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        cli_module._load_embedding_provider_config(profile)


def test_cli_embedding_provider_rejects_inline_secrets() -> None:
    import gabby.cli as cli_module

    with pytest.raises(ConfigError, match="model.api_key"):
        cli_module._create_embedding_provider(
            {"type": "openai_compatible", "model": "fixture", "api_key": "secret"}
        )


def test_cli_embedding_provider_constructs_openai_compatible_adapter() -> None:
    import gabby.cli as cli_module
    from gabby import OpenAICompatibleEmbeddingProvider

    provider = cli_module._create_embedding_provider(
        {"type": "openai_compatible", "model": "fixture"}
    )
    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.model == "fixture"


def test_cli_embedding_provider_constructs_gemini_adapter() -> None:
    import gabby.cli as cli_module
    from gabby import GeminiEmbeddingProvider

    provider = cli_module._create_embedding_provider(
        {
            "type": "gemini",
            "model": "gemini-embedding-001",
            "task_type": "RETRIEVAL_DOCUMENT",
            "dimensions": 768,
        }
    )
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.model == "gemini-embedding-001"
    assert provider.task_type == "RETRIEVAL_DOCUMENT"


def test_cli_embedding_provider_profile_accepts_gemini_options(tmp_path: Path) -> None:
    import gabby.cli as cli_module

    profile = tmp_path / "gemini-provider.json"
    profile.write_text(
        '{"version":1,"provider":{"type":"gemini","model":"gemini-embedding-2","dimensions":768}}',
        encoding="utf-8",
    )
    provider_config, input_format = cli_module._load_embedding_provider_config(profile)
    assert provider_config["type"] == "gemini"
    assert provider_config["dimensions"] == 768
    assert input_format == EmbeddingInputFormat()


def test_cli_train_builds_config_and_emits_json_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def train(config: training_module.SFTConfig) -> dict[str, Any]:
        observed["config"] = config
        return {"format": "gabby-sft-lora-v1", "example_count": 3}

    monkeypatch.setattr(training_module, "train_lora_sft", train)
    result = main(
        [
            "train",
            "--model",
            "org/model",
            "--dataset",
            str(tmp_path / "data.jsonl"),
            "--output",
            str(tmp_path / "adapter"),
            "--revision",
            "abc123",
            "--target-module",
            "q_proj",
            "--target-module",
            "v_proj",
        ]
    )

    assert result == 0
    assert observed["config"].model_id == "org/model"
    assert observed["config"].revision == "abc123"
    assert observed["config"].target_modules == ("q_proj", "v_proj")
    assert json.loads(capsys.readouterr().out) == {
        "format": "gabby-sft-lora-v1",
        "example_count": 3,
    }


def test_cli_train_can_check_dataset_without_model_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training_module,
        "train_lora_sft",
        lambda _config: pytest.fail("dataset check must not start training"),
    )

    result = main(["train", "--dataset", str(dataset), "--check-dataset"])

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["validation"] == "passed"
    assert report["example_count"] == 1
    assert len(report["dataset_sha256"]) == 64
