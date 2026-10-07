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
"""Top-level names used to construct agents and implement providers."""

from pathlib import Path

import gabby

_V1_CORE_EXPORTS = {
    "Agent",
    "AgentDefinition",
    "ResolvedAgentDefinition",
    "SkillDefinition",
    "ResolvedSkillDefinition",
    "RunRequest",
    "ExecutionResult",
    "AgentStreamEvent",
    "ExecutionTrace",
    "TraceEvent",
    "AgentTool",
    "ModelProvider",
    "StreamingModelProvider",
    "GeminiProvider",
    "Tool",
    "ToolRegistry",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "CancellationToken",
    "Environment",
    "FileParser",
    "ParsedPage",
    "ICalendarTextParser",
    "python_run_tool",
    "VCardTextParser",
    "RSSAtomTextParser",
    "ParquetTextParser",
    "MarkupTextParser",
    "Retriever",
    "KnowledgeStore",
    "EmbeddingProvider",
    "VectorStore",
    "Reranker",
    "SkillSelector",
    "Planner",
    "Verifier",
    "Tracer",
    "Authenticator",
    "ApprovalHandler",
    "ApprovalAuditSink",
    "AsyncPostgresExecutor",
    "ApprovalAuditError",
    "ApprovalAuditRecord",
    "AuditedApprovalHandler",
    "SQLiteApprovalAudit",
    "PostgresApprovalAudit",
    "AsyncPostgresPool",
    "PostgresGenerationManifestStore",
    "PostgresKnowledgeStore",
    "PostgresStreamJournal",
    "PostgresVectorStore",
    "NvidiaReranker",
    "EngineAdapter",
    "SkillTrustPolicy",
    "SkillRevocationChecker",
    "SkillRevocationStore",
    "load_skill_trust_keys",
    "create_app",
    "StreamJournal",
    "StreamJournalSnapshot",
}
_OPTIONAL_PUBLIC_EXPORTS = {
    "create_mcp_server",
    "register_mcp_tools",
    "AnthropicProvider",
    "GeminiProvider",
}


def test_v1_core_contracts_are_available_from_package_root() -> None:
    assert set(gabby.__all__) >= _V1_CORE_EXPORTS
    assert all(getattr(gabby, name) is not None for name in _V1_CORE_EXPORTS)


def test_injectable_policy_engine_contract_is_available_from_package_root() -> None:
    names = (
        "PolicyEngine",
        "PolicyEngineProtocol",
        "PolicyEngineFactory",
        "DefaultPolicyEngineFactory",
    )
    assert all(name in gabby.__all__ for name in names)
    assert all(getattr(gabby, name) is not None for name in names)


def test_optional_mcp_adapter_is_a_documented_package_root_export() -> None:
    assert all(getattr(gabby, name) is not None for name in _OPTIONAL_PUBLIC_EXPORTS)
    compatibility = Path(__file__).resolve().parents[1] / "docs" / "COMPATIBILITY.md"
    documented = compatibility.read_text(encoding="utf-8")
    assert all(f"`{name}`" in documented for name in _OPTIONAL_PUBLIC_EXPORTS)


def test_v1_core_contracts_are_named_in_the_compatibility_document() -> None:
    compatibility = Path(__file__).resolve().parents[1] / "docs" / "COMPATIBILITY.md"
    documented = compatibility.read_text(encoding="utf-8")
    undocumented = sorted(name for name in _V1_CORE_EXPORTS if f"`{name}`" not in documented)

    assert not undocumented, f"v1 compatibility contracts are undocumented: {undocumented}"


def test_stream_journal_snapshot_requires_a_complete_ordered_frame_count() -> None:
    valid = gabby.StreamJournalSnapshot(created=False, frames=(b"frame",), done=True, event_count=1)

    assert valid.event_count == 1
    try:
        gabby.StreamJournalSnapshot(created=False, frames=(b"frame",), done=True, event_count=0)
    except ValueError as exc:
        assert "event_count" in str(exc)
    else:
        raise AssertionError("inconsistent stream journal snapshots must be rejected")


def test_agent_definition_skill_and_model_result_are_public_exports() -> None:
    for name in (
        "AgentDefinition",
        "GabbyClient",
        "GabbyAPIError",
        "RunResult",
        "StreamEvent",
        "sqlite_query_tool",
        "SkillDefinition",
        "ModelResponse",
        "ModelSkillSelector",
        "ToolContext",
        "ToolError",
        "ToolErrorCode",
        "RetryableModelError",
        "FileIngestor",
        "CSVTextParser",
        "ParquetTextParser",
        "JSONTextParser",
        "PDFTextParser",
        "DOCXTextParser",
        "RTFTextParser",
        "EPUBTextParser",
        "ODTTextParser",
        "ODSTextParser",
        "XLSXTextParser",
        "EmailTextParser",
        "MboxTextParser",
        "VCardTextParser",
        "EvaluationCase",
        "EvaluationReport",
        "JSONLinesTextParser",
        "XMLTextParser",
        "ImageOCRParser",
        "OCRBackend",
        "OCRPageResult",
        "TesseractOCRBackend",
        "ParsedPage",
        "TextFileIngestor",
        "SQLiteVectorStore",
        "OpenAICompatibleEmbeddingProvider",
        "TransformersEmbeddingProvider",
        "TransformersReranker",
    ):
        assert name in gabby.__all__
        assert getattr(gabby, name) is not None

    assert gabby.TextFileIngestor is gabby.FileIngestor
