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

"""Build and execute specialized, stateless AI agents.

Gabby composes a model with an environment, reusable skills, registered tools, optional
knowledge retrieval, instructions, runtime-enforced policies, and verification. The same
resolved definition can be run directly with :class:`Agent` or exposed through the FastAPI
service factory :func:`create_app`.

Each run receives its task and any caller-owned context explicitly. Gabby keeps transient
planning and tool observations only for that execution; it does not persist conversations or
user memory. Applications decide what state to retain and pass it to a later run.

The package root exports the supported entry points for the current pre-1.0 release. Public
extension contracts may evolve before 1.0; see the extension policy and guides in the project
documentation before depending on an extension interface.
"""

from .agent import Agent
from .approval import ApprovalAuditSink, ApprovalDecision, ApprovalHandler, ApprovalRequest
from .approval_sqlite import (
    ApprovalAuditError,
    ApprovalAuditRecord,
    AuditedApprovalHandler,
    SQLiteApprovalAudit,
)
from .auth import (
    AuthenticationUnavailable,
    Authenticator,
    BearerTokenAuthenticator,
    JWTBearerAuthenticator,
    Principal,
    TokenRevocationChecker,
)
from .client import GabbyAPIError, GabbyClient, RunResult, StreamEvent
from .composition import AgentTool
from .config import (
    AgentDefinition,
    ResolvedAgentDefinition,
    ResolvedSkillDefinition,
    SandboxDefinition,
    SkillDefinition,
    WorkspaceMount,
    validate_agent_definition,
    validate_skill_definition,
)
from .credentials import ScopedCredentialProvider
from .embedding_evaluation import (
    EmbeddingCandidate,
    EmbeddingCaseResult,
    EmbeddingEvaluationCase,
    EmbeddingEvaluationError,
    EmbeddingEvaluationReport,
    EmbeddingInputFormat,
    evaluate_embeddings,
    load_embedding_evaluation_dataset,
)
from .embeddings import (
    EmbeddingError,
    EmbeddingResponseSizeError,
    GeminiEmbeddingProvider,
    HuggingFaceFeatureExtractionProvider,
    OpenAICompatibleEmbeddingProvider,
)
from .environment import Environment, ResolvedEnvironment
from .evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationCheck,
    EvaluationError,
    EvaluationReport,
    evaluate_agent,
    load_evaluation_dataset,
)
from .host_context import HostContext
from .hybrid import HybridRetriever, RerankingRetriever
from .indexing import (
    GenerationBusyError,
    GenerationLease,
    GenerationManifestStore,
    HybridIndexCoordinator,
    PendingIndexGeneration,
    SQLiteGenerationManifestStore,
)
from .ingestion import (
    FileIngestor,
    FileParser,
    IngestionReport,
    MarkupTextParser,
    ParagraphChunker,
    ParsedPage,
    TextChunker,
    TextFileIngestor,
    Utf8TextParser,
)
from .knowledge import (
    ActiveGenerationReader,
    AsymmetricEmbeddingProvider,
    Document,
    EmbeddingProvider,
    GenerationAwareRetriever,
    GenerationAwareVectorStore,
    InMemoryBM25Retriever,
    KnowledgeStore,
    KnowledgeStoreError,
    Reranker,
    Retriever,
    SourceKnowledgeWriter,
    SQLiteFTS5Store,
    VectorStore,
)
from .mcp import create_mcp_server, load_mcp_tools
from .models import (
    AnthropicProvider,
    GeminiProvider,
    HuggingFaceInferenceProvider,
    ModelProvider,
    ModelRequestSizeError,
    ModelResponse,
    ModelResponseSizeError,
    ModelStreamDelta,
    OllamaProvider,
    OpenAICompatibleProvider,
    RetryableModelError,
    StreamingModelProvider,
    ensure_model_request_size,
    ensure_model_response_size,
)
from .orchestrators import crewai_tool, langgraph_node, n8n_request
from .payload_filters import DefaultPayloadFilter, PayloadFilter, PayloadFilterError
from .planning import (
    ExecutionPlan,
    ModelPlanner,
    Planner,
    PlanningCapability,
    PlanningObservation,
    PlanningResult,
    PlanStep,
)
from .python_tools import python_run_tool, sandboxed_python_tool
from .redis_skill_revocation import AsyncRedisCommands, RedisSkillRevocationStore
from .rerankers import (
    CohereReranker,
    JinaReranker,
    NvidiaReranker,
    RerankingError,
    RerankingResponseSizeError,
    VoyageReranker,
)
from .retrieval_evaluation import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalEvaluationError,
    RetrievalEvaluationReport,
    evaluate_retriever,
    load_retrieval_evaluation_dataset,
)
from .revocation import SQLiteTokenRevocationStore
from .runtime import AgentStreamEvent, ExecutionResult, RunRequest, Runtime, StreamEventType
from .sandbox import (
    EngineAdapter,
    SandboxCommandResult,
    SandboxError,
    SandboxResourceMonitor,
    SandboxResourceUsage,
    SandboxSession,
    SandboxTimeout,
    SandboxUnavailable,
)
from .sandbox_tools import sandbox_tools
from .server import StreamJournal, StreamJournalSnapshot, create_app
from .skill_packages import (
    SkillPackageError,
    SkillPackageFile,
    SkillPackageInfo,
    SkillPackageInspection,
    SkillRegistryAuditRecord,
    audit_skill_registry,
    inspect_skill_package,
    install_skill,
    pack_skill,
    uninstall_skill,
    verify_installed_skill,
)
from .skill_registry import (
    SkillCatalogEntry,
    SkillRegistryClient,
    SkillRegistryError,
    StaticSkillRegistryBuild,
    build_static_skill_registry,
)
from .skill_revocation import SQLiteSkillRevocationStore
from .skill_selection import (
    ConfiguredSkillSelector,
    ModelSkillSelector,
    SkillActivation,
    SkillSelection,
    SkillSelector,
)
from .skill_signing import sign_skill_package, verify_skill_package_signature
from .skill_trust import (
    SkillIntegrityError,
    SkillRevocationChecker,
    SkillRevocationStore,
    SkillRevocationUnavailable,
    SkillRevokedError,
    SkillTrustPolicy,
    load_skill_trust_keys,
)
from .sqlite_tools import sqlite_query_tool
from .tools import (
    CancellationToken,
    DefaultPolicyEngineFactory,
    PolicyEngine,
    PolicyEngineFactory,
    PolicyEngineProtocol,
    Tool,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ToolRegistry,
)
from .tracing import (
    CompositeTracer,
    ExecutionTrace,
    LFTraceExporter,
    OpenTelemetryTracer,
    SignedAuditExporter,
    TraceEvent,
    Tracer,
    create_otlp_tracer,
)
from .vector import SQLiteVectorStore
from .verification import PersonaVerifier, VerificationResult, Verifier

__all__ = [
    "Agent",
    "GabbyAPIError",
    "GabbyClient",
    "RunResult",
    "StreamEvent",
    "AgentDefinition",
    "AgentTool",
    "AnthropicProvider",
    "ApprovalDecision",
    "ApprovalAuditSink",
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalAuditError",
    "ApprovalAuditRecord",
    "AuditedApprovalHandler",
    "SQLiteApprovalAudit",
    "AgentStreamEvent",
    "Authenticator",
    "AuthenticationUnavailable",
    "BearerTokenAuthenticator",
    "CohereReranker",
    "JinaReranker",
    "NvidiaReranker",
    "VoyageReranker",
    "JWTBearerAuthenticator",
    "TokenRevocationChecker",
    "Environment",
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationCheck",
    "EvaluationError",
    "EvaluationReport",
    "RetrievalCaseResult",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationError",
    "RetrievalEvaluationReport",
    "EmbeddingError",
    "EmbeddingCandidate",
    "EmbeddingCaseResult",
    "EmbeddingEvaluationCase",
    "EmbeddingEvaluationError",
    "EmbeddingEvaluationReport",
    "EmbeddingInputFormat",
    "EmbeddingResponseSizeError",
    "GeminiEmbeddingProvider",
    "HuggingFaceFeatureExtractionProvider",
    "OpenAICompatibleEmbeddingProvider",
    "ActiveGenerationReader",
    "AsymmetricEmbeddingProvider",
    "EmbeddingProvider",
    "Document",
    "DefaultPolicyEngineFactory",
    "EngineAdapter",
    "ExecutionResult",
    "ExecutionTrace",
    "FileIngestor",
    "FileParser",
    "MarkupTextParser",
    "ModelProvider",
    "ModelRequestSizeError",
    "ModelResponseSizeError",
    "ModelPlanner",
    "ModelResponse",
    "ModelStreamDelta",
    "HuggingFaceInferenceProvider",
    "GeminiProvider",
    "HybridRetriever",
    "RerankingRetriever",
    "RerankingError",
    "RerankingResponseSizeError",
    "SQLiteTokenRevocationStore",
    "AsyncRedisCommands",
    "RedisSkillRevocationStore",
    "HybridIndexCoordinator",
    "GenerationAwareRetriever",
    "GenerationAwareVectorStore",
    "GenerationManifestStore",
    "GenerationLease",
    "GenerationBusyError",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "RetryableModelError",
    "ParagraphChunker",
    "ParsedPage",
    "PlanStep",
    "Planner",
    "PlanningCapability",
    "PlanningObservation",
    "PlanningResult",
    "ExecutionPlan",
    "PendingIndexGeneration",
    "TextChunker",
    "TextFileIngestor",
    "IngestionReport",
    "InMemoryBM25Retriever",
    "KnowledgeStore",
    "KnowledgeStoreError",
    "Principal",
    "PolicyEngine",
    "PolicyEngineFactory",
    "PolicyEngineProtocol",
    "ResolvedAgentDefinition",
    "ResolvedSkillDefinition",
    "SkillDefinition",
    "SkillCatalogEntry",
    "SkillRegistryClient",
    "SkillRegistryError",
    "StaticSkillRegistryBuild",
    "build_static_skill_registry",
    "SkillPackageError",
    "SkillPackageFile",
    "SkillPackageInfo",
    "SkillPackageInspection",
    "SkillRegistryAuditRecord",
    "inspect_skill_package",
    "RunRequest",
    "Runtime",
    "Retriever",
    "ResolvedEnvironment",
    "Reranker",
    "SandboxCommandResult",
    "SandboxDefinition",
    "ScopedCredentialProvider",
    "SandboxError",
    "SandboxResourceUsage",
    "SandboxResourceMonitor",
    "SandboxSession",
    "SandboxTimeout",
    "SandboxUnavailable",
    "StreamingModelProvider",
    "SQLiteFTS5Store",
    "SQLiteVectorStore",
    "SQLiteGenerationManifestStore",
    "StreamEventType",
    "StreamJournal",
    "StreamJournalSnapshot",
    "SourceKnowledgeWriter",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolRegistry",
    "Utf8TextParser",
    "TraceEvent",
    "Tracer",
    "CompositeTracer",
    "LFTraceExporter",
    "OpenTelemetryTracer",
    "SignedAuditExporter",
    "create_mcp_server",
    "load_mcp_tools",
    "langgraph_node",
    "crewai_tool",
    "n8n_request",
    "create_otlp_tracer",
    "VectorStore",
    "WorkspaceMount",
    "VerificationResult",
    "Verifier",
    "PersonaVerifier",
    "HostContext",
    "PayloadFilter",
    "PayloadFilterError",
    "DefaultPayloadFilter",
    "validate_agent_definition",
    "validate_skill_definition",
    "install_skill",
    "audit_skill_registry",
    "verify_installed_skill",
    "evaluate_retriever",
    "load_retrieval_evaluation_dataset",
    "evaluate_embeddings",
    "load_embedding_evaluation_dataset",
    "SkillTrustPolicy",
    "load_skill_trust_keys",
    "SkillIntegrityError",
    "SkillRevocationChecker",
    "SkillRevocationStore",
    "SkillRevocationUnavailable",
    "SkillRevokedError",
    "SQLiteSkillRevocationStore",
    "uninstall_skill",
    "pack_skill",
    "sign_skill_package",
    "verify_skill_package_signature",
    "create_app",
    "ensure_model_request_size",
    "ensure_model_response_size",
    "ConfiguredSkillSelector",
    "CancellationToken",
    "ModelSkillSelector",
    "SkillActivation",
    "SkillSelection",
    "SkillSelector",
    "sandbox_tools",
    "sqlite_query_tool",
    "python_run_tool",
    "sandboxed_python_tool",
    "evaluate_agent",
    "load_evaluation_dataset",
]

__version__ = "0.1.0b5"
