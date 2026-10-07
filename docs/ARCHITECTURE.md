# Gabby Architecture

**Status:** living architecture document for the pre-1.0 implementation. Accepted decisions and known limitations are recorded in the ADRs, roadmap, and [project scorecard](PROJECT_SCORECARD.md). Public extension contracts may change before v1.0; see [the extension API policy](EXTENSIONS.md).

## Product boundary

Gabby constructs and executes reusable agent capabilities. It does not own a user's conversation, project history, or durable memory. A caller sends one request with whatever context it wants the agent to see. Gabby returns a result and optional trace; request-local state is discarded when execution ends.

Runtime composition is specialization: model choice, instructions, environment, skills, tools, knowledge, and policies. It does not imply model-weight training. Fine-tuning is a separate optional subsystem.

## Proposed component model

```text
Agent definition ─┐
Skill registry ────┼─> Definition resolver/compiler ─> Resolved agent
Environment ──────┘                                  │
                                                     v
Caller request ─> Stateless runtime ─> model / retrieval / tools
                         │                         │
                         ├─ policy decisions <─────┘
                         ├─ verification
                         └─ trace + response
```

The definition resolver validates YAML and programmatic `AgentDefinition` values through the same construction-time checks, then produces an immutable, resolved agent plan. Nested configuration mappings accept finite JSON-compatible values and are capped at 10 MiB, 100,000 nodes, and 128 levels; cycles and unsupported Python objects fail with `ConfigError`. The plan snapshots nested config, resolved skill definitions, environment declarations, tool schemas, and registry membership; source file, definition, environment, registry, or skill edits affect a later agent construction only. Agent and skill tool grants must resolve to registered tools before model-provider construction, so a typo fails before an execution request. Configured knowledge sources require an injected `Retriever`, and enabled verification requires an injected `Verifier`; both are checked during construction. The CLI injects its configured local SQLite FTS5 store for `run` and `serve`, while validation and inspection use an empty in-memory retriever unless a database is supplied. Runtime service handles such as model clients, retrievers, verifier instances, and trusted handler closures remain injected references. Missing providers, skills, tools, or incompatible policy should fail during construction rather than halfway through a request. See [ADR 0006](architecture/adr/0006-immutable-agent-plans.md), [ADR 0021](architecture/adr/0021-shared-agent-definition-validation.md), and [ADR 0065](architecture/adr/0065-bounded-immutable-config-snapshots.md).

### Agent definition

Persistent, versioned configuration: name, model reference, environment reference, skill references, tool grants, knowledge sources, instructions, policies, and verification configuration. It contains no live conversation state or credentials. Gabby rejects configured model API keys and credential-bearing headers for both YAML and in-memory definitions; secret values come from environment lookup or a provider injected by the host. See [ADR 0017](architecture/adr/0017-provider-credentials-outside-agent-config.md).

An optional `output_schema` sets a machine-readable final-response contract. The runtime prompts
provider-neutrally, parses strict JSON, validates it locally, and returns the parsed value as result
metadata. Structured SSE text is buffered until validation succeeds. See [ADR 0086](architecture/adr/0086-validated-structured-agent-output.md).

### Environment

An environment describes an operating context and supplies scoped resources and capabilities. It can constrain tools, filesystem roots, network routes, runtimes, and external services. At agent construction, Gabby snapshots the environment description, allowlist, and tool registrations into an agent-local resolved environment so composing one agent does not mutate a reusable environment or another agent's registry. Host resource handles remain shared references in the host process and are never serialized into model context. A host-trusted tool can opt into request-scoped `ToolContext` injection and declare the exact environment resource names it may receive; construction fails if any grant is unavailable. The context also carries authenticated principal, run metadata, and a process-local agent instance identity without adding them to model arguments, plus a cooperative cancellation token that is signaled on tool timeout or run cancellation; see [ADR 0037](architecture/adr/0037-cooperative-tool-cancellation.md) and [ADR 0072](architecture/adr/0072-agent-composition-instance-identity.md). The identity is used only for delegation cycle detection, not authorization or durable references. A capability description is explanatory; runtime policy and constrained handlers enforce the actual boundary.

### Agent composition

`AgentTool` exposes an async child agent as one explicit parent tool. The parent sends only the task
string selected for that tool call; context, memory, metadata, and principal are not shared implicitly.
The child keeps its own resolved definition, model, skills, tools, knowledge, verifier, and policy.
The parent must grant `agent:invoke`, and the host owns and closes each agent. Delegation inherits the
parent tool timeout and byte limits, rejects cycles and chains deeper than eight agents, and forwards
the authenticated principal only when the host opts in. The child result returns its output and trace
ID to the parent model, and the child trace records the parent trace ID for correlation while child
events remain independently exported. See the
[extension contract reference](EXTENSION_CONTRACTS.md).

### Skill

A portable package with a stable identifier and SemVer version, description, instructions/procedure,
dependencies, activation metadata, tool requirements, knowledge references, input/output schema
where useful, and verification guidance. Agent and dependency references can pin exact versions
with `skill-id@MAJOR.MINOR.PATCH`; unpinned references remain compatible and resolve to the local
package version captured in the immutable plan. One agent resolves only one version of each stable
skill ID because selectors and tool grants address skills by ID. Skill resolution is
dependency-aware and deterministic. Selection is a replaceable strategy (configured, rule-based,
model-selected, or hybrid); only selected skill content and granted tools enter a run.
`gabby skill pack` and `gabby skill install` exchange bounded, checksummed local ZIP packages in the
filesystem registry layout. Checksums detect corruption, not publisher identity; packages are not
authenticated and installation does not execute package code. Signed remote discovery and exact-
version dependency installation are available through the static HTTPS registry client and
`gabby skill fetch`; Gabby does not host a registry service. See the
[skill registry guide](SKILL_REGISTRY.md) and [ADR 0082](architecture/adr/0082-static-skill-registry-builder.md).

### Skill selection

Skill selection is an injected async `SkillSelector`. The default `ConfiguredSkillSelector` preserves deterministic activation: skills with no triggers are active, and triggered skills match case-insensitive phrases in the request. The opt-in `ModelSkillSelector` uses an explicitly injected model provider to choose from configured skill IDs and records its usage and duration in the execution trace. The runtime validates selected IDs, expands declared dependencies, and still enforces tool policy; a selector cannot grant tools by itself. A selector shares the run deadline. Model-driven selection adds provider latency, token cost, and provider data handling. See [ADR 0008](architecture/adr/0008-pluggable-skill-selection.md).

Each serialized provider request is limited to 4 MiB by default through
`policies.max_model_request_bytes`. The limit includes messages and tool schemas and is checked
before runtime calls and built-in planner or model-selector calls. It applies per call rather than
as an aggregate run budget. The public `ensure_model_request_size` helper is available to custom
selectors and planners; opaque extension network calls cannot be enforced by the core. See
[ADR 0016](architecture/adr/0016-bounded-model-requests.md).

Provider responses are separately limited to 4 MiB by default through
`policies.max_model_response_bytes`. Built-in OpenAI-compatible JSON and SSE adapters stop reading
when the decoded response body exceeds the cap. The runtime also bounds custom-provider response
strings and stream deltas before they enter context or are forwarded; custom providers must cap
their own network reads before returning a buffered response. The cap is per provider call. See
[ADR 0018](architecture/adr/0018-bounded-model-responses.md).

Built-in OpenAI-compatible adapters do not include upstream error bodies or transport exception
text in raised errors, including for direct `Agent` and provider callers. HTTP status codes remain
available for diagnosis; execution traces retain stable error categories. Injected providers own
their error sanitization contract. See
[ADR 0038](architecture/adr/0038-sanitized-model-provider-errors.md).

The FastAPI request body defaults to a 1,000,000-byte cap and a 30-second total receive deadline,
both enforced while buffering ASGI chunks before JSON parsing. A body that misses its deadline
receives HTTP 408; Gabby closes the HTTP/1 connection or ends the HTTP/2 stream. Embedded apps can set
`create_app(max_request_bytes=..., request_body_timeout_seconds=...)`; the CLI exposes these with
`gabby serve --max-request-bytes` and `--request-body-timeout`. These settings bound each request,
not the number of open connections; deployments still need server and ingress connection limits.
See [ADR 0028](architecture/adr/0028-bounded-request-body-time.md).

The FastAPI service separately caps each serialized HTTP response body at 4 MiB by default,
including `/run` output and traces or the complete SSE stream with keepalives. Configure this with
`create_app(max_response_bytes=...)` or `gabby serve --max-response-bytes`. `/run` returns a bounded
error instead of partial JSON when a result exceeds the cap; an oversized stream ends with a typed
size-limit error when space remains and never emits an oversized completion. See
[ADR 0024](architecture/adr/0024-bounded-http-responses.md).

### Tool

A typed operation with a stable name, JSON input schema, handler, output contract, timeout, permission metadata, execution class, and a serialized result byte limit. The default result limit is 1 MiB per tool; Gabby rejects oversized nested strings before JSON encoding and rejects oversized serialized results before content enters model context. Model output is untrusted input: Gabby validates tool identity and arguments, checks policy, then calls a registered handler. Injected handlers are `host_trusted` by default and execute with the Gabby process's host permissions. Built-in shell and filesystem operations declare `sandboxed` execution and run through the per-run container. `policies.require_sandbox: true` rejects any declared tool that lacks a sandboxed implementation before model execution. Python tools run concurrently only when each tool explicitly sets `parallel_safe=True` and the agent opts into `policies.max_parallel_tool_calls` (default 1, maximum 32). This declaration is for independent host handlers without approval; batches containing any other tool stay sequential. Results return to the model in original call order. Tracer-enabled runs stay sequential so trace callbacks retain their ordered contract. A Python handler is never made safe merely by setting a policy.

### Policy engine

`Agent` accepts an optional `PolicyEngineFactory`. Gabby asynchronously creates a run-scoped policy
engine after skill activation and before exposing the active tools to the model. The factory receives the
active declared tool names, resolved policy mapping, environment allowlist, and authenticated
principal. The built-in `PolicyEngine` remains the default through `DefaultPolicyEngineFactory`.
Custom factories are trusted host code; Gabby sanitizes their failures and continues to enforce tool
schemas, approval, sandbox, timeout, and resource boundaries. See
[ADR 0124](architecture/adr/0124-injectable-policy-engine.md).

### Knowledge

`HTMLTextParser` is the default dependency-free parser for UTF-8 `.html` and `.htm` files. It keeps
titles and structural text boundaries while omitting script, style, template, SVG, and hidden
subtrees; `FileIngestor` enforces the configured file and extracted-text limits around it.
`JSONTextParser` validates strict UTF-8 JSON, rejects duplicate object keys, and checks nesting and
token limits before constructing the decoded value. It preserves the original JSON text so numeric
lexemes and Unicode content remain unchanged in retrieval.
`JSONLinesTextParser` applies the same strict JSON validation to each non-blank `.jsonl` line,
preserves physical line numbers in the rendered records, and limits record count, record size, and
total rendered output before indexing.
`XMLTextParser` streams UTF-8 XML into path-labeled text and attributes, rejects DTD and entity
declarations, and bounds input/output bytes, elements, depth, and attributes without an extra package.
`CSVTextParser` expects a header row, rejects ambiguous headers and records with inconsistent field
counts, and renders bounded rows as labeled values before retrieval chunking.

Retrieval is separate from skills and tools. A retriever returns source-attributed documents with metadata. The core includes an in-memory BM25 implementation for small fixtures, a persistent SQLite FTS5 store, and a host-pooled PostgreSQL full-text store. Both persistent lexical stores support async ingestion, source replacement, exact metadata filters, and lexical ranking; PostgreSQL schema creation remains migration-owned. `FileIngestor` performs file discovery, path checks, and file reads in bounded worker threads. It reads UTF-8 Markdown/plain text and `.log` files as text without log-format interpretation, with bounded YAML front matter exposed as filterable page metadata, RFC 5322 email, iCalendar VEVENT pages, and vCard contact pages with filterable metadata, HTML, JSON, JSON Lines, XML, CSV, bounded RTF text, DOCX packages, PPTX slide text, EPUB chapter text in spine order, ODT document text, ODS sheet values, and XLSX sheet values by default. `ParquetTextParser` is available through the optional `parquet` extra and indexes flat tables in bounded batches. It supports page-attributed PDF text through the optional `pdf` extra, and supports bounded raster image OCR through the optional `ocr` extra and host-installed Tesseract, plus opt-in scanned-image and vector-only PDF OCR through the `pdf-ocr` extra and a replaceable `PDFPageRenderer`; email parsing bounds MIME part count and nesting, indexes selected headers and text bodies, and skips attachments. Parsers and chunking are replaceable, and each file is atomically replaced as one source. `TextFileIngestor` remains a compatibility alias. `SQLiteVectorStore` persists normalized vectors and performs exact cosine search with bounded top-result memory; its linear scan is intended for small or moderate local corpora. The host selects an `EmbeddingProvider` and keeps one compatible embedding space for each index. `OpenAICompatibleEmbeddingProvider` implements bounded batched requests to standard `/embeddings` endpoints, while `HuggingFaceFeatureExtractionProvider` adapts the feature-extraction task response, including configured pooling for token features. `HybridRetriever` combines a lexical retriever with an embedding provider and `VectorStore`, using reciprocal-rank fusion over concurrently fetched candidate lists. `RerankingRetriever` optionally wraps any retriever, fetches a bounded candidate pool, and lets an injected `Reranker` reorder a unique subset without changing the source documents. `HybridIndexCoordinator` stages matching source generations in both backends and commits one generation through the `GenerationManifestStore` only after both are ready. The built-in `SQLiteGenerationManifestStore` keeps this durable commit record in a separate SQLite database. On recovery, a complete pending generation is activated and an incomplete one is deleted from both backends. Retired generations remain available until an operator invokes `prune_retired(readers_quiescent=True)` after readers drain. The built-in SQLite lexical and vector stores support fenced staging, idempotent generation deletion, and generation-filtered retrieval. The PostgreSQL lexical store supports direct ingestion and retrieval, and its generation staging, retrieval filters, and fencing make it suitable for the lexical half of coordinated multi-instance hybrid indexing with `PostgresVectorStore` or another shared vector backend and manifest. `KnowledgeStore`, `EmbeddingProvider`, `VectorStore`, `Reranker`, `FileParser`, `TextChunker`, and `GenerationManifestStore` are independent extension interfaces; Gabby bundles bounded HTTP adapters for OpenAI-compatible embeddings and Hugging Face feature extraction, but no embedding or reranking model. Runs request at most 100 documents and cap rendered retrieval context at 1 MiB of UTF-8 by default; invalid, excess, or oversized custom results fail before prompt construction. The manifest coordinates generation activation but does not provide a distributed transaction across arbitrary backends. Retrieved text is untrusted reference material and must not acquire instruction priority. See [ADR 0026](architecture/adr/0026-bounded-retrieval-context.md), [ADR 0027](architecture/adr/0027-composable-reranking.md), [ADR 0033](architecture/adr/0033-sqlite-exact-cosine-vector-store.md), [ADR 0034](architecture/adr/0034-openai-compatible-embedding-provider.md), and [ADR 0036](architecture/adr/0036-huggingface-feature-extraction-embeddings.md).

Multi-instance hybrid indexing can use `PostgresGenerationManifestStore` with a host-owned asyncpg-compatible pool and the externally applied migration in `sql/postgres_generation_manifest.sql`. PostgreSQL transactions and server time coordinate shared leases, fencing, generation activation, and retirement. This does not turn the built-in SQLite lexical or vector stores into shared multi-host stores; every service must use shared generation-aware backends. Resumable SSE can use `PostgresStreamJournal` with its host-managed migration so workers across replicas share bounded event frames and session identity. See [ADR 0105](architecture/adr/0105-postgres-generation-manifest.md) and [ADR 0110](architecture/adr/0110-postgres-stream-journal.md).

`CohereReranker` is an optional first-party `Reranker` HTTP adapter; hosts explicitly select and inject it, own its credential, and decide whether candidate text may leave the host. It adds no provider SDK or model dependency. See [ADR 0044](architecture/adr/0044-cohere-rerank-provider.md).

## Request lifecycle

1. Validate request size, shape, identity, and caller-provided context. Embedded `Agent.arun()` and
   the HTTP API accept the same 1–100,000 character task limit; context, memory, and metadata are
   JSON-normalized snapshots bounded together by the configured model-request byte limit.
2. Create isolated run state, deadline/cancellation token, and trace context.
3. Resolve effective policy and active skills; construct bounded context.
4. Optionally ask an injected planner for a validated advisory plan using only policy-approved
   skills and tools; the default path skips this provider call. An agent can opt into at most three
   revisions after tool batches, based on bounded observations from those tools.
5. Call the model through a provider interface with the plan as guidance when one was produced.
6. Validate every requested tool call against the resolved tool set, JSON schema, policy, and remaining budget.
7. Execute tools with bounded time/resources; append structured observations.
8. Continue until final response, cancellation, deadline, resource limit, or step limit.
9. Run configured verifiers and report their results distinctly from model claims.
10. Return response, metadata, and optional trace; release run-local state and handles.

Transient model failures may be retried only when `policies.max_model_retries` opts in; the default is zero and each wait is charged to the run deadline. Built-in providers classify retryable HTTP and network failures, while custom providers use `RetryableModelError`. Streaming stops retrying after the first text delta is sent. Tool calls are never retried implicitly because their side effects may not be idempotent. Cancellation and deadlines must flow through provider, retrieval, and tool interfaces.

`Agent.aclose()` marks an async agent as closing, rejects new runs, drains active `arun()` and
`astream()` executions, and then closes owned provider resources. The FastAPI lifespan uses this
contract during shutdown. An async Agent remains bound to the event loop where it first runs; sync
agents retain the separate persistent-loop behavior described in [ADR 0002](architecture/adr/0002-sync-loop-lifecycle.md).
See [ADR 0022](architecture/adr/0022-graceful-agent-shutdown.md).

## Instruction and policy precedence

Policies are executable controls, not instructions. The runtime checks permissions independently of
the model prompt; a prompt cannot grant a tool, filesystem, network, or sandbox capability. The
system message presents requirements in this order:

1. Gabby runtime requirements and the resolved environment capabilities.
2. Optional construction-time global instructions supplied to `Agent`.
3. Agent instructions.
4. Instructions for the skills activated for this run.
5. The caller's task.

Caller context and memory, retrieved documents, and tool observations are supplied as untrusted data.
They cannot change the instruction order or policy. Retrieved documents are labeled as reference
data; the runtime checks tool identity, schema, and policy before execution. This prompt structure
helps the model follow the contract but is not a security boundary. Security relies on runtime
policy enforcement and sandboxing. Provider adapters must preserve the system/user distinction.
Global instructions are immutable string input captured when an `Agent` is constructed; callers
rebuild the agent to change them. They can describe shared application behavior but cannot grant
capabilities or supersede runtime safety requirements or enforced policies.

## Interface and execution model

The core is Python. The runtime is async-first, with synchronous convenience methods for scripts. A sync `Agent` reuses one event loop on its calling thread and cannot mix sync/async modes or move between threads. Provider, retriever, verifier, and async tool implementations are awaited directly; sync callbacks are run off the event loop in a bounded worker pool. A callback still queued when its timeout or cancellation arrives is skipped. Cancellation cannot forcibly stop arbitrary synchronous functions once their worker thread has started. See [ADR 0001](architecture/adr/0001-python-async-fastapi.md) and [ADR 0002](architecture/adr/0002-sync-loop-lifecycle.md).

Container isolation for shell, filesystem, and declared custom executable tools is implemented behind Docker and Podman CLI/API adapters: engine, image, and workspace access come from agent configuration; missing images are pulled on demand; one temporary container is created per agent run; networking is disabled; Linux containers default to non-root UID/GID `65532:65532` with a numeric per-agent override; native Windows containers require Docker Hyper-V isolation and use their image-defined user. The Windows API adapter supports Docker Desktop's local named pipe through an optional `windows-sandbox` dependency. Custom sandboxed tools use a fixed config-owned argv and a bounded JSON request file in a per-run host directory mounted read-only; their implementation runs from the configured image. The consuming application must keep the mounted workspace quiescent for the full run because component checks cannot prevent a concurrent host process from racing separate archive operations. Adapter implementation does not imply production support on a host. Each host/engine/adapter combination is supported only after live acceptance checks cover isolation, resource limits, workspace mounts, networking, and cleanup. Current live evidence covers Docker and Podman on Linux through both adapters, including custom executable tools; other combinations remain unverified/experimental. Injected Python handlers are host-trusted by default, with `policies.require_sandbox: true` to reject them when a run requires sandboxed tools. See [ADR 0003](architecture/adr/0003-container-sandbox.md) and the [operations guide](OPERATIONS.md).

Agent-owned engine adapters are closed even when startup fails before yielding a sandbox session; injected adapters remain host-owned.

Docker-compatible API control responses are read incrementally with a 1 MiB bound, so engine
metadata, status, and error bodies cannot trigger unbounded client buffering. Exec output, archive
transfers, image-pull progress, and CLI output have their own limits.

Sandboxed runs add a best-effort final `sandbox_resource_usage` trace event when the engine reports counters. API adapters report cumulative CPU time, current and peak memory, and current process count where available; CLI adapters report a final CPU percentage, current memory, and process count. These are a final snapshot rather than a time series or billing-grade accounting source; unsupported fields are omitted, and metric collection failure does not fail the run.

Provider interface: complete/stream with structured messages, tool calls, usage, cancellation, and timeout. Built-in OpenAI-compatible, Ollama, Hugging Face, Anthropic, and Gemini adapters require HTTPS for remote endpoints and permit HTTP only on loopback; custom providers own their transport security. Anthropic uses the native Messages API, translates system and tool-result messages, and normalizes streamed text and tool-input events. Gemini uses GenerateContent REST, translates function calling, and round-trips provider IDs and thought signatures needed by stateless multi-step calls. Mock transport tests cover provider contracts; live compatibility with Ollama, hosted providers, and individual model/provider combinations remains unverified. The optional local `TransformersProvider` loads models lazily, uses bounded worker-thread generation, and disables custom model code and pickle-based model weights. Its model-specific chat template must support structured parsing to use tools; it currently returns complete responses rather than token deltas. Fake-backend contracts cover provider mapping and parsing; one CPU completion smoke passed with Transformers 5.18.0 and PyTorch 2.14.1+cpu on a pinned SmolLM2-135M-Instruct revision. One real CPU structured tool-call provider acceptance passed on a pinned Qwen2.5-Coder-0.5B-Instruct revision; full multi-step model behavior, other templates, model quality, throughput, and accelerator combinations remain unverified. See [ADR 0071](architecture/adr/0071-local-transformers-provider.md), [ADR 0007](architecture/adr/0007-huggingface-inference-providers.md), [ADR 0019](architecture/adr/0019-https-for-remote-model-providers.md), and [ADR 0090](architecture/adr/0090-anthropic-messages-provider.md), and [ADR 0096](architecture/adr/0096-native-gemini-provider.md). Tool interface: validated arguments, structured result/error, cancellation, and execution limits. Host-trusted handlers can opt into request-scoped `ToolContext`, with a construction-validated allowlist of environment resources and a cooperative cancellation token, both excluded from model arguments; see [ADR 0030](architecture/adr/0030-scoped-tool-context.md) and [ADR 0037](architecture/adr/0037-cooperative-tool-cancellation.md). Tools marked `requires_approval=True` pause at an injectable host-owned async approval handler; without a handler they are denied. The consumer owns approval UX and persistence. Approval runs within the execution deadline and emits stream and trace events. Verifiers implement the typed `Verifier` protocol and return `VerificationResult` values, which are validated and recorded in response metadata and traces; see [ADR 0010](architecture/adr/0010-typed-verifier-contract.md). An optional injected `Planner` can return a bounded structured plan that is passed as advisory context; the existing reactive tool loop remains the default and enforces all policies. `Agent` also accepts an optional async `Tracer` that receives ordered event snapshots; the existing run trace remains in the result, and exporter failures are bounded and fail-open. The host owns telemetry credentials, transport, retention, and deletion. See [ADR 0023](architecture/adr/0023-injectable-event-tracer.md). Other replaceable interfaces: environment, skill resolver/selector, tool registry, retriever, reranker, policy engine, sandbox, and storage.

Local Transformers can optionally load an existing PEFT adapter for inference with `model.adapter_id` and `model.adapter_revision`; the `transformers-adapters` extra supplies PEFT, while the host selects the appropriate PyTorch build. Adapter loading requires safetensors weights and does not train or modify the adapter. Training remains a separate optional workflow; see [ADR 0120](architecture/adr/0120-inference-only-peft-adapter-loading.md).

`gabby train` is an optional local SFT workflow that produces LoRA adapters from bounded JSONL conversations. It uses the selected model chat template's assistant-token mask, fails when assistant spans are not explicitly marked, and records the source dataset digest without retaining sample text. Training dependencies and accelerator selection stay outside the runtime; see [ADR 0121](architecture/adr/0121-optional-bounded-local-lora-sft.md).

Use explicit capability checks and stable error types. The `/v1` OpenAPI contract has typed run request, result, trace-event, and execution-error schemas. Authentication errors still use FastAPI's standard HTTP error envelope. Avoid passing arbitrary dictionaries between core modules when a public type can make the contract clear. Public extension compatibility follows the pre-1.0 policy in [ADR 0009](architecture/adr/0009-pre-1-0-extension-api-policy.md) and [the extension guide](EXTENSIONS.md).

## Deployment

- **Embedded:** Python library calling the same runtime directly.
- **Local service:** a FastAPI ASGI adapter for IDEs, CLIs, and other local consumers.
- **Hosted:** a separately operated service with authentication, tenant isolation, rate/resource limits, secret management, and durable telemetry controls.

All modes use the same resolved agent and execution contracts. Transport adapters do not own agent logic. Statelessness does not mean runs share no infrastructure: provider clients, immutable definitions, and external knowledge stores may be reused, while request messages/tool observations are run-scoped.

Local IDEs and MCP hosts can launch the optional `create_mcp_server()` stdio adapter or use
`gabby mcp`; every `run_agent` call passes caller-owned state to one execution. The MCP extra is
loaded only when requested. Remote MCP hosting is not exposed; remote clients use the authenticated
FastAPI boundary. See the [MCP guide](MCP.md) and [ADR 0089](architecture/adr/0089-optional-local-mcp-stdio-adapter.md).

Streaming is a transport feature over the same event model. `POST /v1/agents/{name}/stream` emits SSE text deltas, skill and optional plan progress, tool and approval progress, completion, and sanitized error events; it does not persist run state. External Python applications can use the async-first `GabbyClient.arun()` and `.astream()` methods or their sync wrappers; the SDK does not retain caller context, automatically retry POST requests, or own conversation history. Its response limit defaults to 4 MiB, and remote service URLs must use HTTPS. See the [Python client guide](CLIENT.md) and [ADR 0079](architecture/adr/0079-stateless-python-http-client.md). The public API is rooted at `/v1`; `/health` stays unversioned for platform probes. The configured per-process capacity is acquired before request-body buffering and authentication, then held through the response. Full capacity returns HTTP 429 immediately; `/health` bypasses the gate. This bounds the number of simultaneously buffered agent requests as well as active runs and streams, but is not a request-rate or distributed limit. See [ADR 0029](architecture/adr/0029-admission-before-body-buffering.md). API authentication belongs at the service boundary: run and stream endpoints require an injected authenticator by default, the built-in bearer-token and optional issuer-bound JWT authenticators target single-tenant use, and unauthenticated local serving is explicit and loopback-only through `serve`. Embedded apps can enforce separate `run_scopes` and `stream_scopes`; their authenticated principal is passed separately to host approval callbacks and is not added to model context. See [ADR 0004](architecture/adr/0004-api-authentication.md) and [ADR 0035](architecture/adr/0035-route-scoped-api-authorization.md). The initial tenancy contract is one tenant and one configured agent per Gabby service instance; deployments isolate tenants by routing them to separately configured instances. See [ADR 0005](architecture/adr/0005-single-tenant-service.md). TLS termination, request-rate limits, and deployment routing remain service/deployment concerns. A local development server should not be presented as production hosting.

HTTP and SSE execution failures use stable Gabby error types. `AgentRuntimeError` is preserved;
unexpected exceptions from injected providers or extensions are reported as
`AgentExecutionError` without exposing their implementation class names or messages. Capacity and
response-size rejections retain their dedicated error types.

## Observability and data handling

Execution traces are returned with each embedded result and included in `/run` and SSE completion
responses by default. HTTP callers can set `include_trace: false` to omit the trace body while
keeping the trace ID; this does not disable an injected host tracer. There is no automatic redaction
setting today. Ordinary built-in events record
metadata such as input length, context keys, skill and tool names, argument keys, retrieval source
identifiers, timing, and usage, rather than raw prompts, context values, tool arguments, or tool
results. When planning is enabled, the trace includes each structured plan and plan revision.
Replanning is off by default; `policies.max_replans` accepts 0 through 3 and requires an injected
planner. Each revision receives the preceding plan and at most 16 results from the latest tool
batch, with each observation capped at 4 KiB. These excerpts are sent to the planner provider and
carry no tool authority; the runtime continues to enforce the original allowlist and policies. SSE
emits `plan_updated` for revised plans. Verification events
include the verifier's structured result, whose `details` and evidence are supplied by the injected
verifier and may contain sensitive values. Custom tracers receive these structured events during
execution.

Treat result bodies and trace exports as sensitive. Protect response access, decide whether plans
and verifier details are suitable for the caller and telemetry destination, and define retention and
deletion for host-side exports. Gabby does not persist conversations or traces between runs by
default.

## Open-source quality bar

- Published configuration and API schemas with compatibility policy.
- Reproducible packaging, supported Python versions, locked development dependencies, and CI across supported platforms.
- Unit and integration coverage for policy enforcement, tool schemas, cancellation, timeouts, stateless isolation, and provider compatibility.
- Threat model and security reporting process; no claim of sandboxing unless enforced by an OS/container boundary.
- Clear license, contribution guide, code of conduct, changelog, release process, and deprecation policy.
- Examples that run using mock providers by default, with no API key or network needed for core workflows.

## Current limitations and next work

Gabby is pre-1.0 and the implementation is not yet a production certification. Current known limits
and evidence are maintained in [the roadmap](../ROADMAP.md) and [the weighted project scorecard](PROJECT_SCORECARD.md)
so this document does not duplicate a second, quickly-stale checklist. In particular, cross-platform
live sandbox enforcement has only been verified on Linux with Docker and Podman; native Windows and
macOS acceptance is outstanding. Live model and reranking provider checks also remain outstanding.

The design has concrete limits that hosts must account for: arbitrary injected Python handlers are
host-trusted; sync callbacks already running in worker threads cannot be forcibly stopped; the API
capacity bound is per process and not a distributed rate limit; a mounted workspace must remain
quiescent for a run; the SQLite vector store uses exact linear scans for small and moderate indexes;
and Gabby does not provide a hosted multi-tenant control plane. The CLI's shared bearer token does not
support per-caller capabilities. OIDC issuer discovery is available through bounded async startup
configuration. The included Nginx reference has a repeatable smoke check for request-size rejection,
health forwarding, and burst limiting against the hosted agent; TLS certificates, trusted real-IP
configuration, deployment routing, and distributed revocation lifecycle remain host/deployment work.
See the roadmap for the remaining release
gates and evidence status.

## Delivery focus

Continue with the roadmap's highest-risk evidence gaps: live sandbox acceptance on each claimed
platform, live provider compatibility, recovery behavior against a non-SQLite vector backend, and
deployment-level streaming and admission controls. Before v1.0, publish compatibility guarantees for
named extension contracts and run the full CI matrix. The weighted scorecard records the current
implementation coverage separately from those remaining release gates.

Each stage should land as small reviewable changes with acceptance criteria and tests. No hosted-production readiness claim should precede tenant isolation, authentication, resource limits, secrets handling, and operational documentation.
