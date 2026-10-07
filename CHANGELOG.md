# Changelog

User-visible changes are recorded here. Before the first release, entries are grouped under
`Unreleased`; published releases use dated version headings.

## Unreleased

- Added optional local `gabby train` supervised fine-tuning for PEFT LoRA adapters, with bounded
  assistant-only JSONL data, chat-template mask validation, safetensors output, and a data-digest
  training manifest; `--check-dataset` validates data structure without loading ML dependencies.
- Added optional inference-only PEFT adapter loading to the local Transformers provider, with
  separately configurable base and adapter revisions and safetensors-only adapter weights.
- Added bounded OPML 1.0/1.1/2.0 ingestion with one cited page per outline, searchable category
  paths, safe subscription URLs, `.opml` support, and `.xml` root detection without fetching links.
- Added bounded JSON Feed 1.0/1.1 ingestion with one cited page per item, safe link metadata, and
  plain-text extraction from HTML content; ordinary JSON files retain source-preserving behavior.
- Added opt-in `python_run_tool()` for data agents; it stages bounded source through a private
  read-only mount and executes it with the configured interpreter inside the per-run sandbox.
- Added `examples/python_sandboxed_agent.py`, an offline end-to-end example that executes Python
  through the sandboxed tool using a deterministic mock model.
- Extended bounded iCalendar ingestion to index `VTODO` tasks and `VJOURNAL` entries as cited pages
  with type-specific filter metadata while preserving existing `VEVENT` metadata.
- Added bounded OpenDocument Presentation `.odp` ingestion with one cited page per slide and
  filterable slide-name metadata; notes, images, charts, and embedded media are not extracted.
- Added bounded separator-framed `.mbox` archive ingestion with one citation per message and the
  same attachment-excluding MIME extraction used for individual `.eml` messages.
- Added bounded vCard 2.1, 3.0, and 4.0 contact ingestion with per-contact citations and filterable
  metadata; URL fields are indexed but never fetched.
- Added `gabby skill init` to scaffold a validated portable skill source directory with a manifest,
  procedure, and examples; it creates new directories only and leaves tool grants empty.
- Added `gabby init` templates for generic, research, and data-analysis agents, with configurable
  built-in model provider settings and exclusive file creation that preserves existing configs.
- Added UTF-8 `.log` files to built-in knowledge ingestion; log content is indexed as text without
  assuming a timestamp or severity format.
- Added a bounded NVIDIA NeMo hosted reranker with host-owned credentials and validated response mapping.
- Added `gabby knowledge delete ROOT SOURCE --database DB` to remove stale indexed chunks by safe
  relative path, including when the source file has already been removed.
- Added bounded UTF-8 ingestion for AsciiDoc and reStructuredText files. Markup stays searchable as
  source text; directives and external includes are never executed.
- Added `PostgresStreamJournal` for bounded, cross-replica SSE replay using a host-owned async
  PostgreSQL pool and an externally managed schema migration.
- Added an optional bounded SQLite journal for resumable SSE runs, allowing workers on one host to
  replay committed events across processes and service restarts without rerunning inference.
- Added the async `StreamJournal` extension contract so hosts can inject shared SSE journal
  backends; Gabby manages only the optional SQLite backend it constructs itself.
- Added a Kubernetes deployment reference for the single-tenant hosted research agent, with a
  digest-pinned image slot, external secrets, TLS ingress, pod hardening, resource bounds, and
  ingress/DNS/HTTPS network policy.
- Added optional, bounded Parquet ingestion for flat tables, with row-labeled pages and citations,
  resource limits, and no PyArrow dependency in the core installation.
- Added `PostgresGenerationManifestStore` for shared hybrid-index lease, fencing, activation, and
  retirement coordination through a host-owned asyncpg-compatible pool. Added an external SQL
  migration and an opt-in live PostgreSQL acceptance contract; PostgreSQL remains an optional extra.
- Added bounded RSS 1.0/2.0 and Atom feed ingestion as one cited page per entry, including entry
  metadata and `.xml` feed-root detection, without fetching network resources.
- Added the public async `ApprovalAuditSink` contract so hosts can connect shared audit backends to
  `AuditedApprovalHandler`; the bundled SQLite sink remains the single-host option.
- Added `PostgresApprovalAudit`, a bounded parameterized async PostgreSQL sink over a host-owned
  pool, without adding a PostgreSQL driver dependency to Gabby.
- Added `RedisSkillRevocationStore` over a host-owned async Redis client for bounded shared signer
  revocation checks and management, without adding a Redis dependency to Gabby.
- Added opt-in resumable SSE runs keyed by idempotency key, with request/principal binding,
  sequential event IDs, bounded process-local journals, and reconnect support in `GabbyClient`.
- Added bounded standalone TOML knowledge ingestion with strict parsing, structural and output
  limits, normalized dates/times, and searchable dotted key paths and list indexes.
- Added bounded `.yaml` and `.yml` knowledge ingestion with safe duplicate-key rejection, alias-cycle
  and complexity checks, JSON-compatible value validation, and searchable nested key paths.
- Added a bounded native Gemini text embedding provider, including `gemini-embedding-001` retrieval
  task types and `gemini-embedding-2`, and added it to `gabby embeddings evaluate` profiles.
- Added bounded concurrent dispatch for explicitly parallel-safe host tools, with sequential
  defaults, all-safe batch eligibility, deterministic result ordering, and a per-agent concurrency cap.
- Added a host-injected MCP client bridge that imports bounded remote tool catalogs, retains
  host-owned connection lifecycle, applies Gabby tool policy and permissions, and rejects unsupported
  remote result content.
- Added reusable async SQLite approval auditing with bounded argument hashing, optional principal
  subject retention, bounded reads, and fail-closed handling before an approved tool can proceed.
- Added bounded YAML front matter extraction for UTF-8 text ingestion, exposing page metadata for
  retrieval filters while preserving body text and allowing host metadata to override it.
- Added bounded UTF-8 iCalendar ingestion with one cited page per event, RFC 5545 line unfolding and
  text unescaping, and filterable event metadata.
- Added a native Anthropic Messages chat provider with normalized tool use, streamed text and tool
  input, environment-based credentials, bounded responses, and configurable `max_tokens`.
- Added a native Google Gemini GenerateContent provider with bounded request/response bodies,
  function-call ID and thought-signature preservation, and streaming.
- Added an optional MCP Python SDK adapter and `gabby mcp` stdio command that expose one agent as
  a stateless `run_agent` tool with per-call context, memory, metadata, bounded responses, and
  graceful agent shutdown.
- Added bounded Jupyter Notebook v4 ingestion with required structure validation, ordered cell
  citations, and execution outputs excluded from indexed knowledge.
- Added the typed `SkillRevocationStore` extension contract for durable revocation management,
  including shared-backend consistency requirements for multi-instance deployments.
- Extended `gabby evaluate` JSONL and Python cases to compare parsed structured outputs without
  exposing generated values in evaluation reports.
- Added optional JSON Schema validated final responses for data-oriented agents, exposed as parsed
  result metadata and buffered until validation succeeds when streaming.
- Added optional skill-catalog freshness timestamps and configurable age checks for registry clients
  and `gabby skill search`, `versions`, and `fetch`.
- Added per-run integrity verification for trusted installed skills, with traced success events and
  fail-closed, sanitized errors when skill contents change after agent construction.
- Added optional dependency-closure fetching for signed remote skill packages, with exact pins,
  bounded resolution, preflight validation, and safe retries of partially completed installs.
- Added `gabby skill inspect` and `inspect_skill_package()` for bounded, pre-install package review
  with declared capability and file-digest reporting plus optional trusted-key verification.
- Added `gabby skill catalog build` and a Python API for creating a static skill registry from
  signed packages, with trusted-key validation, bounded artifact generation, and safe updates seeded
  from a previously generated static registry.
- Added bounded, dependency-free `.rtf` text ingestion with Unicode escapes and safe skipping of
  hidden text, non-body destinations, and binary payloads.
- Added an opt-in read-only SQLite query tool with explicit environment-resource binding,
  parameterized queries, and row, output, cell, column, and deadline bounds.
- Added an async-first Python HTTP client for stateless run and SSE endpoints, with synchronous
  wrappers, bounded response parsing, and HTTPS enforcement for remote services.
- Added bounded `.xlsx` ingestion with workbook order, shared-string support, and sheet/row/column labels.
- Added bounded OpenDocument Spreadsheet `.ods` ingestion with sheet, row, column, and repeat limits.
- Added a host approval and SQLite audit example that stores decision metadata and an argument digest without persisting raw arguments.
- Added bounded OpenDocument Text `.odt` ingestion for ordered headings, paragraphs, and tables.
- Added bounded EPUB ingestion with OPF spine-order chapter text and one-based chapter citations.
- Added bounded PowerPoint `.pptx` ingestion with presentation-order slide text and one-based slide citations.
- Added opt-in vector-only PDF OCR using bounded in-memory PDFium rendering and the replaceable `PDFPageRenderer` contract.
- Added a stateless agent regression runner and `gabby evaluate` with bounded JSONL datasets and deterministic output/tool-use expectations.
- Added `gabby knowledge evaluate` and a Python retrieval evaluator with bounded source-level ranking metrics (precision, recall, reciprocal rank, and nDCG).
- Added a labeled query/document embedding evaluator with cosine ranking, nDCG, reciprocal rank, and pairwise accuracy through the replaceable async embedding-provider contract, including asymmetric input prefixes and per-call timeouts.
- Added `gabby embeddings evaluate` with bounded, versioned provider profiles for OpenAI-compatible, Hugging Face, and optional local Transformers embedding providers.
- Added `gabby knowledge ingest` and `gabby knowledge search` for persistent local SQLite FTS5 workflows.
- Connected knowledge-enabled `gabby run` and `gabby serve` commands to the shared local SQLite store.
- Added bounded RFC 5322 `.eml` ingestion with selected headers, MIME text extraction, attachment exclusion, and HTML fallback.
- Added a bounded Jina hosted reranker adapter with host-owned credentials and validated response mapping.
- Added a bounded Voyage hosted reranker adapter with host-owned credentials, validated response mapping, truncation disabled by default, and opt-in live acceptance.
- Align package source, issue, documentation, and security metadata with the Orbit Projects
  organization; document Gabby's shared parent-organization OSS maintenance baseline.
- Added a stateless HTTPX SSE consumer example with caller-owned context and memory, typed event
  parsing, and HTTPS enforcement for remote service URLs.
- Added stateless `AgentTool` composition and the `create_app` service factory to the proposed
  v1.0 compatibility surface, with package-root export regression coverage.
- Added optional local Transformers embeddings with bounded asynchronous batch inference, masked
  mean/CLS pooling, unit normalization, and safetensors-only model loading.
- Improved unit normalization for low-precision local model outputs by converting pooled embeddings
  to float32 before normalization; added real-model local embedding acceptance coverage.
- Hardened explicit CLI trusted-key reads against path replacement and FIFO substitution.
- Added bounded strict JSON file ingestion with duplicate-key rejection and deterministic complexity limits.
- Added bounded header-based CSV ingestion with duplicate-header and row-shape validation, labeled retrieval fields, and configurable row, column, and output limits.
- Added an optional local Hugging Face Transformers chat provider with lazy safe-weight loading,
  bounded input/output tokens, worker-thread generation, cancellation checks, and structured
  tool-call parsing for compatible model templates.

- Hardened the Nginx deployment reference to TLS 1.2/1.3, disabled TLS session tickets and version
  disclosure, and added HSTS and browser security headers; the Docker ingress acceptance check now
  verifies those headers.
- Added bounded host-managed `KEY_ID.pub` trust directories for embedded `SkillTrustPolicy` setup
  and the skill install, verify, fetch, and audit CLI workflows; documented routine rotation and
  emergency revocation operations.
- Added v2 skill signatures over archive and canonical manifest digests, persisted signature evidence,
  and trusted-key audits that detect changed installed files while retaining v1 install compatibility.
- Added an optional host-injected `SkillTrustPolicy` that rejects unsigned, legacy, untrusted,
  revoked, or tampered skills during agent construction; trust keys and revocations are snapshotted.
- Added per-run and per-stream signer revocation checks through an async host-owned contract and a
  durable SQLite store. Revoked signers fail with sanitized 403 responses; checker outages and
  deadlines fail closed with sanitized 503 responses.
- Reject cyclic, non-JSON, and oversized in-memory agent configuration before constructing an agent,
  and cap programmatic skill metadata arrays to the manifest-equivalent byte and item limits.
- Documented the proposed v1.0 compatibility surface and a minimum two-minor-release, six-month
  deprecation window; final stable interfaces and historical support policy remain release work.
- Added an optional OpenTelemetry tracer example with event spans and a privacy-focused attribute
  allowlist; SDK and exporter lifecycle remain host-owned.
- Reject skill package paths that collide after case folding or Unicode normalization, both when
  packaging source trees and before extracting untrusted archives.
- Added optional detached Ed25519 skill package signatures with host-managed trust keys, signature
  verification, and CLI flows for signing, checking, and requiring authenticated installs.
- Added opt-in, deadline-bounded retries for explicitly transient model failures, with safe
  pre-output stream retries, provider `Retry-After` handling, and trace/SSE visibility.
- Added public `RetryableModelError` for custom model providers and a validated
  `policies.max_model_retries` setting, defaulting to zero and capped at three.
- Added SemVer versions to portable skill packages and exact `skill-id@version` references for
  agent skills, dependencies, filesystem and in-memory registries, traces, and CLI inspection.
- Reject unknown agent, skill, policy, knowledge, verification, environment, and sandbox keys so
  configuration typos cannot silently fall back to defaults; provider-specific model options and
  named environment resources remain extensible.
- Validate in-memory skill registry entries with the same structural rules as file-loaded skill
  packages, including safe skill IDs, dependency references, and string-list fields.
- Sanitized built-in model-provider HTTP and transport failures for direct callers by omitting
  upstream error bodies and exception text from completion and streaming errors.
- Suppressed transport, malformed-response, and request-serialization exception chains in built-in
  embedding and reranking errors so traceback formatting cannot expose private endpoint or response
  details.
- Bounded agent YAML, skill manifests, and file-backed skill instructions/examples during reads;
  programmatic agent and skill text receives the same UTF-8 byte limits.
- Capped one agent at 256 resolved skills and bounded combined agent/global/skill text against its
  model-request policy with a 16 MiB hard ceiling, including transitive skill dependencies.
- Added a request-scoped cooperative cancellation token to `ToolContext`; timed-out and cancelled
  tool handlers that opt into the context can stop their own work promptly.
- Added a bounded Hugging Face Inference Providers feature-extraction embeddings adapter with
  batching, token-feature pooling, model options, and strict response validation.
- Verified that the Hugging Face adapter composes with durable SQLite hybrid indexing and retrieval.
- Added optional route-specific `run_scopes` and `stream_scopes` authorization to embedded FastAPI
  services, with validated bearer-token scopes and JWT `scope`/`scp` claim extraction.
- Added `OpenAICompatibleEmbeddingProvider` with bounded batches, request and streamed response
  limits, indexed vector ordering, credential validation, and HTTPS enforcement for remote endpoints.
- Added optional page-attributed PDF text ingestion with bounded page and extracted-character
  counts; install the `pdf` extra to enable the built-in pypdf parser.
- Native Windows sandboxes now verify that the Docker server is version 29.1.4 or newer before
  creating a container with networking disabled; CLI, Engine API, and injected adapter checks fail
  closed on unsupported or unverified daemons.
- Added opt-in request-scoped `ToolContext` injection for host tools with per-tool environment
  resource grants, authenticated principal, and construction-time missing-resource validation.
- Reject container image references that begin with an engine option or contain whitespace/control
  characters, preventing malformed agent configuration from changing Docker/Podman CLI flags.
- Moved the per-process run-capacity check ahead of request-body buffering and authentication so
  concurrent request bodies share the existing execution cap; `/health` remains available during
  overload.
- Made Docker/Podman CLI adapters stop an engine command as soon as combined stdout/stderr exceeds
  the shared 1 MiB capture budget; streamed readers now share that limit instead of retaining up to
  1 MiB per stream.
- Bounded direct hybrid retrieval to 100 results, limited candidate overfetch to 100 per backend,
  and rejected lexical or vector stores that return more documents than requested before fusion.
- Added a configurable total request-body receive deadline, defaulting to 30 seconds, with a
  bounded HTTP 408 response before authentication or run admission when a body stalls.
- Added `RerankingRetriever` to compose an injected reranker with lexical, hybrid, or custom
  retrievers. Candidate count is bounded and rerankers can reorder a unique subset without rewriting
  the original retrieved documents.
- Added optional issuer-bound JWT bearer authentication for embedded FastAPI deployments, with
  strict claims and algorithm validation, bounded asynchronous JWKS retrieval, and key caching.
- Timed-out or canceled synchronous callbacks are now removed logically from the bounded queue and
  skipped if a worker has not started them; already-running callbacks remain cooperative.
- Linux container runs now use non-root UID/GID `65532:65532` by default, with validated numeric
  per-agent overrides; both engine adapters apply the identity and live checks assert it.
- Added a subprocess crash-recovery case after lexical cleanup but before vector cleanup and manifest
  retirement; a fresh coordinator repeats cleanup and fences out stale writers.
- Expanded hybrid-index process-crash acceptance to cover a crash after recovery advances backend
  fences but before the manifest activates the generation; repeated recovery still activates it and
  fences out both stale writers.
- Added SSE acceptance for run-deadline cancellation, capacity release, mid-stream provider errors,
  and retries that start a fresh stateless execution.
- Explicitly closed SSE response iterators after ASGI send failures so transport resets cancel the
  active agent run and release its capacity slot.
- Added opt-in live Ollama and Hugging Face acceptance checks for stateless completion and streaming,
  with environment-based setup instructions; live provider execution remains unverified here.
- Standardized HTTP and SSE execution error types so custom provider exception class names are not
  exposed; framework `AgentRuntimeError` remains distinguishable.
- Expanded CI and tagged-release platform gates to test the Python 3.11–3.14 cross-product on
  Linux, macOS, and Windows, and made the workflow checker enforce that matrix.
- Exposed the existing bounded HTTP request-body setting as `gabby serve --max-request-bytes`,
  with the same 1,000,000-byte default and positive-integer validation as `create_app`.
- Limited streamed engine error previews to 500 bytes so failed exec/archive and image-pull responses
  are closed without buffering their full error bodies.
- Made workspace archive listing fail closed on conflicting file types, so descendant entries
  cannot cause an explicit symlink or special file to be treated as a directory during path checks.
- Closed owned container API clients when sandbox initialization fails before a run-scoped container
  is yielded, preventing repeated startup failures from leaking HTTP transports.
- Streamed Docker-compatible engine control responses into a 1 MiB bound instead of buffering
  unbounded metadata and error bodies; archive, command, and image-pull output keep their separate
  limits.
- Added a per-process HTTP response body cap, defaulting to 4 MiB, for serialized `/run` results and
  complete SSE streams including traces and keepalives; oversized runs return a bounded error and
  streams reserve space for a typed terminal size-limit event.
- Propagated client disconnect cancellation from both stateless HTTP execution routes and ensured
  canceled work releases its process-local capacity slot.
- Redacted host-trusted tool handler exception messages and class names from model observations,
  traces, and stream events while preserving Gabby-generated validation and policy details.
- Made asynchronous agent shutdown reject new runs, drain active runs and streams, then close owned
  provider resources; async runs and closure now enforce event-loop affinity.
- Added an injectable async `Tracer` that receives ordered event snapshots during runs, with bounded
  best-effort delivery and sanitized failure recording; trace durations now populate the typed field.
- Enforced the documented HTTPS requirement for remote container-engine API endpoints during agent
  and API-adapter construction, with HTTP limited to loopback and endpoint URL credentials rejected.
- Validated YAML-loaded and in-memory `AgentDefinition` values through one shared construction-time
  validation path, and exposed `validate_agent_definition` for host preflight checks.
- Added tag-triggered release verification, version-matched distribution builds, and protected
  GitHub OIDC Trusted Publishing to PyPI.
- Required HTTPS for remote built-in model endpoints while allowing HTTP for loopback local
  inference; direct adapter construction enforces the same transport boundary.
- Bounded each provider response to 4 MiB by default, configurable through
  `policies.max_model_response_bytes`; built-in JSON/SSE adapters enforce the cap while receiving
  bodies, and the runtime bounds custom-provider responses and accumulated stream output.
- Rejected requests with a declared oversized `Content-Length` before reading their bodies, then
  coalesced accepted body chunks to prevent many tiny ASGI chunks from amplifying memory use.
- Enforced the existing no-inline-credentials rule for in-memory agent definitions and direct
  built-in provider factories, rejecting model API keys, credential-bearing headers, and obvious
  credentials in provider URLs. Validate the referenced environment variable name. Use environment
  lookup or provider injection instead.
- Bound each serialized provider request to 4 MiB by default, configurable with
  `policies.max_model_request_bytes`, across the main runtime and built-in model planner/selector.
- Added an injectable optional planner with bounded structured plans, an opt-in model-backed
  implementation, policy-scoped capabilities, run-deadline enforcement, trace recording, and a
  typed SSE plan event.
- Moved text-ingestor file resolution, directory traversal, and deletion path checks off the async
  event loop.
- Bounded text directory ingestion to 10,000 files and 1 GiB by default, with configurable limits
  and actual-read enforcement in addition to the existing per-file cap.
- Added a deployment operations guide and documented the architect-approved quiescent-workspace
  requirement for mounted sandboxes.
- Added durable per-source generation coordination for hybrid lexical/vector indexes, with
  renewable writer leases, monotonic fencing tokens, a pluggable manifest contract, separate SQLite
  implementation, crash reconciliation, and explicit quiescent cleanup for retired generations.
- Added a per-tool serialized result limit, defaulting to 1 MiB, and strict JSON handling for tool arguments and results.
- Added backend-neutral `VectorStore` and `HybridRetriever` contracts with concurrent lexical/vector search and deterministic Reciprocal Rank Fusion.
- Added a typed verifier protocol and structured verification results in response metadata and traces; enabled verification rejects malformed or failed results.
- Documented the architect-approved pre-1.0 extension API stability policy and current extension-author guidance.
- Added an injectable async `SkillSelector`, deterministic configured/keyword selection by default, and an opt-in `ModelSkillSelector` with strict ID validation, run-deadline enforcement, traced usage, and dependency activation.
- Added a built-in Hugging Face Inference Providers adapter for the OpenAI-compatible chat API, with `HF_TOKEN` and router URL defaults.
- Recorded the architect-approved Hugging Face hosted-inference decision and documented its prompt-data trust boundary.
- Added shared per-process API admission control for run and stream requests, configurable from `create_app` and the CLI, with HTTP 429 on capacity exhaustion.
- Expanded live Docker and Podman acceptance to verify CPU limits and read-only/read-write workspace mounts across CLI and API adapters on Linux.
- Added an initial threat model covering trust boundaries, current enforcement, residual risks, and hosted release gates.
- Validated positive request-body and execution-capacity limits at app construction.
- Added root-confined UTF-8 Markdown/plain-text ingestion with deterministic paragraph chunks and atomic source reindexing.
- Made the domain-neutral agent scope clearer and documented research, support, and data-agent patterns alongside the initial coding tools.
- Made the bounded synchronous callback bridge poll for worker completion so embedded event loops reliably observe cross-thread results.
- Added a machine-readable weighted project scorecard and CI validation to keep category weights, evidence, and reported progress synchronized.
- Added a CI documentation policy for public package docstrings, Markdown structure and links, and unresolved work markers.
- Added an Ollama provider using the OpenAI-compatible chat completion and streaming contract, defaulting to the local `/v1` endpoint.
- Added model-delta streaming and a stateless SSE route at `POST /v1/agents/{agent_name}/stream`, with skill/tool progress, completion, sanitized errors, and disconnect cancellation.
- Added live generator-level SSE checks for keepalive comments and provider cancellation when a consumer disconnects.
- Added typed runtime stream events and a completion envelope carrying the same result and trace as the non-streaming API.
- Resolve agent definitions, skills, environment declarations, and tool registries into immutable construction-time snapshots.
- Classify tools as host-trusted or sandboxed; `policies.require_sandbox` rejects host-trusted tools before a model call.
- Gate production sandbox support claims on live acceptance evidence for each host, engine, and adapter combination.
- Version the public run route under `/v1` and document the API major-version boundary.
- Publish typed OpenAPI schemas for run requests, responses, traces, and execution errors.
- Closed agent-owned model clients and sync-loop resources when CLI commands finish.
- Added single-tenant-per-service architecture guidance and a pluggable FastAPI authentication boundary.
- Added an in-memory BM25 retriever with source metadata, exact metadata filters, and document snapshots for small corpora.
- Added a persistent SQLite FTS5 knowledge store with async ingestion, source replacement, exact metadata filters, and replaceable storage, embedding, and reranking contracts.
- Kept API credentials out of agent YAML and normalized malformed OpenAI-compatible responses to `ModelError`.
- Added best-effort sandbox cleanup by run-scoped name when Docker or Podman create/start responses are lost.
- Added live Podman CLI/API sandbox acceptance on Linux and corrected Podman host OS detection and API cleanup behavior.
- Added portable host-platform CI coverage for Linux, macOS, and Windows; live container acceptance remains pending outside Docker on Linux.
- Tightened agent and skill YAML text-field validation.
- Rejected reserved Windows device names in sandbox workspace paths.
- Isolated each agent's tool registry from caller-owned environments and registries.
- Enforced one absolute deadline across container CLI process startup, input, execution, and output collection.
- Rejected malformed environment, execution-budget, retrieval-limit, and verification values during configuration loading.
- Activated required skill dependencies whenever a dependent skill is selected.
- Fixed read/write workspace policy propagation from container configuration into the sandbox session.
- Adopted an async-first Python runtime with synchronous wrappers and FastAPI serving.
- Added Python 3.11–3.14 support metadata, a `src/` package layout, and a locked `uv` development environment.
- Added architecture records and initial contributor, security, and governance guidance.
- Rejected skill references and symlinks that resolve outside their configured registry directory.
- Added GitHub Actions CI, CodeQL analysis, dependency update automation, and issue/PR templates.
- Added model adapter, CLI, configuration, environment, policy, runtime, and HTTP contract coverage with a 90% branch-aware coverage gate.
- Made repeated synchronous runs share one managed event loop and added sync/async context-manager cleanup.
