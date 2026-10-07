# Roadmap

This roadmap separates current implementation from work needed before Gabby can make stronger
production claims. Dates are omitted; each step depends on reviewable behavior and evidence.

## Current foundation

- Python core, async-first stateless runtime, and synchronous wrapper.
- Shared construction-time validation for YAML and programmatic `AgentDefinition` objects.
- Bounded immutable snapshots for model, environment, knowledge, policy, and verification mappings;
  complete programmatic agent configuration is capped at 10 MiB, 100,000 nodes, and 128 levels;
  programmatic skill metadata matches the 1 MiB manifest and 10,000-item bounds.
- Per-run model-step and tool-invocation budgets; `max_tool_calls` defaults to 64 and can be
  configured from 1 through 1,024, rejecting an over-budget model batch before any handler runs.
- Optional bounded parallel dispatch for complete batches of explicitly `parallel_safe` host tools;
  sequential execution remains the default, results preserve model-call order, and tracer-enabled
  runs stay sequential to preserve ordered event delivery.
- Per-tool serialized result limits reject oversized JSON results before they enter model context;
  oversized nested strings are rejected before JSON encoding to avoid a large temporary escaped
  copy.
- Opt-in transient model retries with `max_model_retries` defaulting to zero and capped at three;
  built-in providers classify transient HTTP/network failures, retry waits count against the run
  deadline, and streams never replay text already sent to a client.
- Optional construction-time global instructions with explicit runtime, global, agent, skill, and
  task precedence, applied to the main runtime and built-in model-based skill selector/planner;
  context, retrieved knowledge, and tool observations remain untrusted data.
- Closed validation for Gabby-owned YAML fields, including policy and sandbox settings; model
  provider-specific fields and named environment resources remain extension points.
- Optional JSON Schema Draft 2020-12 final response contracts for both YAML and in-memory agents;
  runtime validation returns parsed structured output and buffers stream text until the response is
  valid.
- Optional JSON Schema Draft 2020-12 input and output contracts travel with reusable skills,
  validate the stateless `{task, context, memory}` request envelope and active-skill final response,
  compose with agent output contracts, and appear in agent/package inspection. Focused config,
  runtime, package, and streaming acceptance passes; see
  [ADR 0123](docs/architecture/adr/0123-validated-skill-io-schemas.md).
- FastAPI `/v1` request/response and SSE streaming endpoints with request body limits and optional
  trace bodies on `/run` and SSE completion; response trace IDs remain available when bodies are
  omitted, and host-injected tracer exports remain independent.
- Async-first `GabbyClient` for stateless `/run` and SSE calls, sync wrappers, bounded response
  parsing, caller-supplied auth, HTTPS enforcement for remote service URLs, and opt-in idempotent
  stream replay using a bounded process-local event journal, an optional shared SQLite journal for
  same-host workers and process restarts, or `PostgresStreamJournal` for multi-replica deployments,
  with a host-injected `StreamJournal` protocol for custom shared backends.
- Optional MCP Python SDK adapter with a `run_agent` stdio tool and `gabby mcp` command. Each tool
  call passes caller-owned context, memory, and metadata to one stateless agent run; the adapter
  enforces the configured response byte limit and closes the agent when the MCP process exits.
- Host-injected MCP client bridge imports bounded remote tool catalogs into a `ToolRegistry`; the
  embedding application retains client and transport lifecycle, while Gabby applies permission,
  policy, timeout, and result-bound checks. Only JSON and text results are accepted.
- Opt-in `sqlite_query_tool()` for data environments, resolving a host-selected database resource,
  enforcing read-only SQLite authorization, and bounding rows, values, output, parameters, and query
  duration; opt-in `python_run_tool()` executes bounded model-supplied Python source using a
  configured interpreter inside the per-run OS container, with the same network, workspace,
  resource, timeout, and output boundaries.
- Per-process HTTP response body limit, defaulting to 4 MiB, across `/run` JSON and complete SSE
  streams including optional traces and keepalives; size is configurable through `create_app` and CLI.
- Bounded request-body buffering that coalesces accepted ASGI chunks, and provider credentials
  restricted to environment lookup or host-injected providers; request reads also have a bounded
  total receive deadline before authentication and run admission.
- Per-call model request and response byte caps, including incremental response limits in built-in
  JSON and SSE adapters.
- Retrieval requests have a 100-document ceiling, validate custom retriever results, reject
  over-returning hybrid backends before rank fusion, and enforce a configurable rendered-context
  byte cap before prompt construction.
- Built-in model provider URLs require HTTPS remotely; HTTP is restricted to loopback local
  inference, with custom transport behavior available through injected providers.
- Built-in model-provider errors omit upstream HTTP bodies and transport exception text so direct
  callers do not receive prompt echoes or endpoint details through exceptions.
- Shared per-process request/run/stream admission control, acquired before request-body buffering,
  with configurable capacity and an immediate HTTP 429 overload response; `/health` bypasses it.
- Optional run-scoped `PolicyEngineFactory` and typed `PolicyEngineProtocol` receive active tool
  grants, resolved policies, environment allowlists, and the authenticated principal; the existing
  fail-closed engine remains the default, and custom denial/failure details are sanitized. See
  [ADR 0124](docs/architecture/adr/0124-injectable-policy-engine.md).
- Graceful async agent shutdown that rejects new work, drains active runs/streams, and closes owned
  provider resources afterward.
- OpenAI-compatible, Ollama, Hugging Face Inference Providers, native Anthropic Messages and Gemini
  GenerateContent, optional local Transformers chat adapters, YAML agent and skill definitions,
  typed tool schemas, and runtime tool allowlists.
- The optional local Transformers provider can load an existing PEFT adapter for inference using
  separate base-model and adapter revisions; safetensors adapter weights are required. Inference
  adapter loading remains separate from the local training command.
- `gabby train` provides optional local LoRA SFT from bounded JSONL conversations, provides a
  dependency-free dataset check, uses chat-template assistant masks for assistant-only loss, stores
  adapters as safetensors, and records a dataset digest and finite metrics without retaining sample
  text; an opt-in offline tiny-model train/load/Agent.arun acceptance passed on CPU with
  PyTorch 2.14.1, Transformers 5.18.0, PEFT 0.21.2, and Accelerate 1.15.0. Training-set evaluation, QLoRA,
  distributed training, hyperparameter search, and publishing remain outside the initial capability.
- Docker and Podman Linux launchers enforce a numeric non-root `sandbox.user`, defaulting to
  `65532:65532`; the CLI and API adapter contracts assert the configured identity.
- Registered custom sandboxed tools run a config-owned executable inside the per-run container,
  receive bounded JSON through a private read-only host bind mount, and return bounded validated
  JSON; Docker and Podman live acceptance covers both adapters on Linux.
- Explicit stateless agent composition through `AgentTool`, with scoped task forwarding,
  independent child policies, principal forwarding opt-in, timeout/byte bounds, and cycle/depth
  protection verified for direct and indirect delegation cycles.
- Skill dependency resolution, injectable async skill selection with deterministic
  configured/keyword behavior by default and an opt-in model-driven selector, an opt-in injectable
  planner with bounded structured plans, exact-version portable skill packages with deterministic
  local packing, checksums, optional detached Ed25519 publisher signatures with host-managed trust
  keys, bounded installation, and CLI commands,
  retriever/verifier extension points, opt-in bounded planner revisions from recent tool
  observations, and execution traces.
- `gabby skill inspect` and `inspect_skill_package()` validate bounded package contents and report
  declared tools, knowledge, dependencies, verification labels, archive and file digests, with
  optional host-trusted signature verification and no persistent installation.
- Remote exact-version skill fetch can resolve and validate a bounded signed dependency closure,
  then install dependencies before the requested skill; exact pins and retry-safe per-package
  installation preserve the same local version-resolution contract.
- Optional host-injected `SkillTrustPolicy` snapshots trusted keys and revoked IDs, loads bounded
  host-managed `KEY_ID.pub` directories, and rejects
  unsigned, legacy, untrusted, revoked, or tampered resolved filesystem skills during agent
  construction, before model-provider construction. It also revalidates every installed skill
  before each trusted run and stream, records successful checks in traces, and fails closed with a
  sanitized integrity error if a package changes after construction. No policy preserves
  local-development loading.
- Optional async `SkillRevocationChecker` checks trusted signer IDs before every run and stream,
  within the run deadline and before sandbox/model/tool work. `SQLiteSkillRevocationStore` persists
  single-host revocations and passes concurrent independent-process writer acceptance followed by
  shared-store read/reinstate checks;
  the public `SkillRevocationStore` protocol defines management operations for host-owned shared
  backends, which must provide their own durability and propagation guarantees. The optional
  `RedisSkillRevocationStore` adapter uses a host-owned Redis 6.2+ async client without adding a
  Redis dependency to core; its host configures durability, routing, and read-after-write behavior;
  revoked signers receive HTTP 403 and store failure or timeout fails closed with HTTP 503. Active
  runs poll at a configurable interval and cancel on revocation or checker failure; completed tool
  side effects cannot be rolled back, and uncooperative synchronous callbacks may continue.
- File-loaded and programmatic skill definitions share structural validation before dependency
  resolution and immutable plan construction. The async static HTTPS registry client supports
  bounded catalog search, exact SemVer discovery, host-authenticated headers, and signature-required
  downloads. `gabby skill catalog build` and `build_static_skill_registry()` create or update the
  same bounded static layout from trusted, signed packages, validating staged artifacts and any
  cataloged versions from a prior generated registry. Catalogs include an optional generation
  timestamp; consumers can configure a maximum age, while hosting freshness policy and multi-instance
  trust-key distribution remain operator responsibilities.
  New v2 skill signatures bind archive and canonical manifest digests; installs persist the signature
  evidence, and `gabby skill audit --trusted-key KEY_ID=PATH` rechecks installed content, signatures,
  revoked keys, and missing trust. `SkillTrustPolicy` enforces this at construction and before every
  run; dynamic signer revocations are checked on each execution. Local provenance remains mutable.
- Agent and skill config reads are bounded; construction limits each agent to 256 resolved skills
  and caps combined instruction/description/example text by its model-request policy and a 16 MiB
  hard ceiling.
- Request-scoped `ToolContext` includes a cooperative cancellation token; sync handlers can wait on
  it from their callback worker, while async handlers receive cancellation and can inspect the token
  during cleanup. Uncooperative host code remains non-interruptible.
- Injectable event-by-event async tracing with bounded, fail-open exporter callbacks and the
  existing per-run trace retained in execution results; sandboxed runs include best-effort final
  CPU, memory, and process counters when reported by the configured engine adapter.
- Source-attributed documents, an in-memory BM25 retriever, and a persistent SQLite FTS5 store with
  async ingestion, exact metadata filters, and source-scoped replacement.
- Host-pooled `PostgresKnowledgeStore` for shared lexical ingestion, transactional source replacement,
  exact JSONB metadata filtering, bounded full-text retrieval, generation staging, active-generation
  filtering, and source fencing; schema is applied by the host. Live knowledge-store acceptance
  passed against PostgreSQL 18; production corpus and latency benchmarks remain open.
- Host-pooled `PostgresVectorStore` over pgvector with generation staging, durable dimension checks,
  fenced writes, active-generation filtering, and HNSW cosine search with strict-order iterative
  scans for filtered retrieval. A reusable live acceptance job uses the versioned pgvector 0.8.6
  PostgreSQL 17 image; its hosted run and production recall/latency benchmarks remain open.
- `gabby init` creates generic, research, data-analysis, or customer-support agent YAML starters
  without overwriting existing files; provider, model, and credential environment settings are
  configurable. The support starter distinguishes policy facts from tool-confirmed actions.
- `gabby skill init` creates a validated, packable skill source directory with a manifest,
  instruction file, and examples file; tool, knowledge, verification, and dependency lists start empty.
- `gabby knowledge ingest`, `search`, and `delete` expose bounded directory indexing, persistent
  SQLite FTS5 search, and root-confined source cleanup as machine-readable CLI workflows; `run` and
  `serve` inject the same configured database into knowledge-enabled agents.
- `gabby knowledge evaluate` measures source-level precision, recall, reciprocal rank, and nDCG
  over bounded JSONL query datasets, with a replaceable async retriever contract and Python API.
- `evaluate_embeddings` compares provider cosine rankings with bounded, graded query/document
  judgments and reports nDCG, reciprocal rank, and pairwise accuracy through the async provider
  contract, with asymmetric query/document prefixes and bounded per-call timeouts. The
  `gabby embeddings evaluate` CLI supports versioned JSON provider profiles for OpenAI-compatible,
  Gemini, Hugging Face, and optional local Transformers providers.
- `gabby evaluate` runs bounded JSONL regression suites with fresh per-case state, deterministic
  text, parsed structured-output, and tool-use expectations, machine-readable metrics, and CI
  failure status; reports never include generated output values.
- Root-confined UTF-8 Markdown/plain-text/`.log` ingestion with bounded YAML front matter exposed as filterable page
  metadata, bounded AsciiDoc and reStructuredText source ingestion, standalone YAML and TOML data
  ingestion, bounded RFC 5322 email/MIME and mbox archive, iCalendar event, task, and journal records, and vCard contact ingestion,
  bounded RSS 1.0/2.0 and Atom entry ingestion with `.xml` root detection, OPML 1.0/1.1/2.0
  outline ingestion with `.xml` root detection, JSON Feed 1.0/1.1 item ingestion with ordinary
  JSON source preservation, HTML, strict JSON and JSON Lines, bounded Jupyter Notebook
  v4 cell ingestion, bounded XML, CSV, RTF, DOCX, PPTX, ODP, EPUB, ODT, ODS, and XLSX ingestion plus
  optional PDF text extraction and scanned/vector-only PDF OCR with
  deterministic paragraph chunks, page citations, 4 MiB per-stream and 32 MiB aggregate page-content
  decoded-output limits, and atomic per-source reindexing.
- Optional bounded Parquet ingestion for flat tables through PyArrow, with row-range citations and
  caps on compressed input, rows, columns, row groups, uncompressed metadata size, cells, pages, and
  rendered output. Nested and binary columns are rejected.
- Replaceable `KnowledgeStore`, `EmbeddingProvider`, `VectorStore`, and `Reranker` interfaces; built-in
  SQLite implementations provide lexical FTS5 retrieval and exact-cosine vector search for small or
  moderate local indexes, with hybrid rank fusion, bounded OpenAI-compatible, native Gemini, and Hugging Face
  feature-extraction embedding adapters, an optional local Transformers embedding provider with
  masked pooling and safetensors-only loading, host-provided second-stage reranking, bounded Cohere
  v2, Jina, Voyage, and NVIDIA NeMo hosted reranking adapters, and opt-in Cohere, Voyage, and
  NVIDIA live contract acceptance.
- Durable hybrid index generation coordination through a pluggable manifest contract and separate
  SQLite and PostgreSQL implementations, including server-clock leases, fencing, crash
  reconciliation, and operator-controlled retired cleanup. PostgreSQL uses a host-owned pool and
  externally applied migration; multi-host deployments still need shared generation-aware data
  backends.
- Pluggable FastAPI request authentication with bearer-token and optional issuer-bound JWT
  implementations; the built-in JWT verifier has a fixed asymmetric algorithm allowlist, required
  validity/identity claims, bounded asynchronous JWKS retrieval, cached keys, and strict scope claims;
  optional exact issuer-scope-to-capability mapping drops unmapped claims, and embedded services can
  inject a bounded per-request JWT revocation checker; custom authenticator calls also have a
  configurable five-second default timeout that returns a sanitized 503 and releases request
  capacity; a durable SQLite revocation implementation supports
  issuer-scoped revocation and explicit expiry cleanup. `gabby serve` can grant its shared
  bearer token capabilities and enforce separate run/stream requirements.
- Host-injected per-tool human approval with deny-by-default behavior, run-deadline enforcement,
  trace/SSE progress, and authenticated principal propagation outside model context.
- Optional async `ApprovalAuditSink` supports host-owned audit backends;
  `AuditedApprovalHandler` composes it with the review callback. SQLite provides local storage, and
  `PostgresApprovalAudit` uses a host-owned async pool for shared storage. Both record bounded
  canonical argument digests without raw arguments; principal subject storage is opt-in.
- Optional immutable OCI image digest enforcement for deployment configurations.
- Detached signature verification opens sidecars as bounded regular files, checks the opened file
  identity against pre-open metadata, rejects symlinks where supported, and caps bytes read even if
  the path is replaced concurrently.
- A multi-stage, lockfile-based Docker image and a single-tenant Compose reference for the hosted
  research agent; the example binds to loopback, runs non-root, and applies filesystem, capability,
  CPU, memory, process, and temporary-storage limits. A reusable workflow checks that exact Compose
  service on pull requests, main pushes, and tagged releases. Local Compose 5.5.1 acceptance passed;
  hosted CI execution is pending.
- A single-replica Kubernetes research-agent reference with a digest-pinned image slot, external
  Secrets, ClusterIP service, TLS NGINX ingress, hardened pod context, bounded resources, and
  DNS/HTTPS NetworkPolicy. Manifest contract checks pass; cluster/CNI acceptance remains pending.
- Native Windows Docker resource requests use whole-number `CpuCount`; process-limit opt-out is
  explicit and rejected for Linux images. Native Windows launches require Docker 29.1.4+ for the
  disabled-network mode. Live Windows enforcement remains to be verified.
- One tenant and one configured agent per service instance, with deployment-level tenant separation
  recorded in [ADR 0005](docs/architecture/adr/0005-single-tenant-service.md).
- Apache-2.0 license, Python 3.11–3.14 metadata, `src/` layout, and `uv.lock`.
- Tag-triggered release workflow with cross-platform and quality verification, version-matched
  wheel/source builds, signed artifact provenance, a lock-derived CycloneDX runtime SBOM attested
  against both distributions, and a protected PyPI Trusted Publishing job; GitHub/PyPI trust
  configuration remains an administrator setup step.

## Next engineering work

1. Complete the public-contract acceptance matrix across supported Python and host platforms.
   CI and release workflows now run the Python 3.11–3.14 cross-product on Linux, macOS, and
   Windows, and run live Docker/Podman sandbox acceptance on Ubuntu 24.04 for pull requests and
   tagged releases. Current-tree coverage-enabled full suites passed 1,765 tests and skipped 37
   opt-in cases on Linux/Python 3.11.14 and 3.12.14; branch-aware coverage measured 88.42% and
   88.43%, respectively, below the configured 90%
   gate. Focused evaluation and retrieval-evaluation tests pass 79 cases. The latest Linux
   Docker/Podman acceptance run passed all 14 cases across both adapters and workspace modes
   (8 Docker and 6 Podman cases). Four independent processes also appended 80 SQLite journal
   events without loss, and a terminated worker's partial stream was recovered with a terminal
   error. First hosted workflow execution, macOS/Windows live results, native Windows resource
   enforcement, and less common cancellation races remain. Six live PostgreSQL 18 acceptance cases
   passed for the generation manifest, lexical knowledge store, and stream journal. pgvector live
   acceptance remains open because the local PostgreSQL installation lacks pgvector; the hosted
   PostgreSQL workflow now includes the lexical knowledge suite.
   The previous four-version run passed on Python 3.11–3.14 and met the 90% coverage gate, but
   predates the current tree and does not replace current-tree platform acceptance.
2. Run the opt-in Cohere, Jina, Voyage, NVIDIA, Hugging Face, and Ollama provider acceptance checks documented in
   `tests/integration/README.md`; pinned SmolLM2-135M-Instruct CPU completion and embedding smokes
   passed with Transformers and PyTorch installed. A provider-level structured tool-call acceptance
   also passed on pinned Qwen2.5-Coder-0.5B-Instruct. This verifies one model-specific parser path;
   validate additional model templates, full multi-step behavior, and supported accelerators.
   Continue SSE transport
   acceptance for additional network
   failure modes. Mid-stream provider errors, run deadlines, client retries as fresh stateless
   executions, and ASGI send failures now have route-level tests, including cancellation and
   capacity release. The live provider checks are implemented but have not yet been run against
   real endpoints. Planner success, failure, and deadline behavior now have runtime tests.
3. Review and validate the initial threat model and deployment controls for the accepted
   one-tenant-per-instance model; validate the Docker/Podman CLI and API sandbox adapters against
   live engines across Linux, macOS, and Windows. Verify per-run cleanup, resource controls, workspace
   mounts, and disabled networking on each supported engine/host pair. Verify native Windows Docker
   CPU and process-limit behavior, and the Docker 29.1.4 network-isolation floor, against a live
   Windows daemon before claiming enforcement.
4. The v1 core already has broad bounded document ingestion plus a local Transformers reranker and
   hosted Cohere, Jina, Voyage, and NVIDIA rerankers. Additional formats and providers are
   use-case-driven extensions; there is no finite additional parser/provider set currently scoped
   as a v1 requirement. Remaining work in this area is acceptance and release evidence: expand the
   coordinated-index crash matrix beyond partial-cleanup recovery and tested retired-generation
   cleanup boundaries; run live backend and hosted-service provider acceptance; validate
   multi-instance publisher trust-key distribution; and confirm behavioral contracts and tested
   optional provider/platform support in the v1.0 release notes. Further host-specific approval UX,
   audit backends beyond PostgreSQL, and deployment-specific ingress examples are optional
   integrations rather than missing core interfaces. The v1.0 core Python export list and release
   support window are recorded in `docs/COMPATIBILITY.md` and ADR 0064.
5. The operations guide and `deploy/nginx.conf` provide an Nginx reference for edge request-rate
   limits, exact body bounds, TLS 1.2/1.3, security headers, SSE buffering, and timeouts. A
   repeatable Docker-backed ingress check now verifies health forwarding, security headers, 413 body
   rejection, and burst 429 responses against the hosted agent; local execution passed. The reusable
   container gate runs this check in CI and on tagged releases, with its first hosted execution
   pending. Deployment-specific TLS certificates, trusted-proxy configuration, external Trusted
   Publisher setup, and runbooks still need validation before claiming hosted production readiness.

## Explicit limits

Gabby does not own conversation history or durable user memory. Built-in shell and filesystem tools
use per-run container adapters; live acceptance currently covers Docker and Podman through CLI and
API adapters on Linux. Native Windows Docker CLI/API acceptance tests are available, but that host
combination and macOS remain unverified. Arbitrary
registered Python handlers still run in process unless their
owner provides isolation. Each service instance is single-tenant; hosted deployments must enforce
separate instances and resource handles. The consuming application must keep the mounted workspace
quiescent during each run because component checks cannot eliminate races with concurrent host
mutations. OIDC issuer discovery is available; distributed JWT revocation-store lifecycle and consistency need host integration. Embedded services can
enforce route-specific JWT or bearer capabilities, with optional exact issuer-scope mapping in the
JWT authenticator. The CLI uses one shared bearer token, so it cannot assign different capabilities
to different callers. The run-capacity limit is per process and does not rate-limit callers or coordinate
across workers. `FileIngestor` handles UTF-8 Markdown/plain text/`.log` files, AsciiDoc, reStructuredText, YAML, TOML, bounded RFC 5322 email/MIME and mbox, iCalendar, and vCard,
title- and structure-aware HTML,
strict JSON including JSON Feed 1.0/1.1 and JSON Lines, bounded XML including OPML root dispatch,
OPML outlines, CSV, RTF, bounded DOCX, PPTX,
ODP, EPUB, ODT, ODS, and XLSX in core, plus optional page-attributed
PDF text with the `pdf` extra and bounded raster
image OCR with the `ocr` extra and host-installed Tesseract, plus opt-in OCR for scanned and vector-only PDF pages through PDFium and the replaceable `PDFPageRenderer` contract. Other document formats remain future work. The built-in OpenAI-compatible, Hugging Face, and optional local Transformers
embedding adapters do not bundle model weights; representative labeled benchmark results and
additional hosted reranking providers remain future work. SQLite vector exact search is intended for
small and moderate local indexes. `PostgresVectorStore` provides a shared pgvector/HNSW option up to
2,000 dimensions; live server acceptance and corpus-specific recall and latency evidence remain open.
Opt-in live completion/stream acceptance exists for Ollama and Hugging Face but has not yet been run
against real endpoints. The local Transformers provider has deterministic fake-backend contract
coverage, real CPU completion, structured tool-call provider, and embedding smokes using pinned
checkpoint weights. Multi-step model behavior, semantic embedding quality, throughput, and
accelerator behavior remain unverified. Cross-process index recovery now covers lease contention and
child-process crashes during staging, partial cleanup, before activation, after backend fence
advancement, and after each retired-generation backend cleanup boundary before manifest removal;
broader restart cases remain future work. Gabby
does not offer a hosted multi-tenant service. The `/stream`
route supports SSE text, skill, optional planning, tool, and approval progress, final results,
and sanitized errors; additional network failure and deployment acceptance remain. A local
FastAPI app is not proof of safe production hosting.
