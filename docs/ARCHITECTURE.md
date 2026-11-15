<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2026-present Gabby Contributors.
-->

# Gabby Agent Runtime and Governance Framework

**Status:** living architecture document for the pre-1.0 implementation. Accepted decisions and known limitations are recorded in the ADRs, roadmap, and [project scorecard](PROJECT_SCORECARD.md). Public extension contracts may change before v1.0; see [the extension API policy](EXTENSIONS.md).

## Product boundary

Gabby is an agent runtime and governance framework. It gives developers a way to define agent
capabilities and apply runtime governance to their execution: which tools are available,
what policies constrain their use, where approval or verification is required, and how execution is
bounded and observed. Its responsibility is to execute a configured agent request within those
contracts and return a result.

Gabby is complementary to agent orchestration frameworks such as LangChain, CrewAI, and IBM BeeAI.
An application may use one of those frameworks to coordinate agents or workflows and use Gabby to
execute an agent request under explicit tool permissions, runtime policies, approvals, verification,
and resource bounds. Gabby does not provide a competing general-purpose workflow orchestration layer
and is not intended to replace the host application's coordinator. Its boundary is the governed
execution of a configured agent request.

The runtime is stateless between requests. It does not own the consumer's conversation lifecycle,
project history, or durable user memory. A caller sends one request with whatever context it wants the
agent to see. Gabby returns a result and optional trace; request-local state is discarded when
execution ends. The host application owns caller identity, durable state, and deployment-level
isolation.

The `/v1` run and stream payloads optionally accept a typed `host_context` envelope containing user
preferences, a session summary, and retrieved snippets. It is validated and bounded as request data,
added to model context as untrusted reference material, and discarded after the run. The host
authenticates the caller and selects the corresponding data; this envelope is not an identity or
tenant boundary. See the host-context contract in the user guide.

Governance here means controls applied to an agent definition and its execution. It does not mean a
hosted organizational control plane: Gabby does not centrally manage teams, identities, or policies
across deployed applications. The host remains responsible for deciding who may call a deployment
and how separate users or tenants are isolated.

Runtime composition places user-defined character instructions in the model context and applies runtime policies to tool access and execution. Instructions guide the generated response but do not guarantee its tone or content. Gabby does not train models, modify model weights, or load model adapters.

Gabby's behavior is configured through runtime, global, agent, skill, and task instructions;
retrieved material can provide user-scoped reference context, while policy and capability checks
govern execution. The model generates the response, so instructions and retrieved context guide its
behavior but do not guarantee it. Gabby does not provide model training, weight changes, or adapter
loading. A container isolates execution; it does not by itself make agent development reproducible
or reversible.

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

An optional structured `persona` mapping contains `role`, `tone`, `communication_style`,
`escalation_rules`, and `forbidden_behaviors`. Resolution validates and freezes these values, then
compiles them to an immutable prompt section. Persona instructions guide generation but are not
executable policy. Optional post-generation persona checking uses the injected `PersonaVerifier`
when `verification.persona_compliance` is enabled; a missing verifier, verifier error, malformed
result, timeout, or failed result rejects the run. The runtime buffers output until this check passes,
including for SSE, so rejected content is not emitted as response text. This hook detects only what
the host verifier can determine and does not guarantee non-generation of policy-violating content.

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

### Host-orchestrated pipelines

For deterministic multi-stage workflows, the host can call several separate single-agent Gabby
services using the stateless HTTP client. The host owns routing, authenticated caller and tenant
resolution, per-stage authorization, retrieval, durable checkpoints, retries, aggregate budgets,
and final workflow state. Each service enforces its own route authentication, agent policy, tool
allowlist, and per-run bounds. Typed `HostContext` may carry request-scoped preferences, session
summaries, and tenant-filtered RAG snippets; it does not establish identity or tenant authorization.
The reference uses bounded structured triage, host-side scope checks, parallel specialists, and
final synthesis, but does not add a workflow engine or durable state to Gabby. See the
[host pipeline guide](HOST_ORCHESTRATED_PIPELINES.md) and [ADR 0130](architecture/adr/0130-host-orchestrated-multi-agent-pipelines.md).

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
filesystem registry layout. Package install and inspection require detached Ed25519 signature
verification against host-managed keys by default; unsigned packages require an explicit local
development override. Installed packages also require an injected `SkillTrustPolicy` before an
agent can load them by default. Installation does not execute package code. Signed remote discovery
and exact-version dependency installation are available through the static HTTPS registry client and
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

A typed operation with a stable name, JSON input schema, handler, output contract, timeout, permission metadata, execution class, and a serialized result byte limit. The default result limit is 1 MiB per tool; Gabby rejects oversized nested strings before JSON encoding and rejects oversized serialized results before content enters model context. Model output is untrusted input: Gabby validates tool identity and arguments, checks policy, then calls a registered handler. Injected handlers are `host_trusted` by default and execute with the Gabby process's host permissions. Built-in shell and filesystem operations declare `sandboxed` execution and run through the per-run container. Sandbox enforcement is on by default: any declared tool without a sandboxed implementation is rejected before model execution. Hosts can explicitly set `policies.require_sandbox: false` to accept trusted in-process handlers. `sandboxed_python_tool()` runs an importable host-registered Python handler through a fixed worker command inside the run sandbox. Docker and Podman can select an installed runtime such as gVisor `runsc`; a custom `EngineAdapter` can supply a stronger backend such as a Firecracker microVM. Gabby defines the adapter contract but does not ship or certify those backends. Python tools run concurrently only when each tool explicitly sets `parallel_safe=True` and the agent opts into `policies.max_parallel_tool_calls` (default 1, maximum 32). This declaration is for independent host handlers without approval; batches containing any other tool stay sequential. Results return to the model in original call order. Tracer-enabled runs stay sequential so trace callbacks retain their ordered contract. A Python handler is never made safe merely by setting a policy.

### Policy engine

`Agent` accepts an optional `PolicyEngineFactory`. Gabby asynchronously creates a run-scoped policy
engine after skill activation and before exposing the active tools to the model. The factory receives the
active declared tool names, resolved policy mapping, environment allowlist, and authenticated
principal. The built-in `PolicyEngine` remains the default through `DefaultPolicyEngineFactory`.
Custom factories are trusted host code; Gabby sanitizes their failures and continues to enforce tool
schemas, approval, sandbox, timeout, and resource boundaries. See
[ADR 0124](architecture/adr/0124-injectable-policy-engine.md).

### Knowledge

Retrieval is separate from skills and tools. The core includes an in-memory BM25 retriever for fixtures, SQLite FTS5 for lexical search, SQLite vector storage, and a SQLite generation manifest for local indexing. Hosts can implement the storage and retrieval contracts for other deployment needs. `FileIngestor` performs root-confined discovery and bounded reads for UTF-8 plain-text and Markdown files; specialized parsers are host-provided. The host selects embedding and reranking providers. Hybrid retrieval combines lexical and vector results with reciprocal-rank fusion, and the generation coordinator activates a consistent local generation only after its stores are ready. Retrieved text remains untrusted reference material and cannot override instructions or policies. See the relevant retrieval, vector-store, and hybrid-index ADRs in the [decision index](architecture/adr/README.md).

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
3. Agent instructions and the structured core persona.
4. Instructions and optional scoped style guidance for skills activated for this run.
5. The caller's task.

Skill style may specify voice, tone, and formatting rules for that skill's task, but the schema does
not allow a skill to redefine the agent's role, escalation rules, or forbidden behaviors. The
runtime prompt labels skill style as subordinate guidance, so an activated skill can specialize the
presentation without replacing the agent's core character. Like other model instructions, style
guidance is not deterministic enforcement; executable policies remain authoritative.

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

Container isolation for shell, filesystem, and declared custom executable tools is implemented behind Docker and Podman CLI/API adapters, with a custom `EngineAdapter` contract for host-supplied runtimes: engine, image, and workspace access come from agent configuration; missing images are pulled on demand; one temporary container or isolated runtime is created per agent run; networking is disabled; Linux containers default to non-root UID/GID `65532:65532` with a numeric per-agent override; native Windows containers require Docker Hyper-V isolation and use their image-defined user. Docker and Podman can select an installed runtime such as gVisor `runsc`; a host adapter can supply a microVM backend such as Firecracker. The Windows API adapter supports Docker Desktop's local named pipe through an optional `windows-sandbox` dependency. Custom sandboxed tools use a fixed config-owned argv and a bounded JSON request file in a per-run host directory mounted read-only; their implementation runs from the configured image. The consuming application must keep the mounted workspace quiescent for the full run because component checks cannot prevent a concurrent host process from racing separate archive operations. Adapter implementation does not imply production support on a host. Each host/engine/adapter combination is supported only after live acceptance checks cover isolation, resource limits, workspace mounts, networking, and cleanup. Current live evidence covers Docker and Podman on Linux through both adapters, including custom executable tools; other combinations remain unverified/experimental. Injected Python handlers are host-trusted by default, but sandbox enforcement rejects them unless `policies.require_sandbox: false` explicitly accepts in-process execution. `sandboxed_python_tool()` runs importable handlers inside the per-run sandbox. See [ADR 0003](architecture/adr/0003-container-sandbox.md) and the [operations guide](OPERATIONS.md).

Agent-owned engine adapters are closed even when startup fails before yielding a sandbox session; injected adapters remain host-owned.

Docker-compatible API control responses are read incrementally with a 1 MiB bound, so engine
metadata, status, and error bodies cannot trigger unbounded client buffering. Exec output, archive
transfers, image-pull progress, and CLI output have their own limits.

Sandboxed runs add a best-effort final `sandbox_resource_usage` trace event when the engine reports counters. API adapters report cumulative CPU time, current and peak memory, and current process count where available; CLI adapters report a final CPU percentage, current memory, and process count. These are a final snapshot rather than a time series or billing-grade accounting source; unsupported fields are omitted, and metric collection failure does not fail the run.

Provider interface: complete/stream with structured messages, tool calls, usage, cancellation, and timeout. Built-in OpenAI-compatible, Ollama, Hugging Face, Anthropic, and Gemini adapters require HTTPS for remote endpoints and permit HTTP only on loopback; custom providers own their transport security. Anthropic uses the native Messages API, translates system and tool-result messages, and normalizes streamed text and tool-input events. Gemini uses GenerateContent REST, translates function calling, and round-trips provider IDs and thought signatures needed by stateless multi-step calls. Mock transport tests cover provider contracts; live compatibility with Ollama, hosted providers, and individual model/provider combinations remains unverified. Tool interface: validated arguments, structured result/error, cancellation, and execution limits. Host-trusted handlers can opt into request-scoped `ToolContext`, with a construction-validated allowlist of environment resources and a cooperative cancellation token, both excluded from model arguments; see [ADR 0030](architecture/adr/0030-scoped-tool-context.md) and [ADR 0037](architecture/adr/0037-cooperative-tool-cancellation.md). Tools marked `requires_approval=True` pause at an injectable host-owned async approval handler; without a handler they are denied. The consumer owns approval UX and persistence. Approval runs within the execution deadline and emits stream and trace events. Verifiers implement the typed `Verifier` protocol and return `VerificationResult` values, which are validated and recorded in response metadata and traces; see [ADR 0010](architecture/adr/0010-typed-verifier-contract.md). An optional injected `Planner` can return a bounded structured plan that is passed as advisory context; the existing reactive tool loop remains the default and enforces all policies. `Agent` also accepts an optional async `Tracer` that receives ordered event snapshots; the existing run trace remains in the result, and exporter failures are bounded and fail-open. The host owns telemetry credentials, transport, retention, and deletion. See [ADR 0023](architecture/adr/0023-injectable-event-tracer.md). Other replaceable interfaces: environment, skill resolver/selector, tool registry, retriever, reranker, policy engine, sandbox, and storage.

Gabby does not train models, modify model weights, or load model adapters. Character instructions guide the model response; runtime policies enforce tool access and execution boundaries. The former adapter-loading decision is recorded as superseded in the [ADR index](architecture/adr/README.md).

Use explicit capability checks and stable error types. The `/v1` OpenAPI contract has typed run request, result, trace-event, and execution-error schemas. Authentication errors still use FastAPI's standard HTTP error envelope. Avoid passing arbitrary dictionaries between core modules when a public type can make the contract clear. Public extension compatibility follows the pre-1.0 policy in [ADR 0009](architecture/adr/0009-pre-1-0-extension-api-policy.md) and [the extension guide](EXTENSIONS.md).

## Deployment

- **Embedded:** Python library calling the same runtime directly.
- **Local service:** a FastAPI ASGI adapter for IDEs, CLIs, and other local consumers.
- **Hosted:** a separately operated service with authentication, tenant isolation, rate/resource limits, secret management, and durable telemetry controls.

All modes use the same resolved agent and execution contracts. Transport adapters do not own agent logic. Statelessness does not mean runs share no infrastructure: provider clients, immutable definitions, and external knowledge stores may be reused, while request messages/tool observations are run-scoped.

Streaming is a transport feature over the same event model. `POST /v1/agents/{name}/stream` emits SSE text deltas, skill and optional plan progress, tool and approval progress, completion, and sanitized error events; it does not persist run state. External Python applications can use the async-first `GabbyClient.arun()` and `.astream()` methods or their sync wrappers; the SDK does not retain caller context, automatically retry POST requests, or own conversation history. Its response limit defaults to 4 MiB, and remote service URLs must use HTTPS. See the [Python client guide](CLIENT.md) and [ADR 0079](architecture/adr/0079-stateless-python-http-client.md). The public API is rooted at `/v1`; `/health` stays unversioned for platform probes. The configured per-process capacity is acquired before request-body buffering and authentication, then held through the response. Full capacity returns HTTP 429 immediately; `/health` bypasses the gate. This bounds the number of simultaneously buffered agent requests as well as active runs and streams, but is not a request-rate or distributed limit. See [ADR 0029](architecture/adr/0029-admission-before-body-buffering.md). API authentication belongs at the service boundary: run and stream endpoints require an injected authenticator by default, the built-in bearer-token and optional issuer-bound JWT authenticators target single-tenant use, and unauthenticated local serving is explicit and loopback-only through `serve`. Embedded apps can enforce separate `run_scopes` and `stream_scopes`; their authenticated principal is passed separately to host approval callbacks and is not added to model context. See [ADR 0004](architecture/adr/0004-api-authentication.md) and [ADR 0035](architecture/adr/0035-route-scoped-api-authorization.md). The initial tenancy contract is one tenant and one configured agent per Gabby service instance; deployments isolate tenants by routing them to separately configured instances. See [ADR 0005](architecture/adr/0005-single-tenant-service.md). TLS termination, request-rate limits, and deployment routing remain service/deployment concerns. A local development server should not be presented as production hosting.

The authenticated `Principal` also reaches the default policy engine. `policies.scope_rules`
intersects verified scope grants with the agent's base tool and permission allowlists, and can add
scope-specific persona guidance without granting capabilities. Hosts may inject a
`ScopedCredentialProvider`; it resolves bounded, short-lived process environment values per
sandboxed Python or executable tool call. These values are sent only to that invocation and never
to model context or trace metadata. Per-run token and estimated USD budgets aggregate usage reported
by model calls and stop subsequent execution when the configured limit is crossed. Because providers
report usage after completing a request, one indivisible call can cross the threshold; these are
circuit breakers rather than provider billing controls.

HTTP and SSE execution failures use stable Gabby error types. `AgentRuntimeError` is preserved;
unexpected exceptions from injected providers or extensions are reported as
`AgentExecutionError` without exposing their implementation class names or messages. Capacity and
response-size rejections retain their dedicated error types.

## Observability and data handling

Execution traces are returned with each embedded result and included in `/run` and SSE completion
responses by default. HTTP callers can set `include_trace: false` to omit the trace body while
keeping the trace ID; this does not disable an injected host tracer. Text model calls pass through a
default payload filter that redacts common email, phone, and token patterns from outbound requests
and user-facing response text and blocks simple direct instruction-override attempts in user or
tool content. Plain-text stream fragments are redacted without parsing each fragment as a complete
JSON document. JSON-shaped streamed responses are buffered and filtered once complete so JSON keys
and structure are handled consistently. Plain-text patterns split across deltas may evade
redaction. Host-supplied response filters run on the complete response before text or tool arguments
are emitted. These heuristics do not guarantee PII
removal or detect every prompt injection, and do not replace data classification or provider
controls. Ordinary built-in events record metadata such as input length, context keys, skill and
tool names, argument keys, retrieval source identifiers, timing, and usage, rather than raw prompts,
context values, tool arguments, or tool results. When planning is enabled, the trace includes each
structured plan and plan revision.
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
