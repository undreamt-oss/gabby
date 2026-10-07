# Gabby Threat Model

**Status:** initial project threat model; review required before a hosted production claim.  
**Scope:** Gabby core runtime, built-in container-backed shell/filesystem tools, FastAPI service, and
their direct dependencies. The consuming application and deployment add their own assets and
threats.

## Security objectives

Gabby should execute each request with isolated transient state, enforce tool permissions outside
the model, constrain built-in shell and filesystem actions to the configured run environment, bound
common compute and request resources, and avoid exposing provider or tool secrets in API errors.
These controls reduce risk; they do not make arbitrary model output or extension code trustworthy.

## Assets

- Host files, mounted workspaces, knowledge databases, and any data returned by application tools.
- Provider credentials, API bearer tokens, and secrets held by the consuming application.
- The host process, container engine, operating-system kernel, and deployment identity.
- Agent definitions, skills, tool implementations, container images, and Python dependencies.
- Service availability, execution budgets, and any traces or telemetry retained by the deployment.

## Trust boundaries

1. **Caller to service:** requests, supplied context, identity credentials, and request volume are
   untrusted. The HTTP API authenticates execution routes, validates request shape, limits request
   body size and total receive time, coalesces accepted body chunks into one bounded replay buffer,
   and acquires the per-process request/run capacity before reading agent requests. This bounds
   simultaneously buffered agent bodies; `/health` bypasses admission. The `serve` command restricts unauthenticated
   use to loopback; an embedded ASGI deployment that opts out of authentication must enforce its
   own network boundary. The deployment still owns TLS, request-rate limits, ingress controls, and
   distributed capacity management.
2. **Model to runtime:** model text, tool names, arguments, and claimed verification results are
   untrusted. Gabby validates tool identity, arguments, permissions, policy, budgets, and declared
   output schemas before or after the corresponding operation. Instructions and retrieved text are
   not enforcement mechanisms.
3. **Runtime to model, embedding, or vector provider:** request messages and retrieval inputs may
   leave the Gabby process. Hosted model calls can include user input, caller context, selected skill
   instructions, retrieved documents, tool schemas, and tool observations. An embedding provider may
   receive source document text during indexing and task queries during retrieval. A remote vector
   store may retain document text, metadata, and embeddings. Gabby does not control external
   provider retention, routing, or residency. Use only provider and data-handling arrangements
   approved for the data and workload.
4. **Knowledge and environment data to model:** retrieved documents, workspace files, tool output,
   and caller context can contain prompt injection or false information. They are reference data;
   their contents must not change runtime policy or instruction priority.
5. **Gabby process to extension code:** injected Python tools, providers, retrievers, parsers,
   verifiers, and authenticators execute in the host process unless their owner supplies an isolated implementation.
   They are trusted code. `policies.require_sandbox` rejects host-trusted tools but does not move
   Python code into a container.
6. **Runtime to container engine:** the Docker or Podman daemon controls the operating-system
   boundary. Access to its socket or API is highly privileged. A compromised engine, daemon
   credential, kernel, or hypervisor can defeat container isolation.
7. **Container to workspace:** the configured directory is deliberately exposed to the agent. A
   read-only mount permits reads; a read-write mount permits changes to that host directory. The
   mount does not isolate files from the consuming application that owns the directory.
8. **Service to deployment:** Gabby serves one configured agent and one tenant per service instance.
   The deployment must route tenants to separate instances and keep their credentials, storage,
   traces, and sandbox configuration separate.

## Main threats and current controls

| Threat | Current controls | Residual risk |
|---|---|---|
| Prompt injection asks the agent to use an unintended tool | Explicit tool grants, allowlists, permission checks, schemas, step limits, and runtime-enforced policy | A model may still be manipulated into a harmful action that is permitted by the agent configuration. Review the granted capabilities and verify high-impact outcomes. |
| An agent performs a high-impact action without human review | Tools can require a host approval decision after arguments are validated; missing handlers, invalid decisions, errors, and deadline expiry deny execution; authenticated HTTP principal is passed separately to the approval handler | The application owns reviewer identity, approval UX, and audit retention. Embedded callers must provide only a host-authenticated principal; an approval handler should not treat model context or user metadata as identity. |
| Sensitive prompts, source documents, or retrieval queries leave the application through hosted model, embedding, or vector services | Built-in model and embedding providers require HTTPS for remote endpoints and allow HTTP only on loopback; API credentials stay in host-managed environment variables; the host selects providers and stores | Custom providers and stores own transport security. Gabby cannot enforce provider retention, data residency, or downstream routing. The deployment must approve each model, embedding provider, and vector store for the data. |
| Provider errors expose prompts, credentials, or private upstream details | Built-in model, embedding, and reranking adapters omit HTTP error bodies and suppress HTTP transport and response-parsing exception chains from direct errors; traces record stable error categories, and HTTP/SSE routes return bounded sanitized errors | Custom providers own their exception messages and causes. Direct callers should avoid logging custom-provider exceptions without reviewing the provider contract. |
| Malicious code or a hostile file attempts to access the host | Built-in shell/filesystem tools and the opt-in `python_run_tool()` use per-run containers; Python source is staged through a private read-only mount; network is disabled; Linux containers default to non-root UID/GID `65532:65532` with numeric non-root overrides, read-only root filesystems, dropped capabilities, `no-new-privileges`, and a bounded `/tmp`; image references beginning with an engine option or containing whitespace/control characters are rejected; Windows containers request Hyper-V isolation and Gabby requires Docker Engine 29.1.4 or newer before launch; remote engine API URLs require HTTPS and local HTTP is loopback-only; deployments can require an immutable image digest | Operators can choose a different non-root numeric identity and must ensure writable workspaces grant that identity access. Digest pinning is opt-in and operators must enable it and review the selected digest. Windows containers inherit their image-defined user. Native Windows uses whole-number CPU counts and requires an explicit process-limit opt-out because Docker does not document that control for Windows; live enforcement remains unverified. A configured image and its contents remain trusted inputs. Container escape, daemon compromise, kernel flaws, and unsafe engine configuration remain possible. Linux containers share the host kernel. |
| A custom domain tool executes outside its intended boundary or receives forged arguments | `tool.execute` uses a fixed agent-configured argv inside the run container, passes arguments as bounded JSON in a random per-call file on a read-only per-run mount, applies the run deadline and output limit, validates declared output schemas, and removes each request file after use. Linux live checks cover Docker and Podman CLI/API adapters. | The configured image, executable, and tool implementation are trusted inputs. The tool can read every argument passed to it and any workspace content exposed to the run. Other host/engine combinations have not been live-validated. |
| A path escapes the configured workspace | Built-in filesystem operations validate paths, reject symlink components, and fail closed when archive entries conflict about a component's type; the consuming application must keep the workspace quiescent for the full run; live checks verify configured workspace mount modes | Gabby cannot enforce that host-side precondition. If another process mutates the workspace during a run, separate archive path checks and operations may race. A read-write mount grants write access to its entire configured directory. |
| A crafted skill archive writes outside its registry, installs corrupted resources, or impersonates a publisher | `.gabskill` installs reject traversal, absolute and non-portable paths, case-folded or Unicode-normalized path collisions, symlinks, special files, duplicate entries, unsupported compression, oversized archives and files, manifest mismatches, and invalid checksums; optional v2 detached Ed25519 signatures bind the exact archive and canonical file manifest to an explicitly host-trusted key ID before extraction. Content is validated as a skill and renamed into the registry without overwriting an existing version. `gabby skill audit --trusted-key KEY_ID=PATH` verifies signed installed content and flags revoked, unknown, or invalid records. Host applications can inject `SkillTrustPolicy` so `Agent` construction rejects unsigned, legacy, untrusted, locally revoked, or changed resolved skills, and a `SkillRevocationChecker` for fresh checks before each run and stream; active executions poll for revocation and cancel at the next check; the built-in SQLite store persists signer revocations. | Signatures are optional for local installs; deployments handling shared or remote packages must provision trusted public keys out of band and inject the policy. V1 signatures provide install-time archive verification only; v2 signatures allow audit and policy-gated construction to detect changes to installed content. A valid signature establishes publisher key possession, not safety of the skill instructions, which remain untrusted prompt content and can influence a model. Review skill behavior and grant only intended tools and knowledge access. Local provenance can be edited or removed by a registry writer, trusted public-key material remains a construction-time snapshot; polling means revocation detection is delayed by up to the configured interval plus checker latency, completed tool side effects cannot be rolled back, and uncooperative synchronous host callbacks may continue after the request ends. Removing a version does not invalidate an already-constructed `Agent`. Keep skill files stable during construction and reconstruct agents after changing trusted key material. Registry directories, trust-key configuration, and package source trees are host-managed. |
| A malicious or compromised skill registry redirects clients, leaks credentials, or substitutes a package | The async client requires HTTPS remotely, rejects redirects and credential-bearing registry URLs, derives artifact paths from the configured origin rather than catalog URLs, bounds catalog/package/signature responses and request deadlines, requires an exact version, verifies a publisher signature from a host-managed key, and checks signed package identity before final install | Catalog metadata is untrusted and may hide or replay older signed versions. Signing keys establish authority only to the extent that the host trust map is correct; provisioning, rotation, revocation, and registry write authorization remain operator responsibilities. A correctly signed skill may still contain unsafe instructions. |
| Tool code accesses credentials or performs unrelated host actions | Python handlers are labeled `host_trusted`; `require_sandbox` fails closed for those handlers; `python_run_tool()` executes model-supplied code only inside the configured OS container; provider credentials are configured outside agent YAML | The host application can register unsafe handlers or pass secrets to them. Gabby does not sandbox arbitrary Python callbacks. Sandboxed Python can still use everything exposed by its image, mounted workspace, and kernel. |
| Agent execution consumes excessive resources | Agent YAML is limited to 10 MiB, skill manifests to 1 MiB, each skill text resource to 10 MiB with bounded reads, and one agent to 256 resolved skills with combined agent/global/skill text limited to its model-request budget capped at 16 MiB; run deadlines, step limits, tool timeouts, per-tool serialized result caps, a 100-document retrieval ceiling and 1 MiB default retrieval-context cap, a shared 1 MiB sandbox stdout/stderr cap that stops CLI engine output early, per-process API concurrency cap, container CPU/memory/process limits, PDF input-byte, page-count, extracted-character, per-stream and aggregate page-content limits, and a 65,536-dimension ceiling on stored vectors | Retrieval result checks happen after the custom retriever returns, so they do not bound its internal allocations or work before return. Tool result caps likewise apply after a trusted host handler returns. The built-in PDF parser limits decoded output from supported content filters to 4 MiB per stream and 32 MiB aggregate page content per PDF by default, but page extraction and library internals can still consume significant CPU or total memory; custom parsers are trusted in-process code. Exact SQLite vector search scans all eligible records and costs O(records × dimensions), so hosts must bound corpus size or move large indexes to a specialized backend. The host-trusted callback pool has bounded workers and queue, and skips timed-out callbacks that have not started; already-running synchronous tool handlers can observe a `ToolContext` cancellation token when they opt in, but handlers that ignore it cannot be forcibly stopped. Disk usage in a read-write workspace is not quota-limited. |
| Unauthenticated or excessive API requests exhaust the service | Run/stream endpoints require an authenticator unless local unauthenticated mode is explicitly selected; optional JWT auth verifies a fixed asymmetric algorithm, issuer, audience, expiration, and subject from a configured HTTPS JWKS endpoint or bounded OIDC discovery with exact issuer matching; custom authenticator calls have a configurable five-second default timeout and fail with sanitized HTTP 503 on error or timeout; optional issuer-scope mapping translates claims to Gabby capabilities and drops unmapped claims; an injected revocation checker can require `jti` and check it on every request, returning unavailable on backend failure; the built-in SQLite checker persists issuer-scoped revocations and exposes expired-record cleanup; route requirements are checked against the principal before execution; the per-process slot is acquired before buffering a request body, which is then bounded by bytes and total receive time; active run/stream and body admission return HTTP 429 when full; `/health` bypasses admission | Scope mappings and route requirements remain host-configured. Custom authenticators must clean up resources when cancellation propagates. Revocation propagation across distributed stores remains host-owned; requests racing with revocation can pass if the check happens first. Authentication remains single-tenant. Per-process admission does not limit concurrent connections. The host owns request-rate and connection limiting, distributed quotas, TLS, and process supervision. `/health` is public. |
| Container image or package is malicious or changes after review | Agent definitions select images; `require_image_digest` can reject mutable image tags; networking is disabled after container start; Python dependencies use a lockfile; release wheel and source distributions receive signed GitHub build provenance attestations | Digest enforcement is opt-in and image pulls happen through the host engine. Consumers must verify artifact attestations against the expected repository and signer workflow; provenance does not prove that source code or build dependencies are benign. |
| Secrets or private data appear in errors, retrieval, or traces | HTTP execution errors are sanitized; exceptions from host-trusted tool handlers are redacted before reaching model context, traces, or stream events; credentials stay outside agent YAML; retrieved text is marked as untrusted reference data | Successful tool results are intentionally available to the model and may contain sensitive values. Custom providers, tools, and tracers may log or transmit data elsewhere. Trace retention, access controls, and deletion are deployment concerns and need stronger end-to-end review. |
| One tenant reads or affects another tenant's data | One agent and tenant per service instance; run state is request-scoped | A shared deployment can still misroute requests or reuse external stores/secrets across instances. Gabby does not provide shared-process tenant isolation. |

## Explicitly out of scope for current guarantees

- A malicious host administrator, kernel, hypervisor, container daemon, or Python extension.
- Network isolation for host-trusted tools, providers, ingestion processes, or the image-pull phase.
- Allowlisted container egress, distributed request quotas, distributed token
  revocation propagation,
  and multi-tenant authorization.
- Disk quotas for writable workspace mounts or durable application-managed state.
- Automatic safety certification of user-supplied images, skills, tools, or model providers.

## Release acceptance gates

Before claiming hosted production readiness, the project should maintain evidence for:

- Each supported host/engine/adapter combination: network denial, CPU/memory/process limits,
  read-only and read-write mounts, cleanup after success/failure/cancellation, and Windows Hyper-V
  isolation where applicable.
- Deployment evidence that no host process mutates a mounted workspace during an agent run; component
  checks are defense in depth and not a substitute for workspace quiescence.
- Authentication, TLS termination, rate limiting, tenant routing, secret handling, log/trace
  retention, and recovery procedures in the actual deployment architecture.
- Image digest policy, verification of dependency/release provenance, and security response process.
- Resource exhaustion including workspace disk growth, stuck synchronous callbacks, and concurrent
  request bursts.

See [SECURITY.md](../SECURITY.md) for private vulnerability reporting and
[ADR 0003](architecture/adr/0003-container-sandbox.md) for the accepted sandbox design.
