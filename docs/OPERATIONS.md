# Operating Gabby

This guide documents the current service contract and deployment responsibilities. Gabby is
pre-release software; the guidance here is not evidence that a deployment is production-certified.
Review the [threat model](THREAT_MODEL.md) and validate the actual host, container engine, provider,
and network path used in deployment.

## Service boundary

The CLI serves one configured agent for one tenant per service instance:

```sh
GABBY_API_TOKEN=... MODEL_API_KEY=... gabby serve ./agent.yaml \
  --host 0.0.0.0 --port 8787 --max-concurrent-runs 8 \
  --bearer-scope agent:run --bearer-scope agent:stream \
  --run-scope agent:run --stream-scope agent:stream \
  --max-request-bytes 1000000 --request-body-timeout 30 \
  --max-response-bytes 4194304
```

`--max-request-bytes` bounds each accepted request body; the default is 1,000,000 bytes. Set the
limit to match the largest context payload your clients need to submit. `--request-body-timeout`
bounds the total time allowed to receive that body; its default is 30 seconds. An incomplete body
that exceeds this deadline gets HTTP 408 after a request slot is acquired and before authentication
or execution; Gabby closes the HTTP/1 connection or ends the HTTP/2 stream. Embedded `create_app`
callers configure the equivalent `max_request_bytes` and `request_body_timeout_seconds` arguments.
These per-request bounds do not
cap concurrent open connections; configure connection limits in the ASGI server and ingress.

Embedded hosts can set `authenticator_timeout_seconds` on `create_app` or `serve`; it defaults to
five seconds and bounds each custom async authenticator call. Timeout and authenticator failures
return the same sanitized HTTP 503 response. The timeout cancels the authenticator task, so custom
implementations should release resources during cancellation. The built-in JWT verifier retains
its separate JWKS and revocation-check timeouts.

Set the model credential variable named by the agent's model configuration. Keep tokens and
credentials in the deployment's secret manager or environment, never in agent YAML, skills, or
container images. The CLI requires a bearer token for non-loopback binds. Embedded FastAPI hosts can
use the optional `JWTBearerAuthenticator` from `gabby-agent-runtime[auth]`; configure its issuer
and audience from deployment-owned settings. Call `await JWTBearerAuthenticator.from_oidc_issuer(...)`
during async application startup to fetch bounded OIDC metadata and validate its issuer and HTTPS
JWKS URL. Or provide a fixed HTTPS JWKS URL directly to the constructor. Discovery does not accept
redirects and happens once when constructing the authenticator; later signing-key refreshes use the
cached JWKS path. The authenticator extracts the standard
`scope` or list-valued `scp` claim. Set `scope_mapping` on the authenticator to map issuer scopes to
Gabby capabilities; unmapped scopes are dropped when a mapping is configured. Then set `run_scopes`
and `stream_scopes` on `create_app` to require route-specific capabilities. Without a mapping, JWT
scopes pass through as capabilities. Custom authenticators should return Gabby capability names in
`Principal.scopes`. To check JWT revocation on every request, inject a host-owned
`TokenRevocationChecker`; this requires `jti` in the token and performs a bounded asynchronous lookup.
Revoked tokens receive HTTP 401. Checker failure or timeout returns HTTP 503, so a revocation-store
outage fails closed. `SQLiteTokenRevocationStore("./state/revocations.sqlite3")` is available for
durable single-host storage; retain the file on a local filesystem with deployment-appropriate
permissions (new files are owner-only where supported) and schedule `cleanup_expired()`. Use a host-provided checker for network filesystems or
distributed revocation propagation. The host owns store lifecycle and backups. See
[ADR 0043](architecture/adr/0043-pluggable-jwt-revocation-check.md),
[ADR 0052](architecture/adr/0052-bounded-oidc-issuer-discovery.md), and
[ADR 0046](architecture/adr/0046-sqlite-jwt-revocation-store.md).

## Enforce skill publisher revocations

For a service that requires signed installed skills, inject both a `SkillTrustPolicy` and a
`SkillRevocationChecker` when constructing the agent. Before every run and stream, Gabby revalidates
installed skill signatures and content, then checks for revoked signers before sandbox startup or
model/tool activity. It polls for revocation again while each execution is active:

```python
from pathlib import Path

from gabby import Agent, SkillTrustPolicy, SQLiteSkillRevocationStore

Path("./state").mkdir(mode=0o700, parents=True, exist_ok=True)
revocations = SQLiteSkillRevocationStore("./state/skill-revocations.sqlite3")
trust = SkillTrustPolicy(
    trusted_keys={"support-publisher": support_publisher_public_key},
)
agent = Agent.from_file(
    "agent.yaml",
    skill_trust_policy=trust,
    skill_revocation_checker=revocations,
)
```

For file-managed keys, store raw 32-byte public keys in a host-owned directory named
`KEY_ID.pub`. The directory must contain only regular key files and may contain at most 256 keys;
symlinks, subdirectories, malformed IDs, and incorrect key lengths are rejected. Build the
immutable policy snapshot at application startup:

```python
from gabby import Agent, SkillTrustPolicy, load_skill_trust_keys

trusted_keys = load_skill_trust_keys("/etc/gabby/trusted-skill-publishers")
trust = SkillTrustPolicy(trusted_keys)
agent = Agent.from_file("agent.yaml", skill_trust_policy=trust)
```

The CLI accepts the same directory for package verification, signed installation, remote fetch,
and installed-registry audits. Keep signing private keys in the publisher's secret manager, never
alongside runtime public keys:

```sh
gabby skill verify ./support-review.gabskill --trusted-key-dir /etc/gabby/trusted-skill-publishers
gabby skill install ./support-review.gabskill --registry ./skills --require-signature \
  --trusted-key-dir /etc/gabby/trusted-skill-publishers
gabby skill audit --registry ./skills --trusted-key-dir /etc/gabby/trusted-skill-publishers
```

For routine rotation, add the new publisher's public key, build a new `SkillTrustPolicy`, and
reconstruct or restart agents so the immutable snapshot includes it. Publish new packages with the
new key ID, verify installed content with `gabby skill audit`, then remove the old public key and
reconstruct the agents after migration. For a compromised signer, revoke its key ID through the
injected revocation store immediately; active runs stop at their next poll, while removing its
public key from the trust directory prevents new agent constructions from accepting it. Deployments
must distribute trust-directory changes and revocations to every service instance.

After detecting a compromised publisher, call `await revocations.revoke("support-publisher")`;
new executions are denied, and active executions stop at their next revocation poll.
`reinstate(key_id)` removes a revocation and `revoked_key_ids()` lists active records. New SQLite
files are owner-only where the
host OS supports those permissions. The parent directory must exist and should be protected by the
host. Use a local filesystem with SQLite locking semantics. Each check is bounded by the run's
remaining deadline and defaults to a five-second timeout; set `skill_revocation_timeout_seconds`
when constructing `Agent` to tune it (maximum 60 seconds). HTTP run calls return sanitized 403 for
a revoked signer and 503 if the check times out or the store fails. SSE reports the corresponding
typed error event. Active runs poll every second by default; configure
`skill_revocation_poll_interval_seconds` from 0.1 through 60 seconds to match the incident-response
target and store capacity. Detection latency includes the poll interval and checker latency.
Cancellation does not undo completed tool side effects, and an uncooperative synchronous host
callback may continue after the request ends. Distributed deployments need a checker backed by a
store with explicit cross-host consistency guarantees. See [ADR 0069](architecture/adr/0069-interrupt-runs-on-skill-revocation.md).

`RedisSkillRevocationStore` provides a shared implementation for Redis 6.2+ without adding a Redis
dependency to Gabby. Install `redis>=5` in the host application. The host creates and closes the
async Redis client, configures its authentication, TLS, timeouts, persistence, backups, and replica
routing, and gives each trust domain its own Redis key. Read checks use `SMISMEMBER` against the
configured client, so route checks to a primary or another authority with the required
read-after-write consistency. Revocation inspection uses bounded `SSCAN`; the default maximum is
100,000 entries.

```python
import os

from redis.asyncio import Redis

from gabby import Agent, RedisSkillRevocationStore, SkillTrustPolicy

redis = Redis.from_url(
    os.environ["GABBY_REDIS_URL"],
    decode_responses=False,
    socket_connect_timeout=2,
    socket_timeout=2,
)
revocations = RedisSkillRevocationStore(
    redis,
    key="production:customer-support:skill-revocations",
)
agent = Agent.from_file(
    "agent.yaml",
    skill_trust_policy=SkillTrustPolicy.from_directory("/etc/gabby/trusted-publishers"),
    skill_revocation_checker=revocations,
)

# On service shutdown, close the agent and then the host-owned Redis client.
```

Every instance must use the same Redis key and trusted publisher set. This distributes revocation
state only; public signing keys still need deployment-wide rotation. The runnable
[`redis_skill_revocation.py` example](../examples/redis_skill_revocation.py) shows host client
lifecycle and the shared management calls.
The per-run content check runs within the configured execution deadline. HTTP returns a sanitized
403 if the package no longer verifies under its construction-time trusted signer; SSE sends the
same typed failure. Keep trusted skill registries quiescent during executions because the check
cannot eliminate races with concurrent host mutations.

For the CLI bearer-token service, repeat `--bearer-scope` to assign capabilities to the configured
token and `--run-scope` or `--stream-scope` to require them on those routes. The CLI uses one shared
token for every caller; use embedded hosting and a JWT or custom authenticator when different
principals need different capability sets.

The built-in JWT verifier accepts RS256 and ES256 by default, requires `iss`, `aud`, `exp`, and `sub`,
and caps tokens at 16 KiB and JWKS documents at 1 MiB. It allows five seconds for each JWKS fetch,
caches keys for five minutes, and refreshes for an unknown key at most once every 30 seconds. It does
not follow redirects or use proxy environment variables. Invalid credentials receive HTTP 401;
JWKS fetch or configuration failures fail closed with HTTP 503. Tune these settings to the identity
provider's key rotation and availability policy.

The service exposes `GET /health`, `POST /v1/agents/{name}/run`, and
`POST /v1/agents/{name}/stream`. `/health` reports that the HTTP application is responding; it does
not prove that the model provider, knowledge store, container daemon, or a later run is healthy.
Use deployment-specific checks for those dependencies. Run and stream require authentication unless
an embedded application explicitly configures unauthenticated access.

The repository includes a single-tenant Docker Compose example for the hosted research agent in
[the container deployment guide](DEPLOYMENT_CONTAINER.md). It applies loopback-only port publishing,
a read-only root filesystem, dropped Linux capabilities, and bounded CPU, memory, process, and
temporary-storage resources. The included health probe checks only the HTTP process; deployers still
need to verify provider access, TLS and ingress controls, secrets, image provenance, and runtime
availability in their target environment.

The [Kubernetes reference](DEPLOYMENT_KUBERNETES.md) supplies a single-replica Deployment, ClusterIP
Service, TLS NGINX Ingress, bounded pod resources, and ingress/DNS/HTTPS NetworkPolicy. It requires
a NetworkPolicy-enforcing CNI and cluster-specific updates to controller/DNS selectors. Its HTTPS
egress rule allows any destination on port 443; use an FQDN-aware egress gateway when deployment
policy requires provider-domain allowlisting. No Kubernetes cluster acceptance has been run here.

Each process admits at most eight non-health requests by default. Gabby acquires a slot before
buffering a body or authenticating the request, then holds it through the response. This bounds the
number of simultaneous buffered request bodies along with active runs and streams. Requests over
the configured cap receive HTTP 429 immediately; Gabby does not queue them or coordinate capacity
between processes. `/health` bypasses admission so probes remain responsive during saturation.
Capacity therefore scales with the number of service processes. Start with one process
per instance unless the deployment has measured provider, sandbox, storage, and memory capacity for
more. Configure external load balancing, rate limits, retry policy for 429, and process supervision.
On service shutdown, Gabby stops accepting new agent runs and drains active executions before
closing the agent's owned provider resources; the longest run deadline bounds this wait.

The serialized body for each HTTP response is capped at 4 MiB by default, configurable through
`create_app(max_response_bytes=...)` or `gabby serve --max-response-bytes`. This includes the final
trace when `include_trace` is true (the default) and every SSE frame and keepalive in `/stream`.
Set `include_trace` to false to omit the returned trace body while preserving the trace ID; this does
not disable an injected host `Tracer`. Gabby does not truncate JSON: an
oversized `/run` result becomes a bounded HTTP 500 error. An oversized SSE event ends the stream
with a bounded `error` event if the remaining budget can hold it; the `completed` event is omitted.
The cap is per response body and does not include HTTP headers or transport framing.

## Reverse proxies and streaming

Terminate TLS at a trusted ingress or proxy. Restrict direct access to the Gabby listener, enforce
request-rate and connection limits at the edge, and set request-body limits consistent with Gabby's
default one-million-byte bound. Do not expose the local unauthenticated mode outside loopback.
Each outbound model call is also capped at 4 MiB of canonical UTF-8 JSON by default; adjust
`policies.max_model_request_bytes` for a specific agent when its grounded context or tool schemas
need a different bound.
Provider responses are independently capped at 4 MiB by default through
`policies.max_model_response_bytes`. Increase that limit only when a workload needs longer output;
the limit applies to each planner, selector, and runtime provider call.
Transient reasoning-model failures are not retried by default. `policies.max_model_retries` opts in
to at most three retries for a `RetryableModelError`; backoff and bounded `Retry-After` waits count
against the run deadline. Retried requests may incur duplicate inference charges. For SSE calls,
Gabby retries only before sending any text delta, so a failed partial response is never replayed.

SSE sends a comment keepalive every 15 seconds while a run is quiet. Disable proxy response
buffering for `/v1/agents/{name}/stream`, preserve `text/event-stream`, and set upstream idle and
overall timeouts to fit the agent's configured execution deadline. An ordinary stream disconnect,
or a failed ASGI send during `/stream`, cancels the active run and releases its process-local
execution slot. A request using `Idempotency-Key` opts into event journaling: the run continues after
the client disconnects, and the client can reconnect with the same key and its last
`Last-Event-ID`. The default in-memory journal is process-local. For reconnects across workers or
process restarts, configure `create_app(..., stream_journal_path=Path("/var/lib/gabby/streams.sqlite3"))`
or pass `--stream-journal-db` to `gabby serve`. Workers on the same host must use the same SQLite
file on a local filesystem. Gabby stores bounded serialized SSE frames, request fingerprints, and
hashed principal subjects; frames can contain agent output, tool progress, and optional traces, so
protect the file and its parent directory as execution data. Gabby creates the database with
owner-only permissions on POSIX systems. Do not put this SQLite journal on a network filesystem or
share it between hosts. Completed streams survive process restarts until their retention period
expires. Active runs remain owned by the process that started them; another worker can replay
committed events while that process is alive, but Gabby does not restart inference after a process
crash. Abandoned active entries become terminal recovery errors after at least twice the configured
run deadline plus 30 seconds, or the completed-session retention period, whichever is longer; they
remain retained for the configured completed-session period. Synchronous callbacks
still queued when their deadline expires are skipped. Cancellation cannot stop a callback already
running in a worker thread; it may continue until it returns. Once the journal expires, reusing its
key starts a new execution.

For shared storage beyond the same-host SQLite boundary, inject an async `StreamJournal` with
`create_app(..., stream_journal=backend)` or `serve(..., stream_journal=backend)`. The host owns the
backend's initialization, credentials, durability, and shutdown. Implement its atomic session and
event operations as documented in the [extension contract](EXTENSION_CONTRACTS.md#extension-matrix).
Gabby also provides `PostgresStreamJournal` over the host-owned asyncpg-compatible pool. Install the
`postgres` extra and apply the packaged `gabby/sql/postgres_stream_journal.sql` migration through
the application's migration system before injecting `PostgresStreamJournal(pool)`. It uses
PostgreSQL server time for expiry
and serializes global capacity checks across replicas. The host still owns pool lifecycle, TLS,
credentials, backups, availability, and schema migration.

The repository includes a complete [Nginx configuration](../deploy/nginx.conf) with a rate-limited
run and stream route, streaming-safe proxy settings, an unthrottled health route, and a default 404.
It targets a Gabby listener on `127.0.0.1:8787`; adapt the upstream, TLS certificate paths, host
name, rate, and burst to the deployment. It restricts TLS to versions 1.2 and 1.3, disables session
tickets and version disclosure, and adds HSTS, content-type, frame, and referrer response headers.
Review HSTS policy against the domain and certificate lifecycle before deployment. Its body cap is
exactly 1,000,000 bytes, matching Gabby's default API request cap. The rate key is the connecting IP; when Nginx sits behind a load balancer,
configure the real-IP module only for trusted proxy addresses before using client IPs for limits.
The repository check `python scripts/check_nginx_ingress.py` starts a temporary Nginx container,
forwards health, verifies the security response headers, checks a 1,000,001-byte body receives 413,
and confirms burst traffic receives 429. It requires Docker, OpenSSL, and the Gabby service listening
on `127.0.0.1:8787`; the hosted container workflow runs it against the Compose service.

The API bearer credential is forwarded in the `Authorization` header by default; keep Gabby
reachable only from this trusted proxy and do not log request headers or bodies. The proxy's 429 rate limit is separate from
Gabby's per-process 8-run admission cap. The 140-second read timeout leaves room for the default
120-second run deadline and SSE keepalives; raise it when an agent has a longer deadline.

## Sandboxed tools

Built-in shell and filesystem tools create one configured Docker or Podman container per run and
remove it afterward. The agent definition chooses the engine, image, keepalive argv, workspace path,
and read-only or read/write mount mode. Linux containers run as non-root UID/GID `65532:65532` by
default; set `sandbox.user` to another numeric non-root `UID:GID` pair when the image or workspace
permissions require it. Ensure a read/write host workspace grants that identity write permission.
Native Windows containers keep their image-defined user and require Hyper-V isolation. Gabby checks
that a Docker daemon reports version 29.1.4 or newer before starting a Windows container with
networking disabled; older releases could panic on this configuration. A missing
image is pulled by the host engine before the run, so image retrieval uses host/daemon network
policy. Pin reviewed images to immutable digests where available, control who can change agent YAML,
and restrict access to the Docker/Podman socket; daemon access is highly privileged.

For Docker Desktop's Windows named-pipe API, install the optional `windows-sandbox` extra and set
the API endpoint in the agent definition:

```yaml
sandbox:
  engine: docker
  adapter: api
  image: mcr.microsoft.com/windows/servercore:ltsc2022
  keepalive_argv: [powershell.exe, -NoProfile, -Command, "Start-Sleep -Seconds 86400"]
  api:
    named_pipe: '\\.\pipe\docker_engine'
```

The named-pipe transport uses the current Windows user's pipe permissions and the standard Proactor
event loop. The live Windows acceptance suite exercises both CLI and API adapters; run it on the
target host before treating that combination as verified.

Default container limits are 2 CPUs, 2 GiB memory, and 256 processes, with a run deadline. Verify
these controls against the selected engine and host. Linux live acceptance exists for Docker and
Podman through CLI and API adapters; macOS and Windows combinations are not certified by that
evidence. Native Windows containers require Docker Hyper-V isolation. A read/write mount grants the
agent write access to the entire configured workspace directory. The consuming application must
keep the mounted workspace quiescent for the full agent run. Gabby's component checks are defense
in depth; they cannot prevent a host process from racing a path check and archive operation. Each
container command has a combined 1 MiB stdout/stderr capture limit; exceeding it aborts the engine
command and removes the run-scoped container.

Remote Docker-compatible engine API endpoints must use HTTPS. Plain HTTP is allowed only for
loopback endpoints; Unix domain sockets are preferred for local daemons. Do not put API credentials
in endpoint userinfo or query parameters; inject an authenticated client from host-managed secrets.
Treat access to an engine API or socket as host-level privilege and restrict it to the Gabby service
identity.

## Knowledge storage and ingestion

SQLite FTS5 and the SQLite generation manifest are local persistent files. Keep them on a supported
local filesystem, not a network filesystem. Place database files outside workspaces mounted
read/write into agent containers. Back up both lexical and manifest databases consistently, protect
their contents as application data, and define retention and restore procedures. Multi-host indexing
can use `PostgresGenerationManifestStore` with a host-owned pool, externally applied migration,
atomic lease claims, and the PostgreSQL server clock. The host owns TLS, credentials, routing,
migrations, backups, and read-after-write consistency. Multi-host indexing also requires shared
generation-aware lexical and vector backends that durably enforce fencing tokens; the PostgreSQL
manifest alone does not make SQLite data files shareable.

`PostgresKnowledgeStore` supplies the generation-aware lexical backend through the same
host-owned PostgreSQL consistency domain. Apply its migration alongside the generation-manifest
migration. Pair it with a generation-aware vector store that enforces the same fencing tokens;
never run direct `replace_source` or `delete_source` calls against a store managed by
`HybridIndexCoordinator`.

Gabby includes `PostgresVectorStore` for pgvector deployments. The database administrator must
install pgvector 0.8.0 or newer before applying `sql/postgres_vector.sql`; the migration role
must have permission to create the extension when it is absent. Each store schema fixes its
embedding dimension on first use, and all workers must agree on that value and embedding space.
HNSW supports up to 2,000 dimensions; choose a different `VectorStore` backend for embeddings above
that limit. Search enables strict-order iterative scans to compensate when metadata or generation
filters remove nearest candidates. Results can still be limited by the configured maximum scan
tuples. HNSW search is approximate, so evaluate recall and tune `ef_search` and `max_scan_tuples` on
representative data before setting service quality targets.

The bundled text ingestor supports UTF-8 Markdown, plain text, and `.log` files. Log files are
indexed as UTF-8 text; Gabby does not infer timestamps, severity, or application-specific fields.
Its default limits are 10 MiB per
file, 10,000 files, and 1 GiB total per directory import. A directory import preflights count and
file sizes, then atomically replaces each source in sequence; it is not one corpus-wide transaction.
Concurrent file changes or a later I/O failure can leave earlier sources updated. Split large
imports into batches and coordinate them with the generation-aware index API when a corpus-level
activation boundary is required.

At run time, `knowledge.top_k` defaults to five and cannot exceed 100. The rendered retrieval
context defaults to 1 MiB of UTF-8 text per run; configure `knowledge.max_context_bytes` when a
workload needs a different bound. Gabby checks custom retriever result counts, document types, and
rendered size before adding retrieval text to the prompt. This does not constrain memory a custom
retriever allocates before returning; retriever implementations must apply their own storage and
query limits as well.

## Traces and data handling

The run response includes a trace with tool and skill names, timings, usage, verification details,
and, when planning is enabled, the structured plan. Treat response bodies and any application-side
trace exports as potentially sensitive. Restrict access, avoid logging full request/response bodies
by default, define retention and deletion, and redact provider or tool data before exporting it.
Gabby does not persist conversations or traces between runs by default.

## Readiness checklist

Before exposing a service to production traffic, verify the actual deployment's TLS and
authentication path, rate limiting, tenant routing, provider credentials, process limits, log/trace
retention, database backups, and sandbox cleanup. Run live sandbox acceptance for each host/engine/
adapter combination in use. Rehearse provider outage, container-daemon outage, SQLite restore,
429 behavior, request cancellation, and service shutdown. Document how credentials and container
images are rotated. See [ROADMAP.md](../ROADMAP.md) for current evidence gaps.
