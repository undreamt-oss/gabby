# Runnable examples

Run the offline domain examples from the repository root:

```sh
uv run --frozen --extra server python examples/domain_agents.py
```

The script composes customer-support, research, and data-analysis agents from the same Gabby
primitives. Each agent gets a small skill, a scoped environment, and an application-owned tool. A
deterministic mock model requests the tool and uses its result, so the example needs no API key,
network access, external service, or model download.

The sample records are synthetic. Registered Python handlers are trusted code that runs in the
Gabby process; production applications must implement their own data access and authorization
boundaries. Replace `ExampleModel` with a built-in or custom `ModelProvider` to connect a real model.

For a data agent that uses a real local database instead of a mock handler, run:

```sh
uv run --frozen python examples/sqlite_data_agent.py
```

This example creates synthetic sales rows in a temporary SQLite file, grants only the
`database:read` tool permission, executes a parameterized query, and removes the database after the
agent run. Read the [SQLite data tool guide](../docs/SQLITE_DATA.md) before pointing it at an
application database.

To run bounded Python analysis inside a per-run container without a model API key, use:

```sh
uv run --frozen python examples/python_sandboxed_agent.py
```

This requires a local Docker daemon. Gabby pulls `python:3.14-slim` if needed, disables container
network access, executes a deterministic calculation through `python_run_tool()`, and removes the
container after the run. See the [sandboxed Python tool guide](../docs/PYTHON_TOOL.md).

## OpenTelemetry event export

`opentelemetry_tracer.py` adapts Gabby's async `Tracer` contract to OpenTelemetry. It exports one
short span per runtime event and correlates spans with `gabby.trace_id`. It uses an explicit
attribute allowlist; prompts, model output, tool arguments, retrieved text, verifier evidence, and
arbitrary trace details are not exported. Agent names and the opaque run trace ID are included.

Install the optional API and SDK packages, then configure an exporter in the consuming application.
The example deliberately leaves SDK setup and shutdown to the host:

```sh
uv sync --frozen --extra observability
```

Pass the adapter to `Agent(..., tracer=OpenTelemetryEventTracer())`. Hosts that already configure an
SDK can inject their named tracer with `OpenTelemetryEventTracer(trace.get_tracer("my-service"))`.
The adapter does not own or shut down the SDK. Export callbacks remain subject to Gabby's per-event
tracer timeout, so configure a non-blocking SDK processor for production.

For a local console smoke check, configure `TracerProvider` with `SimpleSpanProcessor(ConsoleSpanExporter())`,
set it as the global provider, and call `provider.shutdown()` after closing the agent. Production
services should use a batch processor and their deployment's OTLP or vendor exporter.

Run the offline parent/child composition example with:

```sh
uv run --frozen python examples/agent_composition.py
```

It shows the parent granting `agent:invoke`, delegating one explicit task, and correlating the
specialist's request trace to the parent run without forwarding parent context or memory. Both models
are local deterministic examples; no network access or model credentials are needed.

To see a planner revise its plan after an order lookup, run:

```sh
uv run --frozen python examples/planning_revisions.py
```

The deterministic example prints a `plan_updated` event and a response based on the tool result.
It needs no network access or model credentials. Set `policies.max_replans` to `0` to compare
against the default single-plan behavior.

## FastAPI service with a real model provider

`hosted_service.py` is an embedded FastAPI app factory for one configured agent. It reads the agent
path, API bearer token, and optional service limits from the host environment. The sample agent uses
Hugging Face Inference Providers and a portable research skill; it has no host-side tools, so each
request remains a stateless synthesis of caller-supplied context.

Install the optional server dependency, set `HF_TOKEN` and a deployment-owned API token, then start
the app from the repository root:

```sh
GABBY_AGENT_CONFIG=examples/hosted-agent.yaml \
GABBY_API_TOKEN=replace-with-a-secret \
HF_TOKEN=replace-with-a-hugging-face-token \
uv run --frozen --extra server uvicorn examples.hosted_service:create_app \
  --factory --host 127.0.0.1 --port 8787
```

Check the public health route, then send an authenticated stateless execution request:

```sh
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/v1/agents/research-synthesizer/run \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -H 'Content-Type: application/json' \
  -d '{"input":"What does the research say about tree cover?","context":{"sources":[{"label":"sample-report.md","text":"Tree canopy increased by 8% between 2018 and 2024."}]}}'
```

`GABBY_MAX_CONCURRENT_RUNS`, `GABBY_MAX_REQUEST_BYTES`, and `GABBY_MAX_RESPONSE_BYTES` can tune
the corresponding per-process limits. The default bind above is loopback for a local check. For a
hosted deployment, terminate TLS at a trusted ingress, restrict direct access to the app, provide
secrets through the deployment secret manager, configure external rate limits and process
supervision, and review the [operations guide](../docs/OPERATIONS.md). This sample uses a static
single-tenant bearer token; hosts that need JWT validation can replace the authenticator with
`JWTBearerAuthenticator`. A successful `GET /health` does not test the model provider; the run route
does make a real provider call.

## Stateless SSE consumer

`stateless_sse_client.py` shows an external Python application consuming Gabby's typed SSE stream.
The application supplies `input`, `context`, and `memory` for each request; the client keeps no
conversation state in Gabby. Its HTTPX client owns transport lifecycle, and the bearer token is read
from the host environment.

With the service above running, invoke one streamed task from the repository root:

```sh
GABBY_API_TOKEN=replace-with-a-secret \
uv run --frozen python examples/stateless_sse_client.py \
  --agent research-synthesizer \
  --context-json '{"sources":[{"label":"sample-report.md","text":"Tree canopy increased by 8% between 2018 and 2024."}]}' \
  --memory-json '{"preferred_format":"brief"}' \
  'What does the research say about tree cover?'
```

Set `GABBY_URL` when the service is not at `http://127.0.0.1:8787`. The client prints text deltas,
reports typed progress event names on stderr, and prints the trace ID on completion. Context and
memory are request inputs supplied by the consuming application; persist or update them there if
later requests need state. It requires HTTPS for remote service URLs and allows HTTP only for
loopback development.

## Shared publisher revocation with Redis

[`redis_skill_revocation.py`](redis_skill_revocation.py) demonstrates the host-owned Redis client
boundary and the shared `RedisSkillRevocationStore` management/check operations. Install
`redis>=5`, set `GABBY_REDIS_URL`, and use the same namespaced Redis key from every Gabby process.
The adapter requires Redis 6.2+ and does not own the client lifecycle. See the
[multi-instance revocation operations guide](../docs/OPERATIONS.md#enforce-skill-publisher-revocations) for
consistency, TLS, persistence, and rotation requirements.
