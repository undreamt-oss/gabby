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

# Host-Orchestrated Pipeline Example

This reference shows a host service coordinating four separate Gabby `/v1` services:

1. `triage.yaml` returns a JSON-schema-validated plan with a knowledge query and/or finance task.
2. The host checks caller scopes and retrieves tenant-filtered evidence through its own retriever.
3. `rag.yaml` and `analyst.yaml` run concurrently when authorized and useful.
4. `synthesis.yaml` streams a response based on the worker statuses and bounded results.

Gabby remains stateless between each call. The host owns authentication, tenant routing, the
retriever, execution leases, durable checkpoints, credentials, the finance data tool, and any
workflow-wide budget ledger. This example contains interfaces for token issuance, retrieval, and
checkpoints, not implementations of those infrastructure services. It is a starting point for a
host integration, not a deployable production service.

## Agent service configuration

Each YAML file configures one agent and should be deployed as a separate service or separately
configured application instance:

The templates use Gabby's generic OpenAI-compatible adapter and a placeholder model ID. Before
deploying, configure the model ID, API endpoint, and credentials for the inference provider selected
by the host. The pipeline's orchestration and governance boundaries do not depend on a specific
provider.

| Role | Definition | API capability | Tools |
|---|---|---|---|
| Triage | [`agents/triage.yaml`](agents/triage.yaml) | `pipeline:run` | None |
| Knowledge | [`agents/rag.yaml`](agents/rag.yaml) | `pipeline:run` + `rag:read` | None; receives host-scoped snippets |
| Finance analysis | [`agents/analyst.yaml`](agents/analyst.yaml) | `pipeline:run` + `sql:finance_readonly` | Host-registered `finance_summary` |
| Synthesis | [`agents/synthesis.yaml`](agents/synthesis.yaml) | `pipeline:run` on run and stream routes | None |

Configure `JWTBearerAuthenticator` on each service with its own expected audience and an issuer
that validates the host-minted token. The host token issuer must include the trusted caller's
`tenant_id` as a signed JWT claim as well as its subject and scopes. `JWTBearerAuthenticator`
exposes that verified claim as `ToolContext.principal.tenant_id`; the analyst tool must fail closed
when it is absent and resolve it through the host's tenant registry before querying data. Do not
copy a tenant value from request JSON, prompt text, or tool arguments into the token. Require
`pipeline:run` with `run_scopes`; the synthesis service must also require `pipeline:run` with
`stream_scopes`. Keep the analyst's base tool and permission allowlists narrow, and let
`scope_rules` intersect them with the authenticated principal scopes. An API route scope only
authorizes reaching that route; it does not replace tool policy.

The analyst YAML declares a tool name but does not provide its implementation. Register
`finance_summary` when constructing the analyst `Agent`. It should accept a bounded business query
or typed date range, use parameterized read-only aggregate queries, and derive the tenant predicate
from the verified `ToolContext.principal.tenant_id`, resolving that opaque ID through the host's
tenant registry before querying. Declare `permissions=("sql:finance_readonly",)` and a
`context_parameter` on the `Tool`, then reject calls whose principal or tenant is missing, or whose
principal lacks the required scope. Do not accept arbitrary SQL or a caller-provided tenant
identifier as a tool argument.

Each service factory uses its own audience and one role-specific agent definition. For the analyst,
the factory pattern is:

```python
from gabby import Agent, JWTBearerAuthenticator, create_app


async def create_analyst_app():
    authenticator = await JWTBearerAuthenticator.from_oidc_issuer(
        issuer="https://identity.example.com/",
        audience="gabby-analyst",
    )
    agent = Agent.from_file(
        "examples/host_orchestrated_pipeline/agents/analyst.yaml",
        tools=build_finance_tool_registry(),
    )
    return create_app(
        agent,
        authenticator=authenticator,
        run_scopes=("pipeline:run",),
        stream_scopes=("pipeline:run",),
    )
```

Implement the factory and `build_finance_tool_registry()` in the analyst deployment, not in the
shared host coordinator. Other roles load their matching YAML and audience. Configure scope mapping
at the authenticator if the identity provider's claim names differ from Gabby's capabilities.

## Host wiring

Construct the pipeline once from trusted deployment configuration. `revision` should be a stable
deployment/config digest (for example, an immutable build identifier); update it when the agent
definition, tool implementation, policy, or model configuration changes.

```python
from examples.host_orchestrated_pipeline import (
    AgentEndpoint,
    HostOrchestratedPipeline,
)

pipeline = HostOrchestratedPipeline(
    endpoints={
        "triage": AgentEndpoint(
            "https://gabby-triage.internal",
            "pipeline-triage",
            "gabby-triage",
            "sha256:triage-config",
        ),
        "rag": AgentEndpoint(
            "https://gabby-knowledge.internal",
            "pipeline-knowledge-worker",
            "gabby-rag",
            "sha256:rag-config",
        ),
        "analyst": AgentEndpoint(
            "https://gabby-finance.internal",
            "pipeline-finance-analyst",
            "gabby-analyst",
            "sha256:analyst-config",
        ),
        "synthesis": AgentEndpoint(
            "https://gabby-synthesis.internal",
            "pipeline-synthesis",
            "gabby-synthesis",
            "sha256:synthesis-config",
        ),
    },
    token_issuer=host_token_issuer,
    retriever=tenant_scoped_retriever,
    checkpoints=durable_checkpoint_store,
)
```

The issuer, retriever, and checkpoint objects implement the protocols in
[`orchestrator.py`](orchestrator.py). The hosting HTTP handler should authenticate the request,
load the caller's preferences/session summary, and atomically claim a random execution ID before
calling the pipeline:

```python
caller = await identity_service.resolve(request)
lease = await workflow_store.claim(
    tenant_id=caller.tenant_id,
    subject=caller.subject,
    execution_id=execution_id,
    request_fingerprint=request_fingerprint,
)
try:
    result = await pipeline.run(
        execution_id=execution_id,
        task=request.task,
        caller=caller,
        on_synthesis_delta=stream_to_client,
    )
except BaseException:
    await workflow_store.release(lease, status="interrupted")
    raise
else:
    await workflow_store.complete(lease, result_status=result.status)
```

`identity_service`, `workflow_store`, and `request` above are host application concepts, not Gabby
APIs. The `release` and `complete` calls represent host-specific lease cleanup and terminal status
updates. The lease must prevent two host replicas from concurrently executing the same tenant-scoped
execution ID. Apply an HTTP request-size limit before parsing a potentially large request. The
orchestrator validates its own task size and validates triage output again even though the triage
agent has `output_schema`.

The token issuer receives the already verified `tenant_id`; it must place that value in the signed
`tenant_id` claim. All four worker calls receive that claim so host tools can enforce the same tenant
boundary.

## Recovery and data retention

The example saves an accepted marker, triage plan, bounded retrieval snippets, successful worker
results, and final result under tenant-partitioned checkpoint keys. It reuses only records with
matching request/context identity, caller subject/scopes, workflow version, and service revisions.
The host must still provide an atomic execution lease, durable and conditional checkpoint writes,
encryption for sensitive contents, access controls, and expiry/deletion. It should also decide what
to do with an interrupted `running` execution after the lease expires.

The sample analyst is read-only so repeating an uncheckpointed call is safe. For side-effecting
tools, add an idempotency key and durable operation status before retries. Persist trace IDs for
correlation, but do not place tokens, raw task text, or caller PII in metadata or normal logs.

Per-agent YAML can cap tokens and estimated cost for each run. It does not enforce a single budget
across triage, parallel workers, and synthesis. A hard workflow-wide budget requires a host-owned
atomic reservation and usage-reconciliation ledger. For streamed synthesis, deltas are provisional;
only persist a completed answer after the terminal completion event has been received. The example
applies the same 12,000-byte output limit to streamed deltas and the saved answer, and rejects a
completion whose bounded output does not match the streamed text.

Retrieval defaults to a 10-second timeout (configurable up to 300 seconds) and runs concurrently
with independent analyst work. A timeout cancels the pending retrieval and records the retrieval
stage as unavailable, allowing an authorized analyst to return a partial result.

See the [full architecture guide](../../docs/HOST_ORCHESTRATED_PIPELINES.md) for the security
model, failure table, and deployment guidance.
