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

# Gabby User Guide

This guide explains how to use Gabby's agent runtime and governance capabilities: installation, configuration, providers, skills, runtime policies, knowledge, tools, and APIs. The top-level README is the project overview and quick-start entry point.

## Install

Install Gabby from PyPI:

```sh
python -m pip install gabby-ai
```

The distribution is named `gabby-ai`, while Python code imports `gabby` and the CLI command is
`gabby`. Install optional features with extras, such as `python -m pip install
"gabby-ai[server,auth]"` for the HTTP server and JWT authentication.

For local development inside the repository, install the locked contributor environment instead:

```sh
uv sync --frozen --group dev --extra server --extra auth
```

Use `uv run` for project commands so the locked virtual environment is selected consistently. The `server` extra installs Uvicorn for `gabby serve`; embedded users can install the core package without the server process.

## Define an agent

```yaml
name: coding-helper
description: Helps with code questions using an explicitly supplied environment.

model:
  provider: openai_compatible # or ollama, huggingface, anthropic, gemini
  model: your-model-name      # for example qwen3:8b with Ollama
  # base_url: https://api.example.com/v1
  # api_key_env: MODEL_API_KEY

environment:
  type: coding

instructions: |
  Be concise. Explain assumptions and report what you verified.

persona:
  role: Customer support assistant
  tone: Calm, respectful, and direct
  communication_style: Use plain language and ask one question at a time.
  escalation_rules:
    - Escalate suspected fraud to a human agent.
  forbidden_behaviors:
    - Never request a full payment card number.

skills:
  - debugging

tools: []

policies:
  allowed_tools: []
  max_steps: 8
  max_tool_calls: 64 # per run; may be configured from 1 through 1024
  max_parallel_tool_calls: 1 # opt in up to 32; only explicitly parallel-safe host tools run together
  max_model_retries: 0 # opt in; at most 3 transient retries within the run deadline
  max_replans: 0 # opt in; at most 3 bounded plan revisions after tool observations
  timeout_seconds: 120
  max_model_request_bytes: 4194304  # 4 MiB per provider call
  max_model_response_bytes: 4194304 # 4 MiB per provider response

verification:
  enabled: false
  persona_compliance: false
```

Persona fields are snapshotted at agent construction and compiled into a structured system-prompt
section. They guide model behavior; they are not policy enforcement by themselves. For post-generation
checking, set `verification.persona_compliance: true` and inject a `PersonaVerifier` when constructing
`Agent`. Its `verify_persona(request, output, persona, trace)` method returns a
`VerificationResult`; Gabby rejects the run if it raises, returns an invalid result, or reports
`passed=False`. Persona-checked output is buffered until verification passes, including streamed
responses. Implement the verifier with deterministic checks where possible and an independent
review model or policy service where needed. Company rules that must constrain actions should also
be encoded in runtime policies and tool handlers; a text verifier cannot guarantee that a model
never generated a violating answer, and arbitrary behavioral boundaries may not be reliably
machine-detectable.

API credentials must stay outside agent configuration. Gabby rejects inline `model.api_key`,
credential-bearing model headers, URL userinfo, and provider URL query parameters with
credential-like names in YAML and in-memory `AgentDefinition` objects. Set
`model.api_key_env` to the name of an environment variable (default `OPENAI_API_KEY`,
`GABBY_OLLAMA_API_KEY` for Ollama, `HF_TOKEN` for Hugging Face, `ANTHROPIC_API_KEY` for
Anthropic, or `GEMINI_API_KEY` for Gemini), or inject a `ModelProvider`
backed by the consuming application's secret store. A provider can also be constructed directly
with a secret supplied by the host. `provider: ollama` defaults to
`http://localhost:11434/v1`; override `model.base_url` for a separately hosted Ollama endpoint.
Built-in providers require HTTPS for remote endpoints and allow HTTP only for loopback local
inference. This is checked both when loading an agent and when constructing a built-in provider.
Custom `ModelProvider` implementations own their transport security.

For Hugging Face Inference Providers, set `provider: huggingface`; Gabby uses
`https://router.huggingface.co/v1` and reads `HF_TOKEN` by default. The `model` value is a Hugging
Face model ID and can include the router's provider-selection suffix, for example
`Qwen/Qwen3-8B:fastest`. `model.base_url` and `model.api_key_env` can override the defaults. This
adapter supports the router's OpenAI-compatible chat completion contract. See the upstream
[Hugging Face Inference Providers documentation](https://huggingface.co/docs/inference-providers/index)
for model and provider availability.

For the native Anthropic Messages API, set `provider: anthropic` and use an Anthropic model ID.
Gabby reads `ANTHROPIC_API_KEY` by default; set `model.api_key_env` to use another environment
variable. `model.max_tokens` sets the required output limit and defaults to 1024. The adapter maps
Gabby's tool schemas and results to Anthropic's Messages format and normalizes streamed text and
tool-input events. See the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
for current model availability.

For the native Google Gemini GenerateContent API, set `provider: gemini` and use a Gemini model ID.
Gabby reads `GEMINI_API_KEY` by default and sends it in the `x-goog-api-key` header. Set
`model.api_key_env` to select another host-managed environment variable and
`model.max_output_tokens` to set the generation limit (default 1,024). The adapter maps function
declarations and responses, preserves Gemini function IDs and thought-signature parts across tool
turns, and supports SSE text streaming. It uses the configured HTTPS `model.base_url` or defaults
to `https://generativelanguage.googleapis.com/v1beta`. See the official
[Gemini GenerateContent guide](https://ai.google.dev/gemini-api/docs/generate-content) and
[function calling guide](https://ai.google.dev/gemini-api/docs/generate-content/function-calling).

Opt-in live completion and streaming acceptance checks for Ollama and Hugging Face are documented
in [`tests/integration/README.md`](../tests/integration/README.md). They use credentials from the host
environment and are skipped by the regular test suite unless explicitly enabled.

An optional `skills/debugging/skill.yaml` can define a reusable skill:

```yaml
name: debugging
version: 1.0.0
description: Diagnose and resolve software failures.
instructions: |
  Reproduce the failure, inspect evidence, identify the root cause, and propose
  the smallest safe fix. State which checks were actually run.
triggers: [dependency, regression, failing test]
style:
  voice: Calm, factual, and reassuring.
  tone: Concise and non-alarmist.
  formatting_rules:
    - Start with the current impact.
    - Use bullets for mitigation steps.
tools: []
input_schema:
  type: object
  properties:
    task: {type: string}
    context: {type: object}
    memory: {type: object}
  required: [task, context, memory]
  additionalProperties: false
output_schema:
  type: object
  properties:
    diagnosis: {type: string}
    evidence: {type: array, items: {type: string}}
  required: [diagnosis, evidence]
  additionalProperties: false
```

An optional `style` block can add task-specific `voice`, `tone`, and `formatting_rules` to a skill
package. Gabby includes these rules only when that skill is active. They refine how the agent handles
the skill's work; they cannot replace the agent's role, escalation rules, forbidden behaviors, or
runtime policies. Style fields guide model output and do not guarantee compliance.

Skill schemas are optional runtime contracts. `input_schema` validates the `{task, context,
memory}` request envelope when the skill activates. `output_schema` validates the agent's final
JSON response while that skill is active; if multiple active skills define output schemas, the
response must satisfy all of them. These checks complement an agent-level `output_schema`.

For longer procedures, put UTF-8 instructions in `skills/debugging/instructions.md`; Gabby loads that
conventional file when inline `instructions` is absent. A skill may instead set
`instructions_file: procedures/debugging.md`. Put text examples in `examples.md`, or select another
file with `examples_file`; Gabby adds those examples only when the skill is active. Resolved text
files must remain inside the skill package. Runnable sample files and skill-specific tests can live
in `examples/` and `tests/`; those directories are package resources and are not loaded into model
context. Create a validated starter directory with `gabby skill init`; it contains a manifest,
instructions, and examples. Review and edit the generated procedure and capability lists before
packaging; it grants no tools by default.

```sh
gabby skill init debugging --output ./skills/debugging \
  --description "Diagnose and resolve software failures."
mkdir -p ./dist
gabby skill pack ./skills/debugging --output ./dist/debugging.gabskill
gabby skill install ./dist/debugging.gabskill --registry ./skills
```

Initialization refuses to replace an existing directory. Package and install an exact version with
the commands shown above.

Installation writes `skills/<id>/<version>/skill.yaml` and refuses to overwrite an existing
version. Package checksums detect file corruption. Gabby requires detached Ed25519 signatures for
package installation and inspection by default, verified against host-managed public keys. Install
the optional `skill-signing` extra to enable signature verification. Skill contents remain untrusted
instructions and should be reviewed before attaching them to an agent. Static HTTPS registries support
catalog search, exact version discovery, and signature-required installation; see the
[skill registry guide](../docs/SKILL_REGISTRY.md). You can also place a package directory beside an
agent definition or add its parent directory to `skill_paths`.

To sign and require a publisher key, install the extra and provide the raw 32-byte private key as
base64 through your secret manager's environment injection. Keep the public key in a host-managed
trust directory:

```sh
uv sync --frozen --extra skill-signing
gabby skill sign ./dist/debugging.gabskill \
  --key-id gabby-release-2026 \
  --private-key-env GABBY_SKILL_SIGNING_KEY
gabby skill verify ./dist/debugging.gabskill \
  --trusted-key gabby-release-2026=./trusted-keys/gabby-release.pub
gabby skill install ./dist/debugging.gabskill --registry ./skills \
  --trusted-key gabby-release-2026=./trusted-keys/gabby-release.pub
gabby skill audit --registry ./skills --revoked-key old-release-key
gabby skill uninstall debugging 1.0.0 --registry ./skills --yes
```

The signature is a detached `ARCHIVE.sig` file and covers the exact archive digest and key ID.
Unsigned packages are rejected unless `--allow-unsigned` is explicitly selected for local
development. Agents also require a host `SkillTrustPolicy` to load installed packages by default;
`allow_unsigned_skills=True` is a deliberate development override. Each install records
its signer ID and archive digest for inventory; `gabby skill audit` flags revoked, unknown, or
invalid records. This metadata is advisory and does not disable installed skills or prove their
current contents are unchanged. `gabby skill uninstall` removes one validated exact skill version
and requires `--yes`. Hosts distributing skills should keep private keys in a KMS or
secret manager and manage trusted-key distribution and revocation. See the
[skill package guide](../docs/EXTENSIONS.md#portable-skill-packages) and
[registry operations guide](../docs/SKILL_REGISTRY.md).

Skill manifests use semantic versions. Existing manifests without `version` default to `0.1.0`.
Pin an exact version in the agent with `skills: [debugging@1.0.0]`; skill dependencies can also use
exact references such as `base-analysis@2.1.0`. The legacy layout `skills/<id>/skill.yaml` remains
supported; local registries can hold multiple versions at `skills/<id>/<version>/skill.yaml`.
Unpinned references use the legacy package when present or the only versioned package; if multiple
versioned packages exist, Gabby requires an exact pin. An agent cannot compose two versions of the
same skill ID because selectors and tool grants address skills by stable ID.

Agent files use YAML. Skills are resolved from `skills/<name>/skill.yaml` beside the agent file or from paths in `skill_paths`. Skills without `triggers` are always active; skills with `triggers` activate when a keyword appears in the request. Selecting a skill also activates its declared dependencies, even if a dependency's own trigger does not match. The async `SkillSelector` is injectable through `Agent(..., skill_selector=...)`; the default `ConfiguredSkillSelector` activates skills without triggers and matches configured trigger phrases case-insensitively. For semantic selection, opt in to `ModelSkillSelector(provider, model_id)`. It asks the injected provider for a JSON list of configured skill IDs, includes the agent's global and agent instructions within its fixed selection contract, records its model, latency, and usage in the trace, and fails the run if the response is malformed or names an unavailable skill. The runtime expands dependencies and enforces the normal tool policy after selection; selector output cannot grant tools. Model-based selection adds a provider call, cost, latency, and provider data handling to the run.

Planning is separately optional. Inject `ModelPlanner(provider, model_id)` to create a bounded structured plan from the task, supplied context and memory, selected skills, and policy-approved tools. The built-in planner receives global and agent instructions while preserving its fixed output and capability constraints. The plan is recorded in the trace and sent to the reasoning model as advisory context; it cannot grant capabilities or override runtime policy. This adds a provider call and sends the supplied context and memory to the planner's provider. Without a planner, the existing reactive model/tool loop runs as before.

Each outbound provider request has a UTF-8 serialized size limit of 4 MiB by default. Set
`policies.max_model_request_bytes` to a different positive byte limit. The count includes messages,
tool schemas, and request options and is enforced independently for runtime, planner, and built-in
model-selector calls. A request over the limit fails before the provider call. Custom selectors or
planners that call providers are responsible for using the public
`ensure_model_request_size` helper with the agent's configured limit.

Each provider response is also limited to 4 MiB by default through
`policies.max_model_response_bytes`. The built-in OpenAI-compatible adapter enforces this while
reading JSON and SSE response bodies; the runtime bounds custom-provider responses before using
their text or tool calls. Custom providers that buffer HTTP responses must cap transport reads
themselves because the runtime can only inspect a response after the extension returns it.

Transient model failures are not retried by default. Set `policies.max_model_retries` from 1 through
3 to retry failures explicitly marked transient by a built-in provider or by a custom provider
raising `RetryableModelError`. The runtime respects bounded delta-seconds `Retry-After` values,
otherwise uses short exponential backoff, and keeps every retry inside the run deadline. It does not
retry validation, size-limit, authentication, or other unclassified failures. A streamed call can
retry before emitting text; after the first text delta it fails normally instead of replaying partial
output. Retries may cause another billable inference request, so opt in only when that tradeoff is
acceptable. Retry attempts appear in traces and as `model_retry` SSE events.

At construction, `Agent` resolves the definition, skills, environment declarations, tool schemas, and tool registry into an immutable snapshot. YAML and programmatic `AgentDefinition` values pass through the same structural and policy validation; hosts can also call `validate_agent_definition(definition)` for preflight checks. Later edits to the YAML, source definition, skill registry, or caller-owned tool registry apply only when constructing another agent. Runtime handles such as model clients, retrievers, verifier instances, and Python handler closures stay injected references. File-backed skills are the default; an in-memory registry can be supplied with `Agent(..., skill_registry={"review": skill_definition})`.

## Agent types

Gabby’s runtime is not tied to coding. For example, a developer can build:

- A **knowledge-grounded support agent** that retrieves product guides and calls a registered ticket lookup or escalation tool.
- A **research agent** that searches an application-provided source, summarizes evidence, and returns citations. Gabby supplies retrieval contracts; a web search/browser tool must be registered by the application.
- A **data agent** that queries approved datasets through registered SQL or analysis tools. The opt-in [`sqlite_query_tool()`](../docs/SQLITE_DATA.md) provides bounded, read-only local SQLite queries. The sandboxed [`python_run_tool()`](../docs/PYTHON_TOOL.md) runs analysis code in the configured per-run container; other database access and authorization belong in application tools and their environment.
- A **coding agent** that uses Gabby’s built-in container-backed shell and workspace tools.

These are agent configurations and integrations, not four prebuilt products shipped by Gabby. Coding is the most turnkey initial path because shell and filesystem are its first built-in tools. Other domains can use the same model, skill, policy, execution, and API contracts, with application-provided tools.

Agents that feed other software can declare a JSON Schema for their final output. Gabby validates
the result and exposes the parsed value in `result.metadata["structured_output"]`; the same value is
available in `/run` and SSE completion metadata. Streaming calls hold back response text until it
passes validation. See the [structured output guide](../docs/STRUCTURED_OUTPUT.md).

Runnable support, research, and data-analysis examples use an offline mock model and synthetic
application tools. Run them with `uv run --frozen python examples/domain_agents.py`. The separate
`examples/agent_composition.py` demonstrates bounded parent/child delegation and correlated traces;
both need no credentials or network. See the [examples guide](../examples/README.md). The public Python
package exports `AgentDefinition`, `SkillDefinition`, and `ModelResponse` for in-memory composition
and custom model providers.

## Compose specialized agents

Wrap an async specialist as a parent tool when one agent should delegate a bounded task to another.
Each agent retains its own model, skills, tools, knowledge, and policy. The parent must grant the
delegation permission explicitly:

```python
from gabby import Agent, AgentDefinition, AgentTool, ToolRegistry

researcher = Agent.from_file("researcher.yaml")
tools = ToolRegistry()
tools.register(
    AgentTool(
        researcher,
        name="research",
        description="Ask the research specialist to investigate a focused question.",
    ).to_tool()
)
coordinator_definition = AgentDefinition(
    name="coordinator",
    model={"provider": "openai_compatible", "model": "..."},
    tools=["research"],
    policies={"allowed_permissions": ["agent:invoke"]},
)
coordinator = Agent(coordinator_definition, tools=tools)

async with researcher, coordinator:
    result = await coordinator.arun("Compare the strongest evidence for these options.")
```

Only the model-produced task string is sent to the child. Parent context, memory, metadata, and
caller identity are not forwarded by default. Set `forward_principal=True` only when the child
approval handler should receive the authenticated caller identity. Child calls inherit the parent
tool timeout and byte bounds; cycles and chains deeper than eight agents are rejected. The host owns
and closes every agent in the composition. See the [extension contract reference](../docs/EXTENSION_CONTRACTS.md).

## Tool trust and isolation

Tools run as `host_trusted` by default. Their handlers execute inside the Gabby process with its host operating-system permissions; registering a handler grants the agent a way to invoke that code. `Tool.execution` exposes this classification. Gabby requires sandboxed execution by default and rejects host-trusted tools before the model is called. Set `policies.require_sandbox: false` only when the host intentionally accepts in-process handlers. The built-in shell and filesystem tools declare `execution: sandboxed` and run in the configured per-run container. Use `sandboxed_python_tool()` for importable Python handlers that should run inside that sandbox; the selected image must include Gabby and the handler package.

See the [threat model](../docs/THREAT_MODEL.md) for trust boundaries, current controls, residual risks,
and the evidence required before hosted-production claims. See the [operations guide](../docs/OPERATIONS.md)
for service, storage, streaming, and sandbox deployment responsibilities.

## Model payload filters

Gabby applies a default provider-boundary filter to reasoning, planning, and skill-selection calls.
It redacts common email, phone, and token patterns from outbound messages, tool schemas, and
user-facing response text, and rejects simple direct attempts in user or tool content to override higher-priority
instructions. These are lightweight heuristics, not semantic prompt-injection detection or a
guarantee that all sensitive data is removed. Hosts can add synchronous or asynchronous filters with
`Agent(..., payload_filters=[...])` for organization-specific checks before provider calls and before
model responses reach runtime handling. Filtered requests are rechecked against the configured
`max_model_request_bytes` limit. The built-in heuristic filter runs on each stream delta so
SSE remains incremental; patterns split across provider chunks may evade its redaction. Adding a
host filter buffers the full stream until that filter approves the complete response, which can add
latency but prevents partial output from escaping before approval.
The default filter leaves structured tool-call arguments unchanged so it cannot silently change an
operation; add a host filter that rejects tool arguments when policy requires that check.

## Run from Python

```python
import asyncio

from gabby import Agent

agent = Agent.from_file(
    "agent.yaml",
    global_instructions="Use the organization's terminology and cite source documents.",
)


async def main():
    async with agent:
        result = await agent.arun("Explain this error", context={"error": "..."})
        print(result.output)
        print(result.trace.trace_id)


asyncio.run(main())
```

The embedded and HTTP interfaces enforce the same 1–100,000 character input range. `context`,
`memory`, and `metadata` must be dictionaries with string top-level keys; Gabby snapshots and
JSON-normalizes them before execution, within the agent's `max_model_request_bytes` budget. This
keeps caller mutation from changing a run after it starts and rejects oversized data before provider
calls.

`global_instructions` is optional shared application guidance captured when the agent is
constructed. Gabby places it above the agent's and active skills' instructions, but below runtime
requirements, and includes it in built-in model-based skill selection and planning. It cannot grant
capabilities or override runtime-enforced policies; rebuild the agent to change it.

For synchronous scripts, `agent.run(...)` wraps the async runtime. Inside an existing event loop, use `await agent.arun(...)`.
Repeated synchronous calls on one thread reuse the same event loop so provider resources stay loop-affine. Use either `run()` or `arun()` consistently for an agent instance; close it with the matching `close()` or `aclose()` method. Context managers handle cleanup:

```python
with Agent.from_file("agent.yaml") as agent:
    result = agent.run("Explain this error", context={"error": "..."})
    print(result.output)
```

Register only the tools the consuming application intends to expose. Tool calls must be declared by the agent and allowed by its policy:

```python
from gabby import Agent, Tool, ToolRegistry

tools = ToolRegistry()
tools.register(
    Tool(
        name="lookup_issue",
        description="Look up an issue in the host application's issue store.",
        parameters={
            "type": "object",
            "properties": {"issue_id": {"type": "string"}},
            "required": ["issue_id"],
        },
        handler=lambda issue_id: {"issue_id": issue_id, "status": "open"},
        output_schema={"type": "object", "required": ["issue_id", "status"]},
    )
)
agent = Agent.from_file("agent.yaml", tools=tools)
```

Each tool is limited to a 1 MiB serialized result by default. Set `max_result_bytes` on the
`Tool` to choose a different per-tool bound. An oversized result becomes a tool error before its
content reaches the model; Gabby does not truncate structured results. Tool error details are also
bounded; unusually long `ToolError` messages are replaced with a short generic message. The limit
bounds the model observation after the handler returns, so trusted host handlers should also bound
their own work. Every `Tool` must declare an `output_schema` JSON Schema. Gabby validates each
handler result before adding it to model context, and rejects tools without that contract at
construction. Tool observations, trace events, and `tool_failed` stream events also include a
stable `error_code` such as `invalid_arguments`, `policy_denied`, or `result_too_large`. The
`ToolError` and `ToolErrorCode` types are public for custom tools; Gabby preserves a handler's
failure code while redacting its exception message from model context. See
[ADR 0053](../docs/architecture/adr/0053-required-tool-output-schemas.md) and
[ADR 0054](../docs/architecture/adr/0054-stable-tool-error-codes.md).
See [ADR 0013](../docs/architecture/adr/0013-tool-result-size-limits.md) for the contract and tradeoffs.

Mark side-effecting tools with `requires_approval=True` and inject a host-owned approval handler.
The runtime validates arguments first, waits within the run deadline, and denies execution when no
handler is configured or the host does not explicitly approve. The consumer owns the approval UI,
identity checks, and any durable approval record:

```python
from gabby import ApprovalDecision, ApprovalRequest, Tool


class AppApproval:
    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        allowed = await my_application_approval_flow(request)
        return ApprovalDecision(approved=allowed)


send_tool = Tool(
    name="send_report",
    description="Send a report to a recipient.",
    parameters={"type": "object", "properties": {"recipient": {"type": "string"}}},
    handler=send_report,
    requires_approval=True,
)
agent = Agent.from_file("agent.yaml", tools=tools, approval_handler=AppApproval())
```

See [ADR 0025](../docs/architecture/adr/0025-host-owned-tool-approval.md). Approval decisions are
per-call and per-run; Gabby does not retain them. The [approval guide](../docs/APPROVALS.md) shows
how to connect host review to the reusable SQLite audit helper without storing raw arguments.

## Container sandbox tools

When an agent definition includes `sandbox`, Gabby registers the built-in shell and workspace tools.
They appear in model context only when listed under `tools` and permitted by `policies`:

```yaml
tools:
  - shell_run
  - filesystem_read_file
  - filesystem_write_file
  - filesystem_list_dir

policies:
  allowed_tools:
    - shell_run
    - filesystem_read_file
    - filesystem_write_file
    - filesystem_list_dir
  allowed_permissions:
    - shell.execute
    - filesystem.read
    - filesystem.write
  require_sandbox: true # default; set false only for deliberately trusted in-process handlers
  max_steps: 8
  max_tool_calls: 64
  timeout_seconds: 120

sandbox:
  engine: docker
  adapter: cli
  runtime: runsc # optional Docker runtime, for example gVisor's runsc
  image: python:3.12-slim
  require_image_digest: false # set true in production for immutable image identity
  user: "65532:65532" # default; Linux only, numeric non-root UID:GID
  keepalive_argv: [python, -c, "import time; time.sleep(86400)"]
  workspace:
    path: ./workspace
    container_path: /workspace
    access: read_write
  resources:
    cpus: 2
    memory_bytes: 2147483648
    process_limit: 256 # set null only for native Windows containers
```

Each run creates one container, pulls the configured image if it is missing, and removes the container
when the run finishes. The container has networking disabled. Its workspace mount is read-only unless
the agent config grants `read_write`. The default resource limits are 2 CPUs, 2 GiB of memory, and
256 processes; Linux containers run as the non-root UID/GID `65532:65532` by default. Set
`sandbox.user` to another numeric non-root `UID:GID` pair for Linux images and workspace permissions.
Native Windows containers retain their image-defined user under Hyper-V isolation. Docker's Windows
CPU control uses a whole-number CPU count; fractional CPU values are rejected. Docker does not
document a process-count limit for Windows containers, so set `process_limit: null` explicitly for
native Windows images. Gabby rejects that opt-out for Linux images, which retain the 256-process
default. The run deadline
also bounds container lifetime. `shell_run` takes an argv array and
does not invoke a shell implicitly. Images must provide the configured keepalive executable and any
programs the agent needs to run. Native Windows containers require Docker with Hyper-V isolation;
Gabby checks the Docker server version and requires 29.1.4 or newer before starting one, because
earlier Docker releases could panic when asked to start a Windows container with networking disabled.
Podman uses Linux containers on every host.

Docker and Podman can select an installed runtime such as gVisor's `runsc` using `sandbox.runtime`.
For a microVM backend such as Firecracker, configure `engine: custom` and `adapter: custom`, then
inject a host implementation of Gabby's `EngineAdapter` through `Agent(sandbox_adapter=...)`. Gabby
provides the adapter contract, not a Firecracker or gVisor installation; the host adapter is
responsible for creating, bounding, and cleaning up the isolated environment. A custom adapter must
honor workspace, resource, network, deadline, and cleanup restrictions.

Set `sandbox.require_image_digest: true` to require an immutable OCI reference ending in
`@sha256:<64 hex characters>`. This is opt-in for local development; production deployments should
pin and review the image digest they run.

Docker and Podman expose both CLI and API adapters. Linux live acceptance checks pass for both
engines and adapters, covering isolation, resource limits, workspace mounts, networking, and cleanup.
Other host, engine, and adapter combinations remain experimental until they pass the same checks.

Set `adapter: api` with `sandbox.api.base_url` or `sandbox.api.unix_socket` to use the API adapter.
Remote API endpoints must use HTTPS. Engine API access is highly privileged and must be scoped to a
trusted local or secured remote endpoint. Image pulls use the host engine's registry configuration;
network isolation applies to the container itself.

### User-defined sandboxed tools

An image can provide domain-specific tools without running their handlers in the Gabby process.
Declare the executable argv in Python; the runtime appends a temporary JSON input-file path as its
last argument. The executable reads that file and writes one JSON value to stdout:

```python
from gabby import Agent, Tool, ToolRegistry

classify_invoice = Tool(
    name="classify_invoice",
    description="Classify an invoice using the bundled domain model.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    sandbox_action="tool.execute",
    execution="sandboxed",
    sandbox_command=("/opt/gabby/classify-invoice",),
    max_input_bytes=512 * 1024,
    output_schema={"type": "object", "required": ["category"]},
)
tools = ToolRegistry()
tools.register(classify_invoice)
agent = Agent.from_file("agent.yaml", tools=tools)
```

The runtime writes each request file into a private per-run host scratch directory and mounts that
directory read-only into the container only when a custom sandboxed tool is registered. On POSIX
hosts, the directory permits traversal without listing and each request file is read-only before
execution, so the default unprivileged container user can read it. Gabby removes the file after
each call and the directory after the container is removed. Input size defaults to 1 MiB per tool
and is configurable; stdout and stderr share a 1 MiB sandbox output limit. Exceeding it terminates
the engine command and run container. Results are also checked against `output_schema` when
provided. The executable and argv come from trusted agent configuration,
never from model-supplied tool arguments. See
[ADR 0003](../docs/architecture/adr/0003-container-sandbox.md).

For Python handlers packaged in the selected image, `sandboxed_python_tool()` creates the fixed
worker command and keeps the callable out of the main Gabby process:

```python
from gabby import sandboxed_python_tool

summarize = sandboxed_python_tool(
    name="summarize_record",
    description="Summarize one validated record.",
    handler="my_agent_tools:summarize_record",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    output_schema={"type": "object", "required": ["summary"]},
)
```

The referenced function must be importable in the sandbox image and accept schema-defined
arguments as keyword arguments. Its return value must be JSON serializable and satisfy the output
schema. Local functions and closures are rejected so the host cannot accidentally serialize runtime
state into the worker.

An environment can be supplied as a reusable object. Agent construction snapshots its capability description, tool registrations, and allowlist into an agent-local registry; resource handles are shared with the supplied environment. An additional `ToolRegistry` passed to `Agent` is merged into the local registry without mutating either caller-owned registry. The environment allowlist is intersected with `policies.allowed_tools` at runtime; resources are not copied into model context automatically.

```python
from gabby import Agent, Environment

environment = Environment(
    type="research",
    description="Read-only access to the research catalog.",
    capabilities=["catalog lookup"],
    resources={"catalog": catalog_client},
    tools=tools,
    allowed_tools=["lookup_issue"],
)
agent = Agent.from_file("agent.yaml", environment=environment)
```

A host tool can opt into request-scoped environment access with `context_parameter` and an explicit
`context_resources` allowlist. Gabby checks those resource names when constructing the agent and
passes only those handles in a `ToolContext`; the context parameter is excluded from the model's
tool schema. The context also carries the run ID, agent/environment description, and authenticated
principal when one is available. It is for trusted in-process handlers and should not be retained
after the run:

```python
from gabby import Tool, ToolContext


def lookup_issue(issue_id: str, gabby: ToolContext) -> dict[str, str]:
    catalog = gabby.resources["catalog"]
    return catalog.lookup(issue_id)


lookup_tool = Tool(
    name="lookup_issue",
    description="Look up an issue in the configured catalog.",
    handler=lookup_issue,
    parameters={
        "type": "object",
        "properties": {"issue_id": {"type": "string"}},
        "required": ["issue_id"],
        "additionalProperties": False,
    },
    context_parameter="gabby",
    context_resources=("catalog",),
)
```

See [ADR 0030](../docs/architecture/adr/0030-scoped-tool-context.md) for the trust boundary and
compatibility contract. Context-enabled handlers also receive a cooperative cancellation token;
see [ADR 0037](../docs/architecture/adr/0037-cooperative-tool-cancellation.md). A synchronous handler
can wait on `gabby.cancellation.wait(timeout)` from its worker thread. Handlers that ignore the
signal may continue after the run returns, so use container-backed tools when forced termination
is required.

## Knowledge retrieval

Gabby's RAG path is a runtime pipeline: the host selects an agent configuration and appropriately
scoped retriever, the retriever finds relevant documents for the request, Gabby validates and bounds
the returned context, and the model receives it as reference material. Document storage and lifetime
remain outside an agent run. Gabby includes an in-memory BM25 retriever for fixtures and a persistent
SQLite FTS5 store for local or single-service use.

An agent configuration can describe a user-requested character and enable knowledge retrieval:

```yaml
name: writing-companion
description: Adapts its writing help to the user's requested style.
model:
  provider: openai_compatible
  model: your-model-name
instructions: |
  Character: be warm, concise, and practical.
  Adapt the tone to the user preferences supplied by the host.
knowledge:
  sources: [user-profile, writing-guides]
  top_k: 5
policies:
  allowed_tools: []
  max_steps: 8
```

For per-user adaptation, the host authenticates the caller and selects that user's configuration
and retrieval scope. Supplying validated preferences through `RunRequest.context` or
`RunRequest.memory` can personalize a request without retrieval. Those fields do not filter retrieved
documents: for private or tenant-specific documents, the host must inject a retriever already scoped
to the authenticated caller. The retriever contract receives the request text and result limit, not
an authenticated user identity. Never use an untrusted request field as the identity or tenant
boundary. The host owns storage and persistence of user preferences because Gabby is stateless
between runs.

RAG supplies relevant context; it does not enforce a character or policy. Gabby marks retrieved
documents as untrusted reference data, while configured instructions shape the response and runtime
policies govern tool access and execution. A retrieved document cannot grant a capability or
override those controls.

The retriever is injected by the consuming application:

```python
from gabby import Agent, Document, InMemoryBM25Retriever

retriever = InMemoryBM25Retriever(
    [
        Document(
            text="Rotate the signing key and update the issuer configuration.",
            source="docs/authentication.md",
            metadata={"area": "identity"},
        )
    ]
)
agent = Agent.from_file("agent.yaml", retriever=retriever)
```

Each run requests five documents by default and validates that a custom retriever returns no more
than requested. `knowledge.top_k` can be set from 1 through 100. The rendered retrieval context has
a 1 MiB UTF-8 byte limit by default; tune `knowledge.max_context_bytes` for an agent that needs a
different bound. Gabby rejects oversized or malformed retriever results before constructing the
prompt instead of silently truncating them. This caps Gabby's prompt assembly, but cannot bound
memory already allocated inside a custom retriever before it returns.

The SQLite store uses FTS5 BM25 ranking, exact metadata filters, stable document IDs, and
source-scoped replacement. Markdown and plain-text files can be ingested with deterministic
paragraph chunks (2,000 characters per base chunk and a 160-character overlap by default). Each
file is a source and is atomically replaced when reingested. Each database operation runs off the
event loop and uses its own short-lived connection. File discovery, path checks, and reads also run
in Gabby's bounded synchronous-callback workers, keeping directory walks and disk I/O off the async
event loop:

```python
from gabby import Agent, FileIngestor, SQLiteFTS5Store

knowledge = SQLiteFTS5Store("./.gabby/knowledge.db")
ingestor = FileIngestor(knowledge, root="./docs")
await ingestor.ingest_directory(metadata={"collection": "product-docs"})
agent = Agent.from_file("agent.yaml", retriever=knowledge)
```

`FileIngestor` handles UTF-8 `.txt`, `.log`, `.md`, and `.markdown` files below an explicit root.
Markdown is indexed as source text with its markup preserved. Inputs, extracted text, directory file
count, and total directory bytes are bounded. Each source is replaced atomically when reingested;
parsers and the paragraph chunker are replaceable. The default parser set stays deliberately small;
hosts can inject a trusted `FileParser` for formats they need, and should isolate complex parsers
when processing untrusted files. `TextFileIngestor` remains an alias of `FileIngestor`.

`FileParser` receives file bytes and returns ordered `ParsedPage` values. `FileIngestor` owns
root-confined paths, byte and output limits, chunking, metadata, and source replacement. It does not
include PDF, OCR, office-document, email, archive, or structured-data parsers.
See [ADR 0125](architecture/adr/0125-plain-text-markdown-ingestion.md) for the scope decision.

`KnowledgeStore`, `EmbeddingProvider`, `VectorStore`, and `Reranker` are replaceable
interfaces. `HybridRetriever` combines any lexical `Retriever` with a host-provided embedding
provider and vector store using deterministic Reciprocal Rank Fusion (RRF). It embeds a query once,
fetches both candidate lists concurrently, deduplicates by document ID, and returns the fused top
results. The defaults fetch up to 20 candidates per backend and use the standard RRF constant of 60;
the candidate limit is configurable from 1 through 100. Direct result limits also stop at 100, and
the hybrid wrapper rejects a backend that returns more candidates than requested.
`RerankingRetriever` adds an optional second stage to any retriever,
including a hybrid retriever:

```python
from gabby import RerankingRetriever, SQLiteFTS5Store


class MyReranker:
    async def rerank(self, query, documents, *, limit):
        # Call a host-selected reranking model or service and return candidate Documents in rank order.
        ...


retriever = RerankingRetriever(
    SQLiteFTS5Store("./.gabby/knowledge.db"),
    MyReranker(),
    candidate_limit=20,
)
```

The wrapper fetches at most 100 candidates (20 by default), passes them to the reranker, and accepts
only a unique, ordered subset of those candidates. Gabby returns copies of the original documents,
so a reranker can change order but cannot rewrite retrieved evidence. `CohereReranker`,
`JinaReranker`, `NvidiaReranker`, and `VoyageReranker` are optional hosted implementations. They send candidate text
to their respective rerank APIs, read `COHERE_API_KEY`, `JINA_API_KEY`, `NVIDIA_API_KEY`, or `VOYAGE_API_KEY` from the
host environment by default, preserve local source metadata by mapping validated indexes, and
enforce HTTPS for remote endpoints plus 4 MiB request/response caps. Voyage defaults to provider-side
truncation disabled. Candidate text leaves Gabby when any hosted adapter is used. See [the extension
guide](../docs/EXTENSIONS.md), [Cohere's rerank API reference](https://docs.cohere.com/v2/reference/rerank),
[Jina's reranker API](https://jina.ai/en-US/reranker/), and
[Voyage's reranker API reference](https://docs.voyageai.com/reference/reranker-api), and
[NVIDIA's reranking API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nim-rerankqa-mistral-4b-v3-infer). Opt-in live acceptance checks are available for Cohere, Jina, Voyage, and NVIDIA; they verify provider request and identity mapping, not ranking quality. Gabby does not select an embedding model.

Gabby includes `SQLiteVectorStore`, a persistent exact-cosine backend for small and moderate local
collections. It has no embedding-model dependency; the host still supplies an `EmbeddingProvider`.
Search scans the index linearly, so use a specialized `VectorStore` for larger corpora or approximate
nearest-neighbor search. Each vector database has a fixed dimension. Create a fresh index when
changing the embedding model, including same-dimension model changes; the store cannot identify the
provider itself. `OpenAICompatibleEmbeddingProvider` calls a standard `/embeddings` endpoint with
batched text, indexed response validation, HTTPS enforcement for remote endpoints, and 4 MiB request
and response caps per provider call. Credentials come from an environment variable or host injection.
For local Ollama, point `base_url` at `http://localhost:11434/v1` and choose an installed embedding
model; set `api_key_env` to an unset Gabby-specific variable to avoid sending ambient credentials.
`GeminiEmbeddingProvider` calls Google's native `batchEmbedContents` endpoint with the same bounds.
It defaults to `gemini-embedding-2`; `gemini-embedding-001` can set a task type such as
`RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY`. Set separate `query_task_type` and
`document_task_type` values to have hybrid retrieval apply each role correctly through one provider.
Embedding 2 uses task instructions in the input text instead of API task types, so configure
`query_prefix` and `document_prefix` for its retrieval prompts. Gemini credentials come from
`GEMINI_API_KEY` by default:

```python
from gabby import GeminiEmbeddingProvider

embeddings = GeminiEmbeddingProvider.from_config(
    {
        "model": "gemini-embedding-001",
        "query_task_type": "RETRIEVAL_QUERY",
        "document_task_type": "RETRIEVAL_DOCUMENT",
        "dimensions": 768,
    }
)
vectors = await embeddings.embed(["Gabby builds reusable stateless agents."])
await embeddings.aclose()
```

`HybridIndexCoordinator` uses `embed_documents()` while `HybridRetriever` uses `embed_queries()`;
the provider is also compatible with direct `embed()` calls. The model
spaces for `gemini-embedding-001` and `gemini-embedding-2` are incompatible, so changing between
them requires a fresh vector index. See [Google's embedding API](https://ai.google.dev/api/embeddings)
and [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) for model limits and task
guidance.

The host indexes the same document IDs in the lexical and vector stores. For coordinated reindexing,
`HybridIndexCoordinator` stages both backends under a new generation and activates that generation
through a durable manifest only after both stores report success. `SQLiteGenerationManifestStore` is
the built-in manifest implementation; it uses a separate SQLite file so it does not depend on the
lexical store schema:

```python
from gabby import (
    HybridIndexCoordinator,
    OpenAICompatibleEmbeddingProvider,
    SQLiteFTS5Store,
    SQLiteGenerationManifestStore,
    SQLiteVectorStore,
)

coordinator = HybridIndexCoordinator(
    lexical=SQLiteFTS5Store("./.gabby/lexical.db"),
    embeddings=OpenAICompatibleEmbeddingProvider(model="text-embedding-3-small"),
    vectors=SQLiteVectorStore("./.gabby/vectors.db"),
    manifest=SQLiteGenerationManifestStore("./.gabby/index-manifest.db"),
)
await coordinator.replace_source("docs/guide.md", documents)
results = await coordinator.retrieve("How does authentication work?")
```

For Hugging Face Inference Providers, use `HuggingFaceFeatureExtractionProvider`; it reads
`HF_TOKEN` from the host environment by default and supports bounded batches, model-specific
truncation options, and mean or CLS pooling for token-level feature output:

```python
from gabby import HuggingFaceFeatureExtractionProvider

embeddings = HuggingFaceFeatureExtractionProvider.from_config({"model": "BAAI/bge-small-en-v1.5"})
vectors = await embeddings.embed(["How does authentication work?"])
await embeddings.aclose()
```

The selected model and inference provider receive indexed text. Select pooling and prompt options
according to the model documentation. Gabby does not bundle weights or choose an embedding model.

Use the same embedding model and pooling settings for indexing and queries. Rebuild the vector index
when changing models or pooling because equal vector dimensions do not imply compatible embedding
spaces.

Pass the coordinator to `FileIngestor` to use the same generation protocol for file reindexing.
At startup or before a write, it reconciles only pending generations whose writer lease has
expired: complete staging is activated, while incomplete staging is removed. Active writers renew
a 30-second lease every 10 seconds by default; configure `lease_ttl_seconds` on the coordinator to
change the timeout. Each lease takeover advances a per-source fencing token, and both backends must
reject stale staging/discard tokens. The built-in SQLite lexical and vector stores persist and enforce
this fence. Retired generations stay stored so in-flight
retrievals can finish on the generation they selected. Call `prune_retired(readers_quiescent=True)`
during a maintenance window after readers have drained. Backend generation deletion is idempotent,
so interrupted cleanup can be retried. The manifest coordinates writes across stores; it does not
make arbitrary backend implementations transactional or automatically clean up retired data. The
built-in SQLite manifest and indexes are intended for cooperating processes on one host; database files
must not be shared over a network filesystem. A multi-host deployment should inject a manifest store
that provides atomic lease claims and an authoritative clock. The built-in SQLite indexes are
single-host components. Other vector databases, embedding implementations, and reranking providers
remain extension work.
The database stores source text and metadata as ordinary SQLite data; the consuming application
chooses each path and controls filesystem access, backups, and retention. Python's SQLite build must
include FTS5 support for the lexical index.

## CLI and API

```sh
gabby init --template customer-support ./agent.yaml
gabby validate ./agent.yaml
gabby inspect ./agent.yaml
gabby skills ./agent.yaml
gabby knowledge ingest ./docs --database ./.gabby/knowledge.db --metadata '{"collection":"product-docs"}'
gabby knowledge delete ./docs retired/guide.md --database ./.gabby/knowledge.db
gabby knowledge search "installation requirements" --database ./.gabby/knowledge.db --limit 5
gabby knowledge evaluate --dataset ./retrieval.jsonl --database ./.gabby/knowledge.db
gabby embeddings evaluate --dataset ./embedding.jsonl --provider-config ./embedding-provider.json
gabby run ./agent.yaml "Summarize this task" --knowledge-db ./.gabby/knowledge.db
gabby evaluate ./agent.yaml --dataset ./evaluation.jsonl
mkdir -p ./dist
gabby skill pack ./skills/debugging --output ./dist/debugging.gabskill
gabby skill install ./dist/debugging.gabskill --registry ./skills
MODEL_API_KEY=... gabby run ./agent.yaml "Summarize this task"
MODEL_API_KEY=... GABBY_API_TOKEN=... gabby serve ./agent.yaml --host 127.0.0.1 --port 8787 \
  --bearer-scope agent:run --bearer-scope agent:stream \
  --run-scope agent:run --stream-scope agent:stream \
  --max-request-bytes 1000000 --request-body-timeout 30 \
  --max-response-bytes 4194304 --max-concurrent-runs 8
```

`gabby init` creates a starter YAML definition for `generic`, `research`, `data-analysis`, or
`customer-support` environments. The customer-support starter instructs the agent to ground
answers in approved support material and avoid claiming actions that tools have not confirmed.
Choose a built-in model provider with `--model-provider`, set its model identifier
with `--model`, and select the credential environment variable with `--api-key-env`. The default
model identifier is a placeholder that must be replaced. Initialization never overwrites an
existing file, and generated agents declare no tools or skills; register application-owned tools
and resources in the host before granting them to an agent.

### Host context for personalization

The `/v1` run and stream endpoints accept an optional typed `host_context` envelope. The host chooses
which authenticated caller's preferences, summary, and retrieved snippets to attach on each request.
Gabby validates and size-bounds this input, passes it to the model as untrusted reference data, and
discards it when the run ends.

```json
{
  "input": "Explain the timeout setting",
  "host_context": {
    "user_preferences": {
      "display_name": "Sam",
      "locale": "en-AU",
      "response_style": "Concise; use Australian spelling",
      "accessibility_needs": ["Avoid unexplained acronyms"]
    },
    "session_summary": {
      "summary": "The user is configuring a local service.",
      "recent_topics": ["installation", "timeouts"]
    },
    "rag_snippets": [
      {"text": "The request timeout is configured in seconds.", "source": "operations.md"}
    ]
  }
}
```

`user_preferences` supports `display_name`, `locale`, `timezone`, `response_style`,
`accessibility_needs`, and scalar `custom` values. `session_summary` has required `summary`, optional
ISO-8601 `updated_at`, and `recent_topics`. A RAG snippet has required `text` and `source`, with
optional `title`, finite `score`, and scalar `metadata`. Unknown fields are rejected; text, list
sizes, and the whole request are bounded. Python callers can pass a `HostContext` instance or a
mapping to `Agent.arun()` / `Agent.astream()` or `GabbyClient.arun()` / `GabbyClient.astream()`.

The host must authenticate the caller and select the matching data before submitting the request;
`host_context` does not establish identity or tenant authorization. Treat preferences, summaries, and
snippets as untrusted content. Snippets cannot grant tools or override runtime policies. Gabby does
not persist this envelope or isolate tenants on the host's behalf.

`gabby evaluate` runs a bounded JSON Lines regression dataset sequentially against the configured
agent. Each case is a fresh stateless request with its own `context` and `memory`. Expectations are
deterministic: exact output, required output substrings, expected parsed structured output, required
tool calls, and forbidden tool calls. `expected_structured_output` compares JSON values from
`ExecutionResult.metadata["structured_output"]`, so formatting changes in the raw response do not
break the case. A case must declare at least one expectation. The command prints JSON metrics and
exits with status 1 if any case fails, which makes it usable as a CI gate. Datasets are limited to
10 MiB and 1,000 cases; reports include checks, duration, trace IDs, and tool names, but omit
generated text and structured values.

```jsonl
{"id":"refund-policy","input":"How long does a travel refund take?","context":{"region":"US"},"expected_contains":["five business days"],"required_tools":["lookup_policy"]}
{"id":"greeting","input":"Say hello","expected_output":"Hello."}
{"id":"classification","input":"Classify this request","expected_structured_output":{"category":"billing","urgent":false}}
```

Python callers can use the same runner directly:

```python
from gabby import EvaluationCase, evaluate_agent

report = await evaluate_agent(
    agent,
    [
        EvaluationCase(
            id="classification",
            input="Classify this request",
            expected_structured_output={"category": "billing", "urgent": False},
        )
    ],
)
print(report.score)
```

`gabby knowledge evaluate` benchmarks source ranking against labeled JSONL queries. Each row has a
`query`, a list of `relevant_sources`, and optional metadata `filters` and `limit`. The JSON report
includes macro precision, recall, mean reciprocal rank, and nDCG, plus per-query results. The
dataset is bounded to 10 MiB and 1,000 cases. The command exits nonzero if the retriever fails for a
case; ranking metrics are reported for developers to set their own acceptance thresholds. This
measures retrieval against source-level judgments and does not establish intrinsic embedding quality.

```jsonl
{"id":"refund-policy","query":"How long do travel refunds take?","relevant_sources":["refund-policy.md"],"filters":{"region":"US"},"limit":5}
```

```python
from gabby import RetrievalEvaluationCase, evaluate_retriever

report = await evaluate_retriever(
    store,
    [RetrievalEvaluationCase("refund", "refund timing", ("refund-policy.md",))],
)
print(report.mean_reciprocal_rank, report.mean_ndcg)
```

For evaluating an embedding model directly, `evaluate_embeddings` ranks labeled candidate documents
by cosine similarity from the provider's vectors. Relevance grades range from 0 (irrelevant) through
5 (highly relevant); the report includes nDCG, reciprocal rank, and pairwise ranking accuracy. This
evaluates the embedding space on your labels; it does not supply or certify a universal benchmark.
Provider calls have a configurable 120-second default timeout, and failed cases are reported by
exception type while the remaining dataset continues. `EmbeddingInputFormat` supports model-specific
query and document prefixes for asymmetric encoders; custom providers can implement richer input
requirements behind the same async provider interface.

```jsonl
{"id":"refund","query":"When will the travel refund arrive?","documents":[{"id":"policy","text":"Travel refunds take five business days.","relevance":5},{"id":"hours","text":"Support is open on weekdays.","relevance":0}],"limit":2}
```

```python
from gabby import (
    EmbeddingInputFormat,
    OpenAICompatibleEmbeddingProvider,
    evaluate_embeddings,
    load_embedding_evaluation_dataset,
)

provider = OpenAICompatibleEmbeddingProvider.from_config({"model": "text-embedding-3-small"})
cases = load_embedding_evaluation_dataset("embedding.jsonl")
report = await evaluate_embeddings(
    provider,
    cases,
    input_format=EmbeddingInputFormat(query_prefix="query: ", document_prefix="passage: "),
)
await provider.aclose()
print(report.mean_ndcg, report.mean_pairwise_accuracy)
```

For repeatable CLI runs, `gabby embeddings evaluate` accepts a versioned JSON provider profile. It
supports `openai_compatible`, `gemini`, and `huggingface` providers; credentials
must come from the provider's configured host environment variable. The profile is limited to 64
KiB, rejects duplicate or unknown keys, and cannot contain inline API keys.

```json
{
  "version": 1,
  "provider": {
    "type": "gemini",
    "model": "gemini-embedding-001",
    "query_task_type": "RETRIEVAL_QUERY",
    "document_task_type": "RETRIEVAL_DOCUMENT",
    "dimensions": 768,
    "api_key_env": "GEMINI_API_KEY"
  },
  "input_format": {
    "query_prefix": "query: ",
    "document_prefix": "passage: "
  }
}
```

The command prints the same machine-readable report as the Python API and exits with status 1 if
provider calls fail. Ranking scores are descriptive; set acceptance thresholds in your own workflow
based on representative labeled data.

`gabby knowledge ingest` indexes supported files under one directory into the persistent SQLite
FTS5 store, replacing each file's prior source atomically. Its output is one JSON report with source
and chunk counts. `gabby knowledge search` returns matching text, source IDs, and metadata as JSON;
the default result limit is five and the maximum is 100. The ingestion root is explicit and the
database path is application-owned. `gabby knowledge delete ROOT SOURCE --database DB` removes
that relative source's indexed chunks, even if the file has already been removed. It applies the
same root and symlink checks as ingestion and leaves the source file untouched. Python callers can use the same `FileIngestor` and
`SQLiteFTS5Store` interfaces directly. `gabby run` and `gabby serve` attach the same store to agents
whose definitions configure knowledge, using `./.gabby/knowledge.db` by default or the
`--knowledge-db` path supplied on the command. They fail with an ingestion hint if the database is
missing. `knowledge search` also fails with that hint when the database does not exist, rather than
silently creating an empty index. `validate`, `inspect`, and `skills` use an empty in-memory retriever
unless a database path is provided, so configuration inspection does not require an indexed corpus.

The FastAPI service accepts `POST /v1/agents/{agent_name}/run` with JSON such as `{"input":"...","context":{}}`; its OpenAPI schema defines the result, trace events, and execution-error response. It returns output, metadata, and a trace ID. Set `include_trace` to `false` to omit the trace body while keeping the correlation ID; the default is `true`. The `/v1` prefix is the public API major version. `/health` remains unversioned for deployment probes. Request bodies are limited to 1,000,000 bytes and 30 seconds of total receive time by default; configure these with `create_app(max_request_bytes=..., request_body_timeout_seconds=...)` or `gabby serve --max-request-bytes --request-body-timeout`. An incomplete body that exceeds its deadline receives HTTP 408; Gabby closes the HTTP/1 connection or ends the HTTP/2 stream. The per-process execution cap defaults to eight and is configured through `create_app(max_concurrent_runs=...)` or `gabby serve --max-concurrent-runs`. Ordinary runs and non-resumable streams reserve a slot before body buffering and authentication and hold it through the response. Keyed stream requests use a separate bounded request-admission cap while buffering and authenticating; they reserve an execution slot only when starting a new run, so reattachment works when execution capacity is full. `/health` probes bypass both caps. Requests above a cap receive HTTP 429. The default CLI bind is loopback and unauthenticated. Non-loopback binds require a bearer token from `GABBY_API_TOKEN` or the variable selected with `--bearer-token-env`:

For incremental output and execution progress, clients can `POST /v1/agents/{agent_name}/stream` with the same request body. The response is Server-Sent Events (`text/event-stream`); each event has an `event:` name and a JSON envelope shaped as `{"type":"...","data":{...}}`. Setting `include_trace` to `false` omits the trace from the `completed` result while retaining its `trace_id`. This affects the response body only; an injected host `Tracer` continues to receive events.

| Event | Data fields |
|---|---|
| `run_started` | `trace_id` |
| `skill_activated` | `name`, `method` |
| `plan_created` | `plan` with a summary and bounded objective/success-criteria steps |
| `plan_updated` | `replan_number`, `plan` with the revised bounded steps |
| `text_delta` | `text` |
| `model_retry` | `attempt`, `purpose`, `delay_ms`, `error_type` |
| `tool_started` | `name`, `call_id` |
| `approval_required` | `name`, `call_id` |
| `approval_granted` | `name`, `call_id` |
| `approval_denied` | `name`, `call_id` |
| `tool_completed` | `name`, `call_id`, `duration_ms` |
| `tool_failed` | `name`, `call_id`, `error_type`, `error_code` |
| `completed` | `result` containing output, metadata, and `trace_id`; `trace` is optional |
| `error` | Sanitized `error` and stable Gabby `error_type`; unexpected provider or extension failures use `AgentExecutionError` |

Without an idempotency key, disconnecting the client cancels the in-flight run. Clients that need to
resume across a network interruption can send an `Idempotency-Key` (1–128 visible ASCII characters)
on the initial `POST`, then retry the same request with the same key and `Last-Event-ID` set to the
last event received. Keyed events include sequential SSE IDs. The key is bound to the authenticated
principal and a fingerprint of the full request, so changing the input or caller returns HTTP 409;
unknown sessions return 404. The in-memory journal is per process, retains at most 4 MiB of event
frames per run by default, admits at most four times the run capacity in sessions, and keeps
completed sessions for ten minutes by default. Configure those limits with
`create_app(max_resumable_streams=..., stream_session_ttl_seconds=...)` or `gabby serve
--max-resumable-streams --stream-session-ttl`. Reconnect to the same process during this window;
process restarts and requests routed to another worker cannot resume. A key can start a new execution
after its session expires. Providers with a `stream()` method stream model deltas; other providers
fall back to one text delta after completion.

For same-host process sharing or restart replay, configure `stream_journal_path` with a local SQLite
file. Other storage backends can implement the `StreamJournal` contract; the host owns their
consistency and lifecycle.

```sh
curl -N http://127.0.0.1:8787/v1/agents/coding-agent/stream \
  -H 'Content-Type: application/json' \\
  -d '{"input":"Review this change"}'
```

The run response remains available at `POST /v1/agents/{agent_name}/run` for clients that do not need incremental events.

```sh
GABBY_API_TOKEN=... MODEL_API_KEY=... gabby serve ./agent.yaml --host 0.0.0.0
```

For embedded ASGI hosting, `create_app(agent, authenticator=...)` requires a pluggable authenticator:

```python
from gabby import Agent, BearerTokenAuthenticator, create_app

agent = Agent.from_file("agent.yaml")
app = create_app(
    agent,
    authenticator=BearerTokenAuthenticator.from_env(
        scopes=frozenset({"agent:run", "agent:stream"})
    ),
    run_scopes=("agent:run",),
    stream_scopes=("agent:stream",),
)
```

For a containerized single-tenant service, see the [Docker Compose deployment reference](../docs/DEPLOYMENT_CONTAINER.md).
It builds the bundled research agent from the lockfile, binds the service to loopback, and applies
container resource and filesystem limits. Treat it as a starting point and set deployment-owned
image digests, credentials, ingress, and operational controls before exposing a service.
For a Kubernetes reference with a ClusterIP service, NGINX TLS ingress, pod security context, resource
bounds, and NetworkPolicy, see [the Kubernetes deployment guide](../docs/DEPLOYMENT_KUBERNETES.md).

For deployments issuing signed JWT access tokens, install with
`python -m pip install 'gabby-ai[auth]'`, then configure an issuer and audience. The
authenticator can use a fixed HTTPS JWKS endpoint or discover it from the configured OIDC issuer:

```python
from gabby import Agent, JWTBearerAuthenticator, create_app


async def create_application():
    authenticator = await JWTBearerAuthenticator.from_oidc_issuer(
        issuer="https://identity.example.com/",
        audience="gabby-api",
        scope_mapping={
            "jobs:execute": "agent:run",
            "jobs:watch": "agent:stream",
        },
    )
    return create_app(
        Agent.from_file("agent.yaml"),
        authenticator=authenticator,
        run_scopes=("agent:run",),
        stream_scopes=("agent:stream",),
    )
```

`JWTBearerAuthenticator` verifies signatures against its configured asymmetric algorithm allowlist,
issuer, audience, expiration, and subject. If the signed token includes a `tenant_id` claim, the
authenticator validates it and exposes it as `Principal.tenant_id`; it is optional, and the host must
make tenant-aware tools reject requests where the claim is required but missing. `from_oidc_issuer`
performs bounded, no-redirect metadata discovery once during async application startup and requires
an exact issuer match and an HTTPS JWKS URL. Alternatively, pass `jwks_url` directly to the
constructor. Both paths use bounded JWKS caching and do not follow redirects. The authenticator maps
a standard space-delimited `scope` claim or list-valued `scp` claim onto the authenticated
`Principal`. Set `scope_mapping` to translate identity-provider scopes to Gabby
capabilities; when set, unmapped issuer scopes are dropped. A source scope may map to one capability
or a tuple of capabilities. Without a mapping, scopes pass through unchanged. `create_app` can
require separate capabilities for run and stream routes; bearer tokens can be assigned capabilities
when the authenticator is constructed. The host remains responsible for TLS at ingress, rate limits,
and tenant separation. To check JWT revocation, inject a `TokenRevocationChecker`; this requires a
bounded ASCII `jti` claim and performs an uncached lookup for every request. Revoked tokens are
rejected, and checker failures or timeouts return HTTP 503. The host owns checker lifecycle and
distributed storage consistency. `SQLiteTokenRevocationStore` provides durable issuer-scoped local
storage and explicit expired-record cleanup without an extra dependency. The host owns its database
path, permissions, backup, and cleanup schedule; use an injected checker for distributed or
network-filesystem deployments. `gabby serve` can assign capabilities to its one bearer
token with repeatable `--bearer-scope` flags and require route capabilities with `--run-scope` and
`--stream-scope`. Because the CLI configures one shared token, it cannot assign different
capabilities to different callers; use embedded `create_app` with JWT or a custom authenticator for
per-principal scope sets. For local development, explicitly pass `allow_unauthenticated=True`;
`/health` remains public. Custom authenticators return a `Principal` with Gabby capability scopes,
an optional trusted tenant ID, or `None`. The concurrency cap is per process; it is not a distributed
limit or request-rate limiter.

`JWTBearerAuthenticator` verifies a token against a host-configured issuer and audience and passes
its verified scope set and optional tenant ID into `Principal`. The default policy engine can use
`policies.scope_rules` to restrict tools and permissions and add configured persona guidance for
matching scopes. Scope rules only narrow the agent's base `allowed_tools` and
`allowed_permissions`; an unmatched caller is denied tools when any scope rule defines tool grants.
Never trust identity or scopes supplied in the JSON request body.

```yaml
tools: [lookup_record, redact_record]
policies:
  allowed_tools: [lookup_record, redact_record]
  allowed_permissions: [records:read, records:redact]
  scope_rules:
    records:reader:
      allowed_tools: [lookup_record]
      allowed_permissions: [records:read]
      persona_instructions: Answer as a read-only support agent.
    records:operator:
      allowed_tools: [lookup_record, redact_record]
      allowed_permissions: [records:read, records:redact]
      persona_instructions: Use the operator workflow and explain each redaction.
```

The host must authenticate the token and configure token scopes or scope mapping at the auth
boundary. Scope-based persona guidance is presentation guidance; it does not grant capabilities.

## Per-run usage budgets

Configure `max_total_tokens` to stop a run after aggregated provider-reported usage crosses the
limit. For estimated USD limits, set `max_cost_usd` together with the model's
`input_cost_per_1m_tokens` and `output_cost_per_1m_tokens` rates. Gabby checks selection, planning,
and response usage after each completed provider call and stops before a subsequent run step. The
provider call that first crosses a threshold has already completed and may exceed the limit by that
call's usage; these are circuit breakers, not provider-side billing guarantees. Cost is an estimate
from the configured rates and token counts, so configure rates for the deployed model and pricing
plan. A provider that does not report the required usage causes the run to fail closed.

```yaml
policies:
  max_total_tokens: 12000
  max_cost_usd: 0.25
  input_cost_per_1m_tokens: 0.15
  output_cost_per_1m_tokens: 0.60
```

## Scoped sandbox credentials

For a sandboxed Python or custom executable tool, inject an async `ScopedCredentialProvider` into
`Agent`. Gabby resolves credentials separately for each tool call, bounds the returned environment
mapping, and passes it only to that sandbox process. The host provider should mint credentials for
the specific principal and tool, give them a short expiry and minimum permissions, and revoke or
expire them promptly. Gabby never adds values to model context or traces. Host-trusted Python
handlers should use their own resource boundary rather than this sandbox-only mechanism.

