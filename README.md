# Gabby

Gabby is a domain-neutral framework for defining and executing specialized, stateless agents. An agent is a composition of a model, environment, skills, tools, knowledge sources, instructions, policies, and verification steps. Applications own conversation history and durable user state; each Gabby run receives the context it needs and returns a result.

Gabby is a Python core with an async-first runtime and synchronous convenience methods. Its FastAPI service, YAML agent and skill definitions, OpenAI-compatible, Ollama, Hugging Face Inference Providers, native Anthropic Messages and Google Gemini, and local Transformers chat adapters, explicit tool registration and policy checks, and execution traces all use the same stateless execution contract. Providers, tools, retrieval, policy engines, verifiers, and tracing can be replaced through Python interfaces.

External Python applications can use the async-first [stateless HTTP client](docs/CLIENT.md) for run results and SSE events, or follow the [SSE consumer example](examples/README.md#stateless-sse-consumer) to integrate another language or HTTP stack. The optional [OpenTelemetry example](examples/README.md#opentelemetry-event-export) shows how a host can export runtime events without adding a telemetry backend to Gabby's core. It filters exported span attributes through an explicit allowlist.

Local IDEs and other MCP hosts can launch an agent as a stdio tool server with the optional `mcp` extra; see the [MCP integration guide](docs/MCP.md). Remote deployments should use Gabby's authenticated FastAPI service.

Gabby is currently pre-1.0. Public extension APIs can change incompatibly before v1.0; see the
[extension API policy](docs/EXTENSIONS.md) and [proposed v1 compatibility contract](docs/COMPATIBILITY.md).

## Install

```sh
uv sync --frozen --group dev --extra server --extra auth
```

Use `uv run` for project commands so the locked virtual environment is selected consistently. The `server` extra installs Uvicorn for `gabby serve`; the `mcp` extra enables `gabby mcp`; embedded users can install the core package without either server process.

## Define an agent

```yaml
name: coding-helper
description: Helps with code questions using an explicitly supplied environment.

model:
  provider: openai_compatible # or ollama, huggingface, anthropic, gemini, transformers
  model: your-model-name      # for example qwen3:8b with Ollama
  # base_url: https://api.example.com/v1
  # api_key_env: MODEL_API_KEY

environment:
  type: coding

instructions: |
  Be concise. Explain assumptions and report what you verified.

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
```

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

Optional local supervised fine-tuning can produce PEFT LoRA adapters using `gabby train`; this is
separate from runtime agent specialization. See the [local adapter training guide](docs/TRAINING.md)
for the bounded JSONL format, install steps, and a command example.

For local Hugging Face Transformers inference, install `gabby-agent-runtime[transformers]` and a
PyTorch 2.5+ build selected for your CPU/GPU platform. Set `provider: transformers`; model weights are
loaded lazily on first execution. The default device is CPU. Configure `revision` to pin a Hub
commit, `local_files_only: true` to prohibit model downloads, and `max_input_tokens` /
`max_new_tokens` to bound each prompt and generation. Gabby disables custom model code and accepts
only safetensors weights. Models need a Transformers chat template; agents with tools also need a
compatible structured response parser. This provider returns a complete response, so HTTP SSE does
not receive token-by-token deltas from local generation. Install PyTorch using the
[official platform selector](https://pytorch.org/get-started/locally/) because CPU, CUDA, and other
accelerator builds differ.

To load a previously trained PEFT adapter, also install
`gabby-agent-runtime[transformers-adapters]` and set `adapter_id` in the model configuration.
`adapter_id` can be a local adapter directory or a Hub repository; `adapter_revision` optionally
pins the adapter revision separately from the base model's `revision`. The adapter is loaded for
inference only and Gabby requires safetensors adapter weights. Install the PyTorch build for your
platform before the adapter extra so the package resolver reuses that build. This provider does not
train or update weights; the separate `gabby train` command can produce a LoRA adapter locally.
Keep the base model and adapter compatible, and pin Hub revisions for reproducible deployments.
Adapter files and their producer are separate trust inputs from the base model configuration.

```yaml
model:
  provider: transformers
  model: HuggingFaceTB/SmolLM3-3B
  revision: <commit-sha>
  # adapter_id: organization/adapter-repository
  # adapter_revision: <adapter-commit-sha>
  device: cpu
  max_input_tokens: 16384
  max_new_tokens: 1024
```

Opt-in live completion and streaming acceptance checks for Ollama and Hugging Face are documented
in [`tests/integration/README.md`](tests/integration/README.md). They use credentials from the host
environment and are skipped by the regular test suite unless explicitly enabled.

An optional `skills/debugging/skill.yaml` can define a reusable skill:

```yaml
name: debugging
version: 1.0.0
description: Diagnose and resolve software failures.
instructions: |
  Reproduce the failure, inspect evidence, identify the root cause, and propose
  the smallest safe fix. State which checks were actually run.
tools: []
triggers: [dependency, regression, failing test]
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
version. Package checksums detect file corruption. The optional `skill-signing` extra adds detached
Ed25519 signatures verified against host-managed public keys. Skill contents remain untrusted
instructions and should be reviewed before attaching them to an agent. Static HTTPS registries support
catalog search, exact version discovery, and signature-required installation; see the
[skill registry guide](docs/SKILL_REGISTRY.md). You can also place a package directory beside an
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
  --require-signature \
  --trusted-key gabby-release-2026=./trusted-keys/gabby-release.pub
gabby skill audit --registry ./skills --revoked-key old-release-key
gabby skill uninstall debugging 1.0.0 --registry ./skills --yes
```

The signature is a detached `ARCHIVE.sig` file and covers the exact archive digest and key ID.
Unsigned local installs remain available when `--require-signature` is omitted. Each install records
its signer ID and archive digest for inventory; `gabby skill audit` flags revoked, unknown, or
invalid records. This metadata is advisory and does not disable installed skills or prove their
current contents are unchanged. `gabby skill uninstall` removes one validated exact skill version
and requires `--yes`. Hosts distributing skills should keep private keys in a KMS or
secret manager and manage trusted-key distribution and revocation. See the
[skill package guide](docs/EXTENSIONS.md#portable-skill-packages) and
[registry operations guide](docs/SKILL_REGISTRY.md).

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
- A **data agent** that queries approved datasets through registered SQL or analysis tools. The opt-in [`sqlite_query_tool()`](docs/SQLITE_DATA.md) provides bounded, read-only local SQLite queries. The sandboxed [`python_run_tool()`](docs/PYTHON_TOOL.md) runs analysis code in the configured per-run container; other database access and authorization belong in application tools and their environment.
- A **coding agent** that uses Gabby’s built-in container-backed shell and workspace tools.

These are agent configurations and integrations, not four prebuilt products shipped by Gabby. Coding is the most turnkey initial path because shell and filesystem are its first built-in tools. Other domains can use the same model, skill, policy, execution, and API contracts, with application-provided tools.

Agents that feed other software can declare a JSON Schema for their final output. Gabby validates
the result and exposes the parsed value in `result.metadata["structured_output"]`; the same value is
available in `/run` and SSE completion metadata. Streaming calls hold back response text until it
passes validation. See the [structured output guide](docs/STRUCTURED_OUTPUT.md).

Runnable support, research, and data-analysis examples use an offline mock model and synthetic
application tools. Run them with `uv run --frozen python examples/domain_agents.py`. The separate
`examples/agent_composition.py` demonstrates bounded parent/child delegation and correlated traces;
both need no credentials or network. See the [examples guide](examples/README.md). The public Python
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
and closes every agent in the composition. See the [extension contract reference](docs/EXTENSION_CONTRACTS.md).

## Tool trust and isolation

Tools run as `host_trusted` by default. Their handlers execute inside the Gabby process with its host operating-system permissions; registering a handler grants the agent a way to invoke that code. `Tool.execution` exposes this classification. The built-in shell and filesystem tools declare `execution: sandboxed` and run in the configured per-run container. Set `policies.require_sandbox: true` to reject any declared tool without a sandboxed implementation before the model is called. This setting does not move arbitrary Python handlers into a container.

See the [threat model](docs/THREAT_MODEL.md) for trust boundaries, current controls, residual risks,
and the evidence required before hosted-production claims. See the [operations guide](docs/OPERATIONS.md)
for service, storage, streaming, and sandbox deployment responsibilities.

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
[ADR 0053](docs/architecture/adr/0053-required-tool-output-schemas.md) and
[ADR 0054](docs/architecture/adr/0054-stable-tool-error-codes.md).
See [ADR 0013](docs/architecture/adr/0013-tool-result-size-limits.md) for the contract and tradeoffs.

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

See [ADR 0025](docs/architecture/adr/0025-host-owned-tool-approval.md). Approval decisions are
per-call and per-run; Gabby does not retain them. The [approval guide](docs/APPROVALS.md) shows
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
  require_sandbox: true
  max_steps: 8
  max_tool_calls: 64
  timeout_seconds: 120

sandbox:
  engine: docker
  adapter: cli
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
[ADR 0003](docs/architecture/adr/0003-container-sandbox.md).

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

See [ADR 0030](docs/architecture/adr/0030-scoped-tool-context.md) for the trust boundary and
compatibility contract. Context-enabled handlers also receive a cooperative cancellation token;
see [ADR 0037](docs/architecture/adr/0037-cooperative-tool-cancellation.md). A synchronous handler
can wait on `gabby.cancellation.wait(timeout)` from its worker thread. Handlers that ignore the
signal may continue after the run returns, so use container-backed tools when forced termination
is required.

## Knowledge retrieval

The retriever is injected by the consuming application, so document storage and lifetime remain
outside an agent run. Gabby includes an in-memory BM25 retriever for fixtures and a persistent
SQLite FTS5 store for local or single-service use:

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
source-scoped replacement. Markdown, plain-text, YAML, TOML, HTML, and JSON files can be ingested with deterministic
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

For shared lexical knowledge across service workers, `PostgresKnowledgeStore` uses a host-owned
async PostgreSQL pool. Apply `sql/postgres_knowledge.sql` with the application's migration system,
then inject the store into `FileIngestor` and the agent. The optional `postgres` extra supplies
asyncpg; Gabby does not own or close the pool:

```python
from gabby import Agent, FileIngestor, PostgresKnowledgeStore

knowledge = PostgresKnowledgeStore(host_owned_asyncpg_pool)
await FileIngestor(knowledge, root="./docs").ingest_directory()
agent = Agent.from_file("agent.yaml", retriever=knowledge)
```

This backend provides PostgreSQL full-text retrieval, exact JSON metadata filters, and generation
staging with fencing. For multi-instance hybrid indexing, pair it with a shared generation-aware
vector store and `PostgresGenerationManifestStore`; all three must use the same manifest generation
contract.

`FileIngestor` accepts UTF-8 `.md`, `.markdown`, `.txt`, `.log`, `.adoc`, `.asciidoc`, `.rst`, `.yaml`, `.yml`, `.toml`, `.eml`, `.mbox`, `.ics`, `.vcf`, `.rss`, `.atom`, `.html`, `.htm`, `.json`, `.jsonl`,
`.ipynb`, `.xml`, `.opml`, `.csv`, `.rtf`, `.docx`, `.pptx`, `.odp`, `.epub`, `.odt`, `.ods`, and `.xlsx` files below its configured root, plus
optional `.parquet` with `gabby-agent-runtime[parquet]`, `.pdf`, and raster image OCR.
`MarkupTextParser` indexes reStructuredText (`.rst`) and AsciiDoc (`.adoc`, `.asciidoc`) as bounded
UTF-8 source, preserving markup for retrieval. It never evaluates directives or follows include
paths; limits default to 10 MiB input and 10 million extracted characters.
`ICalendarTextParser` indexes each `VEVENT`, `VTODO`, and `VJOURNAL` as a separate cited page in source order,
unfolds folded lines, decodes RFC 5545 text escapes, and exposes type-specific identifiers and
selected fields as filterable metadata. Task pages include due/completed times, status, and priority
when present; journal pages include summary, start, description, status, and UID when present.
Nested alarms and other subcomponents are excluded. The parser requires valid UTF-8 and balanced
calendar components, and defaults to 10 MiB input, 1,000 events, 1,000 tasks, 1,000 journals,
1,000 attendees per record, 100,000 content lines, and 10 million extracted characters.
`VCardTextParser` indexes `.vcf` files as one cited page per contact, supporting UTF-8 vCard
2.1, 3.0, and 4.0 records with folded lines and escaped text. Names, organizations, email addresses,
telephone numbers, URLs, and other selected fields are searchable; selected identifiers and contact
fields are also available as exact metadata filters. It does not fetch URLs or decode quoted-printable
values. Input defaults to 10 MiB, 1,000 cards, 100,000 lines, 1,000 properties per card, and 10
million extracted characters.
`RSSAtomTextParser` indexes RSS 1.0/2.0 and Atom feed entries as individual cited pages from `.rss`
and `.atom` files; `.xml` files with RSS, RDF, or Atom roots are detected by `XMLTextParser` too.
`OPMLTextParser` indexes nested `.opml` subscription and outline entries with category paths and
safe URL metadata; `.xml` files with an OPML root are detected by `XMLTextParser` too.
`JSONTextParser` detects JSON Feed 1.0/1.1 in `.json` files and indexes each item as a cited page;
ordinary JSON files continue to preserve their source text. Feed links are citations only and are
never fetched.
Entry titles, links, dates, authors, and categories become searchable text or filterable metadata;
HTML summaries are reduced to visible text, and only absolute HTTP(S) links without embedded
credentials are retained as citations. The parser does not fetch feeds or follow entry links.
It rejects DTDs and entity declarations and bounds input, output, item count, XML depth, elements,
attributes, and categories.
`MboxTextParser` indexes separator-framed `.mbox` archives as one cited page per message and
reuses `EmailTextParser` so MIME attachments stay excluded. It requires a valid `From ` envelope
before each message, does not fetch content, and bounds total input, line count, message count,
aggregate output, MIME parts, and MIME nesting. Content-Length-framed mbox variants are rejected;
separator-framed archives should escape body lines beginning with `From ` according to mbox rules.
`EmailTextParser` indexes selected message headers and text bodies, skips attachments, and prefers
plain text over duplicate HTML alternatives. For HTML-only messages it extracts visible text. It
indexes From, To, Cc, Date, Subject, and Message-ID headers, and bounds input to 10 MiB, MIME parts
to 1,000, nesting to 32 levels, and output to 10 million characters by default. `HTMLTextParser`
preserves the title, headings, paragraphs, and list boundaries,
and omits script, style, template, SVG, and hidden subtrees. `JSONTextParser` validates strict JSON,
rejects duplicate object keys and excessive nesting or token counts, and preserves the source text
for retrieval. `YAMLTextParser` renders bounded `.yaml` and `.yml` documents as searchable dotted
key paths and list indexes, rejecting duplicate keys, alias cycles, merge keys, unsafe tags, multiple
documents, and excessive structure or output. `TOMLTextParser` renders bounded `.toml` documents as
searchable dotted key paths and list indexes, normalizes TOML dates and times to ISO-8601 strings,
and rejects malformed input, excessive structure, and oversized output. `JSONLinesTextParser` validates
one strict JSON value on each non-blank line, retains physical line numbers in the indexed text, and
caps records, record bytes, and total rendered bytes.
It uses the same depth and token checks as `JSONTextParser`. `CSVTextParser` expects a header row, rejects duplicate headers and inconsistent
records, and renders bounded rows as labeled fields so retrieval keeps each value tied to its column.
`ParquetTextParser` is available with `pip install 'gabby-agent-runtime[parquet]'`. It reads flat
Parquet tables in bounded batches, renders field-labeled rows with row-range citations, and applies
limits to input, row and column counts, row groups, aggregate uncompressed data, cell and row size,
page count, and total rendered output. Nested and binary columns are rejected so results stay predictable
and searchable; flatten those fields before ingestion when needed.
`NotebookTextParser` validates the required Jupyter notebook v4 structure (including v4.5+ cell IDs),
indexes source in cell order with one-based cell citations, and excludes cell outputs and execution
metadata. It bounds input, cell count, source bytes per cell, total rendered output, JSON depth, and
tokens. In `FileIngestor`, its cell limit follows the configured `max_pages` limit. It uses no
notebook dependency and never executes notebook code. See the
[Jupyter notebook format specification](https://nbformat.readthedocs.io/en/latest/format_description.html).
It defaults to 100,000 rows, 1,000 columns, and 10 MiB of rendered text; JSON defaults to depth 64 and
200,000 tokens. `XMLTextParser` streams UTF-8 XML into path-labeled text and attributes without
requiring another package. It rejects DTD/entity declarations and bounds input/output bytes, nesting,
elements, and attributes. Defaults are 10 MiB input, 12 MiB rendered output, 250,000 elements,
depth 128, and 256 attributes per element. `DOCXTextParser` reads paragraph and table text in document order with tabs and line
breaks preserved. `RTFTextParser` extracts visible text, Unicode escapes, paragraph breaks, and tabs;
it skips known non-body destinations, hidden text, and bounded binary payloads. It uses the standard
library and defaults to 10 MiB input, 128 nested groups, 1 million control words, and 10 million
extracted characters. It does not render layout, images, or embedded objects. See the
[Microsoft RTF 1.9.1 specification](https://interoperability.blob.core.windows.net/files/Archive_References/%5BMSFT-RTF%5D.pdf).
`DOCXTextParser` reads paragraph and table text in document order with tabs and line breaks
preserved. It requires no additional package, extracts no images or deleted revision text,
and has no page attribution. `PPTXTextParser` reads slide text in presentation order and assigns
one-based slide numbers as `page_number` citations; it does not extract notes, images, or chart data.
Its defaults cap the archive at 10 MiB, uncompressed content at 32 MiB, each slide XML part at 4 MiB,
slides at 100, and extracted text at 10 million characters. `ODPTextParser` reads OpenDocument
Presentation slide paragraphs and headings in source order, assigning slide-number citations and
retaining slide names as filterable metadata. It uses no additional package and rejects encrypted
archives, unsafe or duplicate paths, DTD/entity declarations, and oversized content; defaults cap the
archive at 10 MiB, expanded content at 32 MiB, content XML at 16 MiB, XML elements at 250,000,
slides at 100, and extracted text at 10 million characters. Notes, images, charts, and embedded
media are not extracted. Both OpenXML parsers reject encrypted archives, unsafe or duplicate entry
names, DTD/entity declarations, and inputs over their limits.
DOCX defaults cap archives at 10 MiB, uncompressed content at 32 MiB,
package metadata XML at 1 MiB per part, and document XML at 16 MiB,
250,000 XML elements, 100,000 paragraphs, and 10 million extracted characters. `EPUBTextParser`
reads visible HTML/XHTML chapters in OPF spine order and assigns one-based chapter-order citations
as `page_number`. It uses no extra package, rejects unsafe archive paths and remote spine references,
and caps the archive at 10 MiB, expanded content at 32 MiB, metadata parts at 1 MiB, chapters at
4 MiB each, chapter count at 1,000, and extracted text at 10 million characters. It does not
extract images, stylesheets, or embedded media. `ODTTextParser` reads OpenDocument Text headings,
paragraphs, tables, tabs, line breaks, and repeated spaces in document order. It uses no extra package,
rejects encrypted archives, unsafe or duplicate paths, DTD/entities, and oversized content; defaults
cap the archive at 10 MiB, expanded content at 32 MiB, content XML at 16 MiB, XML elements at
250,000, paragraphs at 100,000, and extracted text at 10 million characters. `ODSTextParser` renders non-empty cells from each sheet
in source order as sheet- and row-labeled column values, including bounded repeated rows and
columns. It rejects encrypted archives, unsafe or duplicate paths, DTD/entities, and oversized
content; defaults cap the archive at 10 MiB, expanded content at 32 MiB, content XML at 16 MiB,
XML elements at 250,000, rows at 100,000, columns at 1,000, cells at 1 million, sheets at 1,000,
and extracted text at 10 million characters. It does not recalculate formulas or extract embedded
media. `XLSXTextParser` follows workbook sheet order, resolves bounded worksheet/shared-string parts,
and renders cached or inline values with sheet, row, and column labels. It rejects external or unsafe
relationships, encrypted archives, duplicate paths, DTD/entities, malformed XML, and oversized
input. Defaults cap the archive at 10 MiB, expanded content at 32 MiB, worksheet XML at 4 MiB per
sheet, shared-string XML at 16 MiB, sheets at 1,000, rows at 100,000, columns at 1,000, cells and
shared strings at 1 million, and extracted text at 10 million characters. Formula values are not
recalculated. Tune limits by passing custom parsers through
`FileIngestor(parsers=...)`; that argument replaces the full default parser set.
Install the optional `ocr` extra (`pip install 'gabby-agent-runtime[ocr]'`) and the Tesseract
executable to OCR `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`, `.tiff`, and `.webp` images. The default
`TesseractOCRBackend` processes bounded image frames without writing them to disk and records
one-based page numbers. It defaults to at most 100 frames, 40 million decoded pixels total, and
30 seconds per frame; extracted text uses the common 10 million character limit. Configure
`max_ocr_pages`, `max_image_pixels`, and `ocr_timeout_seconds` on `FileIngestor`, or inject an
`OCRBackend` into `ImageOCRParser` for a different engine. The Python packages are optional; the
Tesseract executable must also be installed on the host. For image-only PDF pages that contain
embedded raster scans, install the optional `pdf-ocr` extra and pass
`pdf_ocr_backend=TesseractOCRBackend(language="eng")` to `FileIngestor`; set `language` to a
Tesseract language code installed on the host. Gabby OCRs only blank-text pages, using their first
embedded image or, when none exists, a bounded PDFium rendering of the page. Vector rendering is
opt-in with OCR and runs at a configurable scale within the existing pixel budget. OCR keeps the
original PDF page number for citations and applies the configured page, pixel, image-byte, text, and
Tesseract timeout limits. The `pdf-ocr` extra includes PDFium; the Tesseract executable and language
data must also be installed on the host. Rendering runs synchronously in process and cannot be
interrupted by the OCR timeout, so isolate parsing when processing hostile PDFs.
Install the optional `pdf` extra (`pip install 'gabby-agent-runtime[pdf]'`) to ingest `.pdf` files.
PDF chunks include one-based `page_number` metadata for citations. OCR is opt-in; without a configured
backend, image-only PDF pages remain empty. PDF ingestion defaults to at most 1,000 pages, 10 million extracted characters, 4 MiB
of decoded page-content output per stream, and 32 MiB aggregate page content per PDF. One directory
import is limited to 10,000 files and 1 GiB total, in addition to the
10 MiB per-file limit. `max_total_bytes` also applies to a single-file import if configured below
the per-file limit. Configure `max_files`, `max_total_bytes`, and `max_file_bytes` on the ingestor
to match the application; also configure `max_pages`, `max_extracted_chars`, and
`max_content_stream_bytes` and `max_total_content_stream_bytes` for PDF content.
Directory imports that exceed their file and byte bounds are rejected during preflight before
indexing begins. Call `delete_file(path)` after removing a file to delete its indexed source. Each
source replacement is atomic, while a whole directory import is not a single transaction. Parsers
and the chunker are replaceable. `TextFileIngestor` remains an alias of `FileIngestor` for
backward compatibility. `KnowledgeStore`, `EmbeddingProvider`, `VectorStore`, and `Reranker` are replaceable
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
guide](docs/EXTENSIONS.md), [Cohere's rerank API reference](https://docs.cohere.com/v2/reference/rerank),
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

For multi-worker or multi-host indexing, use the PostgreSQL adapters with one host-owned asyncpg
pool and a PostgreSQL server where pgvector is installed. Apply the knowledge, vector, and generation
manifest migrations through the application's migration system. Keep all three tables in the same
PostgreSQL consistency domain:

```python
from gabby import (
    HybridIndexCoordinator,
    PostgresGenerationManifestStore,
    PostgresKnowledgeStore,
    PostgresVectorStore,
)

coordinator = HybridIndexCoordinator(
    lexical=PostgresKnowledgeStore(pool),
    embeddings=embedding_provider,
    vectors=PostgresVectorStore(pool, dimensions=1536),
    manifest=PostgresGenerationManifestStore(pool),
)
await coordinator.replace_source("docs/guide.md", documents)
```

`PostgresVectorStore` uses pgvector's HNSW cosine index and persists the configured embedding
dimension, so every worker must use the same embedding model and dimensions for that index. Its
HNSW configuration supports 1–2,000 dimensions. It enables strict-order iterative scans for filtered
retrieval and exposes bounded `ef_search` and `max_scan_tuples` controls. Install pgvector 0.8.0 or
newer to support iterative scans. The host owns pgvector installation, the pool, credentials,
backups, and migrations.

For a multi-instance deployment, install `gabby[postgres]`, inject a host-owned asyncpg pool, and
use `PostgresGenerationManifestStore`. Apply Gabby's SQL migration through your normal migration
system. The manifest provides shared lease and activation coordination; each service still needs
shared lexical and vector stores that honor the generation fencing contract. See the
[extension guide](docs/EXTENSIONS.md#hybrid-index-generation-contract) for ownership and consistency
requirements.

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

For local semantic retrieval, `TransformersEmbeddingProvider` loads an encoder model through the
optional `transformers` extra and the host-selected PyTorch build. It uses attention-mask-aware mean
pooling by default, supports CLS pooling, normalizes vectors for cosine search, and keeps inference
off the event loop. It does not apply model-specific retrieval prompts or Sentence Transformers
modules; use a custom `EmbeddingProvider` when a model requires those. Pin `revision` for reproducible
deployments, and use `local_files_only=True` when the model must already be present in the local
cache:

```python
from gabby import TransformersEmbeddingProvider

embeddings = TransformersEmbeddingProvider(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="<model-commit-sha>",
    local_files_only=True,
    device="cpu",
)
vectors = await embeddings.embed(["How does authentication work?"])
await embeddings.aclose()
```

The provider caps input at 4 MiB, defaults to 512 tokens per text and batches of 16, caps an
eight-bytes-per-value vector output estimate at 4 MiB by default, and applies a 120-second timeout.
These limits can be configured on the provider.
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
gabby mcp ./agent.yaml --knowledge-db ./.gabby/knowledge.db
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
supports `openai_compatible`, `gemini`, `huggingface`, and optional local `transformers` providers; credentials
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
file. For multiple hosts, install `gabby-agent-runtime[postgres]`, apply
`sql/postgres_stream_journal.sql` through your migration system, then inject
`PostgresStreamJournal(pool)` using `create_app(stream_journal=...)` or `serve(stream_journal=...)`.
The migration is also packaged at `gabby/sql/postgres_stream_journal.sql` for installed consumers.
Gabby does not own or close the host's PostgreSQL pool.

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

For a containerized single-tenant service, see the [Docker Compose deployment reference](docs/DEPLOYMENT_CONTAINER.md).
It builds the bundled research agent from the lockfile, binds the service to loopback, and applies
container resource and filesystem limits. Treat it as a starting point and set deployment-owned
image digests, credentials, ingress, and operational controls before exposing a service.
For a Kubernetes reference with a ClusterIP service, NGINX TLS ingress, pod security context, resource
bounds, and NetworkPolicy, see [the Kubernetes deployment guide](docs/DEPLOYMENT_KUBERNETES.md).

For deployments issuing signed JWT access tokens, install with
`python -m pip install 'gabby-agent-runtime[auth]'`, then configure an issuer and audience. The
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
issuer, audience, expiration, and subject. `from_oidc_issuer` performs bounded, no-redirect metadata
discovery once during async application startup and requires an exact issuer match and an HTTPS
JWKS URL. Alternatively, pass `jwks_url` directly to the constructor. Both paths use bounded JWKS
caching and do not follow redirects. The authenticator maps a standard space-delimited `scope` claim or list-valued `scp` claim onto the
authenticated `Principal`. Set `scope_mapping` to translate identity-provider scopes to Gabby
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
`/health` remains public. Custom authenticators return a `Principal` with Gabby capability scopes or
`None`. The concurrency cap is per process; it is not a distributed limit or request-rate limiter.

## Current boundaries

- The included OpenAI-compatible adapter supports OpenAI-style chat completions. The Ollama provider uses its compatible `/v1` endpoint by default; the adapter has mocked transport coverage, while live compatibility with specific Ollama model tool-call formats is not yet acceptance-tested. The built-in Hugging Face adapter targets Inference Providers' OpenAI-compatible chat endpoint and has mocked contract coverage; live compatibility depends on the selected model/provider combination. Local inference is available through the optional `TransformersProvider`; a CPU completion smoke passed on a pinned SmolLM2-135M-Instruct checkpoint. Structured tool-call templates, model quality, throughput, and accelerator support remain unverified. Gemini GenerateContent has mocked REST and end-to-end streaming tool-cycle coverage, and Gemini text embeddings have mocked batch and hybrid role-routing coverage; no live Gemini acceptance has been run. Multimodal model input/output and Vertex AI are outside these adapters.
- Async tool handlers run on the event loop; synchronous handlers are dispatched to a worker thread. Cancellation cannot forcibly stop synchronous code that has already started. The built-in shell and filesystem tools use Docker/Podman CLI or Docker-compatible API adapters through a per-run container with networking disabled. The Windows API adapter uses Docker Desktop's local named pipe and the optional `windows-sandbox` dependency. Filesystem operations check every path component and reject symlinks before archive reads or writes. The engine, image, and workspace access come from agent config. Native Windows containers require Docker Hyper-V isolation; Podman uses Linux containers. Live checks currently cover Docker and Podman on Linux through CLI and API adapters; the native Windows acceptance suite now covers CLI and API but needs a Windows host run before that combination is considered live-verified. Other host/engine/adapter combinations remain experimental until live acceptance checks cover isolation, resource limits, mounts, networking, and cleanup. Other injected handlers are host-trusted by default; `policies.require_sandbox: true` rejects them unless a sandboxed implementation is supplied.
- The API run and stream endpoints require authentication by default. Built-in bearer-token and optional issuer-bound JWT authenticators support one configured tenant per service; embedded hosts and `gabby serve` can require route-specific `run_scopes` and `stream_scopes`. The JWT adapter can map issuer scopes to capabilities and drops unmapped scopes in mapping mode. `SQLiteTokenRevocationStore` provides durable local revocations, while distributed propagation and multi-tenant authorization remain host responsibilities. OIDC issuer discovery is available through the async startup factory. Public health checks, TLS, request-rate limits, and process supervision still need deployment-specific controls. A per-process execution cap returns 429 when all run slots are occupied.
- The in-memory BM25 and persistent SQLite FTS5 retrievers are lexical. Hybrid retrieval includes a local exact-cosine SQLite vector store and OpenAI-compatible, Gemini, and Hugging Face feature-extraction embedding adapters; Gemini supports role-specific query/document settings. The host still selects and pays for the embedding model/provider. Exact search is for small or moderate corpora. The optional Transformers embedding adapter does not bundle model weights. Bounded raster-image OCR and opt-in OCR for scanned and vector-only PDF pages are available through optional extras; document formats beyond Markdown, plain text, logs, YAML, email, vCard, HTML, JSON, JSON Lines, Jupyter notebooks, CSV, RTF, DOCX, PPTX, EPUB, ODT, ODS, XLSX, PDF, and the listed image formats remain out of scope. Managed vector services are not included; PostgreSQL/pgvector provides an optional self-hosted shared vector backend. Cohere and Jina reranking have bounded built-in HTTP adapters; Cohere has an opt-in live acceptance check.
- Verification callbacks are supported by the runtime; agent YAML does not execute arbitrary verification commands.
- “Specialization” here means runtime composition (instructions, skills, tools, and knowledge), not changing model weights.

## Contributing and project policies

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development environment and quality gates,
[GOVERNANCE.md](GOVERNANCE.md) for project decision-making, [SECURITY.md](SECURITY.md) for
private vulnerability reporting, [HANDOFF.md](HANDOFF.md) for maintainer-owned external setup, and
[CHANGELOG.md](CHANGELOG.md) for user-visible changes.
