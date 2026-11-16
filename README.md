<div align="center">

<img width="200" height="200" alt="Gabby Logo" src="https://github.com/user-attachments/assets/e586a825-3c98-4a33-adec-08aed1263315" />

# Gabby

**An Agent Runtime and Governance Framework**

Gabby enables developers to define agent capabilities, enforce tool-use policies, and execute model-driven tasks within explicit operational boundaries. It combines reusable agent and skill definitions with policy enforcement, human-in-the-loop approvals, bounded tool execution, and auditable traces—operating on a stateless runtime where the host application retains ownership of conversation history and durable state.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15741/badge)](https://www.bestpractices.dev/projects/15741)

[Documentation](docs/index.md) · [Architecture](docs/ARCHITECTURE.md) · [Quick Start](#quick-start) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

</div>

---

## About

Gabby is for developers who need a consistent way to configure agent capabilities, govern tool access, and execute agent work. A host application supplies a request and any context or memory it wants the agent to use. Gabby validates the agent definition, applies its configured runtime contracts, executes model and tool calls within the configured bounds, and returns the output and optional trace. Execution state is temporary; persistence remains with the host by default.

To give an agent a requested character, the host defines its instructions and may supply user-scoped preferences or reference material. Gabby places that guidance in the model context while runtime policies enforce tool access and execution limits; the model still generates the response, so guidance alone cannot guarantee its tone or content. Gabby does not train models, modify model weights, or load model adapters.

Gabby is complementary to agent orchestration frameworks such as LangChain, CrewAI, and IBM BeeAI. Those frameworks help developers compose agents and workflows; Gabby focuses on the runtime boundary for a configured agent: enforcing tool permissions and policies, applying approvals and verification, bounding execution, and returning traceable results. Gabby does not aim to replace an application's orchestration layer. It can serve as the governed execution runtime for an agent or workflow selected and coordinated by the host application.

The runtime is asynchronous first, with synchronous convenience methods and a FastAPI service. The same agent definition can be used from Python or through the versioned HTTP API. Model providers, tools, retrieval, policy engines, verification, and tracing are replaceable through Python interfaces.

Gabby governs agent execution inside the runtime. It is not a hosted multi-tenant control plane, an identity-management system, or the owner of application conversations and durable memory. Host applications remain responsible for caller identity, long-term state, and deployment-level isolation. See the [architecture](docs/ARCHITECTURE.md) and [threat model](docs/THREAT_MODEL.md) for these boundaries.

## Architecture

```text
Host application
  ├── owns conversation history and durable state
  └── sends request + selected context
        ↓
Gabby governance and execution runtime
  ├── resolved agent definition and skills
  ├── model provider and knowledge retrieval
  ├── policy checks and bounded tool execution
  ├── optional approvals and verification
  └── bounded result and optional trace
        ↓
Host application
```

An agent definition composes a model, environment, skills, tools, knowledge sources, instructions, policies, and verification. At construction, Gabby validates and resolves configuration into an immutable execution plan. Each run receives its own transient state and ends with a result; it does not implicitly create a chat session or persistent memory. Stateless execution is the runtime's state-ownership model; governance and controlled tool execution are the product's central purpose.

## Runtime capabilities

| Area | What Gabby provides |
|---|---|
| Execution | Async-first agent loop, sync wrappers, bounded steps and tool calls, deadlines, cancellation, optional planning and retries, per-run token and estimated cost circuit breakers |
| Models | OpenAI-compatible, Ollama, Hugging Face Inference Providers, Anthropic, Gemini |
| Skills | Reusable, composable skill definitions, dependency resolution, package and registry workflows, Ed25519 publisher signature verification required by default |
| Tools and policies | Schema-validated tools, explicit permissions, JWT scope-based RBAC, approval hooks, bounded results, runtime policy enforcement |
| Isolation | Optional per-run Docker or Podman sandboxing for supported shell/filesystem tools; configuration and host determine the effective boundary |
| Knowledge | Bounded plain-text and Markdown ingestion, SQLite FTS5 and vector stores, hybrid retrieval contracts, pluggable embeddings and reranking |
| Interfaces | Python API, FastAPI `/v1` run and SSE endpoints, generated TypeScript/Node.js, Go, and Rust clients |
| Observability | Structured execution traces and injectable event tracing |

Review the [starter bundle pack](bundles/README.md) for source-reviewed support, SQL analyst, and
sandboxed Git examples. Signed release artifacts require a maintainer-provisioned signing key.
Optional prewarmed single-use containers are described in the
[scaling guide](docs/SCALING_AND_OPERATIONS.md#prewarmed-sandbox-pools). Lightweight TypeScript,
Go, and Rust HTTP/SSE clients are available under [sdks/](sdks/README.md).

Gabby’s controls are not all enabled by default. Skill signing is optional, sandboxing requires explicit configuration, and host-provided tool handlers are trusted code unless a sandboxed implementation is supplied. Engine, operating-system, provider, and deployment behavior must be verified for the target environment. See the [threat model](docs/THREAT_MODEL.md), [scaling and operational tradeoffs](docs/SCALING_AND_OPERATIONS.md), and [current limitations](docs/PROJECT_SCORECARD.md).

## Quick start

Install the published package from PyPI with:

```sh
python -m pip install gabby-ai
```

The PyPI distribution is named `gabby-ai`; the Python import and command-line tool remain
`gabby`. Add an extra when you need optional functionality, for example
`python -m pip install "gabby-ai[server,auth]"` to run the HTTP server with JWT authentication.

To work on Gabby's source code, clone the repository and install the checkout in editable mode.
Changes made in the checkout will then be used immediately:

```sh
git clone https://github.com/undreamt-oss/gabby.git
cd gabby
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[server,auth]"
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` instead of
`source .venv/bin/activate`.

To install the full contributor environment and run the checks, use the locked development setup:

```sh
uv sync --frozen --group dev --extra server --extra auth
uv run pytest
```

Create a starter agent and inspect its configuration:

```sh
gabby init --template customer-support ./agent.yaml
gabby validate ./agent.yaml
```

Edit `agent.yaml` to choose a model and configure the agent’s instructions, skills, tools, and policies. Keep credentials outside the file. For example, set the configured provider’s API key in the environment, then run:

```sh
export MODEL_API_KEY="..."
gabby run ./agent.yaml "Summarize this request"
```

In Windows PowerShell, set the variable with `$env:MODEL_API_KEY = "..."` before running Gabby.

The generated starter uses a placeholder model and grants no tools by default. Add the correct provider credential variable and explicitly register tools in the consuming application before granting them to an agent. See the [complete user guide](docs/README_GUIDE.md) for provider-specific setup, configuration, tools, skills, knowledge, Python embedding, API serving, and deployment.

## Project status

Gabby is pre-1.0 beta software (`0.1.0b5`). Public extension APIs may change before v1.0. Local tests and static checks establish only the behaviors they exercise; they do not certify a hosted deployment, every provider/model pairing, or every container host and engine combination. Review the [release notes](CHANGELOG.md), [readiness scorecard](docs/PROJECT_SCORECARD.md), and [threat model](docs/THREAT_MODEL.md) before evaluating it for sensitive or production workloads.

## Open source and contributing

See the [runtime integrations guide](docs/INTEGRATIONS.md) for MCP, OpenTelemetry, signed audit,
and orchestration adapters.

Gabby is maintained in the [undreamt-oss](https://github.com/undreamt-oss) organization. Contributions are welcome across runtime behavior, providers, documentation, examples, accessibility, security, and developer experience.

Before contributing, read the [contributing guide](CONTRIBUTING.md), [governance](GOVERNANCE.md), and [security policy](SECURITY.md). See the [documentation index](docs/index.md) for architecture, extension contracts, operations, and release guidance.
