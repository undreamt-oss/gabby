<div align="center">

# Gabby

### A stateless agent execution runtime

Gabby executes configured AI agents within explicit runtime boundaries. It manages model calls, skills, tools, knowledge retrieval, verification, and policies for each request, then returns a result without owning the consuming application's conversation history or durable user state.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15741/badge)](https://www.bestpractices.dev/projects/15741)

[Documentation](docs/index.md) · [Architecture](docs/ARCHITECTURE.md) · [Quick start](#quick-start) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

</div>

---

## About

Gabby is for developers who need a reusable runtime boundary around agent execution. A host application supplies the request and any context or memory it wants the agent to use. Gabby runs the configured agent, enforces its runtime contracts, and returns the output and optional trace. Execution state is temporary; persistence remains with the host by default.

The runtime is asynchronous first, with synchronous convenience methods and a FastAPI service. The same agent definition can be used from Python, through the versioned HTTP API, or as a local MCP server. Model providers, tools, retrieval, policy engines, verification, and tracing are replaceable through Python interfaces.

## Architecture

```text
Host application
  ├── owns conversation history and durable state
  └── sends request + selected context
        ↓
Gabby stateless execution runtime
  ├── resolved agent definition and skills
  ├── model provider and knowledge retrieval
  ├── policy checks, tool execution, verification
  └── bounded result and optional trace
        ↓
Host application
```

An agent definition composes a model, environment, skills, tools, knowledge sources, instructions, policies, and verification. At construction, Gabby validates and resolves configuration into an immutable execution plan. Each run receives its own transient state and ends with a result; it does not implicitly create a chat session or persistent memory.

## Runtime capabilities

| Area | What Gabby provides |
|---|---|
| Execution | Async-first agent loop, sync wrappers, bounded steps and tool calls, deadlines, cancellation, optional planning and retries |
| Models | OpenAI-compatible, Ollama, Hugging Face Inference Providers, Anthropic, Gemini, and optional local Transformers adapters |
| Skills | Reusable, composable skill definitions, dependency resolution, package and registry workflows, optional Ed25519 publisher signatures |
| Tools and policies | Schema-validated tools, explicit permissions, approval hooks, bounded results, runtime policy enforcement |
| Isolation | Optional per-run Docker or Podman sandboxing for supported shell/filesystem tools; configuration and host determine the effective boundary |
| Knowledge | Bounded ingestion, SQLite FTS5 and vector stores, hybrid retrieval contracts, pluggable embeddings and reranking |
| Interfaces | Python API, FastAPI `/v1` run and SSE endpoints, optional local MCP stdio adapter |
| Observability | Structured execution traces and injectable event tracing |

Gabby’s controls are not all enabled by default. Skill signing is optional, sandboxing requires explicit configuration, and host-provided tool handlers are trusted code unless a sandboxed implementation is supplied. Engine, operating-system, provider, and deployment behavior must be verified for the target environment. See the [threat model](docs/THREAT_MODEL.md), [scaling and operational tradeoffs](docs/SCALING_AND_OPERATIONS.md), and [current limitations](docs/PROJECT_SCORECARD.md).

## Quick start

Clone the repository, then sync the locked development environment:

```sh
git clone https://github.com/orbit-projects/gabby.git
cd gabby
uv sync --frozen --group dev --extra server --extra auth
```

Create a starter agent and inspect its configuration:

```sh
uv run gabby init --template customer-support ./agent.yaml
uv run gabby validate ./agent.yaml
```

Edit `agent.yaml` to choose a model and configure the agent’s instructions, skills, tools, and policies. Keep credentials outside the file. For example, set the configured provider’s API key in the environment, then run:

```sh
export MODEL_API_KEY="..."
uv run gabby run ./agent.yaml "Summarize this request"
```

The generated starter uses a placeholder model and grants no tools by default. Add the correct provider credential variable and explicitly register tools in the consuming application before granting them to an agent. See the [complete user guide](docs/README_GUIDE.md) for provider-specific setup, configuration, tools, skills, knowledge, Python embedding, API serving, and deployment.

## Project status

Gabby is pre-1.0 alpha software. Public extension APIs may change before v1.0. Local tests and static checks establish only the behaviors they exercise; they do not certify a hosted deployment, every provider/model pairing, or every container host and engine combination. Review the [release notes](CHANGELOG.md), [readiness scorecard](docs/PROJECT_SCORECARD.md), and [threat model](docs/THREAT_MODEL.md) before evaluating it for sensitive or production workloads.

## Open source and contributing

Gabby is maintained as part of the [Undreamt](https://github.com/undreamt-hq) open-source community and follows the shared OSS maintenance baseline used by Orbit Core. Contributions are welcome across runtime behavior, providers, documentation, examples, accessibility, security, and developer experience.

Before contributing, read the [contributing guide](CONTRIBUTING.md), [governance](GOVERNANCE.md), and [security policy](SECURITY.md). See the [documentation index](docs/index.md) for architecture, extension contracts, operations, and release guidance.
