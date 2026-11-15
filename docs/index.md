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

# Gabby Documentation

This is the entry point for Gabby's user, architecture, operator, and contributor documentation.
Gabby is an Agent Runtime and Governance Framework: it lets developers define agent
capabilities, govern tool use, and execute requests within explicit runtime contracts. Execution is
stateless between requests; the consuming application owns conversation history and durable memory.
It is designed to complement orchestration frameworks such as LangChain, CrewAI, and IBM BeeAI:
those frameworks can coordinate agents and workflows, while Gabby supplies governed execution for
configured agent requests. Gabby is not intended to replace the host application's orchestration layer.

Gabby is maintained in the `undreamt-oss` GitHub organization. The README introduces the project,
architecture documents explain ownership and contracts, focused guides explain capabilities, and the
roadmap and scorecard separate implementation from release evidence.

## Start here

- [Project overview and quick start](../README.md) — understand Gabby's governance and execution
  boundary and run the offline starter flow.
- [Complete user guide](README_GUIDE.md) — install Gabby, configure providers, define agents, use
  skills, tools, and knowledge, and use the Python and HTTP interfaces.
- [Examples](../examples/README.md) — runnable offline examples and a hosted-service example.
- [CLI and API guide](../README.md#cli-and-api) — use the `gabby` developer command line and
  local service.
- [Python HTTP client](CLIENT.md) — call the stateless HTTP API from another Python application.
- [Multi-language HTTP/SSE clients](../sdks/README.md) — call Gabby from TypeScript/Node.js, Go,
  or Rust.
- [Runtime integrations](INTEGRATIONS.md) — connect MCP servers, OpenTelemetry, signed audit
  evidence, LangGraph, CrewAI, and n8n.

## Architecture and contracts

- [Architecture overview](ARCHITECTURE.md) — component ownership, request lifecycle, state
  boundaries, policy enforcement, and deployment interfaces.
- [Extension contracts](EXTENSION_CONTRACTS.md) — interface responsibilities, lifecycle ownership,
  concurrency, and failure behavior.
- [Extension guide](EXTENSIONS.md) — build providers, tools, retrievers, policies, and other
  replaceable components.
- [Host-orchestrated multi-agent pipelines](HOST_ORCHESTRATED_PIPELINES.md) — coordinate separate
  single-agent Gabby services with host-owned state, caller scopes, tenant retrieval, and recovery.
- [API stability policy](API_STABILITY.md) — current pre-1.0 compatibility rules.
- [Proposed compatibility contract](COMPATIBILITY.md) — candidate public contract for a future 1.0
  release; it is not in force during 0.x.
- [Architecture decision records](architecture/adr/README.md) — accepted design choices and their
  consequences.

## Capabilities

- [Starter skill bundles](../bundles/README.md) — reviewed source templates and signed release
  bundle installation.
- [Skill registry](SKILL_REGISTRY.md) — skill structure, exact versions, packages, signatures,
  trust, and registry operations.
- [Structured output](STRUCTURED_OUTPUT.md) — JSON Schema contracts for agent responses.
- [Approval handling](APPROVALS.md) — host-owned approval decisions and audit records.
- [SQLite data tool](SQLITE_DATA.md) — bounded read-only SQL against a host-selected database.
- [Sandboxed Python tool](PYTHON_TOOL.md) — execute bounded Python in a per-run container.

## Operations and security

- [Threat model](THREAT_MODEL.md) — trust boundaries, implemented controls, and residual risks.
- [Operations guide](OPERATIONS.md) — authentication, resource limits, storage, tracing, and
  deployment responsibilities.
- [Scaling and operational tradeoffs](SCALING_AND_OPERATIONS.md) — sync callbacks, vector scale,
  streaming, platform gates, and schema prototyping.
- [Container deployment](DEPLOYMENT_CONTAINER.md) and [Kubernetes reference](DEPLOYMENT_KUBERNETES.md).
- [Security policy](../SECURITY.md) — private vulnerability reporting.

## Contributing and release readiness

- [Contributing](../CONTRIBUTING.md) — development environment, required checks, and change
  expectations.
- [AI usage guide](AI_USAGE.md) — responsible use of AI tools while contributing, including review,
  validation, and data protection.
- [Repository governance](REPOSITORY_GOVERNANCE.md) — Gabby's contribution safeguards and repository
  settings maintainers must verify.
- [Release procedure](RELEASING.md) — tag validation, build, attestations, and PyPI publishing.
- [Project roadmap](../ROADMAP.md) and [readiness scorecard](PROJECT_SCORECARD.md) — current
  implementation evidence and open release gates.

## Release status

Gabby is pre-1.0 beta software (`0.1.0b5`). Local tests and static checks establish only the
behaviors they exercise; they do not certify a hosted deployment, every model/provider combination,
or every container host and engine. Review the threat model and scorecard before using sensitive
data or making production claims. See the
[0.1.0 beta release notes](../CHANGELOG.md#planned-010-beta-010b5) for the scope of this pre-release.
