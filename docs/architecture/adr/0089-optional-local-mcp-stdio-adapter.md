# ADR 0089: Optional local MCP stdio adapter

- Status: accepted
- Date: 2026-10-03

## Context

IDE and automation hosts increasingly consume capabilities through the Model Context Protocol.
Gabby should make its stateless agent execution usable by those hosts while preserving its
authenticated FastAPI boundary for remote deployments and avoiding an MCP dependency for core
runtime users.

## Decision

Provide an optional adapter backed by the official Python MCP SDK. The adapter exposes one
`run_agent` tool over stdio. Each call supplies `input` and optional `context`, `memory`, and
`metadata`, and invokes one `Agent.arun()` execution. It returns the output, result metadata, and
trace ID as structured content. Serialized responses are bounded. The MCP server lifespan owns the
agent lifecycle and drains/closes it when the process exits; injected provider and host-owned
resources retain the existing ownership rules.

Ship a `mcp` package extra and `gabby mcp AGENT.yaml` command. Keep the adapter lazily imported so
the core package remains usable without the SDK. The first transport is local stdio. Remote MCP
transport is deferred until Gabby defines its authentication, capacity, and deployment contract;
remote consumers can use the authenticated `/v1` FastAPI API meanwhile.

## Consequences

- MCP hosts can launch Gabby without implementing an HTTP client or maintaining agent state in
  Gabby.
- The tool schema follows the SDK's typed function contract and is exercised through its in-process
  client in tests.
- MCP SDK upgrades are constrained to the v2 major line and verified in the supported Python CI
  matrix.
- MCP tool errors must remain sanitized, and the adapter must not return a response beyond its
  configured byte limit.
- MCP compatibility is pre-1.0 and may evolve under the documented extension policy.
