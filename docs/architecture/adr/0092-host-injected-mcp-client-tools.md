# ADR 0092: Host-injected MCP client tools

## Status

Accepted

## Context

Gabby can expose an agent as an MCP server, but embedding applications also need agents to call
tools hosted by other MCP servers. Gabby should support this without owning transport setup,
authentication, reconnection, or client shutdown. Imported handlers must still pass through Gabby's
tool permissions, policy checks, and result-size limits.

## Decision

Provide `register_mcp_tools(registry, client, ...)` as an optional-SDK-compatible bridge. The host
supplies an already connected client and owns its lifecycle. Discovery is bounded by time, tool
count, and schema bytes. Imported tools receive explicit host-supplied permission labels, use the
normal Gabby tool invocation path, and accept only structured JSON or text results. Unsupported
content and remote failures become sanitized typed tool errors. The bridge does not claim to
sandbox remote MCP servers or their calls.

## Consequences

Embedding applications can compose remote MCP capabilities with local Gabby tools and policies.
They retain control of credentials and transport security. The bridge adds an optional MCP SDK
dependency and a public pre-1.0 API that may evolve before v1.0.

## Alternatives considered

- Require every application to write its own MCP-to-Gabby tool adapter. This duplicates schema,
  policy, timeout, and error-boundary behavior.
- Have Gabby create and own remote MCP transports. This couples Gabby to transport configuration,
  authentication, reconnection, and process lifecycle decisions belonging to the host application.
