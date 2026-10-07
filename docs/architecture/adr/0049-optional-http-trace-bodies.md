# ADR 0049: Optional HTTP trace bodies

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

Gabby returns an execution trace in every `/run` response and SSE completion, even for clients that
only need the agent output. Traces can include structured plans and verifier-supplied details. A
caller may need the stable trace ID for support or correlation without receiving the full trace body.

## Decision

- Add `include_trace: bool = true` to the shared run and stream request payload.
- When false, `/run` and the SSE `completed` event return `trace: null` and retain `trace_id`.
- Preserve the current default and response shape when `include_trace` is omitted.
- Do not interpret this response preference as disabling host observability. An injected `Tracer`
  continues receiving events according to its existing contract.
- Continue enforcing the configured HTTP response-size limit after serialization.

## Consequences

Clients can reduce response size and avoid receiving trace details they do not need. This does not
prevent the runtime from using its transient trace internally, disable host-injected trace export, or
change the embedded Python API's `ExecutionResult.trace` contract. Hosts remain responsible for
authenticating access to the API and protecting telemetry.

## Alternatives considered

- **Remove traces from the HTTP API:** rejected because traces are useful by default and existing
  clients rely on them.
- **Make trace bodies opt-in by default:** rejected for this pre-1.0 addition because it would
  silently change existing response behavior.
- **Let callers disable host tracer export:** rejected because observability policy belongs to the
  service host, not an individual execution request.

## Compatibility and evidence

This is an additive request field; omission preserves the existing behavior. Route tests cover both
`/run` and SSE completion with trace inclusion disabled and verify that the trace ID remains. See
the [HTTP API guide](../../../README.md) and [operations guide](../../OPERATIONS.md).
