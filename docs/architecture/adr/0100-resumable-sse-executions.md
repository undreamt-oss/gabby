# ADR 0100: Opt-in resumable SSE executions

## Status

Accepted

## Context

The streaming API cancels an execution when its client disconnects. Retrying the request then
starts another stateless execution, which may repeat tool side effects. Some consumers need to
recover from a transient network interruption without asking Gabby to retain conversations or
making every execution durable.

## Decision

Keep the existing cancellation behavior for requests without an idempotency key. When a client
supplies `Idempotency-Key`, execute the request once and journal its SSE frames in a bounded,
process-local session. Assign monotonically increasing numeric SSE IDs. A reconnect repeats the
same request and key and supplies `Last-Event-ID`; the server replays only events after that ID.
Bind each key to the authenticated principal and a digest of the complete request. Reject key reuse
with a different request. The journal defaults to a 4 MiB event limit per execution, four times the
run-capacity session count, and ten minutes of retention for completed sessions. Hosts can configure
session count and retention. The keyed request bypasses initial run-slot admission only to look up an
existing session; new executions still acquire the same process-local run slot.

## Consequences

- Clients can resume a transient execution after network interruption without rerunning its tools,
  provided they reach the same service process before the session expires.
- The default behavior remains cancellation on disconnect; resumability is explicitly opt-in.
- Event data, including output and traces, remains in process memory for the retention window. The
  journal is bounded and is discarded on shutdown.
- The feature does not resume across process restart or worker routing. A future host-supplied store
  can add that capability without changing the event cursor contract.
- After expiry, the idempotency key may start a new execution; clients should use unique keys per
  logical run and avoid retrying after the advertised retention window.

## References

- [HTTP API and SSE guide](../../../README.md#http-api)
- [Operations guide](../../OPERATIONS.md)
