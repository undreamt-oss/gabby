# ADR 0024: Bounded HTTP response bodies

- Status: Accepted
- Date: 2026-09-29

## Context

Agent API responses include execution traces, verifier evidence, and potentially large output. The
SSE endpoint can also emit an unbounded sequence of progress events and keepalives. Provider
response limits do not bound these HTTP bodies because runtime metadata and traces are assembled
after provider calls.

## Decision

`create_app` and `gabby serve` expose a per-process `max_response_bytes` setting with a 4 MiB
default and a 256-byte minimum. The cap applies to the serialized body of each HTTP response,
including the full `/run` trace and every SSE frame and keepalive. Headers and transport framing
are excluded.

Gabby serializes `/run` output within the cap. If the result is too large, it returns a small,
typed HTTP 500 response instead of truncating JSON. SSE accounts for complete frames before
emission, reserves space for a typed size-limit error while the run is in progress, and ends the
stream without an oversized completion. If the error itself cannot fit in the remaining body
budget, the stream closes at the cap. An ASGI middleware enforces the same body ceiling for other
responses and streaming output.

## Consequences

Hosts can bound per-response memory and bandwidth without imposing limits on the agent's internal
trace object. Applications that need larger results can raise the cap explicitly. Oversized
responses fail as a whole; Gabby does not silently truncate output or trace data. The cap is
process-local configuration and does not limit aggregate traffic across workers.

## Alternatives considered

- Cap individual trace fields: this would leave output and future response fields unbounded and
  distribute transport policy across runtime components.
- Leave sizing to each host: this would produce inconsistent behavior across embedded deployments
  and make the built-in server's memory bound unclear.
