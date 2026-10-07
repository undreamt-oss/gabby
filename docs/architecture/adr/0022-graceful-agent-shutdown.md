# ADR 0022: Graceful agent shutdown

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

An asynchronous agent owns resources such as its model client's HTTP session. Closing those
resources while `arun()` or `astream()` is active can interrupt requests and tools. The FastAPI
lifespan closes the configured agent during service shutdown, so execution and resource lifetimes
need an explicit ordering contract.

## Decision

- `Agent.aclose()` marks the agent as closing and rejects new asynchronous executions.
- Runs already started through `arun()` or `astream()` are allowed to finish. `aclose()` waits for
  all of them to leave the runtime before closing resources owned by the agent.
- Async agent execution and closure must use the same event loop, matching the loop affinity of
  asynchronous provider clients and lifecycle coordination.
- FastAPI application shutdown uses this drain through its existing lifespan hook.

## Consequences

Graceful shutdown does not close a provider under an active request. Shutdown can take as long as
the longest active run's configured deadline. Cancellation remains available to the request caller;
closing the agent does not cancel accepted work. Host-trusted synchronous callbacks that outlive
their deadline may continue in Gabby's bounded daemon worker pool, as Python cannot forcibly stop
their threads.

## Alternatives considered

- Cancel all active executions on close: faster shutdown, but discards accepted work and may
  interrupt tools with side effects.
- Require callers to stop all executions first: avoids tracking active work, but leaves a lifecycle
  race easy to trigger from embedded applications and service shutdown.
