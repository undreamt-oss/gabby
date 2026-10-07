# ADR 0001: Python async core and FastAPI serving

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Gabby must be embeddable in Python applications and serve the same stateless agent through a local or hosted API. Service execution needs to support concurrent requests, asynchronous providers and tools, deadlines, and cancellation without blocking an ASGI event loop. Synchronous scripts still need a simple entry point.

## Decision

- Gabby's framework core is implemented in Python.
- The public runtime is async-first (`Agent.arun`); a synchronous wrapper (`Agent.run`) is provided for callers outside an active event loop.
- Provider, retrieval, verification, and tool contracts support async implementations. Synchronous tool, retrieval, and verification callbacks are dispatched off the event loop.
- FastAPI is the HTTP application framework. Uvicorn is the local development server and the initial CLI host.
- The model adapter uses an async HTTP client and owns its client lifecycle when Gabby constructs it. Applications that inject their own providers retain ownership of those provider resources.

## Consequences

Async applications must use `await agent.arun(...)`; calling the synchronous wrapper from a running event loop fails with a direct instruction to use the async method. A synchronous `Agent` reuses one event loop on the calling thread across repeated `run()` calls. A single instance cannot mix sync and async execution or move between threads; `close()` shuts down sync resources on that same thread. This preserves provider loop affinity without a background-loop thread.

Cancellation is cooperative. Synchronous callbacks canceled while queued are skipped, but Python cannot forcibly stop code already executing in a worker thread. Tool handlers with side effects must enforce their own bounded execution and idempotency behavior. Gabby does not claim that a thread timeout is a sandbox.

The HTTP run and stream routes propagate client disconnects into execution cancellation and release
the process-local admission slot. Streaming cancellation is handled by the ASGI response lifecycle;
the non-streaming route observes the ASGI disconnect while awaiting the run. A canceled request does
not create persistent run state or support reconnect/resume.

FastAPI provides the ASGI transport contract and schema documentation. Gabby provides a pluggable authenticator and a single-tenant bearer-token implementation at the service boundary (see ADR 0004). Tenant authorization, TLS termination, rate limiting, process supervision, and deployment-specific resource limits remain explicit host responsibilities.

## Alternatives considered

- **Synchronous runtime and standard-library HTTP server:** simpler initial implementation, but blocks concurrent ASGI requests and does not give async providers and tools a direct cancellation path.
- **Async runtime without sync wrappers:** clean async surface, but unnecessarily difficult for simple scripts and synchronous integrations.
- **A different ASGI framework:** not selected; the project architect explicitly chose FastAPI.

## Compatibility

This decision is being established before a stable release. The previous prototype's `Agent.run` behavior remains available as a synchronous wrapper. Async adopters use the new `Agent.arun` method. The HTTP run route is versioned as `POST /v1/agents/{name}/run`; deployment health remains at the unversioned `/health` path.
