# ADR 0002: Persistent event loop for synchronous wrappers

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

The architect chose a persistent sync bridge so one `Agent` supports repeated `run()` calls. Providers such as async HTTP clients can bind resources to the event loop where they are first used. A background-thread loop bridge hung under the project's Python 3.14 environment and also complicates thread affinity and shutdown.

## Decision

- Synchronous `Agent.run()` reuses one event loop created lazily on the calling thread.
- One agent instance uses either the sync API or async API for its lifetime; it cannot move between threads.
- Sync callers close with `Agent.close()` or a synchronous context manager. Async callers use `await Agent.aclose()` or an async context manager.
- Calling the sync API from an active event loop remains an error; async callers use `await Agent.arun()`.

## Consequences

Repeated sync calls preserve model-client loop affinity and avoid cross-thread event-loop scheduling. Synchronous calls on the same agent are sequential. Applications that need concurrent calls should use an async agent instance or separate agent instances. The caller owns the sync agent's thread affinity and lifecycle.
