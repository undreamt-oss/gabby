# ADR 0108: Shared SQLite journal for resumable SSE streams

## Status

Accepted

## Context

Resumable SSE runs used an in-memory event journal. That made reconnects depend on worker
affinity and prevented completed streams from surviving a service restart. The API must continue
to bind a journal entry to the request fingerprint and authenticated principal, keep the execution
stateless, and avoid rerunning an agent when a client reconnects.

## Decision

- Keep the bounded in-memory journal as the default.
- Add an opt-in SQLite-backed journal selected by `create_app(stream_journal_path=...)` or
  `gabby serve --stream-journal-db`.
- Persist the complete bounded SSE frames and request fingerprint; persist only a hash of the
  principal subject. SQLite transactions allocate sequential event IDs and enforce the shared
  session-count and per-session byte bounds across workers.
- Use SQLite's local file locking for workers on one host. Do not claim support for network
  filesystems or multi-host sharing.
- Create the database with owner-only POSIX permissions. Operators must protect the database and
  its directory because event frames may contain agent output, tool progress, and traces.
- A worker that owns a live run continues producing it; another worker polls the same journal and
  can replay committed events. Gabby never restarts inference for an existing key.
- Completed sessions persist until the configured retention period. An unfinished session that
  outlives at least twice the agent run deadline plus 30 seconds, or the retention period if longer,
  is closed with a sanitized recovery error and retained for the normal TTL. This bounds abandoned
  records without making an idempotency key silently start duplicate work.
- Database I/O runs through `asyncio.to_thread`; SQLite and storage remain outside agent execution
  state and the default runtime has no new dependency.

## Consequences

Completed streams can resume after worker changes or process restarts when workers share the same
local SQLite file. Live cross-worker replay observes writes through bounded polling. SQLite serializes
journal writes, so this option is intended for a modest local service rather than a distributed event
store. A process crash does not recover or repeat the agent run; reconnecting clients receive the
stored prefix and a recovery error after the owner lease window. Deployments requiring multi-host
replay need a future host-provided shared journal implementation.

## Verification

Contract tests replay a completed stream through a second app instance without a second model call
and attach a second app instance to a live run while events are still being written. The CLI forwards
the configured journal path. Storage remains optional and adds no runtime package dependency.
