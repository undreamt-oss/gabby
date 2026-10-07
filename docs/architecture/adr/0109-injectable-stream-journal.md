# ADR 0109: Injectable resumable stream journal

## Status

Accepted

## Context

ADR 0108 added an optional SQLite journal for durable same-host SSE replay. Tying the FastAPI
runtime to SQLite would prevent a host from using a shared database or distributed event store for
multi-host deployments, despite Gabby's extension-oriented architecture.

## Decision

- Expose the async `StreamJournal` protocol and `StreamJournalSnapshot` from the package root.
- Accept a host-owned `stream_journal` in `create_app` and `serve`; reject configuring it together
  with the SQLite `stream_journal_path` option.
- Keep the SQLite journal as the built-in path selected by `stream_journal_path`; the in-memory
  journal remains the default when neither option is set.
- Require implementations to atomically bind the session key to request fingerprint and hashed
  principal, enforce the shared session cap and cursor validity, allocate contiguous event IDs,
  enforce byte limits during append, reject appends after finish, and return ordered events from
  `read_after`.
- Leave initialization, credentials, connections, and close behavior with the host for injected
  implementations. Gabby owns and closes only the SQLite backend it creates from a path.
- Map injected backend failures during session lookup to a sanitized HTTP 503 response.

## Consequences

Hosts can supply a shared backend for multi-host worker routing without changing agent execution or
the HTTP protocol. The built-in SQLite backend remains a same-host option. A custom backend must
provide atomic operations and its own TTL recovery, cross-worker visibility, durability, security,
and lifecycle guarantees. The interface is pre-1.0 and remains subject to change.

## Verification

Contract coverage continues to verify SQLite replay through the path option. An injection contract
test reuses a host-owned journal across two app lifetimes and confirms Gabby does not close it; the
API package-root contract includes the protocol and snapshot. Process-level and live distributed
backend acceptance remain separate deployment evidence.
