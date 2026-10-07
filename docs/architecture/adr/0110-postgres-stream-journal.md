# ADR 0110: PostgreSQL journal for multi-replica SSE replay

## Status

Accepted

## Context

The default stream journal is process-local, and the optional SQLite journal supports workers on a
single host. Hosts can inject their own `StreamJournal`, but a built-in shared adapter makes the
multi-replica API deployment path usable without tying the server to a database driver or pool.

## Decision

- Provide `PostgresStreamJournal` over the existing asyncpg-compatible `AsyncPostgresPool` contract.
- Keep PostgreSQL optional; the host installs `gabby-agent-runtime[postgres]`, owns the pool, and
  applies `sql/postgres_stream_journal.sql` through its migration system.
- Store principal-bound request fingerprints, response byte limits, ordered SSE frames, and session
  lifecycle timestamps in separate session and event tables.
- Use database server time for expiration and abandoned-run recovery.
- Use a transaction-scoped advisory lock to make global capacity checks and cleanup serialize
  across Gabby replicas; lock sessions while appending so event IDs remain contiguous.
- Keep pool lifecycle, TLS, credentials, HA, backups, and schema migration owned by the host.

## Consequences

Replicas configured with the same PostgreSQL database can reattach to active sessions and replay
completed sessions, subject to the configured TTL and shared capacity. The adapter does not restart
inference after the process that created an active run crashes; it leaves a bounded terminal recovery
event. Advisory capacity locking creates a serialization point for reservations and reaping. Hosts
that need another consistency or throughput tradeoff can still inject a different journal backend.

## Verification

An opt-in live acceptance suite applies the migration to an isolated schema and exercises ordered
append/replay, principal and fingerprint binding, per-session byte caps, idempotent finish, shared
capacity enforcement, and abandoned-run recovery. It requires `GABBY_POSTGRES_DSN` and the optional
`postgres` extra.
