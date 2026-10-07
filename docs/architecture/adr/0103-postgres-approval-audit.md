# ADR 0103: Host-pooled PostgreSQL approval audit adapter

## Status

Accepted

## Context

The async `ApprovalAuditSink` contract allowed hosts to provide shared audit storage, while the
bundled SQLite implementation was limited to a single host. A shared backend is useful when
multiple Gabby service instances execute approved tools and the application needs one audit store.
Database driver, credentials, schema migration, pool lifecycle, and retention remain host concerns.

## Decision

Provide `PostgresApprovalAudit`, which accepts a host-owned async client implementing
`execute(query, *args)`, uses parameterized PostgreSQL placeholders, stores invocation metadata and a
bounded canonical argument digest, and never stores raw arguments. The host creates the documented
table through its migration system and owns the pool lifecycle. The adapter adds no PostgreSQL
driver dependency and does not create schema at runtime. Duplicate `(run_id, call_id)` rows fail
closed through the table's primary key.

## Consequences

Applications can reuse an async PostgreSQL pool for shared approval auditing while retaining control
of credentials, TLS, migrations, and operations. The adapter supports the documented text and boolean
columns with a `TIMESTAMPTZ` default. It offers writes only; hosts use their normal database tools
for query, retention, and export. Pool failures are sanitized as `ApprovalAuditError`, so an
approved tool does not proceed when its decision cannot be confirmed as recorded.

## Alternatives considered

- Require each application to implement the SQL sink. This repeats parameterization, argument
  hashing, bounds, and privacy-sensitive behavior.
- Add asyncpg as a Gabby dependency and construct pools. This would transfer credential and
  connection lifecycle into the framework.
- Use SQLite on a shared network filesystem. SQLite does not provide the intended multi-host
  connection and coordination contract.
