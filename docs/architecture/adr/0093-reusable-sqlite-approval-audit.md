# ADR 0093: Reusable SQLite approval audit adapter

## Status

Accepted

## Context

Gabby exposes a host-owned `ApprovalHandler` so applications can connect their own reviewer
identity and UX. A documented example showed how to write minimal approval metadata to SQLite, but
each adopter otherwise needed to copy its persistence and argument-hashing behavior. Audit writes
must complete before an approval decision returns to the runtime, and raw arguments should not be
stored by default.

## Decision

Ship `SQLiteApprovalAudit` and `AuditedApprovalHandler` as optional standard-library helpers. The
host supplies the database path and async review callback and owns permissions, retention, backups,
and access. Audit I/O runs on worker threads. Records contain invocation identifiers, agent/tool
names, the decision, timestamp, and bounded canonical-JSON SHA-256; subject storage is opt-in. Raw
arguments are never persisted. Audit failure raises a sanitized `ApprovalAuditError`, so the runtime
does not run the approved tool. Reads have a caller-selected limit capped at 1,000 records.

## Consequences

Applications can reuse a small local audit adapter while preserving host ownership of approval UX
and persistence policy. SQLite is suitable for a single host and is not a distributed audit backend.
The helper becomes part of Gabby's proposed v1.0 package-root API.

## Alternatives considered

- Keep only the example. This requires consumers to copy persistence and privacy-sensitive code.
- Add a hosted approval UI or remote database backend to core. This would assign reviewer identity,
  access control, retention, and deployment policy to Gabby instead of the consuming application.
