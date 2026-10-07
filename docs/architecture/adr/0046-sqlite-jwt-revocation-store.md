# ADR 0046: Built-in SQLite JWT revocation store

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

Gabby can check JWT revocations through a host-injected asynchronous protocol, but hosts had to
implement all persistence themselves. A small single-service deployment benefits from durable local
revocations without adding a network database dependency. The JWT verifier should continue to depend
on the protocol, so storage choice does not become part of authentication logic.

## Decision

- Provide `SQLiteTokenRevocationStore` as an optional implementation of `TokenRevocationChecker`.
- Key records by the pair `(issuer, token_id)` and retain each record until the supplied token expiry.
- Make `revoke()` idempotent and retain the later expiry when the same key is revoked again.
- Provide explicit `cleanup_expired()` for host-managed storage maintenance; authentication lookups do
  not mutate the database.
- Open a short-lived SQLite connection per operation and run the blocking work through Gabby's
  bounded synchronous-callback workers. SQLite's transaction and busy-timeout behavior coordinates
  store instances sharing a file.
- Create a new database file with owner-only permissions where the host platform supports POSIX file
  modes; existing database permissions remain host-managed.
- Keep database location, filesystem permissions, backup, restore, and lifecycle with the host.
  Deployments requiring distributed or cross-region revocation propagation should inject a suitable
  shared store through the unchanged protocol.

## Consequences

Single-host services can persist revocations across process and authenticator restarts without a new
runtime dependency. Calls remain bounded by the authenticator's lookup timeout and the store's
SQLite busy timeout. A shared SQLite file is appropriate only where the deployment filesystem
provides SQLite's required locking semantics; network filesystems and multi-region deployments need
another backend. Cleanup is explicit, so operators should schedule it for long-lived databases.

## Alternatives considered

- **Require every host to implement storage:** preserves flexibility but leaves a common local
  deployment without a ready-to-use durable checker.
- **Bundle a network database client:** rejected because it adds operational and dependency costs to
  every installation.
- **Cache revocation lookup results:** rejected because stale positive or negative results delay
  revocation and conflict with the per-request checking contract.

## Compatibility and evidence

This is a pre-1.0 additive API. Tests cover issuer scoping, shared-file behavior across store
instances, concurrent operations, idempotent expiry, cleanup, invalid input, and integration with
JWT authentication after constructing a new store instance. See [ADR 0043](0043-pluggable-jwt-revocation-check.md)
for the host-injected interface and fail-closed authentication behavior.
