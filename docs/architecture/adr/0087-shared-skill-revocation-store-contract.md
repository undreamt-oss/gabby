# ADR 0087: Shared skill revocation store contract

- Status: Accepted
- Date: 2026-10-03

## Context

`SkillRevocationChecker` already lets a host connect execution-time checks to a shared authority,
while `SQLiteSkillRevocationStore` is intentionally limited to one host. The public API did not
describe a complete management contract for backends that also need to revoke, reinstate, or list
publisher IDs. Adding a specific network database to core would impose an operational dependency and
would not establish propagation guarantees for every deployment topology.

## Decision

- Define `SkillRevocationStore` as a typed protocol extending `SkillRevocationChecker` with
  `revoke`, `reinstate`, and `revoked_key_ids` operations.
- Keep `Agent` dependent only on the read/check contract. Store lifecycle and administrative calls
  remain host-owned.
- Have `SQLiteSkillRevocationStore` implement the complete store protocol for durable single-host
  use. Multi-host deployments inject a store or checker backed by their shared authority.
- Require shared implementations to document durable-write and read-after-write propagation bounds.
  Active runs poll on each instance at the configured interval; Gabby does not cache revocation
  decisions or claim distributed consistency on behalf of a backend.

## Consequences

Hosts can implement a typed management and execution boundary for shared revocation services
without adding a hosted-database dependency to core. SQLite remains appropriate for cooperating
processes on one local filesystem, not as a cross-host store. Trust-key configuration remains a
separate deployment concern and changes still require agent reconstruction.

## Alternatives considered

- Add a Redis or SQL backend to core: deferred because backend topology, availability, credentials,
  and consistency are deployment choices, and a concrete backend would not eliminate the need for
  a replaceable contract.
- Keep only `SkillRevocationChecker`: rejected because it omitted the typed management operations
  needed by a reusable durable store implementation.
