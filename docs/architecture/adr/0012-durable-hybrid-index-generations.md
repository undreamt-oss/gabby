# ADR 0012: Durable hybrid index generations

## Status

Accepted

## Context

Hybrid retrieval reads a lexical index and a vector index independently. Replacing each source
independently can expose mismatched versions after a partial write or process crash. The two stores
cannot generally share an atomic transaction, and the host should not have to invent its own recovery
protocol for the built-in composition.

## Decision

- Add a `GenerationManifestStore` interface and a built-in `SQLiteGenerationManifestStore` stored in
  its own database file.
- Give each writer a renewable per-source lease. The default lease is 30 seconds, renewed every
  10 seconds; hosts may configure the TTL.
- Allocate monotonically increasing per-source fencing tokens when writers start and when recovery
  takes over an expired lease.
- Stage each source under a new generation in both indexes while the previous active generation
  remains queryable; each store rejects staging or recovery cleanup with a stale fencing token.
- Persist backend readiness, then atomically activate the new generation in the manifest only after
  both stores finish staging.
- On startup or before a write, inspect pending generations but take over only expired leases.
  Advance both backend fences before activating complete expired generations; remove incomplete
  expired data from both indexes before retiring those manifest entries. A live lease is left
  untouched.
- Filter lexical and vector results through the same active source-to-generation manifest snapshot.
- Keep retired backend data until the operator explicitly runs quiescent cleanup. Backend generation
  deletion must be idempotent so interrupted cleanup can be retried.
- Keep manifest storage replaceable and do not prescribe a vector database or embedding provider.

## Consequences

Generation switching is atomic from the perspective of new retrievals: both searches use a single
manifest snapshot, and the previous generation remains stored for readers that already selected it.
Recovery can safely complete or roll back an interrupted stage. Empty source replacement uses an
active empty generation, so it hides old data immediately while retaining it for in-flight readers.

This is a coordination protocol, not a distributed transaction. A backend must implement atomic
per-generation staging, generation-filtered search, durable per-source fencing, and idempotent
deletion correctly. Retired data uses extra storage until a maintenance window drains readers and
`prune_retired` runs. Lease expiry depends on the manifest store’s clock and availability; a delayed
writer must stop when renewal fails, and backend fencing prevents it from staging or committing after
takeover. The built-in SQLite manifest is for processes sharing a local host filesystem and must not
be placed on a network filesystem. Multi-host deployments need a manifest implementation with
atomic lease claims and an authoritative clock. The manifest file and index databases need compatible
backup/restore treatment; restoring them from different points in time can leave missing active
generations and should be followed by reindexing.

## Alternatives considered

- Put the manifest tables inside the FTS5 database: simpler backups for the built-in lexical store,
  but couples vector coordination to SQLite FTS schema and is unsuitable when lexical retrieval is
  supplied by another backend.
- Require the host to coordinate stores: less code in Gabby but every integration would need to
  handle crash recovery, activation ordering, and cleanup itself.
- Require one atomic combined lexical/vector backend: offers backend-specific transactions but
  removes the backend-neutral hybrid composition accepted in ADR 0011.
- Delete the old generation immediately at activation: reclaims space sooner but can break an
  in-flight retrieval that already read the old manifest snapshot.
- Require a single writer per manifest: simpler but does not suit a service shared by multiple
  worker processes and still needs an enforceable ownership boundary.
- Require each application to supply a distributed lock: flexible, but leaves crash recovery and
  stale-writer fencing inconsistent across integrations.

## Validation required before broader stability

Tests cover readiness checks, durable reopen, activation, lease renewal and takeover, stale writer
fencing, two-process claim contention, and process crashes during staging, after readiness but before
activation, after recovery advances both backend fences but before activation, and after recovery
cleans the lexical generation but before vector cleanup or manifest retirement. Recovery advances
both backend fences again before publishing a complete generation, or repeats incomplete-generation
cleanup idempotently, proving these interrupted operations remain recoverable.
Before v1.0, expand the crash/recovery matrix and run the protocol against at least one concrete
vector backend. The extension API remains pre-1.0 under ADR 0009.
