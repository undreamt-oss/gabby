# ADR 0105: PostgreSQL generation manifest adapter

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

Hybrid indexing needs a durable, shared authority for source leases, fencing tokens, staged
generations, activation, and retirement when multiple Gabby processes coordinate writes. The
SQLite manifest is intended for processes sharing a local host filesystem and must not be placed
on a network filesystem. Multi-host users also need shared generation-aware lexical and vector
stores; changing the manifest alone does not make local SQLite indexes shareable.

## Decision

Provide `PostgresGenerationManifestStore` as an optional adapter over a host-owned asyncpg-compatible
pool. Gabby provides `sql/postgres_generation_manifest.sql` for the host's migration system and
does not create or alter schema at runtime. Each operation uses a PostgreSQL transaction; row locks
serialize per-source state changes, and PostgreSQL server time is authoritative for lease
expiration. Lease recovery advances the source fencing token before a new writer may proceed.

The host owns connection credentials, TLS, pool lifecycle, migration, routing, backups, and
read-after-write consistency. All participating instances must use shared generation-aware lexical
and vector stores that enforce the same fencing contract. `PostgresKnowledgeStore` is Gabby's
shared lexical adapter; a compatible shared vector adapter is still required. The PostgreSQL
manifest coordinates metadata only and does not provide a distributed transaction across data
backends.

## Consequences

- PostgreSQL deployments can coordinate writers without sharing a SQLite manifest file.
- Core installation remains driver-free; users opt into the `postgres` extra.
- Schema changes remain visible and controlled by the deploying application.
- Recovery and activation remain subject to the host's PostgreSQL availability and consistency.
- The adapter does not certify any specific external lexical or vector backend.

## Validation

Opt-in live acceptance uses `GABBY_POSTGRES_DSN`, creates an isolated temporary schema, applies the
provided migration, and verifies exclusive leases, renewals, recovery fencing, stale-owner
rejection, generation activation, retirement, and pending-generation abandonment against a real
PostgreSQL server.
