# ADR 0114: PostgreSQL pgvector generation-aware store

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

`PostgresKnowledgeStore` supplies shared lexical storage and the PostgreSQL generation manifest
coordinates multi-worker source activation. Multi-instance hybrid indexing also needs a shared
vector backend that observes the same generation and fencing contract. SQLite exact-cosine storage
is intended for local small or moderate indexes, not shared network deployments.

## Decision

Provide `PostgresVectorStore` over the host-owned asyncpg-compatible pool and PostgreSQL's pgvector
extension. The host installs pgvector 0.8.0 or newer and applies `sql/postgres_vector.sql` through
its migration system. Gabby uses pgvector's HNSW cosine index and requires a configured embedding dimension from
1 through 2,000. The first store operation persists that dimension in the schema; subsequent
workers with a different dimension fail closed. Embedding values are passed as text and cast by
PostgreSQL, so core does not depend on a Python pgvector client package.

The adapter implements generation staging, active-generation filtering, durable source fencing,
and advisory transaction locks. Searches scope strict-order iterative scans, `ef_search`, and
`max_scan_tuples` to their database transaction to improve recall when filters discard candidates.
For coordinated multi-instance hybrid indexing, use it with
`PostgresKnowledgeStore` and `PostgresGenerationManifestStore` in one PostgreSQL consistency domain.
The host owns server extension installation, connection lifecycle, database credentials, migrations,
backups, tuning, and availability.

## Consequences

- Multi-worker deployments have a built-in shared lexical/vector backend pair for hybrid indexing.
- Core remains free of a PostgreSQL driver dependency; consumers opt into the `postgres` extra.
- HNSW trades exact ranking for approximate search; deployments must evaluate recall on their corpus.
- Filtered HNSW search requires pgvector 0.8.0 or newer for iterative scans; scan tuple limits can
  still cap results for very selective filters.
- HNSW's supported dimension limit excludes some embedding models; those deployments can inject a
  different `GenerationAwareVectorStore` implementation.
- pgvector must be installed on the PostgreSQL server, and the migration role may need extension
  installation privileges.

## Validation

SDK-independent contract tests cover storage IDs, vector shape and finiteness, parameterized search,
metadata and generation filters, source replacement, dimension consistency, fencing, cleanup, and
sanitized errors. Opt-in live acceptance uses a disposable database with pgvector installed and
`GABBY_POSTGRES_DSN`; it applies the packaged migration in an isolated schema and checks ranking,
metadata filtering, source replacement, generations, fencing, and cleanup.
