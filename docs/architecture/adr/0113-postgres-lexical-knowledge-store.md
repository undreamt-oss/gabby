# ADR 0113: Host-pooled PostgreSQL lexical knowledge store

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

SQLite FTS5 is a useful persistent local knowledge store, but it does not provide the shared
network database needed by multi-worker and multi-host services. The PostgreSQL generation
manifest coordinates hybrid-index metadata, but does not itself store retrievable documents.

## Decision

Provide `PostgresKnowledgeStore` as a `KnowledgeStore` and `GenerationAwareRetriever`
implementation over a host-owned asyncpg-compatible pool. It supports stable document IDs,
transactional insertion and source-scoped replacement, PostgreSQL `simple` full-text ranking,
exact JSONB metadata filters, generation staging, active-generation retrieval filters, and durable
per-source fencing.
Gabby supplies `sql/postgres_knowledge.sql`; the host applies schema changes and owns the pool,
credentials, TLS, lifecycle, backups, routing, and availability policy. Gabby does not add a
PostgreSQL driver dependency to core and does not close the injected pool.

The store can serve as the lexical half of a coordinated multi-instance hybrid index when paired
with a compatible shared generation-aware vector store and `PostgresGenerationManifestStore`.
The manifest alone does not provide distributed transactions across the data backends.

## Consequences

- PostgreSQL-backed applications can share lexical documents across Gabby workers.
- Existing `FileIngestor` and `Agent` retriever injection work without runtime-specific coupling.
- Schema lifecycle and database operations remain under host control.
- Search tokenization and ranking use PostgreSQL's `simple` text-search configuration and may not
  rank identically to SQLite FTS5.
- The FTS data, source-fencing records, and generation manifest must use the same PostgreSQL
  consistency domain for coordinated hybrid writes.
- Live PostgreSQL acceptance remains required before claiming deployed backend support.

## Validation

The migration is packaged for downstream migration systems. Opt-in live PostgreSQL acceptance
using an isolated schema, concurrent pool connections, source replacement, metadata filtering,
retrieval bounds, and cleanup remains required before claiming live backend support.
