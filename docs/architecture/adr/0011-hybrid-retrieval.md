# ADR 0011: Backend-neutral hybrid retrieval

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

SQLite FTS5 provides useful persistent lexical search without requiring a model or vector database.
Semantic search needs an embedding model and a vector index, and Gabby is intended to keep both
replaceable. A hybrid retriever should combine lexical and semantic candidates without forcing a
provider, database, or reranking service into the core.

## Decision

- Add a `VectorStore` protocol for document/vector upsert, source replacement, source deletion, and
  nearest-neighbor search with metadata filters.
- Add `HybridRetriever`, composed from any lexical `Retriever`, an `EmbeddingProvider`, and a
  `VectorStore`.
- Embed each query once, run lexical and vector candidate searches concurrently, and fuse ordered
  results using Reciprocal Rank Fusion (RRF). The default candidate limit is 20 per backend and the
  default RRF constant is 60; the candidate limit can be configured from 1 through 100.
- Cap direct requested result counts at 100 and reject either backend if it returns more documents
  than the requested candidate limit before fusion begins.
- Use stable document IDs for de-duplication. Preserve lexical order for deterministic ties and
  return copies of indexed documents to callers.
- Keep concrete embedding models, vector databases, and rerankers out of core dependencies.
- Keep indexing explicit through each store interface. Each backend must replace a source atomically
  on its own, but Gabby does not claim a transaction spanning independent lexical and vector stores.
  Hosts must reconcile both indexes after a partial indexing failure.

## Consequences

Applications can use one retriever interface for lexical-only or hybrid retrieval and choose their
own embedding and vector infrastructure. Hybrid retrieval adds one query embedding operation and two
concurrent backend searches. External embedding providers may receive source documents during
indexing and query text during retrieval; remote vector stores may retain source text, metadata, and
vectors. The host owns index lifecycle and consistency across backends. At the time of this
decision, the built-in SQLite store was lexical-only; the interfaces did not supply a ready-to-run
semantic index. That local capability was later added by
[ADR 0033](0033-sqlite-exact-cosine-vector-store.md) without changing this backend-neutral contract.

## Alternatives considered

- **Bundle a local embedding model and SQLite vector extension:** gives a turnkey local path, but
  adds heavy optional dependencies and commits Gabby to model-loading, device, and database choices.
- **Require an external vector database:** provides production scale options, but makes a specific
  service or adapter a prerequisite for hybrid retrieval.
- **Defer hybrid retrieval:** avoids new contracts, but leaves the existing embedding and reranking
  extension points without a composable retrieval implementation.

## Compatibility and evidence

`VectorStore` and `HybridRetriever` are exported from `gabby` and remain pre-1.0 APIs under
[ADR 0009](0009-pre-1-0-extension-api-policy.md). Tests cover rank fusion, deterministic ties,
de-duplication, filters, embedding validation, candidate limits, backend result validation, and
returned-document isolation. Cross-backend indexing repair and concrete provider integrations remain
unverified work.
