# ADR 0033: SQLite exact-cosine vector store

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-09-30.

## Context

Gabby defines model-independent embedding and vector-store contracts and can compose them with
lexical retrieval, but a developer must supply every vector backend. That makes the local hybrid
retrieval example incomplete even for small knowledge collections. A small, dependency-free local
backend improves that path while preserving the host's choice of embedding model and large-scale
vector service.

## Decision

- Add `SQLiteVectorStore` as a persistent implementation of `GenerationAwareVectorStore`.
- Normalize and validate finite, non-zero vectors, persist them as fixed-width little-endian doubles,
  and rank with exact cosine similarity.
- Scan stored vectors using a SQLite cursor and retain only the best requested results in memory.
  Limit dimensions to 65,536 and result counts to the shared maximum of 100.
- Keep one vector dimension for the lifetime of an index. The host must create a new index when it
  changes its embedding model; Gabby does not choose or infer the model identity.
- Enforce per-source fencing tokens and stage, discard, and delete generations so this store can
  participate in `HybridIndexCoordinator` recovery.
- Keep the SQLite vector index in its own file, separate from lexical and manifest databases, and
  keep the generic `VectorStore` contract as the production extension point for approximate or
  distributed backends.

## Consequences

Local semantic and hybrid retrieval work without a vector-database dependency or provider-specific
configuration. Search is exact and linear in corpus size, so the SQLite implementation is for small
and moderate local corpora, not a substitute for an ANN or distributed index. The host still supplies
an `EmbeddingProvider` and owns decisions about embedding quality, data handling, index rebuilds, and
the workload size at which it moves to another backend.

## Alternatives considered

- **Add a specific vector database dependency:** rejected because it would select a backend and
  deployment model for every Gabby installation.
- **Bundle an embedding model:** rejected because model choice, hardware, and data handling belong
  to the host and an embedding model can be materially larger than Gabby's core.
- **Leave all vector storage to extensions:** rejected for the initial local path because the
  existing hybrid coordinator would have no built-in durable vector backend to demonstrate.

## Compatibility and evidence

`SQLiteVectorStore` is a pre-1.0 public API. Tests cover cosine ranking, stable tie ordering,
metadata filters, persistence, dimension and value validation, source replacement, fencing,
generation filtering, and coordinated lexical/vector reindexing. See the
[knowledge guide](../../../README.md#knowledge-retrieval),
[extension guide](../../EXTENSIONS.md), and [threat model](../../THREAT_MODEL.md).
