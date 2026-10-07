# ADR 0034: Add a bounded OpenAI-compatible embeddings adapter

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-09-30

## Context

Gabby exposes an `EmbeddingProvider` contract and a local vector store, but applications need to
write an adapter before they can run the documented hybrid retrieval flow against common hosted or
local inference endpoints. Embedding implementations still need to remain replaceable, and sending
large documents or receiving unbounded vectors must respect Gabby's provider-call size limits.

## Decision

- Provide `OpenAICompatibleEmbeddingProvider` for services exposing `POST /embeddings`.
- Bind each adapter instance to one model name, accept optional output dimensions, and split inputs
  into bounded batches (64 by default and 100 maximum).
- Enforce UTF-8 request and streamed response byte caps per provider call (4 MiB by default),
  validate vector count, indices, dimensional consistency, and finite numeric values, and restore
  vectors to input order using response indices.
- Read credentials from the host environment or explicit Python injection. Configuration rejects
  inline API keys and validates remote HTTPS URLs, allowing HTTP only for loopback inference.
- Keep token counting, model revision identity, retries, and the embedding model itself outside this
  adapter. Applications must create a fresh index when changing model identity, even if dimensions
  match.

## Consequences

The built-in SQLite vector store and hybrid index coordinator can now be used with a common
OpenAI-style embeddings endpoint without another required dependency. One adapter can also target
compatible local services such as Ollama by setting its loopback `/v1` base URL. This does not make
every OpenAI-compatible service behaviorally identical: hosts must select a model supported by the
endpoint, respect its token limits, and verify compatibility for their deployment.

## Alternatives considered

- **Keep all embedding clients host supplied:** rejected because it leaves the first-party hybrid
  retrieval path incomplete for common endpoints.
- **Bundle a local Transformers model:** deferred because model weights and hardware dependencies
  should remain optional and model-specific.
- **Add a Hugging Face-specific feature-extraction client now:** deferred; it is a separate API and
  should be added as another `EmbeddingProvider` adapter with its own contract and acceptance tests.

## Compatibility and evidence

This is a pre-1.0 public API. Tests cover request batching and byte limits, response streaming and
limits, response ordering and validation, credential handling, HTTPS policy, and an end-to-end
hybrid index using the SQLite lexical store, vector store, and generation manifest. See the
[knowledge guide](../../../README.md#knowledge-retrieval),
[extension guide](../../EXTENSIONS.md), and [threat model](../../THREAT_MODEL.md).
