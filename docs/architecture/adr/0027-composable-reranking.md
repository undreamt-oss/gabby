# ADR 0027: Composable second-stage reranking

**Status:** Accepted; the first-party provider adapter is documented in [ADR 0044](0044-cohere-rerank-provider.md).  
**Date:** 2026-09-30.

## Context

Gabby exposed a `Reranker` protocol but had no adapter to compose it with a retriever. Hosts had to
write their own wrapper, and each wrapper would need to decide how to bound candidates and preserve
retrieved evidence. The protocol should compose with lexical, hybrid, and host-defined retrievers
without adding a reranking provider or database dependency to core.

## Decision

- Provide `RerankingRetriever`, which wraps a `Retriever` and a `Reranker` and implements the
  standard async retrieval contract.
- Fetch a bounded candidate pool. The default is 20 and the configurable range is 1 through 100.
- Ask the reranker for at most the caller's requested result count.
- Accept only a list of unique documents from the original candidate set, with no more entries than
  requested. Fail on malformed results rather than silently dropping invalid entries.
- Give the reranker copies of candidates and return copies of Gabby's preserved originals, so the
  extension can reorder or omit documents without rewriting their evidence.
- Keep model weights and provider selection outside core. Optional first-party HTTP adapters may
  implement the protocol when hosts explicitly inject them; see ADR 0044 for the Cohere adapter.

## Consequences

Hosts can compose reranking with existing retrievers using a shared, bounded contract. Candidate
text may still be sent to the injected reranker, so the host controls that service, its credentials,
transport, and data handling. A reranker may omit candidates and return fewer than requested. ADR
0044 adds one optional Cohere HTTP adapter that preserves this contract and keeps provider selection
explicit; local models and other services remain host extensions.

## Alternatives considered

- **Leave wrapper behavior to every host:** rejected because candidate limits and evidence integrity
  would vary between integrations.
- **Bundle a local reranking model or provider SDK:** rejected because weights or another provider
  dependency would enlarge the install and compatibility surface; ADR 0044 uses HTTPX already in core.
- **Allow the reranker to return rewritten documents:** rejected because ranking should not alter
  source-attributed knowledge presented as evidence.
