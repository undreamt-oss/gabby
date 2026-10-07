# ADR 0044: Optional Cohere reranking adapter

**Status:** Accepted.  
**Date:** 2026-10-01.

## Context

Gabby has a bounded, provider-neutral `Reranker` contract and a `RerankingRetriever`, but every
application had to supply its own service adapter. A first-party implementation provides a concrete
path developers can compose and test while retaining the replaceable interface. Reranking sends
retrieved document text to an external service, so the service choice and credential boundary must
stay explicit.

## Decision

- Include an optional `CohereReranker` that calls Cohere's v2 rerank HTTP endpoint using Gabby's
  existing HTTPX dependency; do not add the Cohere SDK or a bundled model.
- Keep `Reranker` and `RerankingRetriever` provider-neutral. Applications opt in by constructing and
  injecting `CohereReranker`; no agent configuration silently selects it.
- Read its credential from a host environment variable by default. Direct constructor injection is
  available to hosts, while `from_config` rejects inline credentials.
- Send only candidate text, query, model, bounded result count, and configured token limit. Do not
  send local source metadata or document IDs. Map validated response indexes to local documents.
- Require HTTPS for remote endpoints, cap request and response bytes, cap candidate count at 100,
  and redact upstream body and transport details from raised errors.
- Document that candidate text leaves the host and include an opt-in live contract test. Live test
  success is not evidence of ranking quality.

## Consequences

Gabby ships one concrete hosted reranking adapter without adding a provider SDK or changing its
retrieval protocol. Applications can replace it with a local or other hosted reranker. The host must
decide whether its retrieved content may be sent to Cohere and provide the credential and service
account. Provider availability, model access, billing, and ranking quality remain external.

## Alternatives considered

- **Keep all service adapters host-owned:** preserves a smaller core but leaves developers without a
  first-party, end-to-end reranking path.
- **Add the Cohere SDK:** rejected because the existing HTTPX dependency can implement the bounded
  endpoint contract without coupling Gabby to an additional client lifecycle or release cadence.
- **Select Cohere from agent YAML:** rejected because provider credentials, external data flow, and
  service policy should remain host-owned and explicit.
- **Bundle a local model:** rejected for this increment because model weights, accelerator support,
  and runtime dependencies would substantially expand the installation and compatibility surface.
