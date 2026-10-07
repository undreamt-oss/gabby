# ADR 0026: Bounded retrieval context

**Status:** Accepted.  
**Date:** 2026-09-30.

## Context

The runtime passed `knowledge.top_k` to an injected retriever but trusted it to honor the requested
count and accepted arbitrary document sizes. The provider request limit eventually rejected an
oversized serialized request, after Gabby had already joined retrieval text into another large
string. A misbehaving retriever could therefore cause avoidable memory growth during context
construction.

## Decision

- Limit `knowledge.top_k` to an integer from 1 through 100; its default remains five.
- Limit rendered retrieved context to 1 MiB of UTF-8 text by default, configurable through
  `knowledge.max_context_bytes`.
- Validate the retriever's collection type, result count, and each returned document before building
  the prompt.
- Count UTF-8 text in bounded slices and reject invalid, excess, or oversized results without
  truncation before making a model call.
- Treat custom retrievers as trusted extensions for work performed before they return. This runtime
  check cannot undo memory allocated inside a custom retriever.

## Consequences

Agent definitions have deterministic, reviewable retrieval budgets. A custom retriever that breaks
the protocol or returns too much context fails the run clearly rather than silently changing the
evidence the model sees. Applications can raise the byte cap for a known workload, while the
separate model request limit remains the final outbound bound.

## Alternatives considered

- **Trust retrievers to honor `limit`:** rejected because a custom extension could accidentally
  return an unbounded result set.
- **Silently keep only the first documents or truncate text:** rejected because it hides contract
  violations and can remove important evidence without notice.
- **Rely only on the model request cap:** rejected because the runtime would already have built the
  combined retrieval string before applying that cap.
