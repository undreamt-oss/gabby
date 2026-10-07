# ADR 0097: Native Gemini embeddings provider

## Status

Accepted

## Context

Gabby has a model-independent asynchronous `EmbeddingProvider` contract, but its built-in hosted
embedding adapters do not cover Google's Gemini embedding API. Relying on an OpenAI-compatible proxy
adds an unnecessary service and can hide model-specific retrieval semantics. Google's
`gemini-embedding-001` accepts task types such as `RETRIEVAL_QUERY` and `RETRIEVAL_DOCUMENT`, while
`gemini-embedding-2` uses task instructions in the text and does not accept the API task-type field.
The model spaces are incompatible and changing models requires reindexing.

## Decision

Add an async `GeminiEmbeddingProvider` implemented with the existing HTTPX dependency and native
`batchEmbedContents` API. Use `GEMINI_API_KEY` by default, require HTTPS for remote endpoints,
disable redirects, and preserve the existing per-call request and response bounds. Batch at most 100
input texts per request. Validate vector counts, values, dimensions, configuration, and bounded JSON
serialization before accepting results.

Add an optional `AsymmetricEmbeddingProvider` contract with separate query and document methods.
The hybrid retriever, generation coordinator, and labeled embedding evaluator use those methods when
available and retain the original single-method behavior for existing providers. Gemini supports
separate query/document task types and text prefixes so both indexing and retrieval use compatible
task-specific vectors through one provider instance.

Support documented task types on `gemini-embedding-001`; reject task types on `gemini-embedding-2`
instead of silently changing text. Restrict document titles to `RETRIEVAL_DOCUMENT`. Expose the
provider as a Python API and as a strict `gabby embeddings evaluate` profile type. Keep credentials
in the host environment or host-injected provider configuration; reject inline API keys in profiles.

## Consequences

- Gemini-backed hybrid retrieval and labeled embedding evaluation use Gabby's existing provider
  contract without adding a Google SDK dependency.
- The adapter stays text-only. Gemini Embedding 2 multimodal content and asynchronous batch jobs
  remain separate integrations because the current `EmbeddingProvider` accepts text and returns
  results synchronously within one call.
- Hosts must use consistent model and task settings for indexing and queries, and reindex when they
  switch model spaces.
- Unit-level HTTP contract coverage does not establish live provider behavior; the credential-gated
  acceptance remains part of the provider integration checks.

## References

- [Gemini embeddings API](https://ai.google.dev/api/embeddings)
- [Gemini embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings)
