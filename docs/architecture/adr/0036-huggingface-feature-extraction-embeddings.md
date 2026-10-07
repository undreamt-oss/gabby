# ADR 0036: Add a Hugging Face feature-extraction embeddings adapter

- Status: accepted
- Date: 2026-10-01

## Context

Gabby has a provider-neutral `EmbeddingProvider` contract, SQLite lexical and vector stores, and a
bounded OpenAI-compatible embeddings adapter. Hugging Face Inference Providers exposes a separate
feature-extraction task API; treating it as another `/embeddings` implementation would produce an
incorrect contract. The task may return sentence vectors or token-level features, and model families
can require different pooling, prompt, and truncation options.

## Decision

- Add `HuggingFaceFeatureExtractionProvider` implementing the existing `EmbeddingProvider` contract.
- Use the Hugging Face Inference Providers feature-extraction task route and the existing HTTPX
  dependency, rather than adding a required Hugging Face SDK dependency.
- Require a Hub model ID and obtain credentials from `HF_TOKEN` or host injection; reject inline
  credential configuration.
- Preserve the shared request/response byte limits and remote HTTPS policy, and bound batches.
- Accept vector or token-feature response shapes. Mean-pool token features by default, allow CLS
  pooling, and validate dimensions and finite numeric values before exposing vectors.
- Keep this as runtime inference only. Gabby does not bundle model weights, select an embedding
  model, or describe hosted inference as local model training.

## Consequences

The adapter can be composed with any `EmbeddingProvider` consumer, including the durable SQLite
hybrid index, without making a provider or vector database mandatory. Hosted inference transfers
input text to Hugging Face and its selected inference backend; hosts must choose and configure a model
whose pooling behavior matches the index/query contract. Users who need local weights may still
inject an implementation behind `EmbeddingProvider`.

## Alternatives considered

- **Defer the adapter:** preserves the smaller feature set but leaves Hugging Face chat support
  without a first-party route to vector retrieval.
- **Use a Transformers SDK and local weights:** adds large optional dependencies, device/runtime
  policy, and model lifecycle concerns that are separate from this hosted inference adapter.
- **Force task output to one assumed pooled shape:** model APIs vary; explicit mean/CLS pooling and
  strict validation make the adapter's behavior visible and testable.

## References

- [Hugging Face feature-extraction task documentation](https://huggingface.co/docs/inference-providers/en/tasks/feature-extraction)
- [Hugging Face async inference client reference](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client)
