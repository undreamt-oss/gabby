# ADR 0073: Optional local Transformers embedding provider

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Gabby can run a local Transformers chat model, but local semantic retrieval still requires a host
implementation of `EmbeddingProvider` or a remote embedding service. That prevents a fully local
retrieval path even though the knowledge and vector-store contracts are provider-neutral.

## Decision

- Add an optional `TransformersEmbeddingProvider` implementing the existing async
  `EmbeddingProvider` contract. Do not add embeddings to `AgentDefinition`; knowledge construction
  and storage remain explicit host compositions.
- Load `AutoTokenizer` and `AutoModel` lazily with `trust_remote_code=False` and safetensors-only
  model weights. Read private-model credentials from a named host environment variable.
- Run blocking tokenization and inference through Gabby's bounded synchronous callback pool. Serialize
  access to model weights, support cancellation between batches, and retain weights until `aclose`.
- Bound text count, UTF-8 input bytes, per-batch token count, inference batch size, serialized vector
  output estimate, and execution time. Return finite vectors with one stable dimension per call.
- Support attention-mask-aware mean pooling and CLS pooling. Normalize vectors to unit length by
  default for cosine search; document that model-specific prompt formatting remains the host's
  responsibility.
- Keep PyTorch platform selection outside core dependencies. The existing `transformers` extra
  installs Transformers; users install the PyTorch build appropriate to their platform.

## Consequences

Hosts can build local retrieval pipelines with the same `EmbeddingProvider`, `HybridRetriever`, and
`VectorStore` contracts used for remote embeddings. Weight loading and inference do not block the
event loop. The first-party implementation is limited to standard Transformers encoder models and
the documented pooling modes; Sentence Transformers modules, special retrieval prompts, and
task-specific preprocessing remain available through custom providers.

## Alternatives considered

- **Add `sentence-transformers` as a core or extra dependency:** rejected for the initial adapter
  because it introduces another large model stack and dependency lifecycle when ordinary
  Transformers `AutoModel` provides the needed encoder contract.
- **Make local embeddings implicit in agent configuration:** rejected because model choice, vector
  dimensions, persistence, and reindexing belong to host knowledge composition, not a stateless
  agent definition.
- **Leave local embeddings entirely to applications:** rejected because Gabby already owns optional
  local Transformers inference, and safe bounded pooling is useful reusable infrastructure.

## Compatibility and evidence

The class is an additive pre-1.0 export and uses the existing `EmbeddingProvider` contract. Fake
backend tests cover pooling, attention masks, normalization, batching, bounds, safe loading options,
and public imports. Live model loading remains opt-in and platform-dependent. See the
[knowledge guide](../../EXTENSIONS.md) and the [embedding provider reference](../../../README.md).
