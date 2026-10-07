# ADR 0071: Optional local Hugging Face Transformers chat provider

- Status: Accepted
- Date: 2026-10-02

## Context

Gabby supports hosted OpenAI-compatible, Ollama, and Hugging Face Inference Providers chat APIs.
Applications can inject a local provider, but local model execution is a named model-layer path and
should not require every application to implement its own adapter. Transformers and PyTorch have
large, hardware-specific dependency distributions, and model chat templates and tool-call parsers
are not uniform across checkpoints.

## Decision

- Add `TransformersProvider` selected by `model.provider: transformers`. Keep Transformers out of
  Gabby's core dependencies and declare a `transformers` optional extra; users install a PyTorch
  build that matches their host separately.
- Load tokenizer and causal language model lazily on the first completion. Read a Hub token from
  `model.api_key_env` (default `HF_TOKEN`) and support optional `revision`, `cache_dir`,
  `local_files_only`, and `device` settings.
- Disable `trust_remote_code` and require safetensors weights. Operators can set `local_files_only`
  to prevent downloads and should pin a fixed model revision for repeatable deployment.
- Bound prompts with `max_input_tokens` (default 32,768, maximum 1,000,000) and generation with
  `max_new_tokens` (default 1,024, maximum 16,384). The runtime's existing serialized request,
  response, and deadline limits remain in force.
- Run synchronous model loading and generation in Gabby's bounded daemon callback pool, serialize
  generation per provider instance, and request cancellation through a Transformers stopping
  criterion. A model kernel that does not yield to that criterion cannot be forcibly interrupted.
- Require a compatible Transformers chat template. When tools are present, require the tokenizer's
  `parse_response` implementation to produce structured tool calls, then verify names against the
  request's tool schemas before returning them to the runtime. Do not guess executable calls from
  ordinary generated text.
- Expose completion only in this first adapter. The SSE transport emits the finished local response
  as one text delta; token-by-token local streaming is deferred.

## Consequences

Applications can select a local Hugging Face model using the same agent definition and
`ModelProvider` contract as hosted chat models. The core install remains small, and CPU, CUDA, and
other PyTorch builds remain under the operator's hardware-aware installation choice. A fake-backend
contract suite verifies configuration, normalized tool calls, and unsupported-parser behavior.
Actual model/template compatibility, hardware performance, and cancellation latency still require
acceptance runs with selected checkpoints and devices. Models without a structured response parser
can only be used by agents that expose no tools.

## Alternatives considered

- Require applications to inject all local model providers: rejected because local Transformers is
  an explicit supported model family and repeating its lifecycle and security controls in each
  application would fragment behavior.
- Install PyTorch through Gabby's optional extra: rejected because platform, accelerator, and wheel
  indexes are hardware-specific and can add hundreds of megabytes to an otherwise light package.
- Depend on `trust_remote_code=True` or unsafe pickle checkpoints for broader model compatibility:
  rejected because model repositories would be able to execute repository-supplied Python or
  pickle payloads during loading.
- Parse tool calls from arbitrary text with a Gabby-owned regex: rejected because model output
  formats vary and guessed calls could cross the runtime's validated tool boundary.
- Add token streaming in this change: deferred because generation and parsing APIs vary across
  Transformers versions and templates; the first implementation keeps the shared provider contract
  safe and bounded while reporting its complete-response behavior clearly.
