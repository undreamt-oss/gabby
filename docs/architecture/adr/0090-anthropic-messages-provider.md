# ADR 0090: Native Anthropic Messages provider

## Status

Accepted

## Context

Gabby already supports OpenAI-compatible APIs, Ollama, Hugging Face Inference Providers, and local
Transformers. Anthropic offers a distinct Messages API with different system-message placement,
tool schemas, tool-result messages, and streaming events. Requiring every consumer to write an
adapter makes the model-provider contract less portable.

## Decision

- Add a native `AnthropicProvider` implementing Gabby's existing async completion and streaming
  interfaces using HTTPX; do not add the Anthropic SDK as a core dependency.
- Translate normalized function tools and tool results into Anthropic Messages blocks, and map
  text, tool-use, usage, and finish events back to Gabby's provider-neutral model types.
- Read credentials from `ANTHROPIC_API_KEY` by default, support an environment-variable override,
  and reject inline credentials in agent definitions under the existing shared validation.
- Require HTTPS for remote endpoints and apply existing incremental response limits, request bounds,
  sanitized errors, retries, and cancellation behavior.
- Configure the required `max_tokens` field as `model.max_tokens`, defaulting to 1024 and capped at
  200,000. Validate this field in shared YAML and programmatic agent-definition validation.

## Consequences

Anthropic models become directly usable with the same agent definitions, tools, streaming runtime,
and provider injection contract. Gabby does not promise live compatibility for every Claude model or
Anthropic API feature; unsupported content blocks fail with sanitized provider errors. Hosted
acceptance is an explicit release validation task.

## References

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Anthropic tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)
