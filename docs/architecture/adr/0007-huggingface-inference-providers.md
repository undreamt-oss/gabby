# ADR 0007: Hugging Face Inference Providers adapter

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Gabby must keep model providers replaceable while offering useful hosted and local integrations.
Hugging Face Inference Providers exposes an OpenAI-compatible chat-completions router, so Gabby can
add a named provider without introducing a second request, streaming, and tool-call implementation.
The project architect selected the hosted Inference Providers chat API as the first built-in Hugging
Face path. Local Transformers inference has a different dependency and model lifecycle and should
remain separately composable.

## Decision

- Add `provider: huggingface` for Hugging Face Inference Providers chat completions.
- Default the endpoint to `https://router.huggingface.co/v1` and the credential variable to
  `HF_TOKEN`; both are configurable in agent model settings.
- Pass the model ID through to the router, including supported provider-selection suffixes.
- Reuse the existing async OpenAI-compatible adapter for completion, tool calls, streaming,
  cancellation, timeouts, and client cleanup. Do not add the `huggingface_hub` or OpenAI SDK as a
  Gabby dependency for this integration.
- Keep local Hugging Face Transformers inference behind the `ModelProvider` interface for now.
- Do not describe inference routing, prompting, retrieval, or skill composition as model training.

## Consequences

The named integration shares Gabby's existing provider contract and lifecycle. It requires outbound
HTTPS access and a Hugging Face token with permission to call Inference Providers. Prompts, caller
context, selected skill instructions, retrieved content, tool schemas, and prior tool observations
sent to the model leave the consuming process and are handled by Hugging Face and the selected
inference provider. Deployments must assess provider availability, privacy, retention, and model
terms for their data and workload.

The router and individual model/provider combinations may support different tool-call behavior.
Mock transport tests verify Gabby's HTTP contract; they do not establish live availability, response
quality, model compatibility, or service-level guarantees. Those require separately maintained live
acceptance evidence.

## Alternatives considered

- **Generic OpenAI-compatible configuration only:** works with a custom base URL, but makes Hugging
  Face token and endpoint setup less discoverable and leaves no named integration to test.
- **Hugging Face `InferenceClient`:** official client and routing helper, but adds a dependency and
  a separate completion/streaming lifecycle that is unnecessary for the selected chat contract.
- **Local Transformers first:** avoids sending prompts to a hosted provider, but adds large optional
  dependencies, device/model loading lifecycle, and host resource management that should be designed
  as a distinct runtime integration.

## Compatibility and evidence

The YAML provider name is `huggingface`; existing `openai_compatible` and `ollama` configurations
are unchanged. Mock transport tests cover endpoint, bearer-token, tools, response, and streaming
behavior. Live compatibility with individual model/provider pairs remains unverified. See the
[upstream Inference Providers documentation](https://huggingface.co/docs/inference-providers/index)
for the current router contract and model selection behavior.
