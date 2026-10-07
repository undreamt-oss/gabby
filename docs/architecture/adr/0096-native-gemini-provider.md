# ADR 0096: Native Gemini GenerateContent provider

- Status: Accepted
- Date: 2026-10-03

## Context

Gabby supports multiple model providers, but Gemini's native GenerateContent API has a distinct
request/response and streaming protocol. The generic OpenAI-compatible adapter does not preserve
all provider-specific tool state required by Gemini's thinking models. Gemini function calls may
carry provider IDs and thought signatures that the client must return on the following tool turn.

## Decision

Add a dependency-free asynchronous `GeminiProvider` using the GenerateContent REST API. It supports
text, function declarations, function responses, usage, and SSE streaming. Credentials come from
`GEMINI_API_KEY` by default, are sent in the `x-goog-api-key` header, and are never stored in YAML.
Remote endpoints require HTTPS. Requests and responses use the existing 4 MiB provider bounds by
default; callers can configure a bounded output token count.

Extend normalized tool-call records and `ModelStreamDelta` with optional bounded JSON
`provider_metadata`. The runtime carries this metadata with the assistant tool-call record and
passes it back to the same provider on the next request. It does not place this metadata in tool
arguments, traces, or execution results. Gemini stores the exact returned model content parts,
including thought signatures, and maps Gemini function-call IDs to matching function-response IDs.
This lets stateless runtime requests preserve provider-required context without putting a Gemini
concept into the agent definition or persistent conversation store.

Function parameter schemas use Gemini's `parametersJsonSchema` field so Gabby's JSON Schema contract
is passed without silently dropping constraints.

## Consequences

- Gemini uses the same `ModelProvider`/`StreamingModelProvider` contracts and runtime tool policy as
  the other adapters.
- Streaming and non-streaming tool cycles preserve the same provider call identifiers and context.
- Provider metadata is part of bounded model response accounting and remains provider-owned opaque
  JSON state.
- Gemini model availability, quotas, and safety behavior remain provider-managed and require live
  acceptance for each release combination.

## Alternatives considered

- Reuse the OpenAI-compatible adapter. Rejected because Gemini function IDs, thought signatures,
  response content parts, and streaming protocol do not have an equivalent lossless mapping there.
- Add the Google SDK. Deferred because the REST contract supports the required text/function-call
  behavior without another core dependency; broader multimodal and Vertex AI support can be separate
  provider implementations.

## References

- [Gemini GenerateContent API](https://ai.google.dev/api/generate-content)
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)
- [Gemini thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)
