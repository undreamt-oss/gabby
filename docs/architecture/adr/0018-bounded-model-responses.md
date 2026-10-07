# ADR 0018: Bound provider responses

- Status: Accepted
- Date: 2026-09-29

## Context

Gabby bounds outbound model requests and tool results, but an inference endpoint can return a much
larger response. Non-streaming HTTP clients can buffer that entire body before the runtime sees it;
streaming can accumulate unbounded text or tool-call arguments. Both paths consume memory and can
inflate API responses or model context.

## Decision

Each model call has a configurable `policies.max_model_response_bytes` limit, defaulting to 4 MiB.
The built-in OpenAI-compatible adapter counts the decoded UTF-8 response body while reading it and
stops before buffering beyond the cap, for both JSON and SSE responses. Runtime completion and
stream handling also bound normalized text, tool-call fields, and usage strings. Built-in planner
and skill-selector calls use the same per-agent limit.

For custom providers, Gabby validates returned `ModelResponse` values and streamed deltas before
adding them to context or forwarding them. A custom provider that buffers a remote response must
also enforce a transport-level cap before allocating its response object; Gabby cannot reclaim
memory already allocated inside an opaque extension.

## Consequences

- Hosts can tune the cap for long-form output through agent policy.
- Oversized responses fail the run with a bounded runtime error; a streamed run may have emitted
  earlier deltas before the limit is reached.
- Provider error bodies remain separately truncated to a short diagnostic prefix.
- The cap is per call, not a cumulative run-output quota. The run deadline bounds call count.

## Alternatives considered

- Leave response sizing to each provider: rejected because Gabby itself accumulates normalized
  responses and streams, and can provide a consistent per-call contract.
- Use a token budget: deferred because tokenizer choice is model-specific; the UTF-8 byte cap is a
  provider-independent resource boundary.
