# ADR 0055: Bounded transient model retries

## Status

Accepted

## Date

2026-10-01

## Context

Temporary provider overloads and transport failures currently fail the run immediately. Retrying
can improve completion reliability, but hidden retries add latency and may incur duplicate
inference charges. Streaming retries after text has been delivered would also duplicate output.

## Decision

Retries are opt-in through `policies.max_model_retries`, defaulting to zero and capped at three.
Built-in HTTP chat providers classify HTTP 408, 425, 429, 5xx, timeout, and network failures as
retryable; custom providers opt in by raising `RetryableModelError`. Other failures, including
response validation and size-limit failures, are not retried. A bounded `Retry-After` delta is
honored when valid; otherwise Gabby uses short exponential backoff. Every delay and provider call
uses the remaining run deadline.

Streaming can retry until the first text delta is emitted. Once text has reached the client, a
later transient failure ends the run without replaying partial output. Retry attempts include their
stage (`reasoning`, `planning`, or `skill_selection`) in traces and as `model_retry` stream events.
No tool handler is retried by this policy.

## Consequences

- Hosts must explicitly accept possible duplicate provider charges before enabling retries.
- The default keeps current latency and billing behavior unchanged.
- A custom provider must distinguish transient from permanent failures accurately.
- The configured retry count bounds attempts; the run deadline bounds total wait and call time.
- Tool execution and side effects remain outside model retry semantics.

## Alternatives considered

- Always retry a fixed number of times. Rejected because it creates unrequested latency and cost.
- Retry all provider exceptions. Rejected because malformed requests, authentication failures, and
  extension bugs are unlikely to improve on repetition.
- Retry streams after any failure. Rejected because already delivered text cannot be retracted.
- Put retry policy in each adapter. Rejected because it would fragment configuration and trace
  behavior across providers.
