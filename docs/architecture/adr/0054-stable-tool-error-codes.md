# ADR 0054: Stable tool error codes

## Status

Accepted

## Date

2026-10-01

## Context

Tool failures are consumed by the runtime, model, trace backends, and streaming clients. Python
exception class names and free-form messages are poor machine interfaces: class names change when
implementations change, and messages can contain private host details. Consumers need a stable
failure category while custom tool implementations retain a way to signal useful semantics.

## Decision

Gabby exposes `ToolErrorCode`, a string enum, and a `ToolError` that carries one of those codes.
The runtime reports the code in the model's tool observation, the `tool_error` trace event, and
the `tool_failed` SSE event. Built-in runtime failures map to stable categories. A `ToolError`
raised by a host handler keeps its code, while its message is redacted before it reaches model
context. Unexpected handler exceptions become `execution_failed` and use a generic model-facing
message. Existing `error_type` fields remain available as diagnostic compatibility fields.

## Consequences

- Model and host consumers can branch on `error_code` without depending on exception wording or
  Python class names.
- Custom handlers can distinguish retryable or actionable categories without exposing exception
  messages that may contain secrets.
- Consumers should treat enum string values as the public contract and handle unrecognized future
  values safely.
- Gabby does not promise that every provider-specific failure has a finer category than
  `execution_failed`.

## Alternatives considered

- Forward the exception type and message. This leaks potentially sensitive details and couples
  consumers to implementation details.
- Hide every failure behind one generic category. This prevents useful policy, approval, and
  validation handling by callers and models.
- Use numeric codes. String values are easier to inspect in JSON traces and SSE payloads.
