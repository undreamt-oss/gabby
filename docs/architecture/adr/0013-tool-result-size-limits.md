# ADR 0013: Per-tool serialized result limits

## Status

Accepted

## Context

Tool results become model observations and can grow execution context, request cost, and memory
usage. Built-in container tools already bound their output, but registered host callbacks did not
have a common result limit. Truncating JSON or structured data can silently change its meaning.

## Decision

- Add `Tool.max_result_bytes`, a positive integer measured on the UTF-8 serialized JSON result.
- Default the limit to 1 MiB per tool so applications can give tools with different result shapes
  different bounds.
- Serialize with standard JSON rules and reject non-standard `NaN` and infinity values in tool
  arguments and results.
- Check result size before output-schema validation and before adding the observation to model
  context. Report an oversized result as a `ToolError`; do not truncate it.
- Replace `ToolError` messages longer than 512 characters with a fixed generic message before
  adding the failure observation to model context.
- Preserve concise messages from Gabby-generated validation and policy errors, but redact both the
  message and exception class for exceptions raised by host-trusted tool handlers. Those become a
  generic `ToolExecutionError` in model observations, traces, and stream events.
- Preserve the limit when a tool is snapshotted into an agent-local registry.

## Consequences

The runtime bounds tool observations and avoids validating an oversized result against a schema.
Handler exception contents are treated as untrusted because they may contain secrets or private
data. The cap is applied after a host handler returns, so the handler may already have allocated a
large object. Extensions still need their own computation and memory bounds. Callers that need
large results should use pagination, filtered retrieval, or a separate streaming contract.

## Alternatives considered

- Use one agent-wide limit: simpler to configure, but a single value is too restrictive for tools
  with materially different result shapes.
- Leave all limits to tool authors: flexible, but easy to omit and impossible for Gabby to enforce
  consistently.
- Truncate oversized results: preserves some output but can corrupt structured results or hide
  which records were omitted.

## Validation

Tests cover positive-integer validation, preserving the limit in tool snapshots, UTF-8 byte
measurement, rejection before model-context insertion, and strict parsing and serialization of
numeric JSON values. Regression coverage for the oversized-error-message fallback is still needed
before this additional guard is runtime-accepted. The full suite runs on Python 3.11–3.14.
