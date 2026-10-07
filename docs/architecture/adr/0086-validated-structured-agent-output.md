# ADR 0086: Validated structured agent output

- Status: Accepted
- Date: 2026-10-03

## Context

Gabby agents are consumed by applications and automation as well as people. Callers need a stable
way to request and validate machine-readable final results without taking control of the runtime
loop or coupling to a specific model provider.

## Decision

- Add an optional JSON Schema Draft 2020-12 `output_schema` to YAML and programmatic agent
  definitions, using the same validation and immutable snapshot path.
- Include the schema in the runtime's highest-priority final-response requirement. Keep provider
  interfaces model-agnostic; provider-native structured-output modes can be added behind adapters
  later.
- Strictly parse the final response as JSON, reject duplicate object keys and non-finite numbers,
  and validate the value against the snapshotted schema before returning a successful execution.
- Add the parsed value as `metadata.structured_output` while preserving the existing text `output`.
- Buffer streamed model text until the final value validates. Emit the validated JSON text once;
  do not send invalid partial structured output.
- Reject remote `$ref`, `$dynamicRef`, and remote `$id` values. Final structured output is limited
  to 1 MiB.

## Consequences

One contract serves extraction, classification, routing, and other data-oriented agents without
requiring a provider-specific API. Validation is enforced even when a model ignores the schema
instruction. The text `output` remains compatible with existing consumers, while API and embedded
callers can use the parsed metadata value. Streaming structured responses incur end-of-response
latency and are delivered in one text event. Invalid output fails the run; automatic repair or
regeneration is not part of this contract. The schema participates in the bounded model request and
must fit the configured request limit.

## Alternatives considered

- Add provider-specific `response_format` parameters to the shared model protocol: rejected because
  provider capabilities and schema dialects differ and would narrow model portability.
- Validate only tool outputs: rejected because tool schemas do not constrain final agent responses.
- Stream partial JSON before validating: rejected because consumers could act on a result Gabby
  later rejects.
- Automatically retry or repair invalid output: deferred because it adds provider calls, cost,
  deadline behavior, and new failure semantics.

## Compatibility and evidence

This is a pre-1.0 additive configuration field. Tests cover YAML and in-memory validation, immutable
schema snapshots, valid and invalid responses, duplicate-key rejection, trace results, API metadata,
and buffering of structured streaming output. See the
[structured output guide](../../STRUCTURED_OUTPUT.md).
