# ADR 0057: Bounded stateless run requests

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-01.

## Context

The FastAPI `RunPayload` bounded task input and required dictionary context, memory, and metadata.
Embedded `Agent.arun()` and direct `RunRequest` construction relied on type hints; falsey invalid
values could be silently replaced with empty dictionaries, mixed top-level keys could fail during
trace sorting, and caller-owned nested objects could mutate during a run. Large embedded values
could also be copied before the model request-size guard ran.

## Decision

- Use one `MAX_RUN_INPUT_CHARS` constant for embedded and HTTP task input, requiring 1 through
  100,000 characters.
- Require context, memory, and metadata to be dictionaries with string top-level keys.
- JSON-normalize and snapshot those mappings before asynchronous runtime work begins. The normalized
  serialized data shares the agent's configured `max_model_request_bytes` budget; standalone
  `RunRequest` values use the 4 MiB default.
- Reject malformed or oversized request data before skill selection, retrieval, tracing callbacks,
  or provider calls. Preserve `None` as the only shorthand for an empty mapping in `Agent` methods.
- Keep the snapshot request-local. Gabby does not persist context or memory after the run.

## Consequences

Embedded and HTTP callers now receive a consistent, bounded request contract, and mutations to their
original nested dictionaries after the request begins do not change the run. JSON normalization
matches the representation sent to model providers; custom Python objects are converted using their
string representation. Applications that need larger request data must raise the agent's configured
model-request limit. Request metadata is validated under the same bound even though its values are
not sent to the model.

## Alternatives considered

- **Keep validation only at the HTTP boundary:** rejected because embedded agents and direct runtime
  callers could fail late or allocate oversized copies before provider checks.
- **Concatenate caller mappings directly into prompts:** rejected because later caller mutation and
  non-JSON Python values would make executions nondeterministic.
- **Silently coerce invalid falsey values to empty maps:** rejected because it hides caller bugs.

## Compatibility and evidence

This is a pre-1.0 tightening of the embedded request contract. Runtime tests cover shape, key, size,
input-length, serialization, and nested snapshot behavior; server and runtime contract tests verify
the same public boundary at the HTTP and embedded entry points.
