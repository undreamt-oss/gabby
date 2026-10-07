# ADR 0016: Bound serialized model requests

- Status: Accepted
- Date: 2026-09-29

## Context

Agent callers can supply large context or memory, retrieval can add substantial text, and tool
schemas also contribute to provider request size. Without an aggregate request bound, a run can
send an unexpectedly large payload to a provider even when tool results and HTTP request bodies
are bounded. Token limits vary by model and tokenizer, so a tokenizer is not a suitable core
requirement.

## Decision

Gabby rejects each outbound model call when the UTF-8 size of its canonical serialized request
exceeds `policies.max_model_request_bytes`. The default is 4 MiB. The count includes model ID,
messages, tool schemas and choice, temperature, and the streaming flag when present. Each call
is measured independently; planner, selector, and runtime calls are not summed across the run.
The runtime checks before calling the provider. Built-in `ModelPlanner` and `ModelSkillSelector`
apply the same configured limit before their calls.

This is a byte limit, not a token budget. It bounds the request Gabby constructs, though a custom
provider adapter may serialize the canonical fields differently or add provider-specific fields.

## Consequences

- Hosts can lower or raise the per-call limit in their agent policy, subject to a positive integer.
- Oversized built-in requests fail before network I/O with a clear runtime error.
- Custom injected selectors and planners that make their own provider calls must use
  `ensure_model_request_size` with the configured agent limit. Gabby cannot inspect an opaque
  extension's outbound network traffic.
- A tokenizer-aware budgeter may later complement this byte cap without changing its safety role.

## Alternatives considered

- Require an injected tokenizer-aware budgeter: deferred because it couples the core to model
  tokenization and leaves a byte-level payload bound absent.
- Leave sizing to callers/providers: rejected because Gabby owns context construction and can
  prevent accidental oversized calls consistently across its built-in call paths.
