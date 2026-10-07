# ADR 0124: Injectable policy engine

- Status: Accepted
- Date: 2026-10-03

## Context

Agent policy is enforced before tools reach the model, but the runtime directly constructed the
built-in `PolicyEngine`. Hosts could configure allowlists, but could not integrate authorization
logic such as principal-specific grants or organization-owned policy rules through a stable Gabby
extension point.

## Decision

- Define `PolicyEngineProtocol` with async `authorize_tool` and `authorize_permissions`
  operations, and `PolicyEngineFactory` with an async `create` operation.
- Accept an optional `policy_engine_factory` in `Agent`. The factory creates one engine per run
  within the run deadline, after skill activation and before tool schemas are exposed to the model.
- Supply active declared tool names, the immutable agent policy mapping, the environment allowlist,
  and authenticated `Principal` to the factory. Async authorization calls share the run deadline.
- Preserve `PolicyEngine` as Gabby's default implementation and expose it through
  `DefaultPolicyEngineFactory`.
- Sanitize factory and authorization failures. Preserve a policy-denied category without forwarding
  arbitrary custom exception messages into model context or API responses.
- Keep schema validation, configured approval, sandbox requirements, timeouts, and resource bounds
  enforced by the runtime independently of the injected policy engine.

## Consequences

Hosts can apply identity-aware authorization and organization policy without forking the runtime.
The factory and returned engines are trusted host code, created per run, and may be called
concurrently for different executions. They must fail closed, honor cancellation, and avoid blocking
the event loop.
Gabby does not pass caller task text, context, memory, or metadata to the policy extension; the
principal is the only caller-specific identity input.

## Alternatives considered

- Let an injected engine return a complete tool list: rejected because this would combine
  authorization with model capability construction and could bypass Gabby's runtime tool checks.
- Add policy logic to `Tool` handlers: rejected because it would duplicate authorization across
  tools and expose disallowed tool schemas to the model before handler execution.
- Keep policy decisions entirely declarative: rejected because hosts need to apply their own
  identity and organization rules.

## Compatibility and evidence

This is an additive pre-1.0 extension point. The built-in engine remains the default. Contract tests
cover per-run factory inputs, principal delivery, denial before model invocation, and sanitized
failure messages.
