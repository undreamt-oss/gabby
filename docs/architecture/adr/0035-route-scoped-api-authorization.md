# ADR 0035: Add route-scoped API authorization

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

Gabby's HTTP service already authenticates bearer tokens and optional issuer-bound JWTs, but the
built-in boundary establishes identity only. A deployment that serves one agent per service still
needs a way to give different callers access to execution and streaming without implementing a
custom authenticator. Identity-provider claims vary, so Gabby should enforce host-selected scopes
without trying to interpret roles or infer product permissions.

## Decision

- Extend the immutable `Principal` with a set of validated ASCII scope tokens.
- Let `BearerTokenAuthenticator` attach host-configured scopes to its single configured principal.
- Extract either the standard space-delimited JWT `scope` claim or a list-valued `scp` claim. Reject
  malformed claims and tokens that contain both forms rather than guessing which claim wins.
- Add independent `run_scopes` and `stream_scopes` requirements to `create_app`. Each route requires
  all configured scopes and returns HTTP 403 before agent execution if the principal is missing any.
- Keep requirements empty by default for compatibility. Scope checks require authentication and do
  not apply to the public `/health` route.
- Leave issuer-specific role mapping, revocation storage/lifecycle, tenant routing,
  and richer policy evaluation with the host or its identity provider. A host may inject the
  bounded JWT revocation checker defined in [ADR 0043](0043-pluggable-jwt-revocation-check.md).

## Consequences

Embedded services can enforce distinct run and stream permissions using either built-in
authenticator without putting identity or scope claims into model context. The contract is
single-tenant and deployment-configured; it does not make a token's issuer trustworthy or define
which business capabilities a scope should represent. The CLI does not expose route scopes and
continues to use its single bearer-token boundary.

## Alternatives considered

- **Keep all caller authorization host supplied:** rejected for route-level scope checks because the
  HTTP boundary already owns authentication and can enforce a small, explicit scope contract.
- **Interpret provider-specific roles or groups in Gabby:** rejected because claim names and
  semantics vary by identity provider and deployment.
- **Use one shared scope for run and stream:** rejected because streaming can expose progress and
  tool activity to a different caller set than ordinary execution.

## Compatibility and evidence

This is a pre-1.0 public API. Tests cover principal and bearer scope validation, JWT `scope` and
`scp` parsing, malformed and ambiguous claims, route-specific success and denial, construction-time
scope validation, and the OpenAPI 403 response. See the [authentication guide](../../../README.md),
[extension guide](../../EXTENSIONS.md), [operations guide](../../OPERATIONS.md), and
[threat model](../../THREAT_MODEL.md).
