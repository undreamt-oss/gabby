# ADR 0042: Map JWT scopes to Gabby capabilities

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

The JWT verifier validates standard `scope` and `scp` claims, and the FastAPI boundary can require
scopes for execution and streaming. Identity providers choose those claim values; Gabby route
requirements need deployment-owned capability names. Requiring every host to wrap the built-in
authenticator duplicates a security-sensitive translation step, while interpreting provider roles
or groups in core would couple Gabby to issuer-specific schemas.

## Decision

- Add an optional `scope_mapping` to `JWTBearerAuthenticator`.
- Map each exact issuer scope token to one Gabby capability token or a tuple of capability tokens.
- Apply the mapping only after signature, issuer, audience, validity, identity, and scope-claim
  validation succeeds.
- When a mapping is configured, drop issuer scopes that have no mapping. Do not use wildcards,
  prefix matching, or case folding.
- With no mapping configured, preserve the current pass-through behavior for compatibility.
- Treat scopes on a `Principal` as Gabby capabilities at the HTTP authorization boundary. Custom
  authenticators remain responsible for returning those capability names.
- Keep `run_scopes` and `stream_scopes` as exact requirements; each route requires every configured
  capability before agent execution.
- Leave role/group claim parsing, token revocation, and tenant policy to the host or
  identity provider.

## Consequences

Deployments using a known issuer scope vocabulary can translate it in Gabby's built-in JWT adapter
and fail closed for unrelated claims. Existing deployments retain behavior until they configure a
mapping. The mapping is host configuration and should be reviewed alongside route requirements; it
does not establish tenant isolation, revoke tokens, or constrain other application routes.

## Alternatives considered

- **Require a custom authenticator:** preserves host control but repeats a common translation and
  validation task for every deployment.
- **Automatically interpret roles and groups:** rejected because their claim names and semantics are
  issuer-specific and can overgrant when guessed.
- **Drop every scope unless mapped:** stronger by default, but would break existing pre-1.0 users
  who intentionally use identical issuer and Gabby scope names. Mapping mode itself is fail-closed.

## Compatibility and evidence

This is a pre-1.0 API. Tests cover mapping validation, one-to-many mappings, dropping unmapped
claims, pass-through compatibility, and FastAPI route authorization based on mapped capabilities.
See the [authentication guide](../../../README.md), [operations guide](../../OPERATIONS.md), and
[threat model](../../THREAT_MODEL.md).
