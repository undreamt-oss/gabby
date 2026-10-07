# ADR 0043: Add pluggable JWT revocation checks

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

The built-in JWT authenticator verifies signatures and validity claims, but without a revocation
check a stolen token remains valid until `exp`. Token storage and revocation state vary by host, so
Gabby should not add a Redis, SQL, or other state-store dependency to core. Revocation checks are
security-sensitive and must not fail open or silently accept malformed checker results.

## Decision

- Define and export an asynchronous `TokenRevocationChecker` protocol with
  `is_revoked(issuer, token_id, expires_at) -> bool` semantics.
- Allow a host to inject a checker into `JWTBearerAuthenticator`; Gabby does not own or close it.
- When configured, require a non-empty, visible-ASCII `jti` value no longer than 512 characters.
- Query the checker for every successfully validated JWT before producing its principal. Do not
  cache positive or negative revocation results.
- Bound each lookup with a configurable timeout, defaulting to one second.
- Reject revoked tokens as invalid credentials. Treat checker exceptions, timeout, and non-boolean
  results as authentication-service failures; the HTTP boundary returns a sanitized 503.
- Keep revocation disabled when no checker is injected, preserving existing JWT behavior.
- Leave persistence, TTL cleanup, cross-process consistency, revocation delivery, and checker
  lifecycle with the host. A request racing a revocation may pass if its check completes first.

## Consequences

Deployments can connect the verifier to their own revocation store without adding a storage backend
to Gabby's dependencies. Enabling the feature requires issuers to emit unique `jti` claims and the
host store to key entries by issuer and token ID. Each authentication adds one bounded backend call;
store latency and availability are part of the service authentication path.

## Alternatives considered

- **Accept token expiry as the only revocation mechanism:** simple, but does not support prompt
  invalidation of compromised credentials.
- **Bundle a Redis or SQL revocation database:** rejected because it imposes deployment state and
  lifecycle on every Gabby user.
- **Fail open on store errors:** rejected because an outage would disable revocation without an
  operator-visible authentication failure.
- **Cache non-revoked results:** rejected because it creates a revocation delay window that is hard
  to reason about; hosts can make their checker perform its own consistency policy.

## Compatibility and evidence

This is a pre-1.0 optional API. Tests cover required `jti`, issuer-aware lookups on every request,
revoked tokens, malformed checker results, checker exceptions and timeout, sanitized HTTP 503, and
existing behavior when the checker is omitted. See the [authentication guide](../../../README.md),
[extension guide](../../EXTENSIONS.md), [operations guide](../../OPERATIONS.md), and
[threat model](../../THREAT_MODEL.md).
