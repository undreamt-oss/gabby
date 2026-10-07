# ADR 0051: Bounded custom authentication calls

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

`create_app` accepts host-provided asynchronous authenticators. The request admission slot is
acquired before authentication, but Gabby previously awaited an authenticator without a deadline.
A stalled identity service could therefore occupy process capacity until the host or client
disconnected.

## Decision

- Bound each authenticator call with `authenticator_timeout_seconds` on `create_app` and `serve`.
- Default the timeout to five seconds and require a finite positive number.
- Convert timeout and authenticator failures into the same sanitized HTTP 503 response.
- Propagate cancellation through `asyncio.wait_for` so cooperative authenticators can clean up.
- Keep built-in JWT JWKS and revocation timeouts as their own lower-level bounds.

## Consequences

An unavailable or stalled custom identity service fails closed without holding request capacity
indefinitely. Hosts with a slower identity path can configure a larger bound. Authenticators should
honor task cancellation and close any resources they own in `finally` blocks.

## Alternatives considered

- **Leave timeouts to each authenticator:** rejected because the public extension boundary would not
  ensure that an integration has a finite request bound.
- **Return an authentication failure (401):** rejected because a timeout is an identity service
  availability error, not invalid caller credentials.

## Compatibility and evidence

This adds a pre-1.0 `create_app` option. Tests verify invalid timeout rejection, sanitized timeout
responses, cancellation of a cooperative authenticator, and capacity reuse by a subsequent request.
See the [operations guide](../../OPERATIONS.md).
