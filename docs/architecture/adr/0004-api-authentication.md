# ADR 0004: API authentication boundary

**Status:** Accepted by the project architect; implemented.  
**Date:** 2026-09-29.

## Context

Gabby exposes stateless agent execution through FastAPI. The run endpoint can trigger model calls,
retrieval, and tools, so API callers need an identity boundary before execution. The framework should
support hosted use without hard-coding one identity provider into the runtime.

## Decision

- FastAPI accepts an asynchronous, pluggable authenticator at the serving boundary.
- The built-in bearer-token authenticator supports single-tenant deployments. Tokens are provided
  by the host, compared with a constant-time function, and are not stored in agent YAML.
- The optional `JWTBearerAuthenticator` supports signed JWTs from a host-configured issuer, audience,
  and HTTPS JWKS endpoint. It uses a fixed asymmetric algorithm allowlist, requires `iss`, `aud`,
  `exp`, and `sub`; route-specific scope claim parsing and enforcement were added later by
  [ADR 0035](0035-route-scoped-api-authorization.md). Optional issuer-scope-to-capability mapping
  is provided by [ADR 0042](0042-jwt-scope-to-capability-mapping.md). Bounded OIDC issuer discovery
  is provided by [ADR 0052](0052-bounded-oidc-issuer-discovery.md); role/group mapping, revocation
  storage/lifecycle, and caller policy remain host responsibilities. A host can
  inject a bounded per-request revocation checker as described in
  [ADR 0043](0043-pluggable-jwt-revocation-check.md).
- Unauthenticated ASGI app creation requires the caller to explicitly opt in. The `serve` helper
  permits unauthenticated traffic on loopback; non-loopback binds require an authenticator or
  bearer token.
- Both run and SSE stream endpoints are authenticated. `/health` remains public for health checks.
- A successful authenticator returns a `Principal`; a rejected credential returns no principal.
  The principal is attached to request state for application-level use.
- Authentication provider failures fail closed with HTTP 503. Invalid or missing credentials get
  HTTP 401 with a Bearer challenge.
- Revocation storage, TLS, rate limiting, and process supervision remain extension
  or deployment responsibilities. The built-in authenticators do not provide multi-tenant identity
  or authorization.

## Consequences

Embedded ASGI users must provide an authenticator or explicitly allow unauthenticated development
traffic. The CLI can load a bearer token from `GABBY_API_TOKEN` or a named environment variable.
Custom authenticators can integrate with an external identity provider without coupling that
provider to the agent runtime. Public health checks reveal only service liveness, not agent state.
