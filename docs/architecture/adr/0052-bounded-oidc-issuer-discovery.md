# ADR 0052: Bounded OIDC issuer discovery

## Status

Accepted

## Context

`JWTBearerAuthenticator` previously required operators to copy the signing-key URL from their
identity provider into service configuration. This is easy to misconfigure and makes common OIDC
deployments harder to embed. Discovery metadata is remote input and must not be trusted to change
the configured identity issuer or redirect Gabby to an unsafe key endpoint.

## Decision

- Provide an async `JWTBearerAuthenticator.from_oidc_issuer()` factory that accepts a host-configured
  issuer and audience, fetches the issuer's OIDC provider configuration once, and constructs the
  existing verifier.
- Build the metadata URL by appending `/.well-known/openid-configuration` to the configured issuer
  after removing one terminal slash for URL construction. Preserve the original issuer value for
  exact comparisons and JWT validation.
- Bound metadata retrieval to 1 MiB and the configured provider timeout. Disable redirects and
  environment proxy configuration, require `application/json`, reject duplicate JSON object
  members, require an exact metadata issuer match, and accept only an absolute HTTPS `jwks_uri`
  without user info, query, or fragment.
- Continue using the existing bounded JWKS cache and refresh behavior for signing-key rotation.
- Validate the verifier's normal constructor options before making the discovery request, so invalid
  algorithms, timeouts, or scope configuration fail without contacting the issuer.
- Keep the direct `jwks_url` constructor path for operators who manage endpoint configuration
  out-of-band. Never derive an issuer or endpoint from an untrusted token.
- Leave WebFinger identity discovery, dynamic/multi-issuer routing, and role/group claim mapping out
  of scope; Gabby receives the issuer identifier from trusted host configuration.

## Consequences

Applications can configure common identity providers with an issuer and audience, then await
discovery during asynchronous startup before passing the authenticator to `create_app`. Startup
fails closed when discovery cannot be validated. Operators that require no discovery network call
can keep using the direct JWKS URL constructor.

The URL construction and issuer comparison follow the OpenID Connect Discovery configuration
requirements: [OpenID Connect Discovery 1.0, sections 4 and 4.3](https://openid.net/specs/openid-connect-discovery-1_0.html).
