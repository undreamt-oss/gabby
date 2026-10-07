# ADR 0079: Stateless Python HTTP client

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-02.

## Context

Gabby exposes a stateless, versioned HTTP API for local and hosted execution, but consumers must
currently assemble the HTTP and SSE protocol themselves. A reusable client makes the API easier to
embed in applications while keeping persistent conversation state with those applications.

## Decision

- Add a public `GabbyClient` for the `/v1` run and SSE stream routes.
- Make `arun()` and `astream()` the primary API, with `run()` and `stream()` synchronous wrappers
  backed by one persistent event loop per client.
- Require callers to supply context and memory on each request; the client does not retain them.
- Bound response and complete stream bytes, validate JSON response and event envelopes, reject
  insecure remote URLs, and expose sanitized server errors as `GabbyAPIError`.
- Accept bearer authentication or host-supplied headers without adding an authentication vendor.
- Keep this API pre-1.0 until its lifecycle and transport contract is included in the final v1.0
  compatibility surface.

## Consequences

Python consumers can call local and hosted Gabby services with typed results and events. The client
does not retry or persist calls; callers control retries and application state. SSE error events are
yielded so applications can choose their own handling. The response limit defaults to the server's
4 MiB limit and can be configured by consumers.

## Alternatives considered

- **Require each application to use HTTPX directly:** rejected because it duplicates URL, bounds,
  protocol validation, and sync/async lifecycle behavior in each consumer.
- **Add a conversation/session abstraction:** rejected because the server and consuming application
  own their own state boundaries, and Gabby's core execution remains stateless.
- **Add automatic retries:** deferred because retrying a POST can duplicate external tool effects;
  callers decide whether a fresh run is safe.

## Evidence

Mock-transport tests cover request shape, bearer headers, URL encoding, typed run results, bounded
responses, typed HTTP errors, SSE envelopes, and sync wrappers. See [the Python client guide](../../CLIENT.md).
