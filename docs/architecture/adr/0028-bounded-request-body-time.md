# ADR 0028: Bounded request body receive time

**Status:** Accepted.  
**Date:** 2026-09-30.

## Context

The FastAPI request middleware already rejected oversized bodies and bounded its buffer, but it
awaited each ASGI receive indefinitely. A client could hold a connection and middleware task while
sending a partial body slowly, before authentication and execution. Byte limits alone do not bound
this resource lifetime.

## Decision

- Give each request body a configurable total receive deadline, defaulting to 30 seconds.
- Expose the setting as `create_app(request_body_timeout_seconds=...)`, `serve`'s matching keyword,
  and the `gabby serve --request-body-timeout` CLI option.
- Reject non-finite, zero, and negative timeout values at app construction and CLI parsing.
- If the complete body does not arrive by the deadline, return HTTP 408 with a small JSON body and
  do not invoke authentication or a route handler. Add `Connection: close` for HTTP/1.x; omit that
  connection-specific header on HTTP/2.
- The request runs inside an already acquired per-process request-capacity slot; see
  [ADR 0029](0029-admission-before-body-buffering.md).
- Keep the existing maximum body size check and deployment-level connection/rate controls. The
  timeout bounds an individual receive, not the total number of concurrently open connections.

## Consequences

Slow or stalled uploads cannot retain a request task indefinitely in this middleware. Clients with
large contexts on slow links can raise the timeout deliberately. ASGI server and ingress settings
remain necessary to bound simultaneous connections and protect resources before the request enters
Gabby's application.

## Alternatives considered

- **Keep only the byte cap:** rejected because a body can remain below the cap while the sender
  stalls indefinitely.
- **Use an idle timeout that resets after every chunk:** rejected for the core default because a
  client could keep a connection indefinitely by sending occasional bytes.
- **Rely on reverse proxies only:** rejected as the sole bound because embedded FastAPI users may
  not run behind a proxy, though deployment-level limits remain recommended.
