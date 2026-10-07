# ADR 0029: Admission before request-body buffering

**Status:** Accepted.  
**Date:** 2026-09-30.

## Context

The API bounded each request body to 1,000,000 bytes and a 30-second read deadline, but read and
buffered request bodies before checking the active execution limit. Concurrent valid-sized uploads
could therefore multiply process memory while waiting to authenticate or execute. A per-request
byte cap does not bound aggregate buffered bodies.

## Decision

- Acquire the existing per-process `max_concurrent_runs` capacity slot before reading non-health
  requests, buffering their bodies, or authenticating them.
- Keep the slot through the response, including the full SSE stream, and pass that same lease into
  the run/stream route rather than acquiring a second execution slot.
- Return an immediate HTTP 429 with the existing typed capacity error when no slot is available;
  do not read or buffer that request body.
- Let `GET /health` bypass the capacity gate so deployment probes stay responsive during overload.
- Keep the existing per-body byte and receive-time limits. Per-process body-buffer exposure is now
  bounded by the admission cap multiplied by the configured body limit, in addition to ASGI server
  and transport buffers. The host still owns connection limits and request-rate limiting.

## Consequences

Concurrent request bodies cannot accumulate without bound inside Gabby's middleware. Capacity
includes body reception, authentication, execution, and response streaming, so slow uploads consume
an execution slot until the body arrives or times out. This makes the limit meaningful before
request memory is allocated and keeps the existing CLI and `create_app` capacity setting as the
single per-process admission control.

## Alternatives considered

- **Check capacity only in the route:** rejected because FastAPI parses and buffers the request body
  before invoking the route.
- **Add a separate request-body semaphore:** rejected because it creates another capacity setting
  and queueing policy; the existing per-process limit already defines admitted work.
- **Rely only on ingress limits:** rejected as the only bound because embedded FastAPI apps may not
  have an ingress, while deployments still need connection and rate limits at the edge.
