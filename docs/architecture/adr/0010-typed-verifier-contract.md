# ADR 0010: Typed verifier contract

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

The runtime accepts a verifier object with a `verify()` method, but its return value is currently
untyped and only loosely interpreted. This makes it difficult for callers to distinguish a verifier
pass from an explanatory result, and can produce inconsistent metadata and trace events. Verification
is a runtime control when enabled, so malformed verifier output must not be treated as success.

## Decision

- Define a public `Verifier` protocol for `verify(request, output, trace)`.
- Require a structured `VerificationResult` with `passed`, `method`, `details`, and optional
  `evidence` fields.
- Support async verifier methods directly and synchronous implementations through Gabby's bounded
  callback bridge.
- Apply the run deadline to verification and normalize verifier exceptions into a sanitized
  `AgentRuntimeError` while preserving the cause for host-side diagnosis.
- Reject any result that is not a `VerificationResult`. A result with `passed=False` fails the
  configured run; a result with `passed=True` is included in response metadata and trace events.
- Keep model claims separate from verifier outcomes; verifier output records the verifier's
  structured result and does not itself prove the correctness of a verifier implementation.

## Consequences

Callers get a consistent result shape in API metadata and traces, and enabled verification fails
closed when an extension violates the contract. Existing verifier extensions that return booleans or
arbitrary mappings must migrate to `VerificationResult`. Synchronous implementations remain
convenient, but the host still cannot forcibly stop synchronous work once it begins.

## Alternatives considered

- **Boolean-only result:** easy to consume, but loses the verification method, details, and evidence
  needed for debugging and traceability.
- **Keep arbitrary return values:** flexible, but makes pass/fail handling ambiguous and weakens the
  guarantee that enabled verification actually ran successfully.

## Compatibility and evidence

The public names are exported from `gabby`. Tests cover structured pass and fail results, trace and
metadata serialization, rejection of unstructured results, and synchronous verifier callbacks. The
extension API remains pre-1.0 under [ADR 0009](0009-pre-1-0-extension-api-policy.md).
