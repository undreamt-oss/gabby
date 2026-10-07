# ADR 0064: V1 compatibility and deprecation policy

**Status:** Accepted; applies beginning with v1.0.  
**Date:** 2026-10-02.

## Context

Gabby has an explicit pre-1.0 policy but its extension reference says the stable interface set and
deprecation window are still to be decided. External applications need a concrete migration path
before Gabby makes a stable release, especially because Python protocols, HTTP/SSE, configuration,
and CLI behavior have different ways of communicating deprecations.

## Decision

- Keep all `0.x` public interfaces pre-1.0, with no SemVer compatibility guarantee.
- Publish a named v1.0 compatibility surface in the release notes. The candidate set is documented
  in `docs/COMPATIBILITY.md`; exclude interfaces without contract tests or adequate implementation
  and platform evidence.
- For supported `1.x` contracts, use Semantic Versioning and retain deprecated behavior for at least
  two minor releases and six months after the first deprecation notice, whichever is longer.
- Remove a deprecated supported contract only in a major release after that window has elapsed.
- Use `DeprecationWarning` for Python interfaces. For HTTP, YAML, and CLI contracts, publish the
  notice in release notes and the relevant guide; accept both forms during the window when feasible.
- Keep deprecated behavior under contract tests until it is removed. Include migration guidance in
  the removal release.
- Permit an earlier breaking security, privacy, legal, or severe data-integrity fix, with the reason
  and safest migration documented in the release notes and security advisory.
- Support the latest 1.x minor for bug and security fixes. Support the preceding minor for security
  and severe data-integrity fixes for six months after its successor is released; only the latest
  patch of each supported minor receives updates.
- Name the exact Python symbols, `/v1` endpoints, configuration, and CLI surfaces supported in the
  v1.0 release notes. The current named baseline is in `docs/COMPATIBILITY.md` and is guarded by
  package-export contract tests.

## Consequences

Consumers get a predictable migration and support window after the stable release, while Gabby can
still refine its public surface throughout `0.x`. The guarantee is limited to contracts named in
the v1.0 release; documentation presence alone does not freeze an interface. Provider and host
platform support remains conditional on the release's acceptance evidence.

## Alternatives considered

- **Guarantee every current top-level export at v1.0:** rejected because some features lack live
  platform/provider evidence and would prematurely freeze incomplete contracts.
- **Require a major-version bump for every 1.x removal:** rejected because a documented deprecation
  window gives consumers a migration opportunity before removal; major versions remain necessary
  for intentional breaking changes to supported contracts.
- **Use one release-count window only:** rejected because release cadence may vary; combining two
  minor releases with a six-month minimum avoids an unusually fast sequence shortening the window.

## Compatibility and evidence

This ADR documents a release policy; it changes no runtime behavior and does not make a `0.x`
interface stable. The named baseline and support window are recorded in `docs/COMPATIBILITY.md`.
The v1.0 release notes must identify the exact contracts and tested optional integrations included
in that release.
