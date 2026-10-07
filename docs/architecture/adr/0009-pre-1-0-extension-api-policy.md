# ADR 0009: Pre-1.0 extension API stability

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Gabby is an early-stage open-source framework. It exposes multiple extension points for models,
retrieval, tools, authentication, sandboxing, and skill selection, while those contracts are still
being exercised by real consumers. Promising compatibility too early can freeze incomplete
interfaces and make the framework harder to improve.

## Decision

- Treat public extension APIs as pre-1.0 until Gabby reaches v1.0.
- Make no SemVer compatibility guarantee for `0.x` releases. Public signatures, lifecycle behavior,
  error types, and extension points may change incompatibly.
- Treat names re-exported from `gabby` as intentional public entry points, but not as stable
  contracts before v1.0. Internal and underscored modules remain implementation details.
- Record user-visible changes in the changelog and architecture-level decisions in ADRs.
- Publish explicit compatibility guarantees for named interfaces as part of the v1.0 release work.
- Document the current policy and extension author guidance in `docs/EXTENSIONS.md`.

## Consequences

Early consumers can experiment with Gabby without relying on accidental internal APIs, and the
project can improve extension contracts before freezing them. Consumers should pin a tested `0.x`
version range and review the changelog and ADRs when upgrading. The project must define the named
stable contracts and deprecation policy before declaring v1.0.

## Alternatives considered

- **Stabilize selected contracts now:** would help early plugin authors, but the interfaces have not
  received enough compatibility and lifecycle review to justify a guarantee.
- **Leave stability undocumented:** avoids committing to a policy, but leaves consumers unable to
  distinguish deliberate API from implementation details.

## Compatibility and evidence

This decision clarifies the project's existing pre-1.0 status; it does not change runtime behavior.
The public entry points can be inspected in `src/gabby/__init__.py`. The policy and remaining v1.0
compatibility work are documented in `docs/EXTENSIONS.md` and the project roadmap.
