# ADR 0065: Bounded immutable configuration snapshots

**Status:** Accepted.  
**Date:** 2026-10-02.

## Context

`AgentDefinition` is mutable so applications can assemble configuration before construction;
`ResolvedAgentDefinition` is intended to be an immutable execution snapshot. Recursive freezing
already copied mappings, lists, tuples, and sets, but left arbitrary Python objects untouched. A
programmatic definition could therefore retain mutable provider-option objects or create cyclic or
deep structures that raised `RecursionError` during resolution. YAML file-size limits did not bound
the equivalent in-memory construction path.

## Decision

- Resolved configuration mappings accept finite JSON-compatible scalars, string-keyed mappings,
  lists, and tuples. Mappings become read-only mapping proxies and sequences become tuples.
- Reject custom objects, sets, non-string mapping keys, non-finite floats, invalid Unicode, and
  cycles before the resolved plan is used.
- Bound a snapshot to 10 MiB, 100,000 values, and 128 nesting levels. Raise `ConfigError` when a
  value is invalid or exceeds a limit.
- Apply the shared snapshot path to agent identity, descriptions and instructions, model,
  environment, skill references and paths, tool names, knowledge, policy, verification, and sandbox
  settings. Host resource handles supplied through an injected `Environment` continue to be retained
  by reference as host-owned resources and do not enter the declarative configuration snapshot.
- Keep programmatic `SkillDefinition` metadata aligned with the 1 MiB manifest limit by capping its
  list-valued tool, knowledge, constraint, dependency, verification, and trigger fields at 1 MiB and
  10,000 entries in aggregate. Skill instruction and example resources retain their separate
  10 MiB-per-resource and resolved request-text bounds.

## Consequences

Provider-specific model options remain extensible but must be JSON-compatible and bounded. YAML and
programmatic definitions now fail with the same typed error for unsupported nested values, cycles,
and excessive depth/size. The limits bound traversal and retained configuration data during agent
construction; they do not limit custom Python objects injected through host-owned environment
resources.

## Alternatives considered

- **Deep-copy arbitrary objects:** rejected because a general deep copy may fail, invoke custom code,
  or still return an object that callers can mutate through the resolved plan.
- **Freeze only the outer mappings:** rejected because nested mutable aliases would violate the
  construction-time snapshot guarantee.
- **Leave programmatic values unrestricted:** rejected because it would make in-memory definitions
  behave differently from bounded declarative configuration and permit unbounded recursion.

## Compatibility and evidence

This tightens pre-1.0 validation for programmatic definitions. Provider option values that were
previously accepted but were not JSON-compatible now raise `ConfigError`. Tests cover cycles,
unsupported values, non-string keys, nesting and size limits, skill metadata limits, plus
source-mutation isolation for valid configurations.
