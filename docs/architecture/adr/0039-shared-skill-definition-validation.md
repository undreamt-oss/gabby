# ADR 0039: Share validation for file and programmatic skills

- Status: accepted
- Date: 2026-10-01

## Context

File-backed skill packages were parsed through YAML helpers that validated their text fields and
lists. Programmatic skills supplied through `skill_registry` only received an instance and name
check. Invalid list values, path-like names, or malformed dependencies could therefore reach
resolution and immutable plan construction through a different path.

## Decision

- Define one `validate_skill_definition` contract for both loaded and programmatic skills.
- Validate canonical safe relative IDs, text fields, string-list fields, dependency IDs, and
  optional source paths before dependencies are traversed or the resolved plan is created.
- Expose the validator for host applications that want to preflight packages independently.

## Consequences

Malformed skills fail at construction with `ConfigError`, regardless of whether they came from a
YAML package or an in-memory registry. Existing valid skill definitions keep their shape and
composition API. Remote package discovery, version constraints, and publisher trust remain future
work.

## Alternatives considered

- **Validate only during YAML parsing:** leaves the public programmatic registry path permissive.
- **Rely on type annotations:** Python callers can pass runtime values that violate annotations.
- **Add package version constraints in the same change:** version selection requires a registry
  identity and dependency pinning contract; it should be designed with package publishing.

## Compatibility and evidence

This pre-1.0 change rejects malformed in-memory skill definitions earlier and adds a public
preflight helper. Ruff, mypy, and documentation checks pass. The full test suite has not been run
against this change; existing tests cover valid file-backed and programmatic skill composition.
