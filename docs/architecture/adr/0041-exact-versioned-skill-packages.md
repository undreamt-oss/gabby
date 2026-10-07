# ADR 0041: Version portable skill packages and allow exact pins

- Status: accepted
- Date: 2026-10-01

## Context

Skills were reusable directory packages but had no version field. Agent and dependency references
resolved by ID alone, so a path or registry change could silently substitute different procedures
when an agent was reconstructed. The architecture described skills as versioned, while manifests,
resolved definitions, and CLI inspection did not expose versions.

## Decision

- Add a SemVer `version` field to file-backed and in-memory skill definitions; older manifests
  default to `0.1.0`.
- Accept exact references in the form `skill-id@MAJOR.MINOR.PATCH`, including SemVer prerelease or
  build metadata. Agent `skills` and package `dependencies` use the same syntax.
- Keep unpinned IDs for compatibility. Resolve exact registry keys first, then a stable-ID registry
  entry; unpinned references to multiple registry versions fail and require an exact pin.
- Verify requested versions against loaded package manifests before dependency resolution.
- Support a filesystem registry layout of `<root>/<skill-id>/<version>/skill.yaml` in addition to
  the existing `<root>/<skill-id>/skill.yaml` layout. Unpinned lookup prefers a legacy package,
  otherwise requires there to be only one versioned package.
- Resolve at most one version per stable ID in one agent because selectors and tool grants address
  capabilities by stable ID. Reject attempts to mix versions.
- Include the selected package version in immutable agent skills, model selection context, skill
  activation traces/events, and `gabby skills` output.
- Do not implement version ranges or remote package discovery in this change.

## Consequences

Hosts can pin skill behavior reproducibly and keep multiple versions in an in-memory registry.
Existing agent files and package manifests remain valid but are unpinned unless they specify a
version. The package content and referenced version are not cryptographically authenticated; the
host still controls local files and registry values.

## Alternatives considered

- **Resolve latest compatible versions:** requires a range language and dependency solver and can
  change behavior as packages are published.
- **Allow two versions of one skill ID in one agent:** the current selector protocol and tool grants
  use stable IDs, so same-ID versions would be ambiguous during activation.
- **Require a version immediately:** would break all existing skill manifests and agent definitions
  without providing an existing package catalog to migrate from.

## Compatibility and evidence

The feature is pre-1.0 and keeps a default version for old manifests. Tests cover SemVer validation,
manifest/agent/dependency pins, registry selection of multiple versions, and mismatch rejection.
