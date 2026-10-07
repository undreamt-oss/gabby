# ADR 0083: Bounded skill package inspection

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-03

## Context

Skill packages can contain executable-looking procedures, declared tools, knowledge references,
and nested resources. Publishers and consumers need a way to review the declared capabilities and
content inventory before installing a package into an agent's skill registry.

## Decision

- Add `inspect_skill_package()` and `gabby skill inspect` to validate and report a package without
  installing it into a caller-owned registry.
- Reuse the package snapshot, manifest, archive-path, per-file, aggregate-content, and skill YAML
  validators used by normal installation.
- Report package identity, description, declared tools and knowledge, dependencies, verification
  labels, archive digest, and bounded file inventory; omit instructions and examples from the result.
- Report signature presence separately from verification. Verify a present signature only against
  caller-supplied trusted keys, and require those keys when `--require-signature` is selected.
- Treat package metadata and skill contents as untrusted even after signature verification; the
  command is an inspection aid, not a sandbox or runtime authorization decision.

## Consequences

Reviewers can see the capability claims and exact packaged files before installation, and automation
can consume machine-readable JSON. Signature verification remains explicitly tied to host-owned
trust material. Inspection temporarily extracts validated files to private temporary storage and
does not retain an installation.

## Alternatives considered

- **Print package contents or instructions by default:** rejected because large or terminal-control
  content would make review noisy and can expose authored material unexpectedly.
- **Treat a present signature as trusted without a host key:** rejected because the archive cannot
  establish its own publisher identity.
- **Inspect only the ZIP manifest:** rejected because the embedded skill definition must also pass
  Gabby's structural and path validation before its claims are useful.

## Compatibility and evidence

This is a pre-1.0 additive Python and CLI feature. Package contract tests cover unsigned and trusted
signed inspection, declared capability reporting, digest and archive validation, and CLI JSON
output. See the [skill registry and package guide](../../SKILL_REGISTRY.md),
[package format ADR](0047-checksummed-local-skill-packages.md), and
[signature format ADR](0060-ed25519-skill-package-signatures.md).
