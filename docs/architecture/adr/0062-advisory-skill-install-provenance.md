# ADR 0062: Record advisory skill installation provenance

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Gabby verifies detached package signatures when a signed skill is installed, but previously kept
the signer ID only in the immediate `SkillPackageInfo` result. A later key revocation could block
future installs but could not help operators identify packages already present in a local registry.

## Decision

- Atomically write a reserved `.gabby-install.json` file into each installed version directory.
  Record the package archive digest, signer key ID when verified, and whether signature verification
  succeeded. Reject that reserved path, including portable case-folded collisions, in package
  archives so package content cannot supply the provenance record.
- Add `audit_skill_registry()` and `gabby skill audit` to report signer status, missing legacy
  provenance, invalid metadata, and matches against host-supplied revoked key IDs. Bound the scan by
  a maximum entry count and metadata byte size; do not follow symlinks.
- Treat the metadata as mutable local provenance, not as an authoritative log. V1 signatures do
  not authenticate installed content, and a revoked key does not automatically disable a skill.
  V2 content-signature auditing is defined in [ADR 0066](0066-auditable-skill-content-signatures.md).
  Hosts remain responsible for trust-map distribution and for disabling or removing affected skill
  versions before redeploying agents.

## Consequences

Operators can find recorded installations signed by a revoked key and identify older or manually
installed skills that lack provenance. The audit may report `unknown` for legacy layouts and fails
its CLI command when it finds revoked, unknown, or invalid records. This ADR's v1 audit did not
recheck installed contents; ADR 0066 adds that ability for v2 signatures. Tamper-proof provenance,
runtime enforcement, automatic removal, publisher key lifecycle, and cross-process revocation remain
outside this change.

## Alternatives considered

- **Keep signer IDs only in the install return value:** rejected because command-line and remote
  installs may not have a durable host record available for later incident response.
- **Treat the local marker as authoritative trust state:** rejected because a writable skill registry
  can be modified by the same actor who can alter its skill files.
- **Automatically delete revoked installations:** rejected because registry deletion is destructive
  and could remove a skill version still needed by an operator; the audit reports exact paths for a
  host-controlled response.
- **Reverify packages at every agent construction:** deferred because installed registries do not
  retain the original archive and signature, and local development skills are intentionally allowed.

## Compatibility and evidence

This is a pre-1.0 additive change. Current full-suite acceptance covers signed and unsigned
provenance, revoked signer detection, legacy and malformed records, symlinks, entry and metadata
bounds, reserved archive paths, and CLI exit behavior. See the
[skill registry operations guide](../../SKILL_REGISTRY.md) and
[skill signature decision](0060-ed25519-skill-package-signatures.md).
