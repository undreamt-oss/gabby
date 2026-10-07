# ADR 0063: Explicit exact-version skill uninstall

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Skill package installation and audit exist, but operators had to manually delete directories to
remove a compromised, revoked, or obsolete skill version. A path typo or ambiguous skill ID could
remove the wrong files, and deleting one version must not affect sibling versions.

## Decision

- Provide `uninstall_skill(registry, name, version)` and `gabby skill uninstall NAME VERSION
  --registry PATH --yes` for one exact version only.
- Validate the skill ID, exact SemVer, registry root, each path component, and the installed
  `skill.yaml` identity before removing the version directory. Refuse registry, path, or manifest
  symlinks and fail closed when the declared identity does not match the requested path.
- Require the CLI's explicit `--yes` confirmation. Do not automatically remove skills when a signer
  is revoked; the audit reports affected paths and the operator chooses which version to uninstall.

## Consequences

Incident responders can remove an exact affected version without touching sibling versions. Agent
definitions that still pin the removed version will fail validation when reconstructed; running
`Agent` instances keep their immutable construction-time plan. The local registry must not be
concurrently mutated during uninstall. This command does not provide filesystem race protection
against another process with write access to the registry.

## Alternatives considered

- **Automatically delete every package signed by a revoked key:** rejected because provenance may
  be incomplete or tampered with, and automatic registry deletion could remove a version still
  needed during incident response.
- **Delete every installed version sharing a skill ID:** rejected because pinned agents may still
  rely on unaffected sibling versions.
- **Expose arbitrary path deletion:** rejected because caller-provided paths create traversal and
  wrong-target risk.

## Compatibility and evidence

This is a pre-1.0 additive API. Tests cover exact-version removal, preservation of sibling versions,
confirmation behavior, path and version validation, manifest identity checks, and symlink rejection.
See the [skill registry operations guide](../../SKILL_REGISTRY.md) and
[advisory installation provenance decision](0062-advisory-skill-install-provenance.md).
