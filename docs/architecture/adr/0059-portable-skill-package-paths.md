# ADR 0059: Reject cross-platform skill package path collisions

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Gabby skill archives are intended to install on Linux, macOS, and Windows. Existing path validation
rejects traversal, reserved Windows names, and other non-portable components, but it still permits
distinct paths that alias on case-insensitive or Unicode-normalizing filesystems. Such an archive
can install on one host and fail or address the same file twice on another.

## Decision

- Validate source trees before packaging and archive entry names before extraction.
- Reject paths that collide after per-component Unicode NFC normalization and case folding.
- Reject file/directory prefix conflicts after the same normalization.
- Include Gabby's reserved package manifest name in source-tree collision checks.
- Keep archive names and contents unchanged; do not silently rename resources.

## Consequences

The same archive has consistent path behavior across supported host filesystems. Some archives
that work only on case-sensitive filesystems are rejected during packaging or installation. This is
a pre-1.0 validation tightening; package format version 1 does not change.

## Alternatives considered

- **Allow collisions and rely on the install host:** rejected because the same package would have
  different validity and contents by platform.
- **Rename colliding files during extraction:** rejected because skill manifests and instructions
  may refer to those paths, and automatic renaming would change package semantics.

## Compatibility and evidence

Tests cover source packaging, crafted archive installation, case-folded names, Unicode-normalized
names, file/directory conflicts, and cleanup after rejection. See the
[extension guide](../../EXTENSIONS.md) and [threat model](../../THREAT_MODEL.md).
