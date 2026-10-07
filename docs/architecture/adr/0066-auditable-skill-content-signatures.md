# ADR 0066: Auditable skill content signatures

- Status: Accepted
- Date: 2026-10-02

## Context

The v1 detached skill signature authenticates exact archive bytes during installation. Gabby then
discards the signature sidecar. The local `.gabby-install.json` file records that verification
occurred, but the registry owner can edit that record, and a later audit cannot verify either the
original signature or whether installed files have changed.

## Decision

- New skill signatures use format v2 and sign the key ID, exact archive SHA-256, and SHA-256 of the
  canonical validated package manifest. The manifest binds skill identity and every package file's
  normalized path, size, and content digest.
- Installation persists the detached signature document with the existing advisory provenance.
  Existing v1 sidecars remain installable and verifiable against their original archives.
- `audit_skill_registry` accepts host-owned trusted public keys. For v2 installs, it reconstructs
  the canonical manifest from current installed files and verifies both its digest and the
  Ed25519 signature. A mismatch or invalid signature is reported as `invalid`; a missing trust key
  is `unknown`; a revoked key is `revoked`.
- V1 installs retain the `recorded-signed` status because their signatures do not bind a
  reconstructable installed-file manifest. Auditing without a trust map also reports v2 installs
  as `recorded-signed`, making clear that this is install-time evidence only.
- The CLI accepts repeated `--trusted-key KEY_ID=PATH` arguments for audits. Trust remains
  deployment-owned.

## Consequences

Audits can detect edits to installed v2 skill files without retaining package archives. Provenance
remains mutable by a local registry writer, and audit does not disable already constructed agents or
provide a tamper-proof event log. Hosts must run audit as a deployment gate and enforce their own
trust-map distribution, revocation, and runtime admission policies.

## Alternatives considered

- Retain the original package archive and sidecar: increases registry storage and introduces a
  second archive copy per installed skill.
- Sign only the archive digest and label metadata advisory: preserves the current behavior but
  cannot verify installed content later.
- Require a signed transparency log: stronger publisher history, but needs a log service and
  consistency protocol outside Gabby's current local registry scope.
