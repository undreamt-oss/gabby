# ADR 0047: Checksummed local skill packages

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

Skills already have stable IDs, SemVer versions, exact dependency pins, and a filesystem registry
layout, but developers had to copy directories by hand. Manual copying is error-prone and does not
detect accidental corruption. The first distribution step should stay local and dependency-free;
publisher authentication and remote discovery require a separate trust and registry design.

## Decision

- Add deterministic `.gabskill` ZIP archives created by `gabby skill pack` from a validated skill
  directory, and install them with `gabby skill install` into `<registry>/<skill-id>/<version>`.
- Include a format-versioned JSON manifest listing each file's relative path, byte length, and
  SHA-256 digest. The installer requires an exact match between archive members and manifest,
  validates the skill ID/version from `skill.yaml`, and verifies every file before activation.
- Bound packages to 1000 files, 10 MiB per file, 100 MiB total uncompressed content, 10000 tree
  entries, and a 100 MiB archive. Reject symlinks, special files, duplicate paths, unsafe or
  non-portable paths, unsupported compression, malformed metadata, and checksum mismatches.
- Extract into a staging directory, validate the skill with Gabby's normal loader, then rename the
  completed tree into the registry. Never overwrite an existing exact version.
- Treat checksums as corruption detection, not publisher authentication. Skill instructions and
  resources remain untrusted. Users should review package content and separately verify its source.
- Keep remote registries, signatures, and publisher identity out of this local package format.

## Consequences

Skill authors can create reproducible local artifacts, exchange them through their chosen channel,
and install exact versions without copying directories manually. Installation does not execute
package code. A checksum does not prove who created an artifact, and an installed skill can still
influence model behavior through its instructions. Hosts remain responsible for trust decisions and
for configuring the installed skill's tools and knowledge access.

## Alternatives considered

- **Continue manual directory copying:** simple but error-prone and offers no integrity check.
- **Install directly from a remote Git or HTTP source:** deferred until publisher identity, immutable
  references, network policy, cache behavior, and revocation are designed together.
- **Sign packages in v1:** deferred because signing-key discovery and publisher trust policy do not
  yet exist; an unsigned signature field could create a false sense of authenticity.

## Compatibility and evidence

This adds a pre-1.0 package format and CLI commands. Tests cover deterministic artifacts, nested
versioned IDs, instruction and resource preservation, checksums, traversal, duplicate/existing
versions, source symlinks, and command-line pack/install. See the
[extension guide](../../EXTENSIONS.md) and [ADR 0041](0041-exact-versioned-skill-packages.md).
