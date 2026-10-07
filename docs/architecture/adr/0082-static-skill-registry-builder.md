# ADR 0082: Build static skill registry artifacts locally

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-03

## Context

Gabby clients can search and install from a static skill registry, and publishers can pack and sign
individual skill archives. Publishers still had to assemble the catalog and exact-version file tree
by hand, which made it easy to publish malformed metadata or a package whose identity did not match
its path.

## Decision

- Add a local registry builder that accepts publisher-signed package archives and a host-supplied
  trust map, verifies the staged archive and sidecar, validates the extracted skill definition, and
  checks package identity before publishing the artifact into the generated file tree.
- Generate only the existing v1 static registry format. Do not add a registry server, mutable
  version aliases, trust-key discovery, or signature-free catalog publication.
- Build in a private sibling staging directory, enforce package-count, aggregate archive, per-file,
  signature, and catalog bounds, and require a new output directory. An optional prior static
  registry may seed a build; only catalog-referenced artifacts with verified signatures and matching
  package identities are retained.
- Keep registry hosting, access control, trust distribution, and deployment atomicity with the
  publisher's hosting platform.
- Expose the workflow through `gabby skill catalog build` and
  `build_static_skill_registry()`.

## Consequences

Publishers can produce or update the exact layout consumed by `SkillRegistryClient`, preserving
older signed versions without manual artifact copying. The catalog remains untrusted discovery
metadata; package signatures and consumer-owned trust keys remain the authenticity boundary. Large
registries still require a future paginated catalog format.

## Alternatives considered

- **Continue manual catalog and path assembly:** rejected because it duplicates a machine-readable
  contract and makes package/path mismatches easy.
- **Add a Gabby registry server:** deferred; static hosting remains sufficient for the current
  format and avoids introducing a hosted service and its operational contract.
- **Trust the package's self-reported key or signature without a host trust map:** rejected because
  the publisher key must be authenticated outside registry-controlled files.

## Compatibility and evidence

This is a pre-1.0 additive CLI and Python API. Tests cover initial and seeded builds, tampered
existing artifacts, package identity, and bounded catalog validation. The builder reuses the
existing bounded package installer, signature verifier, and catalog parser. See the
[registry format and publisher guide](../../SKILL_REGISTRY.md) and
[static registry client decision](0061-static-signed-skill-registries.md).
