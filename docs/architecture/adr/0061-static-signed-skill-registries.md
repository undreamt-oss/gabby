# ADR 0061: Static HTTPS skill registries

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Portable skill packages have checksums and optional detached Ed25519 publisher signatures, but
developers still need to copy artifacts manually. Gabby needs a discovery and distribution contract
that remains usable without operating a new registry service and that does not trust catalog data to
select arbitrary network destinations.

## Decision

- Define a static registry format with a bounded `GET /v1/catalog.json` and fixed immutable
  `/v1/skills/{skill-id}/versions/{exact-semver}/package.gabskill[.sig]` artifact paths.
- Keep discovery metadata unsigned and explicitly untrusted. Never accept a package URL from catalog
  content; derive encoded artifact paths from the configured registry origin, skill ID, and exact
  version.
- Implement an async `SkillRegistryClient` over the existing HTTPX dependency. Allow host-injected
  headers, an injectable transport, explicit close/context-manager lifecycle, total request deadlines,
  byte-bounded streaming, HTTPS for remote registries, loopback-only HTTP, configurable environment
  proxy use, and no redirects.
- Require an exact SemVer version and a non-empty host-owned Ed25519 trust map for remote installs.
  Download into private temporary files, validate and authenticate in a scratch registry, verify the
  package ID and version match the request, then use the ordinary atomic package installer.
- Expose CLI `skill search`, `skill versions`, and `skill fetch`; the CLI reads optional bearer tokens
  from an environment variable and publisher keys from bounded raw-key files.
- Include an optional `generated_at` timestamp in catalogs produced by Gabby's builder. Let clients
  opt into a maximum catalog age; reject missing timestamps when freshness is required and reject
  timestamps more than five minutes ahead of the local clock.
- Keep registry hosting, publisher authorization, public-key distribution and rotation, and
  revocation under deployment/operator control. Catalog timestamps are unsigned and provide a
  staleness check, not publisher authentication or anti-downgrade protection.

## Consequences

Static hosts can serve registries without a Gabby-specific server or database. The client authenticates
package bytes independently of the hosting provider, but a valid signature does not make the skill
instructions safe. An untrusted or stale catalog can hide packages or advertise older signed versions;
consumers that require a release must choose and pin an exact version. The single catalog is bounded
to 4 MiB, so large registries will need a future pagination or sharding format revision.

## Alternatives considered

- **Accept arbitrary artifact URLs from the catalog:** rejected because catalog data could direct
  bearer credentials or package fetches to an attacker-controlled host.
- **Require a Gabby-hosted registry service:** rejected for v1 because a static site can distribute
  immutable signed packages with less operational infrastructure.
- **Resolve `latest` or version ranges remotely:** deferred because it makes installs mutable and
  creates downgrade and reproducibility decisions that belong to the consuming application.
- **Trust TLS or catalog checksums as publisher authentication:** rejected because hosting identity
  and transport integrity do not replace an explicit publisher trust map.

## Compatibility and evidence

This is a pre-1.0 additive API. Mock-transport tests cover bounded discovery, authentication header
injection, exact-version artifact paths, signed install, mismatched identity rejection, redirect
rejection, duplicate catalog keys, and URL policy. See the
[registry format and operator guide](../../SKILL_REGISTRY.md),
[package signature decision](0060-ed25519-skill-package-signatures.md), and
[threat model](../../THREAT_MODEL.md).
