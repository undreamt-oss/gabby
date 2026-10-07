# ADR 0084: Fetch signed skill dependency closures

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-03

## Context

Agent construction already resolves skill dependencies locally, and portable packages carry those
dependencies as versioned skill references. Remote `skill fetch` installed only the requested
archive, leaving consumers to discover and fetch the dependency closure manually. This made
composed skills harder to reuse and allowed dependencies to be missed before agent construction.

## Decision

- Add an opt-in `--with-dependencies` mode and async
  `SkillRegistryClient.install_with_dependencies()` method.
- Read one bounded catalog, require exact package identity and trusted detached signatures, and
  validate the full dependency graph before writing to the local skill registry.
- Require exact `skill-id@version` references for automatic remote dependency resolution. Do not
  silently select a mutable latest version for an unpinned dependency.
- Reject cycles, multiple versions of one stable skill ID, catalog misses, invalid signatures, and
  conflicting installed exact versions before beginning installs.
- Cap a dependency closure at 256 packages, 1 GiB of archives, and five minutes. Install packages
  in dependency-first order through the existing atomic single-package installer.
- Reuse an already-installed exact version only after verifying its trusted v2 content signature
  and matching its archive digest. A partially completed install can be safely resumed by repeating
  the same request.

## Consequences

Consumers can install the portable skill dependency closure using the same exact-version and
publisher-trust contracts that agent construction uses. The entire set is not a single filesystem
transaction: each version is atomically installed, and a process or filesystem failure can leave a
valid prefix. Repeating the request validates and skips matching versions. Deployments requiring
all-or-nothing registry updates should build a new registry tree and switch it through their own
atomic deployment mechanism.

## Alternatives considered

- **Resolve unpinned dependencies to the only version currently in the catalog:** rejected because
  catalog changes could alter future resolution and conflict with multiple versions already in the
  local registry.
- **Install each package as soon as it is downloaded:** rejected because a missing or invalid later
  dependency would leave an avoidable partial closure.
- **Require callers to issue one `fetch` command per dependency:** rejected because it repeats graph
  discovery and makes the composition contract manual.
- **Atomically replace the entire caller-owned registry:** deferred because the registry may contain
  unrelated packages and portable filesystems do not provide one cross-platform transaction for a
  multi-directory update.

## Compatibility and evidence

This is a pre-1.0 additive API and CLI option; single-package `fetch` keeps its existing behavior.
Contract tests cover signed dependency-first installation, idempotent retry, and rejection of
unpinned dependencies before local registry mutation. See the
[skill registry guide](../../SKILL_REGISTRY.md),
[exact-version package ADR](0041-exact-versioned-skill-packages.md), and
[static registry ADR](0061-static-signed-skill-registries.md).
