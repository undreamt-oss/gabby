# ADR 0031: Minimum Docker version for native Windows sandbox networking

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-09-30.

## Context

Gabby requires its tool containers to start with networking disabled. Native Windows Docker
containers use Docker's `none` network mode through either the CLI or Engine API adapter. Moby
29.1.3 and earlier could panic while starting a Windows container in this mode; Moby 29.1.4 fixed
that failure ([release notes](https://github.com/moby/moby/discussions/51834)). Starting an
unsupported container can affect the daemon beyond the current Gabby run.

## Decision

- Before starting a native Windows container, the built-in Docker CLI and API adapters read and
  validate the Docker server version.
- Require Docker Engine 29.1.4 or newer. Reject unknown, malformed, and older versions before
  container creation; do not weaken the network mode or retry with networking enabled.
- Extend the `EngineAdapter` contract with an explicit native Windows support check. An injected
  adapter that cannot provide that check fails closed for Windows images.
- Linux container startup is unchanged.

## Consequences

Native Windows sandbox users need a Docker daemon at or above the documented minimum. The minimum
is checked from the server version, so a newer CLI talking to an older daemon does not bypass the
guard. This version check avoids the known daemon crash; it does not replace live acceptance of
network isolation, Hyper-V isolation, resource controls, mounts, and cleanup on Windows.

## Alternatives considered

- **Start the container and rely on Docker's error handling:** rejected because versions before the
  fix could panic the daemon.
- **Fall back to the default network:** rejected because it violates Gabby's disabled-network
  sandbox policy.
- **Reject all native Windows containers:** rejected because the accepted initial platform scope
  includes Docker native Windows containers with Hyper-V isolation.

## Compatibility and evidence

This is a pre-1.0 `EngineAdapter` contract addition. Unit tests cover supported, too-old, malformed,
and prerelease versions through CLI and API adapters, and verify that an unsupported fake engine is
rejected before container startup. Live Windows validation remains outstanding.
