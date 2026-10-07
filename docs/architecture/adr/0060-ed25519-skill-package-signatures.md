# ADR 0060: Detached Ed25519 signatures for skill packages

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Portable `.gabskill` archives carry per-file SHA-256 checksums. Those checksums detect accidental
or malicious modification only when compared with an independently trusted digest; they do not
identify the publisher. Remote registries are not implemented, but signed packages are a prerequisite
for safely distributing skills through shared registries.

## Decision

- Add an optional `skill-signing` extra based on `cryptography` and Ed25519.
- Sign the exact archive SHA-256 digest together with a domain separator and publisher key ID.
- Store signatures in a detached JSON sidecar named `ARCHIVE.sig` by default; never embed private or
  public keys into the package or signature document.
- Require the host to provide a key-ID-to-raw-public-key trust map for verification. Trust roots,
  key rotation, and revocation stay under host control.
- Allow unsigned local installs by default for current workflows. Callers that handle shared or
  remote packages can set `require_signature=True`; supplying an explicit signature or trust map
  also triggers verification.
- Expose CLI `skill sign`, `skill verify`, and signature-required `skill install` operations. The
  CLI reads private keys from a base64-encoded environment variable and public keys from bounded
  host-managed raw-key files.

## Consequences

The same package format can be integrity-checked and authenticated without making cryptography a
core installation dependency. A successful signature proves that the signed bytes were authorized
by one configured key; it does not make skill instructions safe or guarantee publisher identity
beyond the host's trust mapping. Remote registry transport and discovery remain separate work.

## Alternatives considered

- **Treat the embedded SHA-256 list as publisher authentication:** rejected because an attacker who
  replaces a package can replace its unsigned checksums too.
- **Embed a public key in the archive:** rejected because a self-supplied key does not establish
  trust and would complicate rotation.
- **Make signatures mandatory for every local install:** rejected because it would break local
  authoring and package development before hosts have established trust roots.
- **Add a registry service and trust model in the same change:** deferred; transport, discovery,
  registry authentication, and key lifecycle need separate contracts.

## Compatibility and evidence

This is a pre-1.0 additive API. Tests cover raw key handling, signature generation and verification,
tampered packages, untrusted keys, signature-required installs, CLI secret injection, and installer
cleanup. See the [extension guide](../../EXTENSIONS.md), [threat model](../../THREAT_MODEL.md), and
[agent configuration examples](../../../README.md#define-an-agent).
