# ADR 0067: Host-injected skill trust policy

- Status: Accepted
- Date: 2026-10-02

## Context

Signature verification at install and an operator-run audit provide useful supply-chain checks,
but neither prevents an application from constructing an agent from an unsigned or subsequently
modified skill directory. A hosted consumer needs a fail-closed way to apply its own signer trust
and revocation decisions when it constructs an agent.

## Decision

- Add optional `SkillTrustPolicy` injection to `Agent`. It is host-owned and cannot be configured
  through agent YAML.
- The policy snapshots the supplied raw Ed25519 public keys and revoked key IDs at construction.
- When supplied, the policy requires every resolved skill to have valid install provenance with a
  trusted, non-revoked v2 signature. It verifies the current installed file tree against the signed
  canonical manifest before the `Agent` is resolved. Unsigned, in-memory, legacy v1, untrusted,
  revoked, or tampered skills fail construction. The runtime revalidates installed contents before
  every run and stream; see [ADR 0085](0085-per-run-skill-integrity.md).
- When no policy is supplied, local and legacy skill loading keeps its existing behavior. This is
  useful for development and does not claim publisher authentication.
- Trust keys and revoked IDs are construction-time snapshots. Rotating trust or revoking a key
  requires reconstructing agents before new requests are admitted; installed content is checked
  again at each run boundary.

## Consequences

Hosted applications can make trusted v2 skill signatures an agent-admission requirement and fail
before model-provider construction. Public trust keys remain deployment-owned and are not stored in
agent definitions. Hosts must keep package directories quiescent during construction and execution,
and reconstruct agents after trust changes. Per-run checks detect modified content before execution
but do not provide a transparency log, continuous revocation, or an atomic filesystem snapshot
against a hostile concurrent writer.

## Alternatives considered

- Always require signatures: rejected because local development skills and existing integrations
  must remain usable without a publisher trust infrastructure.
- Treat an earlier CLI audit as authorization: rejected because files can change between audit and
  agent construction.
- Put public keys in agent YAML: rejected because trust belongs to the consuming application and
  deployment, not to the publisher-controlled agent definition.
