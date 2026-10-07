# ADR 0068: Per-execution skill revocation checks

- Status: Accepted
- Date: 2026-10-02

## Context

`SkillTrustPolicy` snapshots trusted keys and local revocations when an agent is constructed. A
long-lived agent therefore continues accepting a signer if the host revokes that publisher later.
Reconstructing every agent immediately may not be practical for a running service.

## Decision

- Add an optional async `SkillRevocationChecker` to `Agent`. It is accepted only alongside a
  `SkillTrustPolicy`; the checker receives only the signer IDs used by that agent's resolved skills.
- `Runtime` checks the current revocation state before sandbox startup, model calls, or tool calls
  for every `/run`, async run, and stream execution. The check defaults to a five-second timeout,
  configurable per agent from above zero through 60 seconds, and never exceeds the run's remaining
  deadline. Timeout, malformed checker behavior, or backend failure denies execution.
- A revoked signer raises `SkillRevokedError`; an unavailable or timed-out checker raises
  `SkillRevocationUnavailable`. HTTP run requests receive sanitized 403 and 503 responses
  respectively. A stream reports a sanitized typed SSE error because its response has begun.
- Add `SQLiteSkillRevocationStore` for durable single-host key revocation, reinstate operations,
  and inspection. Host applications can inject a different checker for distributed state.
- ADR 0069 supersedes admission-only behavior with bounded polling that interrupts active
  executions when a signer is revoked or the checker becomes unavailable.

## Consequences

Long-lived agents can observe new revocations without reconstruction, while every request pays one
bounded trust-store lookup. Store failure is fail-closed. SQLite requires a local filesystem with
SQLite locking semantics and does not guarantee propagation across hosts. Trust key rotation and
agent reconstruction remain deployment responsibilities. See ADR 0069 for active-run polling and
cancellation semantics.

## Alternatives considered

- Reconstruct agents after every revocation: retained as the way to apply trust-key changes, but
  insufficient as the only control for long-lived services.
- Poll during model and tool steps: rejected because it adds latency and failure points throughout
  one execution; admission-time checking gives each transient run one consistent decision.
- Put signer revocations in agent YAML: rejected because publishers must not control host trust.
