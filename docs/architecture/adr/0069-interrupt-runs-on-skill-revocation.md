# ADR 0069: Interrupt active runs on skill publisher revocation

- Status: Accepted
- Date: 2026-10-02

## Context

ADR 0068 checked publisher revocations only when a run was admitted. A long-running model call,
tool call, or stream could therefore continue using a skill after its publisher was revoked. A
service needs bounded detection for active executions while preserving the per-run deadline and
typed failure behavior.

## Decision

- When an agent has both a `SkillTrustPolicy` and a `SkillRevocationChecker`, poll the checker
  during each active execution. The default interval is one second; construction accepts values
  from 0.1 through 60 seconds.
- Bound each poll by the configured revocation-check timeout and the run's remaining deadline.
  A revoked signer or unavailable checker cancels the execution task and becomes the existing
  typed `SkillRevokedError` or `SkillRevocationUnavailable` result. Runs without a checker do
  not create a polling task.
- Async model and tool operations receive normal task cancellation. `Agent.astream` observes
  execution-task completion as well as queued events, so a translated cancellation reaches the
  stream consumer instead of leaving it waiting for another event. HTTP streams emit the existing
  sanitized SSE error event.
- Cancellation does not roll back completed tool side effects. Async handlers can clean up on
  cancellation; synchronous host callbacks must cooperate with the provided cancellation token,
  and an uncooperative callback may continue in its worker after the agent request ends.
- Checkers and their storage must provide the consistency promised by the deployment. Polling is
  process-local; Gabby does not create distributed revocation propagation guarantees.

## Consequences

Revocation detection latency is bounded by the poll interval plus checker latency, subject to the
run deadline. Lower intervals increase checker load; operators should choose an interval that
matches their incident response needs and backing-store capacity. Fail-closed checker errors stop
active work as well as denying new runs.

## Alternatives considered

- Admission-only checks from ADR 0068: rejected as insufficient for long-running executions after
  a key compromise.
- Check only between model/tool steps: rejected because a single model or tool call may itself be
  long-running.
- Cancel immediately from an external revocation notification: deferred because the checker
  contract has no push subscription or cross-process cancellation lifecycle.
