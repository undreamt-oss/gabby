# ADR 0025: Host-owned tool approval

## Status

Accepted

## Decision

Tools can declare `requires_approval=True`. Before invoking such a tool, the runtime calls an
injected async `ApprovalHandler` with a typed request containing the agent, run, tool, call ID, and
validated arguments. Only an explicit `ApprovalDecision(approved=True)` allows execution. If a
tool requires approval and no handler is configured, Gabby denies the call. Handler failure,
invalid responses, and deadline expiry also fail closed.

For HTTP executions, the authenticated `Principal` is passed separately from caller-controlled
context and metadata to the approval handler. It is not added to model context. Embedded callers may
provide their host-authenticated principal through `Agent.arun` or `Agent.astream`.

Approval callbacks run within the run deadline. The runtime emits `approval_required`,
`approval_granted`, and `approval_denied` stream events and records the decision in the transient
execution trace. It does not retain approval state after the run. The consuming application owns
the approval interface, identity checks, persistence, and policy for deciding who can approve.

## Consequences

- Embedded applications can provide synchronous or asynchronous host approval logic; synchronous
  callbacks use the existing bounded callback bridge.
- Service hosts can build approval UX around the injected callback and progress events without
  Gabby owning users, sessions, or conversation history.
- An approval is scoped to one tool call and one run. It is not a reusable grant.
- Gabby does not claim a user-facing approver identity unless the host includes and audits it in its
  own approval system.

## Validation

Unit tests verify approval-before-execution, deny-by-default behavior, and trace recording. An
authenticated API integration test verifies the request principal reaches the approval handler
without appearing in model context. Building a user-facing approval UX and durable audit backend
remain host application responsibilities.
