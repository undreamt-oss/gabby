# ADR 0122: Bounded plan revisions from tool observations

**Status:** Accepted by the project architect.  
**Date:** 2026-10-03.

## Context

An optional initial plan gives the reasoning model a task decomposition, but it cannot account for
new information returned by tools. The runtime must support replanning without allowing model
output to expand capabilities or adding planner calls to every agent run.

## Decision

- Keep replanning opt-in with `policies.max_replans`, default `0` and maximum `3`.
- Require an injected `Planner` when the configured replan count is nonzero.
- After each complete tool-call batch, pass the planner the previous validated plan and bounded
  observations from that batch. Include at most 16 tool results and at most 4 KiB per result.
- Treat prior plans and observations as untrusted input. The plan remains advisory, and the
  resolved tool allowlist, permissions, approval requirements, and per-run budgets remain enforced
  on every model tool call.
- Count planning and replanning provider calls against the run deadline and retry policy. A
  configured revision failure fails the run with the existing sanitized planning error contract.
- Record each revised plan in a `replanning` trace event and emit `plan_updated` to stream clients.
- Expose `PlanningObservation` in the pre-1.0 extension API. Custom planners must accept
  `previous_plan` and `observations` to use revisions.
- Document that bounded tool-result excerpts are sent to the configured planner provider.

## Consequences

Agents can revise high-level plans when tool observations change the next steps. Replanning adds
provider latency, token use, and data transfer only when explicitly configured. A provider may
receive up to 64 KiB of recent tool-result excerpts per revision, in addition to the original task,
context, memory, and authorized capability descriptions. A planner failure after a tool call fails
the configured run; tool side effects already completed cannot be rolled back.

## Alternatives considered

- **Replan after every tool call by default:** rejected because it adds cost, latency, and a new
  failure mode to reactive runs.
- **Let the reasoning model revise its own plan without a planner contract:** this remains the
  default model behavior; it does not provide a separate structured planning trace or a bounded
  planner interface.
- **Send the full conversation to the planner:** rejected because it can duplicate large context
  and unrelated observations. Gabby sends only the most recent bounded tool batch.

## Compatibility and evidence

`policies.max_replans` and the added planner keywords are pre-1.0 contracts. Evidence must cover
unchanged behavior at the default zero, bounded observation handling, policy enforcement after plan
revision, planner failures, deadline/retry behavior, and the `plan_updated` stream event.
