# ADR 0014: Optional structured planning

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Some agents benefit from a visible task decomposition before tool use. Making a planning call
mandatory would add provider latency, cost, data transfer, and another failure mode to every run.
Planning must not replace runtime policy or tool authorization.

## Decision

- Define an injectable async `Planner` contract returning a `PlanningResult` with a bounded,
  structured `ExecutionPlan`.
- Keep planning disabled by default. The current reactive model and tool loop remains the default.
- Ship an opt-in `ModelPlanner` that receives the task, caller context and memory, active skill
  descriptions, and only tools that have already passed policy authorization.
- Bound model plans to 16 steps, 1,200 characters per text field, and a 16 KiB response.
- Validate plans returned by every planner implementation before use.
- Supply a produced plan to the reasoning model as advisory, untrusted context. It cannot grant
  skills or tools, bypass policy, or claim that actions have already happened.
- Record the plan and planner latency/usage in the execution trace. Planning time consumes the same
  per-run deadline as all other work.
- Emit a typed `plan_created` event to SSE consumers when an optional plan is ready.
- Treat the planning provider as an explicit data recipient: opt-in planning sends caller context
  and memory to that provider.

## Consequences

Applications can enable structured decomposition without changing agent YAML or the default
runtime behavior. Planner failures fail that opted-in run clearly; Gabby does not silently continue
without the requested planning stage. Planner implementations are trusted Python extensions and
remain under the pre-1.0 extension stability policy.

## Alternatives considered

- **Plan every run:** simpler configuration, but imposes cost and data transfer even when the task
  does not benefit from decomposition.
- **Keep planning inside the ordinary model loop:** avoids a separate abstraction, but provides no
  explicit plan contract, bounded validation, or trace point.
- **Let plans choose tools directly:** rejected because planning output must not grant capabilities;
  the existing runtime tool allowlist and policy engine remain authoritative.

## Compatibility and evidence

`Agent(..., planner=...)` injects the optional strategy. `ModelPlanner` uses the shared
`ModelProvider` contract. Public planner types are pre-1.0. Runtime acceptance evidence must cover
malformed plans, provider failures, cancellation/deadlines, policy-scoped capabilities, and unchanged
default behavior before this feature is treated as production-accepted.
