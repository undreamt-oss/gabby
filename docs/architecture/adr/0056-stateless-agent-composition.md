# ADR 0056: Bounded stateless agent composition

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-01.

## Context

Gabby agents need a direct composition path for workflows where one specialization delegates a
focused task to another. Treating agents as ordinary model context would share state and policy
implicitly; exposing arbitrary Python callbacks would leave delegation limits and data forwarding
to each application.

## Decision

- Add `AgentTool`, which adapts an async child `Agent` into Gabby's existing `Tool` contract.
- Require an explicit task string as the tool input. Do not forward caller context, memory, metadata,
  or principal by default. Principal forwarding is an explicit host option for child approval flows.
- Keep child agent definitions, models, tools, knowledge sources, verification, and policies
  independent. The host owns both agent lifecycles.
- Require the parent definition to declare the tool and grant its `agent:invoke` permission.
- Bound input and result bytes and delegate within the parent tool timeout and run deadline. Reject
  delegation cycles and nesting deeper than eight agents.
- Return the child output and trace ID to the parent tool loop. Set the child trace's
  `parent_trace_id` metadata and request event to the parent run ID. The child trace remains
  independently exported by its configured tracer; Gabby does not merge the trace trees.

## Consequences

Applications can compose stateless specialists using the same runtime and policy enforcement as
other tools. Each boundary is visible to the parent model as a named capability and is covered by
the parent's tool budgets. The host must manage and close every agent in the composition. Child
context sharing, result shaping beyond output and trace ID, remote agent references, and merged
trace views remain application concerns or future work.

## Alternatives considered

- **Pass the parent request and memory automatically:** rejected because that silently transfers
  caller data to another model/provider and makes data boundaries hard to review.
- **Let every application write a custom delegation handler:** rejected because timeout, byte,
  policy, lifecycle, and cycle behavior would vary between applications.
- **Create a separate multi-agent scheduler and graph DSL:** deferred because the tool contract
  already provides bounded, observable execution and does not require persistent graph state.

## Compatibility and evidence

`AgentTool` is a pre-1.0 public API. Tests cover task-only forwarding, parent/child execution,
principal forwarding defaults and opt-in, direct cycle rejection, and child cancellation at the
parent tool timeout. See the [composition documentation](../../EXTENSIONS.md) and
[extension contract reference](../../EXTENSION_CONTRACTS.md).
