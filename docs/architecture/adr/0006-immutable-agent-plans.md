# ADR 0006: Immutable resolved agent plans

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

An `AgentDefinition`, skill definition, reusable environment, and tool registry can all be changed by
their owner. If a constructed agent continues reading those mutable objects during execution, a later
edit can silently change the agent's instructions, policy, model settings, exposed tools, or skill
procedures. It also makes validation timing unclear: errors may appear only after the first request.

## Decision

- Constructing an `Agent` resolves the source definition and declared skills into an immutable
  `ResolvedAgentDefinition` plan.
- Nested configuration mappings become read-only mappings; configuration sequences become tuples.
- Skill references resolve once, in deterministic dependency order. Skill procedures and metadata are
  copied into frozen resolved skill values. File edits and in-memory registry edits affect only later
  agent constructions.
- Environment descriptions, capabilities, resource mapping membership, allowlists, tool schemas,
  and tool registry membership are snapshotted into the agent-local resolved environment/registry.
- Every tool granted by the agent or any attached skill must exist in the resolved registry;
  missing tool references fail construction before a model provider is constructed.
- Configured knowledge sources require an injected `Retriever`; `verification.enabled: true`
  requires an injected `Verifier`. Missing runtime services fail construction.
- `Tool` declarations are immutable and schema snapshots are used both for validation and model
  exposure. The handler callable itself is retained by reference as an injected runtime capability.
- Model, retriever, verifier, sandbox adapter, and handler implementations are runtime service
  references; the plan freezes the configuration that selects them, not their internal service state.
- Dynamic skill selection chooses among the resolved skills for each request. It does not reload
  modified skill files during execution.

## Consequences

Agent construction is the validation and snapshot boundary. To apply config or skill changes, build a
new `Agent`. The plan is safe to share across repeated stateless runs as long as injected runtime
services satisfy their own concurrency contracts. Host resource handles and handler closures remain
shared references by design and must be treated as trusted integrations. Resolved definition and skill
types are public inspection contracts; their fields must remain immutable and versioned deliberately.

The API adds `skill_registry` injection for in-memory reusable skill definitions. Tool registries are
copied into an agent-local registry and frozen after construction, so caller-owned registry changes do
not alter an existing agent.
