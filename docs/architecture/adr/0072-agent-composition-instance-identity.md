# ADR 0072: Agent composition uses runtime instance identity for cycle detection

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

`AgentTool` prevents recursive delegation by tracking agent names. Names are configuration labels,
not unique runtime identities. Two distinct agents with the same name therefore appear to form a
cycle even when the parent delegates to a separate child instance.

## Decision

- Add a host-only, process-local `agent_instance_id` to runtime-created `ToolContext` values.
- Track this identity in the temporary delegation path and compare child instances by object
  identity. Continue using configured names for diagnostics and returned results.
- Keep name-based cycle detection for manually constructed `ToolContext` values that omit the
  optional identity field, preserving its use in direct extension tests and host code.
- Do not expose this identity to model context or use it for authorization, persistence, or
  cross-process references.

## Consequences

Distinct parent and child instances may share the same configured name without a false cycle error.
Recursive references to the same `Agent` instance remain rejected, and delegation depth stays
bounded. The identity is valid only in the current process and object lifetime.

## Alternatives considered

- **Require globally unique agent names:** rejected because it adds an unnecessary naming constraint
  to independently constructed agents and composed applications.
- **Track names plus configuration hashes:** rejected because equal configurations do not imply the
  same runtime object or lifecycle, and hashing does not solve identity.
- **Track identities only in `AgentTool`:** rejected because each tool handler needs the current
  parent identity from the runtime context to detect direct and indirect cycles reliably.

## Compatibility and evidence

`ToolContext` is a pre-1.0 public extension contract. The new optional field is additive. Regression
coverage verifies same-name parent/child delegation and retains direct cycle rejection. See the
[composition guide](../../EXTENSIONS.md), [extension contracts](../../EXTENSION_CONTRACTS.md), and
[ADR 0056](0056-stateless-agent-composition.md).
