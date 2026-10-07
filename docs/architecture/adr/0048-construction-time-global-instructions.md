# ADR 0048: Construction-time global instructions

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

The runtime already supports agent and skill instructions and receives a caller task. Applications
that compose multiple agents need a shared instruction layer for organization-wide behavior without
copying the same text into every agent definition. This layer must remain distinct from runtime
policy, which is enforced by code and cannot be delegated to prompt text.

## Decision

- Add an optional `global_instructions` string to `Agent` construction and `Agent.from_file()`.
- Capture the string at construction so one agent's behavior does not change if the caller later
  changes its source variable.
- Present instructions in this order: Gabby runtime requirements and environment capabilities,
  global instructions, agent instructions, active skill instructions, then the caller task.
- Apply global and agent instructions to Gabby's built-in model-based skill selector and planner;
  each auxiliary prompt retains its fixed output and capability constraints. Injected custom
  selectors and planners own their internal prompt construction.
- Keep caller context and memory, retrieved documents, and tool observations as untrusted data.
- State explicitly that global instructions cannot grant tools or override runtime requirements or
  enforced policies. Policies continue to be checked by the runtime independently of the model.
- Reject non-string values with `ConfigError`; bound the complete serialized request through the
  existing per-provider-call request limit.

## Consequences

Applications can share behavioral guidance across agents while keeping each definition portable.
Built-in auxiliary model calls receive the same global and agent guidance as the main reasoning call.
Changing global instructions requires constructing a new agent, consistent with immutable resolved
agent behavior. This is a prompt-composition aid, not a security boundary; the model may still fail
to follow instructions.

## Alternatives considered

- **Copy shared instructions into every agent file:** duplicates content and allows definitions to
  drift.
- **Make global instructions a policy mechanism:** rejected because text cannot enforce permission
  boundaries; the policy engine remains authoritative.
- **Load global instructions from a process-global mutable singleton:** rejected because it can make
  already-constructed agents change behavior implicitly and complicates isolation between services.

## Compatibility and evidence

This is a pre-1.0 additive constructor argument. Runtime contract tests verify its presence and its
ordering relative to agent and skill instructions; constructor tests reject invalid types. See the
[extension guide](../../EXTENSIONS.md).
