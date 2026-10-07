# ADR 0123: Validated skill input and output schemas

- Status: Accepted
- Date: 2026-10-03

## Context

Skills are portable capabilities that may be reused by several agents. Some skills need a stable
request shape and a machine-readable result contract, independent of the agent that activates them.
Putting these contracts only on `AgentDefinition` makes them difficult to carry with a skill package
and reuse safely.

## Decision

- Add optional JSON Schema Draft 2020-12 `input_schema` and `output_schema` fields to file-backed
  and programmatic `SkillDefinition` objects.
- Validate both schemas using the same construction-time validator and immutable resolution path as
  agent output schemas. Reject remote references and remote identifiers; bound schema size.
- Validate `input_schema` against the JSON object containing `task`, `context`, and `memory` after
  skill selection and dependency activation, before the main reasoning loop or tool execution.
- Validate `output_schema` against the final response as JSON whenever that skill is active.
  Multiple active skill schemas and an agent-level output schema all apply to that response.
- Buffer streamed text until all active output schemas pass. Keep the existing response text and
  parsed structured-output metadata contract.
- Include schemas in the skill manifest so package checksums and signatures cover them. Expose
  contracts through agent and package inspection commands.

## Consequences

Skill authors can describe reusable data contracts alongside instructions, procedures, tools, and
verification requirements. A skill's input contract applies to the full stateless request envelope,
not only the plain-text task. Output schemas compose by intersection: one response must satisfy every
active contract. These schemas constrain data shape; they do not authorize tools or override runtime
policies. Streaming a structured result waits until validation completes.

## Alternatives considered

- Keep schemas only on the agent: rejected because the contract would not travel with a reusable
  skill package.
- Treat active skill output schemas as suggestions or pick only one: rejected because that would
  silently weaken declared contracts when skills compose.
- Validate skill inputs before selection: rejected because the runtime does not yet know which
  skill-specific input contracts apply.

## Compatibility and evidence

This is an additive pre-1.0 configuration change. YAML manifests and programmatic definitions use
the same validator. Focused config, runtime, package, and streaming acceptance passes. CLI inspection
coverage is also included in the skill contract suite.
