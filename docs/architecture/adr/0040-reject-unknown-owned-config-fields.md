# ADR 0040: Reject unknown Gabby-owned configuration fields

- Status: accepted
- Date: 2026-10-01

## Context

Agent and skill YAML loaders previously selected known keys and silently ignored all others. Nested
policy and sandbox mappings did the same. A typo in a setting such as `require_sandbox`, workspace
access, or a resource limit could therefore leave the default in effect without telling the
operator. The same validation functions process in-memory `AgentDefinition` values, so permissive
unknown policy keys affected both construction paths.

## Decision

- Reject unknown top-level agent and skill keys.
- Reject unknown fields in Gabby-owned environment, knowledge, policy, verification, and sandbox
  mappings, including sandbox workspace, resource, and API submappings.
- Keep `model` provider-specific options open so injected providers can define their own fields.
- Keep names inside `environment.resources` open because they identify host-owned resources.
- Require extensions to validate their own configuration fields.

## Consequences

Configuration typos fail during loading or construction instead of silently falling back to defaults.
Adding a Gabby-owned field now requires updating the allowed-key set and its validation. This is a
pre-1.0 compatibility change; previously ignored fields now produce `ConfigError`.

## Alternatives considered

- **Continue ignoring unknown fields:** allows misspelled security and resource controls to appear
  accepted while having no effect.
- **Reject unknown keys in every mapping, including `model` and `environment.resources`:** would
  prevent provider adapters and host environment integrations from using namespaced extensions.
- **Warnings only:** depends on operators capturing warnings and does not fail closed for policies.

## Compatibility and evidence

Ruff, mypy, and documentation checks pass for the implementation. The full runtime test suite has
not been run against this change. Strict unknown-field validation is limited to Gabby-owned
mappings; provider-specific model configuration and host resource names remain open by design.
