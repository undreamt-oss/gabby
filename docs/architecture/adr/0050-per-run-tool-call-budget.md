# ADR 0050: Per-run tool invocation budget

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-01

## Context

`max_steps` limits model rounds, not the number of tools a model can request in one response. A
bounded model response can contain a large tool-call batch, so step limits alone do not bound the
number of handler invocations during one run.

## Decision

- Add the agent policy `max_tool_calls`, counting tool invocations across all model rounds.
- Default the per-run limit to 64; allow agent configurations to set an integer from 1 through 1,024.
- Before validating or executing a model tool-call batch, reject it if the entire batch would exceed
  the remaining run budget. No handler in an over-budget batch runs.
- Keep the existing run deadline and per-tool timeout as independent limits.

## Consequences

The runtime bounds the number of handler invocations even when one model response requests many
tools. Callers can raise the limit for workflows that require more actions, up to the hard maximum.
The count does not limit cost or duration of a single call; deadlines, resource policies, result
bounds, and sandbox controls remain necessary.

## Alternatives considered

- **Rely on `max_steps` and the overall deadline:** rejected because one provider response can
  request many actions before another model step.
- **Count only successful tools:** rejected because failed calls still consume runtime and may have
  observable side effects.
- **Accept arbitrarily large configured limits:** rejected to preserve a finite upper bound.

## Compatibility and evidence

This adds a pre-1.0 policy. Shared YAML and in-memory definition validation reject values outside
the supported range. Runtime tests verify an over-budget batch fails before any handler executes.
See the [agent configuration guide](../../../README.md) and [ADR 0040](0040-reject-unknown-owned-config-fields.md).
