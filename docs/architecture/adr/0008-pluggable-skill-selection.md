# ADR 0008: Pluggable skill selection

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Skills are reusable procedures that should be selected for a task without requiring every skill's
instructions and tools in every request. Applications may need configured, rule-based, model-based,
or hybrid selection. A model-based selector adds a provider call, latency, token cost, and another
failure mode; those costs should not be imposed on all agents.

## Decision

- Define an async `SkillSelector` interface injected into `Agent` construction.
- Ship `ConfiguredSkillSelector` as the deterministic built-in implementation: skills without
  triggers are always selected; skills with triggers are selected when a configured trigger phrase
  appears case-insensitively in the task.
- Ship `ModelSkillSelector` as an explicit opt-in strategy. It uses an injected `ModelProvider`,
  returns only configured skill IDs, and reports provider model and usage metadata with its result.
- Return explicit `SkillActivation` records in a `SkillSelection`; record each activation method in
  the execution trace.
- Keep dependency expansion in the runtime. A selected skill activates its configured dependencies
  even when a dependency's own trigger does not match.
- Validate selector results against the agent's resolved skill set before model execution. Selection
  does not grant tools; the normal declaration, environment, and policy checks still apply.
- Bound selector work by the run deadline. The deterministic selector remains the default and does
  not call a model; applications opt into the additional provider call by injecting
  `ModelSkillSelector`.

## Consequences

Agent definitions remain portable across selector strategies, while hosts can inject a selector
without changing the agent YAML or runtime loop. The default remains deterministic and does not
require a second model call. `ModelSkillSelector` can choose a narrower set of skills, but its
provider, cost, availability, and data handling become part of the host's runtime configuration.
Selector implementations are trusted Python extensions running in the host process.

## Alternatives considered

- **Hard-code keyword activation in the runtime:** simple, but couples task policy to orchestration
  and makes custom strategies require replacing or forking the runtime.
- **Use model selection by default:** potentially better semantic matching, but adds latency, cost,
  provider dependencies, and another failure path to every run. Model selection therefore stays
  opt-in.
- **Load every configured skill:** predictable, but needlessly increases prompt context and exposes
  tools from skills that are not relevant to the current task.

## Compatibility and evidence

`Agent(..., skill_selector=...)` accepts custom async implementations. The default selector preserves
existing configured and keyword-trigger behavior. Runtime tests cover custom selection, dependency
activation, trace method reporting, and rejection of unknown or duplicate skill activations. Public
selector interfaces are currently part of the pre-1.0 API and do not yet have a formal compatibility
guarantee.
