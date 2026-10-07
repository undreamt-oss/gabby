# ADR 0030: Scoped request context for host tools

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-09-30.

## Context

`Environment` stored host-owned resource handles, but a registered tool could not access them
through Gabby's runtime contract. Applications had to close over resources when constructing tool
handlers. This obscured which resources a tool needed, prevented construction-time checks, and made
authenticated caller identity unavailable to ordinary host tools. Passing the resources or
identity in model arguments would expose them as model-controlled input and was not acceptable.

## Decision

- Add a host-only, request-scoped `ToolContext` containing agent/run identifiers, environment
  description and capabilities, selected resource handles, and the authenticated principal when
  one is available.
- A host-trusted `Tool` opts in with a Python `context_parameter` name and declares exact
  `context_resources`. Tools that do not opt in keep the existing handler signature and receive no
  context.
- Reject a context parameter that is also declared as a model input. Expose only the selected
  resources through a read-only mapping; fail agent construction if any declared resource is
  unavailable in its environment.
- Keep the context outside the model tool schema, request messages, observations, and automatic
  trace data. A handler remains trusted host code and can still transmit data through effects it
  performs itself.
- Treat the context as valid only during the tool invocation. The host owns resource lifetime and
  must not retain the request context after execution.
- Invoke callbacks with their argument mapping separate from Gabby's timeout control value, so
  ordinary tool parameters such as `timeout` do not collide with runtime bookkeeping.

## Consequences

Environment resources become usable through the framework's tool execution path, with explicit
per-tool grants checked during agent construction. Existing tools remain source-compatible unless
they opt into the new context. The resource mapping prevents handlers from changing its keys, but
the underlying resource objects retain their own mutability and trust boundary. Applications may
continue to use closures when that better fits their resource ownership model.

## Alternatives considered

- **Continue requiring closures:** preserves current behavior but leaves resource requirements
  implicit and gives the runtime no opportunity to validate grants.
- **Add resources to model arguments:** rejected because resource handles are host objects and
  identity must not be model-controlled or exposed in prompt context.
- **Inject every environment resource into every handler:** rejected because it grants ambient
  access and prevents least-privilege tool composition.

## Compatibility and evidence

The new `ToolContext`, `context_parameter`, and `context_resources` fields are pre-1.0 extension API
additions. Contract tests verify scoped resource injection, caller identity, schema separation,
missing-resource rejection, and a model-declared `timeout` input. Existing handlers without the
opt-in retain their previous behavior.
