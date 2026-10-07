# ADR 0053: Required tool output schemas

## Status

Accepted

## Context

The public tool contract calls for input and output schemas. Gabby already required input schemas,
but treated output schemas as optional. A tool without one could return any JSON value without
runtime validation, allowing malformed observations to reach a model or downstream consumer.

## Decision

- Require every `Tool` to declare an output JSON Schema at construction, alongside its input schema.
- Validate the schema at construction and validate each handler result before it becomes model
  context. Result serialization and the configured byte cap still apply.
- Keep output schemas out of the model's tool-call parameters; the schema is a runtime contract,
  not permission for the model to shape host output.
- Apply the requirement to host-trusted and sandboxed tools, including custom `tool.execute`
  handlers. Built-in shell and filesystem tools already have explicit result schemas.
- Treat this as a pre-1.0 API tightening. Missing schemas fail during construction; no permissive
  implicit schema is supplied.

## Consequences

Tool authors must describe the values their handlers return. Invalid results become typed tool
errors before model context construction, and tool registries hold only definitions with a checked
input/output contract. Broad schemas remain possible, but host applications should use specific
properties and required fields when the result shape affects application behavior.
