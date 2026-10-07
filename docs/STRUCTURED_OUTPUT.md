# Structured agent outputs

Agents can declare a JSON Schema for their final response. Gabby includes the schema in the runtime
instructions, validates the model's final JSON locally, and returns both the original JSON text and
the parsed value in result metadata. This supports extraction, classification, routing, and other
machine-consumed results while leaving conversation state with the caller.

```yaml
name: request-classifier
model:
  provider: openai_compatible
  model: your-model
output_schema:
  type: object
  properties:
    category:
      type: string
      enum: [billing, account, technical, other]
    priority:
      type: integer
      minimum: 1
      maximum: 3
  required: [category, priority]
  additionalProperties: false
```

The same contract works with programmatic definitions:

```python
from gabby import Agent
from gabby.config import AgentDefinition

agent = Agent(
    AgentDefinition(
        name="request-classifier",
        model={"provider": "openai_compatible", "model": "your-model"},
        output_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 3},
            },
            "required": ["category", "priority"],
            "additionalProperties": False,
        },
    )
)


async def classify_request() -> dict[str, object]:
    async with agent:
        result = await agent.arun("The customer cannot sign in after resetting a password.")
        return result.metadata["structured_output"]
```

`result.output` remains the model's JSON text for existing consumers. When `output_schema` is
configured, `result.metadata["structured_output"]` is the validated JSON value. The `/run` API and
SSE `completed` event expose it in the result metadata. Without a schema, response behavior is
unchanged.

Only JSON Schema Draft 2020-12 object schemas are accepted. Schema references must stay local; remote
references and remote schema identifiers are rejected so output validation never fetches a URL.
Construction snapshots the schema alongside the resolved agent definition. Final output must be
strict JSON with no duplicate object keys or non-finite numbers, and is limited to 1 MiB. An invalid
response fails the run with a sanitized runtime error. Streaming agents buffer model text until the
final value passes validation, then emit it as one `text_delta`; invalid partial JSON is never sent
as result text. The schema guides model generation, while runtime validation enforces the returned
contract.

See [ADR 0086](architecture/adr/0086-validated-structured-agent-output.md) for the contract and
tradeoffs.
