# ADR 0120: Inference-only PEFT adapter loading

- Status: accepted
- Date: 2026-10-03

## Context

Runtime specialization through instructions, tools, skills, and retrieval does not change model
weights. Some deployments also need to run a separately trained parameter-efficient adapter with a
local Transformers base model. Loading an adapter should remain an optional provider capability and
must not turn ordinary agent construction into a training operation.

## Decision

`TransformersProvider` accepts an optional `adapter_id` and `adapter_revision`. When configured, it
loads the base model first and attaches the PEFT adapter in inference-only mode. It passes the
provider's host-managed Hugging Face token, cache directory, and local-only setting to adapter
loading, and requires safetensors adapter weights. The adapter remains an explicit separate trust
input. A pinned base and adapter revision are recommended for reproducibility.

The `transformers-adapters` extra supplies PEFT without changing the dependencies required by other
providers. PyTorch installation remains platform-selected by the host. This feature loads an
existing adapter only; dataset preparation and weight training remain outside Gabby's runtime.

## Consequences

- Local Transformers agents can use compatible PEFT adapters without changing runtime or agent
  execution semantics.
- Existing agents and non-Transformers providers do not install PEFT or alter their behavior.
- Adapter compatibility, provenance, license, and hosted artifact availability remain the
  deployment owner's responsibility.
- Training, evaluation of adaptation quality, quantization, and adapter publishing are not part of
  this interface.

## Alternatives considered

- Load adapters only through a custom provider: rejected because this is a useful local inference
  capability that can be added without changing the provider contract.
- Add a training pipeline to core: deferred because training dependencies and accelerator setup are
  platform-specific and separate from stateless agent execution.
- Merge adapter weights into the base model: rejected because it changes the base artifact and
  removes the ability to select and audit the adapter independently.
