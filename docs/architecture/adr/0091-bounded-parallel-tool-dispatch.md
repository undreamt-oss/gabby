# ADR 0091: Bounded parallel tool dispatch

## Status

Accepted

## Context

Models can return a batch of tool calls in one response. Running all calls sequentially adds
latency when the calls are independent, but automatically parallelizing arbitrary host handlers can
race shared state and external side effects. Runtime callbacks also have an ordered tracing
contract.

## Decision

- Keep sequential dispatch as the default through `policies.max_parallel_tool_calls: 1`.
- Add an explicit `Tool.parallel_safe` declaration, defaulting to false. It is valid only for a
  host-trusted in-process handler that does not require approval; sandbox tools and approval-gated
  tools remain sequential.
- Allow parallel dispatch only when the model's entire tool-call batch resolves to registered tools
  that all declare `parallel_safe=True`. Mixed batches run sequentially so the runtime does not
  silently reorder calls around potentially stateful tools.
- Bound concurrent tasks with `policies.max_parallel_tool_calls`, from 1 through 32. Tool result
  messages are returned to the model in original model-call order even when handlers finish in a
  different order. The existing per-run tool-call budget and shared execution deadline still apply.
- Keep tracer-enabled runs sequential so event-by-event tracer callbacks retain their ordered
  delivery contract.

## Consequences

Hosts can opt independent asynchronous tools into lower-latency batches without changing the
default behavior for existing tools. A `parallel_safe=True` declaration is an explicit host promise:
Gabby cannot inspect a handler's closure or external service to prove it is free of shared state.
Synchronous handlers still use the bounded callback worker mechanism, and their configured tool and
run deadlines apply. See the tool concurrency contract in
[the extension reference](../../EXTENSION_CONTRACTS.md).
