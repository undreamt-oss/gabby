# ADR 0037: Add cooperative cancellation to host tool context

- Status: accepted
- Date: 2026-10-01

## Context

Synchronous host tools run in Gabby's bounded callback worker pool so they do not block the async
runtime. A run deadline can stop awaiting a callback and mark queued callbacks as cancelled, but
Python cannot safely terminate an arbitrary callback that has already started. Such a callback had
no request-scoped signal with which to stop its own I/O or side effects.

## Decision

- Add a read-only `CancellationToken` to the opt-in, request-scoped `ToolContext`.
- Signal the token when Gabby times out the callback or cancels its invocation, including request
  cancellation.
- Let synchronous handlers call `wait(timeout)` from the worker thread or poll `is_cancelled`.
- Continue to cancel async handlers through normal coroutine cancellation; expose the same
  `is_cancelled` property for cleanup paths.
- Keep the token inside host-only `ToolContext`, outside model schemas, prompts, observations, and
  automatic traces.
- Document this as cooperative cancellation. Use a sandboxed tool when the runtime must be able to
  stop a process that ignores cancellation.

## Consequences

Opted-in host tools can release resources and stop side effects promptly after a deadline or caller
cancellation. Existing tools that do not opt into `ToolContext` keep their handler contract. The
runtime cannot force an uncooperative synchronous handler to exit; it remains subject to the bounded
callback worker capacity and may continue after the run has returned.

## Alternatives considered

- **Kill the Python worker thread:** unsafe and unsupported by Python; it can leave locks and shared
  state corrupted.
- **Move all Python callbacks into subprocesses:** arbitrary closures and application resources do
  not have a portable serialization or ownership contract.
- **Require all tools to run in containers:** would remove existing trusted host integration points
  and make in-process access to host-owned resources unavailable.

## Compatibility and evidence

This pre-1.0 addition extends the public `ToolContext` type. Runtime tests verify that a timed-out
synchronous handler waiting on the token observes cancellation while Gabby continues the run with
the configured timeout behavior.
