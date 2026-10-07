# ADR 0102: Pluggable approval audit sink

## Status

Accepted

## Context

`AuditedApprovalHandler` combines a host's human review callback with durable decision recording,
but it accepted only Gabby's bundled `SQLiteApprovalAudit`. That prevented applications from
reusing the same composition with an existing shared database or audit service without copying the
wrapper. Approval audit storage, durability, retention, and access policy belong to the host.

## Decision

Define and export an async `ApprovalAuditSink` protocol with one `record(request, *, approved)`
operation. `AuditedApprovalHandler` accepts this protocol and awaits the write before returning the
decision. The existing `SQLiteApprovalAudit` implements the contract and remains a single-host,
standard-library option. Sink failures propagate to the runtime, which sanitizes them and fails
closed before tool execution. Hosts own sink lifecycle and must make writes durable before returning.

## Consequences

Hosts can integrate shared or distributed audit storage without Gabby depending on a database or
service. Gabby does not prescribe record schemas, retry behavior, idempotency, durability guarantees,
or retention for custom sinks. The request may contain tool arguments and an authenticated principal;
the host must apply its own privacy and access controls.

## Alternatives considered

- Keep requiring `SQLiteApprovalAudit`. This excludes existing host audit backends.
- Bundle remote database or approval-service clients. This would couple core to backend lifecycle,
  credentials, and deployment policy.
- Provide only an example. Consumers would need to recreate the composition and fail-closed ordering.
