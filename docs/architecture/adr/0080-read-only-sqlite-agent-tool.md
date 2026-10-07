# ADR 0080: Read-only SQLite agent tool

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-02.

## Context

Gabby supports domain-neutral tools and environments, but data-agent examples only demonstrate
application-specific mock tools. A standard-library SQLite adapter gives developers a useful local
data-environment integration without making a database dependency mandatory or turning SQLite into
a required storage layer.

## Decision

- Add an opt-in `sqlite_query_tool()` factory that returns a normal Gabby `Tool`.
- Resolve its database from an explicitly named `ToolContext` environment resource; never accept a
  database path from the model.
- Require the `database:read` permission and open the configured file in SQLite read-only mode.
- Use SQLite's authorizer to permit read/query actions and deny writes, PRAGMA operations, database
  attachment, and extension loading.
- Bound SQL input, bind-parameter count, columns, returned rows, cell/row size, serialized output,
  query duration, and the enclosing tool timeout.
- Leave database paths, authentication around the containing service, and access policy to the host.

## Consequences

Developers can build local data agents with parameterized read queries using the standard library.
SQLite query handlers remain trusted host code; the tool is not an OS sandbox or a remote database
client. Other database systems can provide tools using the same `Tool` contract. The implementation
does not create or migrate the database.

## Alternatives considered

- **Accept a database path or connection from model input:** rejected because a model must not choose
  host resources or connection capabilities.
- **Bundle a driver and connection pools for server databases:** deferred because it introduces
  vendor dependencies and deployment-specific credential lifecycle into core.
- **Leave every local data agent to implement its own SQLite policy:** rejected because read-only
  mode, authorizer policy, parameter binding, and resource limits should have one reusable contract.

## Evidence

Tests cover parameterized reads, read-only enforcement, denied PRAGMA/ATTACH/extension loading,
row and byte bounds, deadlines, and missing-resource failures. The runnable synthetic-data example
exercises the tool through a complete stateless agent run. See [SQLite data tool guide](../../SQLITE_DATA.md).
