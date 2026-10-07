# Host approval and audit

Gabby pauses a sensitive tool call and asks the injected `ApprovalHandler` for a decision. The
application owns reviewer identity, notification, UI, database path, retention, and access controls.
Approval decisions are transient and scoped to one tool call; Gabby does not retain them as agent
memory.

## Select an audit backend

`ApprovalAuditSink` is the async extension contract for durable decisions. `SQLiteApprovalAudit` is
the bundled single-host implementation, and `AuditedApprovalHandler` composes any sink with the
host's asynchronous review callback:

```python
from gabby import AuditedApprovalHandler, SQLiteApprovalAudit

audit = SQLiteApprovalAudit("./private/approvals.sqlite3")
approval_handler = AuditedApprovalHandler(review_in_my_app, audit)
agent = Agent.from_file("agent.yaml", approval_handler=approval_handler)
```

The helper records the run and call IDs, agent and tool names, decision, timestamp, and SHA-256 of
the canonical JSON arguments. It does not persist raw arguments. Principal subjects are omitted by
default; set `include_principal_subject=True` only when the application's audit policy needs them.
Argument serialization is capped at 1 MiB by default, audit calls run outside the event loop, and
audit write failures propagate as a sanitized approval failure so the tool does not execute. Reads
are asynchronous and limited to 1,000 records per call. Invocation IDs and names are bounded before
storage; oversized or non-JSON arguments fail closed.

The database remains host-owned. The application must set filesystem permissions, choose retention
and backup policies, and restrict access. This is a local single-host SQLite option.

For shared PostgreSQL storage, Gabby includes `PostgresApprovalAudit`. It accepts a host-owned async
pool with an `execute(query, *args)` operation; `asyncpg.Pool` is compatible. Gabby does not install
an SQL driver, create tables, read credentials, or close the pool. Create this table once through
your migration system:

```sql
CREATE TABLE gabby_tool_approval_audit (
    run_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    principal_subject TEXT,
    arguments_sha256 CHAR(64) NOT NULL,
    approved BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, call_id)
);
```

Then inject the pool-backed sink:

```python
import asyncpg  # Host dependency; Gabby does not require it.
from gabby import AuditedApprovalHandler, PostgresApprovalAudit

pool = await asyncpg.create_pool(dsn=database_dsn)
audit = PostgresApprovalAudit(pool)
approval_handler = AuditedApprovalHandler(review_in_my_app, audit)
```

The host creates the pool with credentials and TLS settings from its secret manager, runs schema
migrations, and closes the pool during application shutdown. PostgreSQL keeps only invocation
metadata, the decision, and a SHA-256 of canonical arguments; raw arguments are never stored.
Principal subjects remain omitted unless `include_principal_subject=True`. Duplicate `(run_id,
call_id)` values violate the primary key and fail closed rather than silently replacing an earlier
decision.

Both sinks implement `ApprovalAuditSink`. A sink must finish recording before returning. If it
raises or times out, Gabby fails closed and does not execute the tool. The host remains responsible
for backend durability, idempotency, availability, retention, and access control.
`AuditedApprovalHandler` does not catch backend exceptions; Gabby's runtime turns them into a
sanitized approval-unavailable tool failure.

See [the runnable example](../examples/approval_audit.py),
[the approval decision](architecture/adr/0025-host-owned-tool-approval.md), and
[ADR 0093](architecture/adr/0093-reusable-sqlite-approval-audit.md) and
[ADR 0102](architecture/adr/0102-approval-audit-sink-contract.md) and
[ADR 0103](architecture/adr/0103-postgres-approval-audit.md).
