# Read-only SQLite tool

Gabby includes `sqlite_query_tool()` as an opt-in data-environment tool. It runs parameterized SQL
against a SQLite database file supplied as a named host environment resource. It does not create,
own, or persist a database, and it does not add database access to an agent unless the host registers
the tool and grants its permission.

```python
from gabby import Agent, AgentDefinition, Environment, ToolRegistry, sqlite_query_tool

tools = ToolRegistry()
tools.register(sqlite_query_tool(resource_name="reporting_db"))
environment = Environment(
    type="data",
    description="Approved reporting database",
    resources={"reporting_db": "/srv/reports/warehouse.sqlite3"},
    tools=tools,
    allowed_tools=["sqlite_query"],
)
definition = AgentDefinition(
    name="sales-analyst",
    model={"provider": "openai-compatible", "model": "your-model"},
    tools=["sqlite_query"],
    policies={
        "allowed_tools": ["sqlite_query"],
        "allowed_permissions": ["database:read"],
    },
)
agent = Agent(definition, environment=environment)
```

The model supplies a SQL string and optional JSON bind parameters. The database is opened read-only;
an SQLite authorizer permits reads, `SELECT`, and functions while denying mutation, attachment,
PRAGMA, and extension-loading actions. SQLite's progress handler interrupts queries at the configured
deadline. Defaults cap a query at 2 seconds, 500 returned rows, 256 KiB for an SQLite value or row,
256 result columns, 256 bind parameters, and 768 KiB of serialized results. Result rows are marked
`truncated` when row or byte limits stop output. Configure these bounds when constructing the tool.

The application owns the file path, database contents, filesystem permissions, backups, and access
policy. The tool is trusted in-process Python and does not isolate SQLite or bound separate CPU or
memory resources beyond SQLite's query, value, and output limits. Use a container or external service
when database execution requires a stronger operating-system boundary. Parameter binding should be
used for values; schema/table names should come from trusted application logic or be selected from
known values.

Run the offline working example with:

```sh
uv run --frozen python examples/sqlite_data_agent.py
```

The example creates synthetic sales data in a temporary database, runs one stateless agent request,
and removes the database after the run. See [ADR 0080](architecture/adr/0080-read-only-sqlite-agent-tool.md).
