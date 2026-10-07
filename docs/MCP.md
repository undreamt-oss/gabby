# MCP integration

Gabby can expose one configured agent as a local Model Context Protocol (MCP) stdio server. An IDE
or other MCP host starts the process and invokes its `run_agent` tool. Gabby keeps each invocation
stateless: the host supplies the task, context, memory, and metadata on every tool call and owns any
conversation history or durable user state.

## Install and run

Install the optional SDK extra:

```sh
uv sync --extra mcp
# or, when installing the published package:
python -m pip install 'gabby-agent-runtime[mcp]'
```

Start the server from a local agent definition:

```sh
gabby mcp ./agent.yaml --knowledge-db ./.gabby/knowledge.db
```

The process communicates over stdin/stdout using MCP stdio framing. Do not wrap its stdout with
human-readable logs. The server loads the same agent configuration and knowledge store used by the
`run` and `serve` commands, and closes the agent gracefully when its process exits. Configure
`--max-response-bytes` to change the serialized `run_agent` result limit; the default is 4 MiB.

An MCP host configuration will typically point its process command at the Gabby executable and
pass the agent path as an argument. For example, an editor that accepts JSON server configuration
can use:

```json
{
  "mcpServers": {
    "gabby": {
      "command": "gabby",
      "args": ["mcp", "/absolute/path/to/agent.yaml"]
    }
  }
}
```

The exposed tool accepts `input` plus optional JSON objects `context`, `memory`, and `metadata`.
It returns structured `output`, result `metadata`, and a unique `trace_id`. Those request maps are
passed to one `Agent.arun()` call; the MCP adapter does not retain them between tool calls. Agent
execution failures and oversized or invalid responses are returned as sanitized MCP tool errors.

This adapter currently supports local stdio. Use Gabby's authenticated `/v1` FastAPI API for
remote service access. The MCP SDK also supports remote transports, but exposing them here would
require a separately specified authentication and deployment boundary. The SDK is an optional
dependency and is not imported when using Gabby's core runtime or FastAPI service.

## Import tools from another MCP server

An embedding application can also expose tools from an MCP server to a Gabby agent. The host
creates and connects the MCP `Client`, registers the remote tools before constructing the agent,
and owns closing the connection:

```python
from gabby import Agent, ToolRegistry, register_mcp_tools

registry = ToolRegistry()
async with connected_mcp_client as client:
    await register_mcp_tools(
        registry,
        client,
        server_name="research",
        permissions=["mcp:research:read"],
    )
    agent = Agent("agent.yaml", tools=registry)
    try:
        result = await agent.arun("Find the latest report")
    finally:
        await agent.aclose()
```

The bridge imports a bounded tool catalog and maps names to `mcp_<server>_<tool>`. It limits
discovery time, tool count, schema size, and each call's result size. Structured JSON and text
content can enter the agent context; other MCP result content types fail closed. Imported handlers
are host-trusted: attach explicit permission labels, allow only the needed tools in the agent
policy, and do not use this bridge as a sandbox boundary. Gabby does not own, reconnect, or close
the MCP client. Install the optional `mcp` extra to use the SDK client.
