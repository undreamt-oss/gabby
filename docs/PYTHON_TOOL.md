# Sandboxed Python tool

`python_run_tool()` gives data-oriented agents a Python execution capability inside their configured
per-run Docker or Podman sandbox. It does not execute model-supplied code in the Gabby host process.
Register it explicitly and require sandboxed tool execution:

```python
from gabby import Agent, ToolRegistry, python_run_tool

tools = ToolRegistry()
tools.register(python_run_tool(executable="python3", timeout_seconds=20))
agent = Agent.from_file("data-agent.yaml", tools=tools)
result = await agent.arun("Calculate the mean of these values: 3, 7, 8")
```

The agent definition must configure a sandbox image that includes the selected interpreter and
grant the tool and its permission:

```yaml
tools:
  - python_run
policies:
  allowed_tools: [python_run]
  allowed_permissions: [sandbox:python]
  require_sandbox: true
sandbox:
  engine: docker
  image: python:3.14-slim
  keepalive_argv: [python3, -c, "import time; time.sleep(3600)"]
  workspace:
    path: ./workspace
    access: read_write
    container_path: /workspace
```

For a complete offline run with a deterministic mock model and a real Docker container, use
[`examples/python_sandboxed_agent.py`](../examples/python_sandboxed_agent.py). It pulls the example
image when needed and removes the per-run container after completion.

Gabby writes each bounded UTF-8 source file to the private read-only tool-input mount, runs the
configured interpreter with Python isolated mode, and removes the file after the call. The tool
returns `exit_code`, `stdout`, and `stderr`. Defaults cap source at 256 KiB, serialized tool output
at 1 MiB, and execution at 30 seconds; the agent run deadline and container resource limits also
apply. The container image controls which Python packages are available. On images where Python is
not named `python3`, pass its executable to `python_run_tool`, such as `python.exe`.

The code can access the mounted workspace only to the extent granted by its configured mount. The
container's network policy and OS isolation apply; Python isolated mode does not itself provide a
security boundary. Treat code and images as untrusted inputs and select a reviewed container image.
For read-only SQLite access with query-level limits, prefer [`sqlite_query_tool()`](SQLITE_DATA.md).
