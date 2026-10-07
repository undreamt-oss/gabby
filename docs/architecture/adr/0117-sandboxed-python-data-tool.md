# ADR 0117: Sandboxed Python data tool

- Status: Accepted
- Date: 2026-10-03

## Context

Data agents can use the opt-in read-only SQLite tool or host-registered Python handlers. The latter
run inside the Gabby process and are trusted extensions, so they are unsuitable for model-supplied
analysis code. A Python capability is useful for data environments, but code execution must inherit
the same OS boundary as other agent tools.

## Decision

Add an opt-in `python_run_tool()` that uses the agent's configured per-run sandbox. The host chooses
the interpreter executable and container image. Gabby stages bounded UTF-8 source in the existing
private read-only tool-input mount, executes the interpreter with isolated mode and the staged file,
then removes the source. The tool returns bounded stdout, stderr, and exit status. Its permission is
`sandbox:python`; it requires a sandbox and is denied by `require_sandbox` when no sandbox is
configured. Workspace access, networking, resources, cancellation, and the run deadline follow the
agent sandbox configuration. No interpreter or package manager is installed by Gabby.

## Consequences

- Data-analysis agents can execute Python without running model-supplied code in Gabby's host
  process.
- The host must select an image containing a reviewed Python interpreter and any desired packages.
- Python code can use only the workspace paths and other capabilities exposed by that container;
  Python isolated mode is not itself a security boundary.
- Output and source remain bounded by the tool and sandbox limits.
- The public tool is opt-in and does not change existing host-handler execution semantics.

## Alternatives considered

- Execute Python in a Gabby worker process. Rejected because it would still expose host resources and
  would not satisfy the accepted OS-container boundary.
- Provide a generic script runner configured by arbitrary argv. Deferred because that overlaps with
  the existing sandboxed custom-tool executable contract and does not provide a Python-specific
  input or output contract.
- Require applications to register their own sandboxed Python tool. Deferred because it duplicates
  the shared private input mount, timeout, cancellation, and result-bound behavior already owned by
  Gabby's runtime.
