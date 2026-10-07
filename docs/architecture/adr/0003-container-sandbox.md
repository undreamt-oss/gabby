# ADR 0003: Container-backed tool isolation

**Status:** Accepted by the project architect; initial adapters implemented, platform validation pending.  
**Date:** 2026-09-29.

## Context

The architect selected OS/container isolation as a core security boundary for tools, with Docker and Podman as the initial engines and Linux, macOS, and Windows as target host platforms. Tool policy must be enforced outside the model; prompt instructions are not a security boundary.

## Decision

- Every tool declares an execution class: injected Python handlers default to `host_trusted`; tools backed by the container sandbox declare `sandboxed`.
- `policies.require_sandbox: true` denies declared host-trusted tools before model execution.
- The initial supported engine family is Docker and Podman, with Linux, macOS, and Windows as target platforms.
- Image digests are optional for local development. `sandbox.require_image_digest: true` rejects mutable tags and requires an OCI SHA-256 digest; operators should enable it for production deployments.
- Docker may run Linux containers on Linux/macOS/Windows and native Windows containers on Windows. Podman targets Linux containers, including its Linux VM on macOS/Windows.
- Native Windows containers use Docker Hyper-V isolation; Gabby requests Hyper-V isolation and fails closed when the engine cannot provide it.
- Agent configuration selects the engine and must specify the container image.
- Linux containers run as numeric non-root UID/GID `65532:65532` by default; agent configuration may
  set `sandbox.user` to another numeric non-root UID:GID pair. Native Windows containers retain the
  image-defined user because the Linux numeric identity does not apply.
- Gabby pulls the configured image on demand when it is not present locally.
- Both CLI and API adapters are included for Docker and Podman behind a shared engine contract.
- Remote engine API endpoints require HTTPS; plain HTTP is accepted only for loopback endpoints.
  Unix domain socket connections remain supported for local engines. API URLs cannot carry
  userinfo or credential-like query parameters; authenticated clients are injected by the host.
- On Windows, the API adapter can connect to a local named pipe such as
  `\\.\pipe\docker_engine`; this uses the optional `windows-sandbox` extra and Windows Proactor
  event loop. Pipe access is governed by the host's named-pipe ACL.
- Agent configuration must specify a keepalive argv for the container image so one container remains available across tool calls within a run.
- Built-in shell and filesystem tools run in the container. Injected Python handlers are explicitly host-trusted by default; policies can require sandboxed tools and reject host-trusted tools before execution.
- User-defined sandboxed tools may declare `sandbox_action: "tool.execute"` and a fixed
  `sandbox_command` argv. When such a tool is registered, the runtime creates a private per-run
  host scratch directory and bind-mounts it read-only into the container. On POSIX hosts, the
  directory grants traversal without listing and each bounded JSON request file is changed to
  read-only before execution so the default non-root container user can read it. The configured
  argv receives that path as its final argument, must return one JSON value on stdout, and is
  subject to the run deadline and sandbox output limit. Gabby validates the required output schema,
  removes the request file after the call, and removes the directory after container removal. The
  tool implementation must be present in the configured image; its executable and command remain
  configuration-owned and never come from model arguments.
- Container networking is disabled in the initial implementation; allowlisted egress is deferred.
- Agent configuration chooses whether the workspace mount is read-only or read/write.
- The consuming application must keep the configured workspace quiescent for the full agent run.
  Component checks are defense in depth and do not prevent a concurrent host process from racing
  separate archive operations.
- Create one container per agent run and remove it after the run; container state is temporary execution state.
- If container creation or startup fails after the daemon may have accepted the request, adapters attempt cleanup using the unique run-scoped container name even when the response containing its ID was lost.
- Default resource limits are 2 CPUs, 2 GiB of memory, and 256 processes; agent configuration may override them. The run deadline always bounds container lifetime.
- Bound captured command output to 1 MiB across stdout and stderr. Docker and Podman CLI adapters stop draining after the shared limit is exceeded and terminate the engine command promptly; the run-scoped container is then removed. API adapters enforce the same limit while streaming exec output.
- Sandbox engine, image, mount, network, resource, and availability behavior must be explicit and fail closed before claiming isolation.

## Open questions

- Engine-backed acceptance runs for Docker and Podman CLI and API adapters on supported Linux, macOS, and Windows configurations. A native Windows suite now exercises both Docker adapters, including the local named-pipe API transport; it still needs to run on a Windows host before that matrix cell is considered live-verified.
- Confirm resource-limit behavior and workspace bind-mount syntax for each supported daemon/host combination. Native Windows Docker runs now use `CpuCount`, require whole-number CPU settings, and require `process_limit: null` as an explicit opt-out because Docker does not document `PidsLimit` for Windows. Linux rejects a null process limit. Mock tests prove the request shape only, not host enforcement; live Windows validation remains open.
- Revisit a race-free, atomic filesystem helper if Gabby later needs to support concurrent host-side
  workspace mutation during a run.
- Repeat live custom-tool checks on macOS and Windows hosts; Linux Docker and Podman CLI/API checks
  now validate the JSON tool protocol, mount permissions, bounds, schema validation, and cleanup.

## Consequences

The runtime exposes a replaceable sandbox interface with Docker and Podman CLI and Docker-compatible API adapters. Agent configuration selects the engine, requires an image, and specifies a keepalive argv. Missing images are pulled on demand. Docker supports native Windows container images on Windows with Hyper-V isolation; Podman uses Linux containers, including through a Linux VM on macOS/Windows. Built-in shell and filesystem tools use a temporary container per agent run, with networking disabled and workspace mount access selected in agent configuration. Linux runs use UID/GID `65532:65532` by default; a numeric non-root UID:GID override can accommodate image and workspace permissions. Native Windows containers retain their image-defined user under Hyper-V isolation. Resource defaults are 2 CPUs, 2 GiB memory, and 256 processes with per-agent overrides and an absolute run deadline. Native Windows uses Docker's whole-number `CpuCount`; it requires explicit `process_limit: null` because Docker does not document `PidsLimit` as a Windows control. Adapters attempt name-based cleanup when a create/start request may have reached the daemon but Gabby did not receive a container ID. Injected Python handlers default to host-trusted execution; a policy can require sandboxed execution and fail closed for those handlers. Unit tests cover adapter behavior with fake processes and HTTP transports. Live support is a per-host/engine/adapter claim: a combination is production-supported only after live acceptance checks prove isolation, resource controls, mounts, network behavior, and cleanup. Current live evidence covers Docker and Podman on Linux through CLI and API adapters, including CPU/memory/process limits and read-only/read-write mounts; other combinations remain experimental.

For user-defined isolated tools, the declared `sandbox_command` runs inside the container with a
temporary JSON request file appended to its argv. It returns a single JSON value on stdout. Gabby
bounds input bytes, command output, and the run deadline, validates the required output schema, and
removes the request file after invocation. On POSIX hosts, the private host directory grants
traversal without listing and request files are read-only before the container reads them. The
scratch directory is a read-only, per-run bind mount that Gabby creates and removes on the host.
