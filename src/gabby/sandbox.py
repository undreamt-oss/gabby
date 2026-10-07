# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-execution container sandbox contracts and lifecycle management."""

from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from .config import DEFAULT_SANDBOX_USER, SandboxDefinition


class SandboxError(RuntimeError):
    """A sandbox could not be created or a sandbox operation failed."""


class SandboxUnavailable(SandboxError):
    """The configured container engine could not be reached or used."""


class SandboxTimeout(SandboxError):
    """A sandbox engine or command operation exceeded its deadline."""


class SandboxOutputLimit(SandboxError):
    """A sandbox command produced more output than Gabby will retain."""


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
_WINDOWS_DEVICE_NAMES = {"con", "prn", "aux", "nul"}.union(
    f"{prefix}{digit}" for prefix in ("com", "lpt") for digit in "123456789¹²³"
)


@dataclass(frozen=True)
class SandboxCommandResult:
    """Captured result of a process executed inside the container."""

    exit_code: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class SandboxResourceUsage:
    """Best-effort per-run resource counters reported by the container engine."""

    cpu_time_ns: int | None = None
    cpu_percent: float | None = None
    memory_current_bytes: int | None = None
    memory_peak_bytes: int | None = None
    process_count: int | None = None

    def as_dict(self) -> dict[str, int | float]:
        """Return only counters supported by the active engine and platform."""
        return {
            name: value
            for name, value in (
                ("cpu_time_ns", self.cpu_time_ns),
                ("cpu_percent", self.cpu_percent),
                ("memory_current_bytes", self.memory_current_bytes),
                ("memory_peak_bytes", self.memory_peak_bytes),
                ("process_count", self.process_count),
            )
            if value is not None
        }


@dataclass(frozen=True)
class ContainerSpec:
    """Validated settings used to create one temporary execution container."""

    run_id: str
    image: str
    image_os: str
    daemon_os: str
    keepalive_argv: tuple[str, ...]
    workspace_host_path: str | None
    workspace_container_path: str | None
    workspace_read_only: bool
    cpus: float
    memory_bytes: int
    process_limit: int | None
    tool_input_host_path: str | None = None
    tool_input_container_path: str | None = None
    user: str = DEFAULT_SANDBOX_USER


class EngineAdapter(Protocol):
    """Pluggable low-level adapter for a container engine."""

    async def daemon_os(self, *, timeout: float) -> str:
        """Return the operating system reported by the container daemon."""
        ...

    async def validate_windows_sandbox_support(self, *, timeout: float) -> None:
        """Fail unless the adapter can safely enforce native Windows isolation."""
        ...

    async def image_os(self, image: str, *, timeout: float) -> str | None:
        """Inspect an image's operating system, returning ``None`` if unavailable locally."""
        ...

    async def pull_image(self, image: str, *, timeout: float) -> None:
        """Pull the configured image within the operation timeout."""
        ...

    async def start_container(self, spec: ContainerSpec, *, timeout: float) -> str:
        """Start one run-scoped container and return its engine identifier."""
        ...

    async def exec(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        timeout: float,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        """Execute argv in a running container and capture bounded output."""
        ...

    async def read_archive(self, container_id: str, path: str, *, timeout: float) -> bytes:
        """Read a tar archive from a container path."""
        ...

    async def write_archive(
        self, container_id: str, path: str, archive: bytes, *, timeout: float
    ) -> None:
        """Write a tar archive to a container path."""
        ...

    async def remove_container(self, container_id: str, *, timeout: float) -> None:
        """Stop and remove a run-scoped container."""
        ...

    async def aclose(self) -> None:
        """Release adapter-owned clients without shutting down the engine daemon."""
        ...


class SandboxResourceMonitor(Protocol):
    """Optional resource-counter capability for an ``EngineAdapter``."""

    async def resource_usage(
        self, container_id: str, *, timeout: float
    ) -> SandboxResourceUsage | None:
        """Return bounded best-effort resource counters for a run-scoped container."""
        ...


@dataclass
class SandboxSession:
    """Run-scoped access to a container and its bounded operations."""

    adapter: EngineAdapter
    container_id: str
    container_os: str
    workspace_path: str | None
    deadline: float
    output_limit_bytes: int = 1024 * 1024
    workspace_read_only: bool = True
    tool_input_host_path: str | None = None
    tool_input_container_path: str | None = None
    _terminated: bool = False

    def _remaining(self, requested: float | None) -> float:
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise SandboxTimeout("The agent run deadline expired")
        return min(remaining, requested) if requested is not None else remaining

    def _operation_deadline(self, requested: float | None) -> float:
        return asyncio.get_running_loop().time() + self._remaining(requested)

    def _operation_remaining(self, deadline: float) -> float:
        remaining = min(self.deadline, deadline) - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise SandboxTimeout("The sandbox operation exceeded its deadline")
        return remaining

    async def read_resource_usage(self) -> SandboxResourceUsage | None:
        """Read engine counters without allowing telemetry to fail the agent run."""
        reader = getattr(self.adapter, "resource_usage", None)
        if reader is None:
            return None
        try:
            result = await asyncio.wait_for(
                reader(self.container_id, timeout=min(1.0, self._remaining(None))),
                timeout=min(1.0, self._remaining(None)),
            )
        except Exception:
            return None
        return result if isinstance(result, SandboxResourceUsage) else None

    async def exec(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        """Execute an argv vector without involving a host shell."""
        if self._terminated:
            raise SandboxError("The sandbox was terminated after a previous command failure")
        if not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise SandboxError("Sandbox commands must be a non-empty argv of strings")
        try:
            result = await self.adapter.exec(
                self.container_id,
                argv,
                timeout=self._remaining(timeout),
                working_directory=working_directory,
                environment=environment,
            )
            if len(result.stdout) + len(result.stderr) > self.output_limit_bytes:
                raise SandboxOutputLimit(
                    f"Sandbox command output exceeded {self.output_limit_bytes} bytes"
                )
        except (SandboxTimeout, SandboxOutputLimit) as error:
            self._terminated = True
            try:
                await self.adapter.remove_container(self.container_id, timeout=10.0)
            except Exception as cleanup_error:
                error.add_note(f"Could not terminate the timed-out sandbox: {cleanup_error}")
            raise
        return result

    async def read_archive(self, path: str, *, timeout: float | None = None) -> bytes:
        """Read a container path as an archive stream."""
        if self._terminated:
            raise SandboxError("The sandbox is terminated")
        return await self.adapter.read_archive(
            self.container_id, path, timeout=self._remaining(timeout)
        )

    async def write_archive(
        self, path: str, archive: bytes, *, timeout: float | None = None
    ) -> None:
        """Upload an archive into a container directory."""
        if self._terminated:
            raise SandboxError("The sandbox is terminated")
        if len(archive) > MAX_ARCHIVE_BYTES:
            raise SandboxOutputLimit("Sandbox archive exceeded its size limit")
        await self.adapter.write_archive(
            self.container_id, path, archive, timeout=self._remaining(timeout)
        )

    def _workspace_target(self, relative_path: str, *, allow_root: bool = False) -> str:
        if self.workspace_path is None:
            raise SandboxError("Filesystem tools require sandbox.workspace.path")
        if not isinstance(relative_path, str) or "\x00" in relative_path:
            raise SandboxError("Workspace path must be a string without NUL characters")
        normalized = relative_path.replace("\\", "/")
        parts = normalized.split("/")
        if (
            normalized.startswith("/")
            or (len(normalized) >= 2 and normalized[1] == ":")
            or any(part in ("..", "") for part in parts if part != ".")
            or any(ord(char) < 32 for char in normalized)
        ):
            raise SandboxError("Workspace paths must be relative and cannot escape the workspace")
        if self.container_os == "windows" and any(
            any(char in part for char in '<>:"|?*') or part.endswith((".", " ")) for part in parts
        ):
            raise SandboxError("Workspace path contains a character invalid on Windows")
        if self.container_os == "windows" and any(_windows_device_name(part) for part in parts):
            raise SandboxError("Workspace path contains a reserved Windows device name")
        safe_parts = [part for part in parts if part != "."]
        if not safe_parts and not allow_root:
            raise SandboxError("A non-empty workspace path is required")
        separator = "\\" if self.container_os == "windows" else "/"
        return separator.join([self.workspace_path.rstrip("\\/")] + safe_parts)

    async def invoke(
        self,
        action: str,
        arguments: dict[str, object],
        *,
        timeout: float | None = None,
        command: Sequence[str] | None = None,
        max_input_bytes: int = MAX_FILE_BYTES,
    ) -> object:
        """Execute one of Gabby's built-in sandbox operations."""
        operation_deadline = self._operation_deadline(timeout)
        if action == "python.run":
            code = arguments.get("code")
            if not isinstance(code, str) or "\x00" in code:
                raise SandboxError("Python code must be a string without NUL characters")
            if command is None or not command:
                raise SandboxError("Python execution requires a configured interpreter")
            return await self.invoke_python(
                command,
                code,
                max_input_bytes=max_input_bytes,
                timeout=self._operation_remaining(operation_deadline),
            )
        if action == "shell.run":
            argv = arguments.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
                raise SandboxError("shell_run argv must be a non-empty list of strings")
            cwd_value = arguments.get("cwd")
            cwd = self._workspace_target(cwd_value) if isinstance(cwd_value, str) else None
            result = await self.exec(
                argv,
                timeout=self._operation_remaining(operation_deadline),
                working_directory=cwd,
            )
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
            }
        if action == "filesystem.read_file":
            target = self._workspace_target_value(arguments)
            await self._check_workspace_components(
                str(arguments["path"]),
                final_type="file",
                allow_missing_final=False,
                deadline=operation_deadline,
            )
            archive = await self.read_archive(
                target, timeout=self._operation_remaining(operation_deadline)
            )
            content = _single_file_from_archive(archive)
            if len(content) > min(self.output_limit_bytes, MAX_FILE_BYTES):
                raise SandboxOutputLimit("Workspace file exceeded the configured output limit")
            return {"path": arguments["path"], "content": content.decode("utf-8")}
        if action == "filesystem.write_file":
            if self.workspace_read_only:
                raise SandboxError("The configured workspace mount is read-only")
            target = self._workspace_target_value(arguments)
            content_value = arguments.get("content")
            if not isinstance(content_value, str):
                raise SandboxError("filesystem_write_file content must be a string")
            content = content_value.encode("utf-8")
            if len(content) > self.output_limit_bytes:
                raise SandboxOutputLimit("Workspace write exceeded the configured size limit")
            separator = "\\" if self.container_os == "windows" else "/"
            await self._check_workspace_components(
                str(arguments["path"]),
                final_type="file",
                allow_missing_final=True,
                deadline=operation_deadline,
            )
            parent, _, name = target.rpartition(separator)
            if not name:
                raise SandboxError("Cannot write over the workspace mount root")
            self._validate_archive_leaf(name)
            archive = _file_archive(name, content)
            await self.write_archive(
                parent or self.workspace_path or target,
                archive,
                timeout=self._operation_remaining(operation_deadline),
            )
            return {"path": arguments["path"], "bytes_written": len(content)}
        if action == "filesystem.list_dir":
            target = self._workspace_target_value(arguments, allow_root=True)
            await self._check_workspace_components(
                str(arguments.get("path", ".")),
                final_type="directory",
                allow_missing_final=False,
                deadline=operation_deadline,
            )
            archive = await self.read_archive(
                target, timeout=self._operation_remaining(operation_deadline)
            )
            entries = _archive_entries(archive, root_name=_path_leaf(target))
            if len(entries) > 10_000:
                raise SandboxOutputLimit("Workspace directory contains too many entries")
            return {"path": arguments.get("path", "."), "entries": entries}
        raise SandboxError(f"Unsupported sandbox operation: {action}")

    async def invoke_python(
        self,
        command: Sequence[str],
        code: str,
        *,
        max_input_bytes: int,
        timeout: float | None = None,
    ) -> dict[str, object]:
        """Stage bounded source privately and execute it through the configured interpreter."""
        if self._terminated:
            raise SandboxError("The sandbox was terminated after a previous command failure")
        if (
            isinstance(max_input_bytes, bool)
            or not isinstance(max_input_bytes, int)
            or max_input_bytes < 1
        ):
            raise SandboxError("Python source limit must be a positive integer")
        if not command or any(not isinstance(value, str) or not value for value in command):
            raise SandboxError("Python execution requires a configured interpreter")
        if not isinstance(code, str) or "\x00" in code:
            raise SandboxError("Python code must be a string without NUL characters")
        try:
            payload = code.encode("utf-8")
        except UnicodeEncodeError:
            raise SandboxError("Python code must contain valid Unicode") from None
        if len(payload) > max_input_bytes:
            raise SandboxOutputLimit(f"Python code exceeded max_input_bytes={max_input_bytes}")
        if self.tool_input_host_path is None or self.tool_input_container_path is None:
            raise SandboxError("Python execution requires the private tool-input mount")

        token = uuid.uuid4().hex
        filename = f"gabby-python-{token}.py"
        host_file = Path(self.tool_input_host_path) / filename
        separator = "\\" if self.container_os == "windows" else "/"
        script_path = self.tool_input_container_path.rstrip("\\/") + separator + filename
        failure: BaseException | None = None
        try:
            descriptor = os.open(host_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with os.fdopen(descriptor, "wb") as source:
                source.write(payload)
            if os.name == "posix":
                host_file.chmod(0o444)
            result = await self.exec(
                [*command, "-I", script_path],
                timeout=timeout,
                working_directory=self.workspace_path,
            )
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
            }
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                host_file.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if failure is not None:
                    failure.add_note("Could not remove the temporary Python source file")
                else:
                    raise SandboxError("Could not remove the temporary Python source file") from (
                        cleanup_error
                    )

    async def invoke_tool(
        self,
        command: Sequence[str],
        arguments: dict[str, object],
        *,
        max_input_bytes: int,
        timeout: float | None = None,
    ) -> object:
        """Run an image-provided JSON tool with its input in a temporary container file.

        The configured argv runs inside the container and receives the generated request-file path
        as its last argument. It must write exactly one JSON value to stdout. Request files are
        written to a per-run host scratch directory mounted read-only into the container and
        removed after invocation.
        """
        if self._terminated:
            raise SandboxError("The sandbox is terminated")
        if not command or any(
            not isinstance(part, str) or not part or "\x00" in part for part in command
        ):
            raise SandboxError("Sandboxed tool command must be a non-empty argv of strings")
        if isinstance(max_input_bytes, bool) or not isinstance(max_input_bytes, int):
            raise SandboxError("Sandboxed tool max_input_bytes must be a positive integer")
        if max_input_bytes < 1:
            raise SandboxError("Sandboxed tool max_input_bytes must be a positive integer")
        try:
            payload = json.dumps(
                arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SandboxError("Sandboxed tool arguments are not valid JSON") from exc
        if len(payload) > max_input_bytes:
            raise SandboxOutputLimit(
                f"Sandboxed tool input exceeded max_input_bytes={max_input_bytes}"
            )

        token = uuid.uuid4().hex
        filename = f"gabby-tool-{token}.json"
        if self.tool_input_host_path is None or self.tool_input_container_path is None:
            raise SandboxError("Sandboxed custom tools require a configured tool input mount")
        host_file = Path(self.tool_input_host_path) / filename
        separator = "\\" if self.container_os == "windows" else "/"
        request_path = self.tool_input_container_path.rstrip("\\/") + separator + filename
        try:
            host_file.write_bytes(payload)
            if os.name == "posix":
                # Linux container users need to traverse the bind mount and read one request.
                # The per-run directory is not listable; each request becomes read-only before
                # the container process is started.
                host_file.chmod(0o444)
        except OSError as exc:
            with suppress(OSError):
                host_file.unlink(missing_ok=True)
            raise SandboxError("Could not stage sandboxed tool input") from exc

        failure: BaseException | None = None
        try:
            result = await self.exec([*command, request_path], timeout=timeout)
            if result.exit_code != 0:
                raise SandboxError("Sandboxed tool command returned a non-zero exit code")
            try:
                return json.loads(
                    result.stdout.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise SandboxError(
                    "Sandboxed tool command did not return valid UTF-8 JSON"
                ) from exc
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                host_file.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if failure is not None:
                    failure.add_note("Could not remove the temporary sandbox tool request file")
                else:
                    raise SandboxError(
                        "Could not remove the temporary sandbox tool request file"
                    ) from cleanup_error

    async def _check_workspace_components(
        self,
        relative_path: str,
        *,
        final_type: str,
        allow_missing_final: bool,
        deadline: float,
    ) -> None:
        """Reject symlinks at every traversed component before archive operations."""
        if self.workspace_path is None:
            raise SandboxError("Filesystem tools require sandbox.workspace.path")
        parts = [
            part for part in relative_path.replace("\\", "/").split("/") if part not in ("", ".")
        ]
        separator = "\\" if self.container_os == "windows" else "/"
        current = self.workspace_path.rstrip("\\/")
        for index, part in enumerate(parts):
            archive = await self.read_archive(current, timeout=self._operation_remaining(deadline))
            entries = {
                entry["name"]: entry
                for entry in _archive_entries(archive, root_name=_path_leaf(current))
            }
            entry = entries.get(part)
            is_final = index == len(parts) - 1
            if entry is None:
                if is_final and allow_missing_final:
                    return
                raise SandboxError(f"Workspace path component does not exist: {part}")
            expected = final_type if is_final else "directory"
            if entry["type"] != expected:
                raise SandboxError(f"Workspace path component must be a regular {expected}: {part}")
            current = current + separator + part

    def _workspace_target_value(
        self, arguments: dict[str, object], *, allow_root: bool = False
    ) -> str:
        path = arguments.get("path", "." if allow_root else None)
        if not isinstance(path, str):
            raise SandboxError("Filesystem path must be a string")
        return self._workspace_target(path, allow_root=allow_root)

    @staticmethod
    def _validate_archive_leaf(name: str) -> None:
        if name in ("", ".", "..") or "/" in name or "\\" in name or "\x00" in name:
            raise SandboxError("Invalid workspace filename")


def _single_file_from_archive(archive: bytes) -> bytes:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            files = [member for member in tar.getmembers() if member.isfile()]
            if len(files) != 1:
                raise SandboxError("Expected exactly one regular file in the workspace archive")
            _archive_parts(files[0].name)
            handle = tar.extractfile(files[0])
            if handle is None:
                raise SandboxError("Could not read the workspace file from the archive")
            return handle.read(1024 * 1024 + 1)
    except (tarfile.TarError, OSError) as exc:
        raise SandboxError("Container engine returned an invalid workspace archive") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _file_archive(name: str, content: bytes) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _archive_entries(archive: bytes, *, root_name: str | None = None) -> list[dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            members = [(member, _archive_parts(member.name)) for member in tar.getmembers()]
            root = next(
                (
                    parts[0]
                    for member, parts in members
                    if len(parts) == 1
                    and member.isdir()
                    and root_name is not None
                    and parts[0] == root_name
                    and all(not other or other[0] == parts[0] for _, other in members)
                ),
                None,
            )
            for member, parts in members:
                if root is not None and parts and parts[0] == root:
                    parts = parts[1:]
                if not parts:
                    continue
                leaf = parts[0]
                if len(parts) > 1 or member.isdir():
                    entry_type = "directory"
                    size = 0
                elif member.isfile():
                    entry_type = "file"
                    size = member.size
                else:
                    entry_type = "other"
                    size = 0

                previous = entries.get(leaf)
                if previous is not None and previous["type"] != entry_type:
                    raise SandboxError("Container archive contains conflicting path types")
                # Descendant paths imply a directory only when no explicit entry
                # for that component exists. In particular, descendants must
                # never turn an explicit symlink or special file into a directory.
                if previous is None or len(parts) == 1:
                    entries[leaf] = {"name": leaf, "type": entry_type, "size": size}
    except (tarfile.TarError, OSError) as exc:
        raise SandboxError("Container engine returned an invalid workspace archive") from exc
    return [entries[name] for name in sorted(entries)]


def _archive_parts(name: str) -> list[str]:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise SandboxError("Container archive contained an absolute path")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts) or (parts and ":" in parts[0]):
        raise SandboxError("Container archive contained an unsafe path")
    return parts


def _path_leaf(path: str) -> str:
    return path.rstrip("\\/").replace("\\", "/").rsplit("/", 1)[-1]


def _windows_device_name(part: str) -> bool:
    stem = part.split(".", 1)[0].rstrip(" .").casefold()
    return stem in _WINDOWS_DEVICE_NAMES


def _adapter_for(config: SandboxDefinition) -> EngineAdapter:
    from .sandbox_engines import api_adapter, cli_adapter

    if config.adapter == "cli":
        return cli_adapter(config.engine)
    return api_adapter(config)


@asynccontextmanager
async def open_sandbox(
    config: SandboxDefinition,
    *,
    deadline: float,
    adapter: EngineAdapter | None = None,
    enable_tool_input: bool = False,
) -> AsyncIterator[SandboxSession]:
    """Create, yield, and remove one isolated container for an agent run."""
    engine = adapter or _adapter_for(config)
    owns_engine = adapter is None
    run_id = uuid.uuid4().hex
    container_id: str | None = None
    tool_input_directory: tempfile.TemporaryDirectory[str] | None = None
    failure: BaseException | None = None

    def remaining() -> float:
        value = deadline - asyncio.get_running_loop().time()
        if value <= 0:
            raise SandboxTimeout("The agent run deadline expired during sandbox startup")
        return value

    async def close_owned_engine_after_startup_failure(error: BaseException) -> None:
        if tool_input_directory is not None:
            try:
                tool_input_directory.cleanup()
            except Exception:
                error.add_note("Could not remove the sandbox tool input directory")
        if not owns_engine:
            return
        try:
            await engine.aclose()
        except Exception:
            error.add_note("Could not close the sandbox engine adapter after startup failure")

    try:
        daemon_os = (await engine.daemon_os(timeout=remaining())).lower()
        if daemon_os not in ("linux", "windows"):
            raise SandboxUnavailable(f"Unsupported container engine host OS: {daemon_os}")

        image_os = await engine.image_os(config.image, timeout=remaining())
        if image_os is None:
            await engine.pull_image(config.image, timeout=remaining())
            image_os = await engine.image_os(config.image, timeout=remaining())
        if image_os is None:
            raise SandboxUnavailable(f"Image is unavailable after pull: {config.image}")
        image_os = image_os.lower()
        if image_os not in ("linux", "windows") or image_os != daemon_os:
            raise SandboxUnavailable(
                f"Image OS {image_os!r} does not match engine host OS {daemon_os!r}"
            )
        if image_os == "windows" and config.engine != "docker":
            raise SandboxUnavailable("Native Windows containers require the Docker engine")
        if image_os == "windows":
            validate_windows = getattr(engine, "validate_windows_sandbox_support", None)
            if validate_windows is None:
                raise SandboxUnavailable(
                    "The configured Docker adapter cannot verify native Windows network isolation"
                )
            await validate_windows(timeout=remaining())
        if image_os == "windows" and config.user != DEFAULT_SANDBOX_USER:
            raise SandboxError("sandbox.user is supported only for Linux containers")
        if image_os == "windows":
            if config.process_limit is not None:
                raise SandboxError(
                    "Native Windows containers do not support the configured process limit; "
                    "set sandbox.resources.process_limit to null to explicitly opt out"
                )
            if not float(config.cpus).is_integer():
                raise SandboxError(
                    "Native Windows containers require sandbox.resources.cpus to be a whole number"
                )
        elif config.process_limit is None:
            raise SandboxError("Linux containers require sandbox.resources.process_limit")

        if not isinstance(enable_tool_input, bool):
            raise SandboxError("enable_tool_input must be a boolean")
        if enable_tool_input:
            tool_input_directory = tempfile.TemporaryDirectory(prefix="gabby-tool-input-")
            if os.name == "posix":
                Path(tool_input_directory.name).chmod(0o711)
        tool_input_container_path = (
            (
                r"C:\Windows\Temp\GabbyToolInput"
                if image_os == "windows"
                else "/tmp/gabby-tool-input"
            )
            if enable_tool_input
            else None
        )

        workspace = config.workspace
        workspace_container_path = (
            workspace.container_path
            if workspace is not None and workspace.container_path is not None
            else (r"C:\workspace" if image_os == "windows" else "/workspace")
        )
        if workspace is not None:
            if image_os == "windows":
                windows_path = PureWindowsPath(workspace_container_path)
                if not windows_path.is_absolute() or ".." in windows_path.parts:
                    raise SandboxError("Windows workspace mount path must be an absolute safe path")
            else:
                posix_path = PurePosixPath(workspace_container_path)
                if not posix_path.is_absolute() or ".." in posix_path.parts:
                    raise SandboxError("Linux workspace mount path must be an absolute safe path")
        spec = ContainerSpec(
            run_id=run_id,
            image=config.image,
            image_os=image_os,
            daemon_os=daemon_os,
            keepalive_argv=config.keepalive_argv,
            workspace_host_path=str(workspace.host_path) if workspace is not None else None,
            workspace_container_path=workspace_container_path if workspace is not None else None,
            workspace_read_only=workspace is None or workspace.access == "read_only",
            cpus=config.cpus,
            memory_bytes=config.memory_bytes,
            process_limit=config.process_limit,
            tool_input_host_path=(tool_input_directory.name if tool_input_directory else None),
            tool_input_container_path=tool_input_container_path,
            user=config.user,
        )
        container_id = await engine.start_container(spec, timeout=remaining())
    except asyncio.CancelledError as exc:
        failure = exc
        await close_owned_engine_after_startup_failure(exc)
        raise
    except SandboxError as exc:
        failure = exc
        await close_owned_engine_after_startup_failure(exc)
        raise
    except Exception as exc:
        failure = SandboxUnavailable("Could not initialize the configured sandbox")
        await close_owned_engine_after_startup_failure(failure)
        raise failure from exc

    try:
        yield SandboxSession(
            adapter=engine,
            container_id=container_id,
            container_os=image_os,
            workspace_path=spec.workspace_container_path,
            deadline=deadline,
            workspace_read_only=spec.workspace_read_only,
            tool_input_host_path=spec.tool_input_host_path,
            tool_input_container_path=spec.tool_input_container_path,
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        cleanup_error: Exception | None = None
        if container_id is not None:
            try:
                await engine.remove_container(container_id, timeout=10.0)
            except Exception as exc:
                cleanup_error = exc
        if owns_engine:
            try:
                await engine.aclose()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if tool_input_directory is not None:
            try:
                tool_input_directory.cleanup()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            error = SandboxError("Could not clean up the per-run sandbox")
            if isinstance(failure, BaseException):
                failure.add_note(str(error))
            else:
                raise error from cleanup_error
