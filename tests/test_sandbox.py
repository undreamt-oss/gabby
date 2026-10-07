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
"""Sandbox contracts, lifecycle, and engine adapter behavior."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tarfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from gabby.agent import Agent
from gabby.config import AgentDefinition, SandboxDefinition, WorkspaceMount
from gabby.models import ModelResponse
from gabby.python_tools import python_run_tool
from gabby.sandbox import (
    ContainerSpec,
    SandboxCommandResult,
    SandboxError,
    SandboxOutputLimit,
    SandboxResourceUsage,
    SandboxSession,
    SandboxTimeout,
    SandboxUnavailable,
    _archive_entries,
    _single_file_from_archive,
    open_sandbox,
)
from gabby.sandbox_engines import (
    ContainerAPIAdapter,
    ContainerCLIAdapter,
    _api_resource_usage,
    _cli_memory_bytes,
    _cli_resource_usage,
    _decode_engine_stream,
    _NamedPipeHTTPTransport,
    _NamedPipeResponseStream,
    _run_process,
)
from gabby.tools import Tool, ToolRegistry


def _config(
    tmp_path: Path,
    *,
    engine: str = "docker",
    adapter: str = "cli",
    workspace_access: str = "read_write",
) -> SandboxDefinition:
    return SandboxDefinition(
        engine=engine,
        image="example/test:latest",
        keepalive_argv=("sleep", "infinity"),
        adapter=adapter,
        workspace=WorkspaceMount(tmp_path, workspace_access, "/workspace"),
        api_base_url="http://localhost" if adapter == "api" else None,
    )


class FakeAdapter:
    def __init__(self, *, image_os: str | None = "linux", daemon_os: str = "linux") -> None:
        self.image_os_value = image_os
        self.daemon_os_value = daemon_os
        self.pulled: list[str] = []
        self.started: list[ContainerSpec] = []
        self.removed: list[str] = []
        self.closed = False
        self.archive = b""
        self.written: list[tuple[str, bytes]] = []
        self.windows_support_error: SandboxUnavailable | None = None
        self.usage: SandboxResourceUsage | None = None

    async def daemon_os(self, *, timeout: float) -> str:
        assert timeout > 0
        return self.daemon_os_value

    async def validate_windows_sandbox_support(self, *, timeout: float) -> None:
        assert timeout > 0
        if self.windows_support_error is not None:
            raise self.windows_support_error

    async def image_os(self, image: str, *, timeout: float) -> str | None:
        assert image == "example/test:latest"
        assert timeout > 0
        return self.image_os_value

    async def pull_image(self, image: str, *, timeout: float) -> None:
        assert timeout > 0
        self.pulled.append(image)
        self.image_os_value = "linux"

    async def start_container(self, spec: ContainerSpec, *, timeout: float) -> str:
        assert timeout > 0
        self.started.append(spec)
        return "container-id"

    async def exec(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        timeout: float,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        assert container_id == "container-id"
        assert timeout > 0
        assert environment is None or all(isinstance(key, str) for key in environment)
        return SandboxCommandResult(0, b"ok", b"")

    async def read_archive(self, container_id: str, path: str, *, timeout: float) -> bytes:
        assert container_id == "container-id"
        assert timeout > 0
        return self.archive

    async def write_archive(
        self, container_id: str, path: str, archive: bytes, *, timeout: float
    ) -> None:
        assert container_id == "container-id"
        assert timeout > 0
        self.written.append((path, archive))

    async def remove_container(self, container_id: str, *, timeout: float) -> None:
        self.removed.append(container_id)

    async def resource_usage(
        self, container_id: str, *, timeout: float
    ) -> SandboxResourceUsage | None:
        assert container_id == "container-id"
        assert timeout > 0
        return self.usage

    async def aclose(self) -> None:
        self.closed = True


def _archive_file(name: str, content: bytes) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _archive_directory(entries: list[tuple[str, bytes | None]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _archive_with_symlink(name: str, target: str) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.type = tarfile.SYMTYPE
        info.linkname = target
        tar.addfile(info)
    return stream.getvalue()


def test_container_resource_usage_parsers_bound_malformed_and_valid_stats() -> None:
    assert _cli_memory_bytes("1.5 MiB") == 1_572_864
    assert _cli_memory_bytes("1.5 MB") == 1_500_000
    for value in (None, 12, "", "NaN GiB", "1e999 TB"):
        assert _cli_memory_bytes(value) is None

    assert _cli_resource_usage(None) is None
    assert _cli_resource_usage({"CPUPerc": "12.5%", "MemUsage": "2 MiB / 8 GiB", "PIDs": "7"}) == (
        SandboxResourceUsage(
            cpu_percent=12.5,
            memory_current_bytes=2 * 1024 * 1024,
            process_count=7,
        )
    )
    assert _cli_resource_usage({"CPUPerc": "NaN%", "MemUsage": "invalid", "PIDs": "-1"}) is None
    assert _cli_resource_usage({"CPUPerc": 1.0, "MemUsage": 8, "PIDs": True}) is None

    assert _api_resource_usage([]) is None
    assert _api_resource_usage(
        {
            "cpu_stats": {"cpu_usage": {"total_usage": 11}},
            "memory_stats": {"usage": 12, "max_usage": 13},
            "pids_stats": {"current": 14},
        }
    ) == SandboxResourceUsage(
        cpu_time_ns=11,
        memory_current_bytes=12,
        memory_peak_bytes=13,
        process_count=14,
    )
    assert _api_resource_usage(
        {
            "cpu_stats": {"cpu_usage": {"total_usage": True}},
            "memory_stats": {"usage": -1, "max_usage": "13"},
            "pids_stats": {"current": 0},
        }
    ) == SandboxResourceUsage(process_count=0)
    assert _api_resource_usage({"cpu_stats": [], "memory_stats": None, "pids_stats": None}) is None


@pytest.mark.asyncio
async def test_open_sandbox_builds_scoped_container_and_cleans_it(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    config = _config(tmp_path)

    async with open_sandbox(
        config,
        deadline=asyncio.get_running_loop().time() + 5,
        adapter=adapter,
    ) as session:
        result = await session.exec(["python", "-V"])
        assert result.stdout == b"ok"
        assert not session.workspace_read_only

    spec = adapter.started[0]
    assert spec.image_os == spec.daemon_os == "linux"
    assert spec.workspace_host_path == str(tmp_path)
    assert spec.workspace_container_path == "/workspace"
    assert not spec.workspace_read_only
    assert spec.cpus == 2
    assert spec.memory_bytes == 2 * 1024**3
    assert spec.process_limit == 256
    assert spec.user == "65532:65532"
    assert adapter.removed == ["container-id"]
    assert not adapter.closed  # Injected adapters remain owned by the host application.


@pytest.mark.asyncio
async def test_sandbox_session_runs_json_tool_and_removes_its_request_file(tmp_path: Path) -> None:
    class ToolAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[tuple[str, ...]] = []
            self.request_mode: int | None = None

        async def exec(
            self,
            container_id: str,
            argv: Sequence[str],
            *,
            timeout: float,
            working_directory: str | None = None,
            environment: dict[str, str] | None = None,
        ) -> SandboxCommandResult:
            self.commands.append(tuple(argv))
            if argv[0] == "classify-document":
                request = Path(tool_input_dir, Path(argv[-1]).name)
                self.request_mode = request.stat().st_mode & 0o777
                return SandboxCommandResult(0, b'{"category":"billing"}', b"")
            return SandboxCommandResult(0, b"", b"")

    adapter = ToolAdapter()
    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )

    result = await session.invoke_tool(
        ("classify-document",), {"text": "Invoice overdue"}, max_input_bytes=1024
    )

    assert result == {"category": "billing"}
    assert len(adapter.commands) == 1
    assert adapter.commands[0][0] == "classify-document"
    assert adapter.commands[0][1].startswith("/tmp/gabby-tool-input/gabby-tool-")
    if os.name == "posix":
        assert adapter.request_mode == 0o444
    assert list(tool_input_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_sandbox_session_rejects_oversized_custom_tool_input_before_upload() -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
    )

    with pytest.raises(SandboxOutputLimit, match="max_input_bytes"):
        await session.invoke_tool(("tool",), {"value": "too large"}, max_input_bytes=1)
    assert not adapter.written


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "arguments", "max_input_bytes", "message"),
    [
        ((), {}, 10, "non-empty argv"),
        (("tool", "bad\x00arg"), {}, 10, "non-empty argv"),
        (("tool",), {"value": float("nan")}, 10, "valid JSON"),
        (("tool",), {}, True, "positive integer"),
        (("tool",), {}, 0, "positive integer"),
    ],
)
async def test_sandbox_session_rejects_invalid_json_tool_request(
    command: tuple[str, ...],
    arguments: dict[str, object],
    max_input_bytes: int,
    message: str,
) -> None:
    session = SandboxSession(
        FakeAdapter(),
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
    )
    with pytest.raises(SandboxError, match=message):
        await session.invoke_tool(command, arguments, max_input_bytes=max_input_bytes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "stdout", "message"),
    [
        (1, b"{}", "non-zero exit code"),
        (0, b"\xff", "valid UTF-8 JSON"),
        (0, b"NaN", "valid UTF-8 JSON"),
    ],
)
async def test_sandbox_session_rejects_bad_json_tool_output_and_cleans_file(
    tmp_path: Path, exit_code: int, stdout: bytes, message: str
) -> None:
    class ResultAdapter(FakeAdapter):
        async def exec(self, *args: Any, **kwargs: Any) -> SandboxCommandResult:
            return SandboxCommandResult(exit_code, stdout, b"")

    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()
    session = SandboxSession(
        ResultAdapter(),
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )

    with pytest.raises(SandboxError, match=message):
        await session.invoke_tool(("tool",), {}, max_input_bytes=1024)
    assert list(tool_input_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_sandbox_session_cleans_partial_request_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()
    session = SandboxSession(
        FakeAdapter(),
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )
    original_chmod = Path.chmod

    def fail_request_chmod(path: Path, mode: int) -> None:
        if path.parent == tool_input_dir:
            raise OSError("chmod unavailable")
        original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", fail_request_chmod)
    with pytest.raises(SandboxError, match="stage sandboxed tool input"):
        await session.invoke_tool(("tool",), {}, max_input_bytes=1024)
    assert list(tool_input_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_open_sandbox_cleans_tool_input_mount_after_run_and_start_failure(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    async with open_sandbox(
        _config(tmp_path),
        deadline=asyncio.get_running_loop().time() + 5,
        adapter=adapter,
        enable_tool_input=True,
    ) as session:
        host_dir = Path(session.tool_input_host_path or "")
        assert host_dir.is_dir()
        assert adapter.started[0].tool_input_container_path == "/tmp/gabby-tool-input"
        if os.name == "posix":
            assert host_dir.stat().st_mode & 0o777 == 0o711
    assert not host_dir.exists()

    class StartFailure(FakeAdapter):
        async def start_container(self, spec: ContainerSpec, *, timeout: float) -> str:
            await super().start_container(spec, timeout=timeout)
            raise SandboxError("startup rejected")

    failed_adapter = StartFailure()
    with pytest.raises(SandboxError, match="startup rejected"):
        async with open_sandbox(
            _config(tmp_path),
            deadline=asyncio.get_running_loop().time() + 5,
            adapter=failed_adapter,
            enable_tool_input=True,
        ):
            pytest.fail("failed container startup must not yield a sandbox")
    failed_host_dir = Path(failed_adapter.started[0].tool_input_host_path or "")
    assert not failed_host_dir.exists()


@pytest.mark.asyncio
async def test_open_sandbox_closes_owned_adapter_after_unexpected_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gabby import sandbox

    class AdapterFailure(FakeAdapter):
        async def daemon_os(self, *, timeout: float) -> str:
            raise RuntimeError("private engine detail")

    adapter = AdapterFailure()
    monkeypatch.setattr(sandbox, "_adapter_for", lambda _: adapter)
    with pytest.raises(SandboxUnavailable, match="initialize") as error:
        async with open_sandbox(_config(tmp_path), deadline=asyncio.get_running_loop().time() + 5):
            pytest.fail("unexpected adapter failures must be sanitized")
    assert "private engine detail" in str(error.value.__cause__)
    assert adapter.closed


@pytest.mark.asyncio
async def test_sandbox_session_cleans_request_file_when_custom_tool_output_is_invalid(
    tmp_path: Path,
) -> None:
    class InvalidOutputAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[tuple[str, ...]] = []

        async def exec(
            self,
            container_id: str,
            argv: Sequence[str],
            *,
            timeout: float,
            working_directory: str | None = None,
            environment: dict[str, str] | None = None,
        ) -> SandboxCommandResult:
            self.commands.append(tuple(argv))
            return SandboxCommandResult(0, b"not-json", b"")

    adapter = InvalidOutputAdapter()
    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )

    with pytest.raises(SandboxError, match="valid UTF-8 JSON"):
        await session.invoke_tool(("tool",), {"text": "data"}, max_input_bytes=1024)

    assert len(adapter.commands) == 1
    assert list(tool_input_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_open_sandbox_pulls_missing_image_and_closes_owned_adapter(tmp_path: Path) -> None:
    adapter = FakeAdapter(image_os=None)
    config = _config(tmp_path)
    from gabby.sandbox import open_sandbox

    # The adapter is injected, so image lifecycle is observable and ownership is retained.
    async with open_sandbox(
        config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
    ):
        pass

    assert adapter.pulled == [config.image]
    assert adapter.removed == ["container-id"]
    assert not adapter.closed


@pytest.mark.asyncio
async def test_open_sandbox_propagates_configured_linux_user(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    config = replace(_config(tmp_path), user="1200:1300")

    async with open_sandbox(
        config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
    ):
        pass

    assert adapter.started[0].user == "1200:1300"


@pytest.mark.asyncio
async def test_open_sandbox_fails_on_os_mismatch_without_starting_container(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="linux")
    with pytest.raises(SandboxUnavailable, match="does not match"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("sandbox should not start for an OS mismatch")
    assert not adapter.started


@pytest.mark.asyncio
async def test_open_sandbox_rejects_native_windows_on_podman(tmp_path: Path) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    with pytest.raises(SandboxUnavailable, match="require the Docker engine"):
        async with open_sandbox(
            _config(tmp_path, engine="podman"),
            deadline=asyncio.get_running_loop().time() + 5,
            adapter=adapter,
        ):
            pytest.fail("Podman must not start a native Windows image")


@pytest.mark.asyncio
async def test_sandbox_session_limits_paths_and_supports_workspace_archives(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        workspace_read_only=False,
    )
    adapter.archive = _archive_file("demo.txt", b"hello")

    result = await session.invoke("shell.run", {"argv": ["echo", "hello"]})
    assert result == {"exit_code": 0, "stdout": "ok", "stderr": ""}
    assert await session.invoke("filesystem.read_file", {"path": "demo.txt"}) == {
        "path": "demo.txt",
        "content": "hello",
    }
    adapter.archive = _archive_directory([("src", None)])
    written = await session.invoke(
        "filesystem.write_file", {"path": "src/demo.txt", "content": "changed"}
    )
    assert written == {"path": "src/demo.txt", "bytes_written": 7}
    assert adapter.written[0][0] == "/workspace/src"
    adapter.archive = _archive_directory([("demo.txt", b"hello")])
    assert await session.invoke("filesystem.list_dir", {"path": "."}) == {
        "path": ".",
        "entries": [{"name": "demo.txt", "type": "file", "size": 5}],
    }

    with pytest.raises(SandboxError, match="cannot escape"):
        await session.invoke("filesystem.read_file", {"path": "../outside"})


@pytest.mark.asyncio
async def test_sandbox_session_rejects_write_on_read_only_workspace(tmp_path: Path) -> None:
    session = SandboxSession(
        FakeAdapter(),
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
    )
    with pytest.raises(SandboxError, match="read-only"):
        await session.invoke("filesystem.write_file", {"path": "demo.txt", "content": "x"})


@pytest.mark.asyncio
async def test_sandbox_session_rejects_invalid_commands_paths_and_expired_deadline() -> None:
    session = SandboxSession(
        FakeAdapter(),
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
    )
    for argv in ([], ["echo", "bad\x00arg"]):
        with pytest.raises(SandboxError, match="argv"):
            await session.exec(argv)
    for path in ("/etc/passwd", "C:/secret", "./", "a//b", "bad\nname"):
        with pytest.raises(SandboxError):
            session._workspace_target(path)
    with pytest.raises(SandboxError, match="Unsupported"):
        await session.invoke("unknown", {})
    session.deadline = asyncio.get_running_loop().time() - 1
    with pytest.raises(SandboxTimeout, match="deadline"):
        await session.exec(["true"])


@pytest.mark.asyncio
async def test_sandbox_session_terminates_on_timeout_or_output_excess() -> None:
    class FailingAdapter(FakeAdapter):
        mode = "timeout"

        async def exec(self, *args: Any, **kwargs: Any) -> SandboxCommandResult:
            if self.mode == "timeout":
                raise SandboxTimeout("timed out")
            return SandboxCommandResult(0, b"too long", b"")

    adapter = FailingAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        output_limit_bytes=3,
    )
    with pytest.raises(SandboxTimeout):
        await session.exec(["true"])
    assert adapter.removed == ["container-id"]

    with pytest.raises(SandboxError, match="terminated"):
        await session.exec(["true"])
    adapter = FailingAdapter()
    adapter.mode = "output"
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        output_limit_bytes=3,
    )
    with pytest.raises(SandboxOutputLimit):
        await session.exec(["true"])
    assert adapter.removed == ["container-id"]


@pytest.mark.asyncio
async def test_sandbox_session_reports_cleanup_failure_and_archive_guards() -> None:
    class CleanupFailure(FakeAdapter):
        async def remove_container(self, container_id: str, *, timeout: float) -> None:
            raise RuntimeError("cleanup unavailable")

    adapter = CleanupFailure()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        output_limit_bytes=1,
    )
    with pytest.raises(SandboxOutputLimit) as exc:
        await session.exec(["true"])
    assert "cleanup unavailable" in " ".join(exc.value.__notes__)
    with pytest.raises(SandboxError, match="terminated"):
        await session.read_archive("/workspace/file")
    with pytest.raises(SandboxError, match="terminated"):
        await session.write_archive("/workspace", b"x")


@pytest.mark.asyncio
async def test_sandbox_session_checks_filesystem_size_and_listing_limits() -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        output_limit_bytes=2,
        workspace_read_only=False,
    )
    with pytest.raises(SandboxOutputLimit, match="write"):
        await session.invoke("filesystem.write_file", {"path": "a", "content": "long"})
    adapter.archive = _archive_file("a", b"long")
    with pytest.raises(SandboxOutputLimit, match="file"):
        await session.invoke("filesystem.read_file", {"path": "a"})
    adapter.archive = _archive_directory([(f"entry-{i}", b"") for i in range(10_001)])
    with pytest.raises(SandboxOutputLimit, match="too many"):
        await session.invoke("filesystem.list_dir", {"path": "."})


@pytest.mark.asyncio
async def test_sandbox_rejects_bad_tool_arguments_and_archive_leaf() -> None:
    session = SandboxSession(
        FakeAdapter(),
        "id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        workspace_read_only=False,
    )
    with pytest.raises(SandboxError, match="argv"):
        await session.invoke("shell.run", {"argv": "echo hi"})
    with pytest.raises(SandboxError, match="string"):
        await session.invoke("filesystem.read_file", {"path": None})
    for name in ("", "..", "a/b", "a\\b", "x\x00y"):
        with pytest.raises(SandboxError, match="filename"):
            session._validate_archive_leaf(name)
    with pytest.raises(SandboxError, match="non-empty"):
        session._workspace_target(".")


@pytest.mark.asyncio
async def test_sandbox_filesystem_requires_workspace_and_valid_utf8(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        None,
        asyncio.get_running_loop().time() + 5,
        workspace_read_only=False,
    )
    with pytest.raises(SandboxError, match="sandbox.workspace.path"):
        await session.invoke("filesystem.read_file", {"path": "a.txt"})
    session.workspace_path = "/workspace"
    with pytest.raises(UnicodeDecodeError):
        adapter.archive = _archive_file("a.txt", b"\xff")
        await session.invoke("filesystem.read_file", {"path": "a.txt"})
    with pytest.raises(SandboxError, match="string"):
        await session.invoke("filesystem.write_file", {"path": "a", "content": 7})


def test_archive_helpers_reject_malformed_and_non_file_archives() -> None:
    with pytest.raises(SandboxError, match="exactly one"):
        _single_file_from_archive(_archive_directory([("dir", None)]))
    with pytest.raises(SandboxError, match="invalid workspace archive"):
        _archive_entries(b"not a tar archive")
    with pytest.raises(SandboxError, match="absolute path"):
        _archive_entries(_archive_file("/etc/passwd", b"x"))


def test_windows_workspace_paths_are_confined_and_normalized() -> None:
    session = SandboxSession(
        FakeAdapter(),
        "container-id",
        "windows",
        r"C:\workspace",
        100_000,
    )
    assert session._workspace_target("src\\main.py") == r"C:\workspace\src\main.py"
    with pytest.raises(SandboxError, match="invalid on Windows"):
        session._workspace_target("file?.txt")
    for path in ("CON", "nul.txt", "COM1.log", "LPT9", "com¹.bin", "folder/PRN.dat"):
        with pytest.raises(SandboxError, match="reserved Windows device name"):
            session._workspace_target(path)
    assert session._workspace_target("console.txt") == r"C:\workspace\console.txt"


@pytest.mark.asyncio
async def test_open_sandbox_rejects_invalid_os_and_expired_deadline(tmp_path: Path) -> None:
    adapter = FakeAdapter(daemon_os="freebsd")
    with pytest.raises(SandboxUnavailable, match="Unsupported"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("unsupported daemon should not start")
    adapter = FakeAdapter()
    with pytest.raises(SandboxTimeout, match="deadline"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() - 1, adapter=adapter
        ):
            pytest.fail("expired run should not start")


@pytest.mark.asyncio
async def test_open_sandbox_rejects_bad_workspace_mount_after_image_resolution(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    config = _config(tmp_path)
    config = replace(config, workspace=WorkspaceMount(tmp_path, "read_only", r"C:\workspace"))
    with pytest.raises(SandboxError, match="Linux workspace mount path"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("bad mount path should not start")
    assert not adapter.started


@pytest.mark.asyncio
async def test_open_sandbox_rejects_linux_user_override_for_windows_image(tmp_path: Path) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    config = replace(_config(tmp_path), user="1200:1300")

    with pytest.raises(SandboxError, match="only for Linux"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("Linux sandbox.user overrides must not be silently ignored")

    assert not adapter.started


@pytest.mark.asyncio
async def test_open_windows_sandbox_requires_explicit_process_limit_opt_out(tmp_path: Path) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    with pytest.raises(SandboxError, match="process_limit to null"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("Windows process limit must be explicitly acknowledged")
    assert not adapter.started


@pytest.mark.asyncio
async def test_open_windows_sandbox_fails_before_start_when_network_isolation_is_unsupported(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    adapter.windows_support_error = SandboxUnavailable(
        "Native Windows containers with disabled networking require Docker Engine 29.1.4 or newer"
    )
    config = replace(_config(tmp_path), process_limit=None)

    with pytest.raises(SandboxUnavailable, match="Docker Engine 29.1.4"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("unsupported Windows daemon must not start a sandbox")

    assert adapter.started == []


@pytest.mark.asyncio
async def test_open_windows_sandbox_rejects_adapter_without_isolation_preflight(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    cast(Any, adapter).validate_windows_sandbox_support = None
    config = replace(_config(tmp_path), process_limit=None)

    with pytest.raises(SandboxUnavailable, match="cannot verify native Windows"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("an adapter without a Windows isolation preflight must not start")

    assert adapter.started == []


@pytest.mark.asyncio
async def test_open_linux_sandbox_rejects_unbounded_process_limit(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    config = replace(_config(tmp_path), process_limit=None)
    with pytest.raises(SandboxError, match="Linux containers require"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("Linux sandbox must retain its process limit")
    assert not adapter.started


@pytest.mark.asyncio
async def test_open_windows_sandbox_rejects_fractional_cpu_limit(tmp_path: Path) -> None:
    adapter = FakeAdapter(image_os="windows", daemon_os="windows")
    config = replace(_config(tmp_path), process_limit=None, cpus=1.5)
    with pytest.raises(SandboxError, match="whole number"):
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("Windows CPU count must be integral")
    assert not adapter.started


@pytest.mark.asyncio
async def test_open_sandbox_reports_image_still_missing_after_pull(tmp_path: Path) -> None:
    class PullFailure(FakeAdapter):
        async def pull_image(self, image: str, *, timeout: float) -> None:
            self.pulled.append(image)

    adapter = PullFailure(image_os=None)
    with pytest.raises(SandboxUnavailable, match="unavailable after pull"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pytest.fail("missing image should not start")


@pytest.mark.asyncio
async def test_open_sandbox_cleans_up_after_body_error_and_reports_cleanup_failure(
    tmp_path: Path,
) -> None:
    class CleanupFailure(FakeAdapter):
        async def remove_container(self, container_id: str, *, timeout: float) -> None:
            raise RuntimeError("remove failed")

    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="body"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            raise ValueError("body")
    assert adapter.removed == ["container-id"]

    adapter = CleanupFailure()
    with pytest.raises(SandboxError, match="clean up"):
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            pass


@pytest.mark.asyncio
async def test_open_sandbox_removes_container_when_run_is_cancelled(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    entered = asyncio.Event()

    async def run_until_cancelled() -> None:
        async with open_sandbox(
            _config(tmp_path), deadline=asyncio.get_running_loop().time() + 5, adapter=adapter
        ):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run_until_cancelled())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.removed == ["container-id"]


@pytest.mark.asyncio
async def test_cli_process_runner_captures_input_and_bounds_output() -> None:
    result = await _run_process(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdin.read()); print('err', file=sys.stderr)",
        ],
        timeout=30,
        input_data=b"payload",
    )
    assert result.stdout.strip() == b"payload"
    assert result.stderr.strip() == b"err"
    with pytest.raises(SandboxOutputLimit):
        await _run_process(
            [sys.executable, "-c", "print('0123456789')"], timeout=30, output_limit=2
        )


@pytest.mark.asyncio
async def test_cli_process_runner_stops_a_command_when_combined_output_overflows() -> None:
    started = time.monotonic()
    command = (
        "import sys, time; sys.stdout.write('x' * 700); sys.stdout.flush(); "
        "sys.stderr.write('y' * 700); sys.stderr.flush(); time.sleep(5)"
    )

    with pytest.raises(SandboxOutputLimit):
        await _run_process([sys.executable, "-c", command], timeout=10, output_limit=1024)

    assert time.monotonic() - started < 4


@pytest.mark.asyncio
async def test_cli_process_runner_reports_missing_executable_and_timeout(tmp_path: Path) -> None:
    with pytest.raises(SandboxUnavailable, match="not available"):
        await _run_process(["gabby-no-such-executable"], timeout=2)
    with pytest.raises(SandboxTimeout):
        await _run_process([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.02)
    with pytest.raises(SandboxUnavailable):
        await _run_process([str(tmp_path)], timeout=1)


@pytest.mark.asyncio
async def test_cli_process_runner_shares_one_deadline_across_io_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gabby import sandbox_engines

    class SlowStdin:
        def write(self, _: bytes) -> None:
            return None

        async def drain(self) -> None:
            await asyncio.sleep(0.03)

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    class SlowProcess:
        stdin = SlowStdin()
        stdout = object()
        stderr = object()
        returncode: int | None = None
        killed = False

        async def wait(self) -> int:
            if self.returncode is None:
                await asyncio.sleep(0.03)
                self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = SlowProcess()

    async def spawn(*_: Any, **__: Any) -> SlowProcess:
        return process

    async def drain(*_: Any) -> tuple[bytes, bool]:
        return b"", False

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(sandbox_engines, "_drain", drain)

    with pytest.raises(SandboxTimeout):
        await _run_process(["fake-engine"], timeout=0.045, input_data=b"request")

    assert process.killed


@pytest.mark.asyncio
async def test_cli_adapter_handles_inspect_pull_exec_archives_and_removal(tmp_path: Path) -> None:
    class FakeCLI(ContainerCLIAdapter):
        calls: list[tuple[tuple[str, ...], dict[str, Any]]]

        async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
            self.calls.append((args, kwargs))
            if args[:2] == ("image", "inspect"):
                return SandboxCommandResult(1, b"", b"missing")
            if args[:2] == ("info", "--format"):
                return SandboxCommandResult(0, b'{"host":{"os":"linux"}}', b"")
            if args[:2] == ("image", "pull"):
                return SandboxCommandResult(0, b"", b"")
            if args[:1] == ("exec",):
                return SandboxCommandResult(4, b"out", b"err")
            if args[:1] == ("cp",) and args[1] == "-":
                return SandboxCommandResult(0, b"", b"")
            if args[:1] == ("cp",):
                return SandboxCommandResult(0, b"archive", b"")
            if args[:1] == ("rm",):
                return SandboxCommandResult(1, b"", b"No such container")
            return SandboxCommandResult(0, b"", b"")

    adapter = FakeCLI("podman")
    adapter.calls = []
    assert await adapter.daemon_os(timeout=2) == "linux"
    assert await adapter.image_os("x", timeout=2) is None
    await adapter.pull_image("x", timeout=2)
    result = await adapter.exec(
        "id", ["echo", "x"], timeout=2, working_directory="/workspace", environment={"A": "b"}
    )
    assert result == SandboxCommandResult(4, b"out", b"err")
    assert await adapter.read_archive("id", "/workspace/a", timeout=2) == b"archive"
    await adapter.write_archive("id", "/workspace", b"tar", timeout=2)
    await adapter.remove_container("id", timeout=2)
    assert any("A=b" in call[0] for call in adapter.calls)


@pytest.mark.asyncio
async def test_cli_adapter_transports_custom_tool_input_and_output(tmp_path: Path) -> None:
    class ToolCLI(ContainerCLIAdapter):
        def __init__(self) -> None:
            super().__init__("docker")
            self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

        async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
            self.calls.append((args, kwargs))
            if args[:2] == ("exec", "container") and args[2] == "classify-invoice":
                return SandboxCommandResult(0, b'{"label":"invoice"}', b"")
            return SandboxCommandResult(0, b"", b"")

    adapter = ToolCLI()
    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()
    session = SandboxSession(
        adapter,
        "container",
        "linux",
        None,
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )

    result = await session.invoke_tool(
        ("classify-invoice",), {"text": "Invoice 123"}, max_input_bytes=1024
    )

    assert result == {"label": "invoice"}
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0][2] == "classify-invoice"
    assert adapter.calls[0][0][-1].startswith("/tmp/gabby-tool-input/gabby-tool-")
    assert list(tool_input_dir.iterdir()) == []


def test_cli_rejects_invalid_engine_and_unsupported_workspace_mount(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ContainerCLIAdapter("invalid")


@pytest.mark.asyncio
async def test_cli_rejects_bad_environment_and_ambiguous_bind_mount(tmp_path: Path) -> None:
    adapter = RecordingCLI("docker")
    with pytest.raises(SandboxError, match="environment"):
        await adapter.exec("id", ["true"], timeout=1, environment={"BAD=KEY": "x"})
    spec = ContainerSpec(
        "abc",
        "img",
        "linux",
        "linux",
        ("sleep", "1"),
        "/tmp/a,b",
        "/workspace",
        False,
        1,
        1024,
        20,
    )
    with pytest.raises(SandboxError, match="commas"):
        await adapter.start_container(spec, timeout=1)


@pytest.mark.asyncio
async def test_filesystem_listing_returns_only_direct_children_and_rejects_unsafe_tar(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
    )
    adapter.archive = _archive_directory(
        [
            ("workspace", None),
            ("workspace/a.txt", b"a"),
            ("workspace/sub", None),
            ("workspace/sub/b.txt", b"b"),
        ]
    )
    result = await session.invoke("filesystem.list_dir", {"path": "."})
    assert result == {
        "path": ".",
        "entries": [
            {"name": "a.txt", "type": "file", "size": 1},
            {"name": "sub", "type": "directory", "size": 0},
        ],
    }

    adapter.archive = _archive_file("../outside", b"secret")
    with pytest.raises(SandboxError, match="unsafe path"):
        await session.invoke("filesystem.read_file", {"path": "outside"})


@pytest.mark.asyncio
async def test_workspace_filesystem_rejects_symlink_components_before_following() -> None:
    adapter = FakeAdapter()
    session = SandboxSession(
        adapter,
        "container-id",
        "linux",
        "/workspace",
        asyncio.get_running_loop().time() + 5,
        workspace_read_only=False,
    )
    adapter.archive = _archive_with_symlink("secrets", "/etc")
    with pytest.raises(SandboxError, match="regular directory"):
        await session.invoke("filesystem.read_file", {"path": "secrets/passwd"})
    with pytest.raises(SandboxError, match="regular file"):
        await session.invoke("filesystem.read_file", {"path": "secrets"})
    with pytest.raises(SandboxError, match="regular file"):
        await session.invoke("filesystem.write_file", {"path": "secrets", "content": "overwrite"})


@pytest.mark.asyncio
async def test_agent_runs_sandbox_tool_in_one_run_scoped_container(tmp_path: Path) -> None:
    class Model:
        name = "fake"

        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "shell_run",
                                "arguments": '{"argv":["echo","hello"]}',
                            },
                        }
                    ]
                ),
                ModelResponse(content="finished"),
            ]

        async def complete(self, **_: Any) -> ModelResponse:
            return self.responses.pop(0)

    adapter = FakeAdapter()
    adapter.usage = SandboxResourceUsage(
        cpu_time_ns=125_000_000,
        memory_current_bytes=4_096,
        memory_peak_bytes=8_192,
        process_count=3,
    )
    definition = AgentDefinition(
        name="sandbox-agent",
        model={"provider": "fake", "model": "test"},
        tools=["shell_run"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 5,
            "allowed_tools": ["shell_run"],
            "allowed_permissions": ["shell.execute"],
        },
        sandbox=_config(tmp_path),
    )
    agent = Agent(definition, model=Model(), sandbox_adapter=adapter)

    result = await agent.arun("echo hello")

    assert result.output == "finished"
    assert adapter.started[0].run_id
    assert adapter.removed == ["container-id"]
    usage_events = [
        event for event in result.trace.events if event.kind == "sandbox_resource_usage"
    ]
    assert len(usage_events) == 1
    assert usage_events[0].details == {
        "cpu_time_ns": 125_000_000,
        "memory_current_bytes": 4_096,
        "memory_peak_bytes": 8_192,
        "process_count": 3,
    }


@pytest.mark.asyncio
async def test_agent_runs_user_defined_sandbox_tool_in_container(tmp_path: Path) -> None:
    class Model:
        name = "fake"

        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    tool_calls=[
                        {
                            "id": "custom-call",
                            "type": "function",
                            "function": {
                                "name": "classify_document",
                                "arguments": '{"text":"Invoice overdue"}',
                            },
                        }
                    ]
                ),
                ModelResponse(content="It is a billing issue."),
            ]

        async def complete(self, **_: Any) -> ModelResponse:
            return self.responses.pop(0)

    class CustomToolAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[tuple[str, ...]] = []

        async def exec(
            self,
            container_id: str,
            argv: Sequence[str],
            *,
            timeout: float,
            working_directory: str | None = None,
            environment: dict[str, str] | None = None,
        ) -> SandboxCommandResult:
            self.commands.append(tuple(argv))
            if argv[0] == "classify-document":
                return SandboxCommandResult(0, b'{"category":"billing"}', b"")
            return SandboxCommandResult(0, b"", b"")

    tool = Tool(
        name="classify_document",
        description="Classify a document using the bundled domain model.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        sandbox_action="tool.execute",
        execution="sandboxed",
        sandbox_command=("classify-document",),
        output_schema={"type": "object", "required": ["category"]},
    )
    adapter = CustomToolAdapter()
    tool_registry = ToolRegistry()
    tool_registry.register(tool)
    definition = AgentDefinition(
        name="sandboxed-custom-tool-agent",
        model={"provider": "fake", "model": "test"},
        tools=["classify_document"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 5,
            "allowed_tools": ["classify_document"],
            "require_sandbox": True,
        },
        sandbox=_config(tmp_path),
    )
    agent = Agent(definition, model=Model(), tools=tool_registry, sandbox_adapter=adapter)

    result = await agent.arun("classify the invoice")

    assert result.output == "It is a billing issue."
    assert adapter.started[0].tool_input_host_path is not None
    assert adapter.started[0].tool_input_container_path == "/tmp/gabby-tool-input"
    assert len(adapter.commands) == 1
    assert adapter.commands[0][0] == "classify-document"
    assert adapter.removed == ["container-id"]


@pytest.mark.asyncio
async def test_agent_runs_python_tool_from_private_mount_in_sandbox(tmp_path: Path) -> None:
    class Model:
        name = "fake"

        def __init__(self) -> None:
            self.responses = [
                ModelResponse(
                    tool_calls=[
                        {
                            "id": "python-call",
                            "type": "function",
                            "function": {
                                "name": "python_run",
                                "arguments": '{"code":"print(40 + 2)"}',
                            },
                        }
                    ]
                ),
                ModelResponse(content="The calculation returned 42."),
            ]

        async def complete(self, **_: Any) -> ModelResponse:
            return self.responses.pop(0)

    class PythonAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.commands: list[tuple[str, ...]] = []
            self.script_mode: int | None = None

        async def exec(
            self,
            container_id: str,
            argv: Sequence[str],
            *,
            timeout: float,
            working_directory: str | None = None,
            environment: dict[str, str] | None = None,
        ) -> SandboxCommandResult:
            self.commands.append(tuple(argv))
            assert self.started[0].tool_input_host_path is not None
            source = Path(self.started[0].tool_input_host_path, Path(argv[-1]).name)
            assert source.read_text(encoding="utf-8") == "print(40 + 2)"
            self.script_mode = source.stat().st_mode & 0o777
            return SandboxCommandResult(0, b"42\n", b"")

    adapter = PythonAdapter()
    tools = ToolRegistry()
    tools.register(python_run_tool())
    definition = AgentDefinition(
        name="python-data-agent",
        model={"provider": "fake", "model": "test"},
        tools=["python_run"],
        policies={
            "max_steps": 2,
            "timeout_seconds": 5,
            "allowed_tools": ["python_run"],
            "allowed_permissions": ["sandbox:python"],
            "require_sandbox": True,
        },
        sandbox=_config(tmp_path),
    )
    agent = Agent(definition, model=Model(), tools=tools, sandbox_adapter=adapter)

    result = await agent.arun("Calculate forty plus two")

    assert result.output == "The calculation returned 42."
    assert adapter.commands[0][:2] == ("python3", "-I")
    assert adapter.commands[0][2].startswith("/tmp/gabby-tool-input/gabby-python-")
    assert adapter.script_mode == 0o444
    assert adapter.removed == ["container-id"]
    assert adapter.started[0].tool_input_host_path is not None
    assert not Path(adapter.started[0].tool_input_host_path).exists()


class RecordingCLI(ContainerCLIAdapter):
    def __init__(self, engine: str) -> None:
        super().__init__(engine)
        self.calls: list[tuple[str, ...]] = []

    async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
        self.calls.append(args)
        if args[0] == "run":
            return SandboxCommandResult(0, b"container-id\n", b"")
        return SandboxCommandResult(0, b"linux\n", b"")


@pytest.mark.asyncio
async def test_cli_adapter_checks_windows_network_isolation_engine_version() -> None:
    class VersionCLI(ContainerCLIAdapter):
        def __init__(self, version: str) -> None:
            super().__init__("docker")
            self.version = version
            self.calls: list[tuple[str, ...]] = []

        async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
            self.calls.append(args)
            return SandboxCommandResult(0, f"{self.version}\n".encode(), b"")

    supported = VersionCLI("29.1.4")
    await supported.validate_windows_sandbox_support(timeout=1)
    assert supported.calls == [("version", "--format", "{{.Server.Version}}")]

    for version in ("29.1.3", "29.1.4-rc1", "not-a-version"):
        unsupported = VersionCLI(version)
        with pytest.raises(SandboxUnavailable):
            await unsupported.validate_windows_sandbox_support(timeout=1)


@pytest.mark.asyncio
async def test_api_adapter_checks_windows_network_isolation_engine_version(tmp_path: Path) -> None:
    requested_paths: list[str] = []

    def make_adapter(version: str) -> tuple[ContainerAPIAdapter, httpx.AsyncClient]:
        def handler(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            return httpx.Response(200, json={"Version": version})

        client = httpx.AsyncClient(
            base_url="http://container-engine", transport=httpx.MockTransport(handler)
        )
        return ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client), client

    adapter, client = make_adapter("29.1.4")
    await adapter.validate_windows_sandbox_support(timeout=1)
    assert requested_paths == ["/version"]
    await client.aclose()

    for version in ("29.1.3", "29.1.4-rc1", "invalid"):
        old_adapter, old_client = make_adapter(version)
        with pytest.raises(SandboxUnavailable):
            await old_adapter.validate_windows_sandbox_support(timeout=1)
        await old_client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_parses_bounded_container_resource_counters(tmp_path: Path) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.url.params["stream"] == "false"
        return httpx.Response(
            200,
            json={
                "cpu_stats": {"cpu_usage": {"total_usage": 125_000_000}},
                "memory_stats": {"usage": 4_096, "max_usage": 8_192},
                "pids_stats": {"current": 3},
            },
        )

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    usage = await adapter.resource_usage("container-id", timeout=1)
    assert usage == SandboxResourceUsage(
        cpu_time_ns=125_000_000,
        memory_current_bytes=4_096,
        memory_peak_bytes=8_192,
        process_count=3,
    )
    assert paths == ["/v1.40/containers/container-id/stats"]
    await client.aclose()


@pytest.mark.asyncio
async def test_cli_adapter_parses_container_resource_counters() -> None:
    class StatsCLI(ContainerCLIAdapter):
        async def _run(self, *args: str, **kwargs: Any) -> SandboxCommandResult:
            assert args == (
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                "container-id",
            )
            return SandboxCommandResult(
                0,
                b'{"CPUPerc":"12.50%","MemUsage":"10MiB / 2GiB","PIDs":"4"}',
                b"",
            )

    usage = await StatsCLI("docker").resource_usage("container-id", timeout=1)
    assert usage == SandboxResourceUsage(
        cpu_percent=12.5,
        memory_current_bytes=10 * 1024 * 1024,
        process_count=4,
    )


@pytest.mark.asyncio
async def test_cli_adapter_uses_network_and_resource_boundaries(tmp_path: Path) -> None:
    adapter = RecordingCLI("docker")
    spec = ContainerSpec(
        "abc123",
        "example/test:latest",
        "linux",
        "linux",
        ("sleep", "infinity"),
        str(tmp_path),
        "/workspace",
        False,
        2,
        2 * 1024**3,
        256,
        user="1200:1300",
    )
    assert await adapter.start_container(spec, timeout=2) == "container-id"
    command = adapter.calls[0]
    assert command[0:2] == ("run", "--detach")
    assert "none" in command
    assert command[command.index("--user") + 1] == "1200:1300"
    assert "--pids-limit" in command and "256" in command
    assert "--read-only" in command and "--cap-drop" in command
    assert any(item.startswith("type=bind,source=") for item in command)


@pytest.mark.asyncio
async def test_cli_adapter_removes_container_by_name_when_start_response_is_lost() -> None:
    calls: list[tuple[str, ...]] = []

    class LostStartResponseCLI(ContainerCLIAdapter):
        async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
            calls.append(args)
            if args[0] == "run":
                raise SandboxTimeout("start response timed out")
            return SandboxCommandResult(0, b"", b"")

    adapter = LostStartResponseCLI("docker")
    spec = ContainerSpec(
        "run-123", "image", "linux", "linux", ("sleep", "1"), None, None, True, 1, 10, 1
    )

    with pytest.raises(SandboxTimeout, match="timed out"):
        await adapter.start_container(spec, timeout=1)

    assert ("rm", "--force", "gabby-run-123") in calls


@pytest.mark.asyncio
async def test_cli_adapter_requires_hyperv_for_native_windows(tmp_path: Path) -> None:
    adapter = RecordingCLI("docker")
    spec = ContainerSpec(
        "abc123",
        "mcr.microsoft.com/windows/servercore:ltsc2022",
        "windows",
        "windows",
        ("powershell.exe", "-Command", "Start-Sleep -Seconds 999999"),
        None,
        None,
        True,
        2,
        2 * 1024**3,
        None,
    )
    await adapter.start_container(spec, timeout=2)
    command = adapter.calls[0]
    assert "--isolation" in command
    assert command[command.index("--isolation") + 1] == "hyperv"
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--memory" in command and command[command.index("--memory") + 1] == str(2 * 1024**3)
    assert "--cpu-count" in command and command[command.index("--cpu-count") + 1] == "2"
    assert "--cpus" not in command
    assert "--pids-limit" not in command
    assert "--user" not in command
    assert "--read-only" not in command
    assert "--cap-drop" not in command
    assert "--security-opt" not in command
    assert "--tmpfs" not in command


def test_decode_engine_exec_stream_separates_stdout_and_stderr() -> None:
    def frame(stream: int, value: bytes) -> bytes:
        return bytes([stream, 0, 0, 0]) + len(value).to_bytes(4, "big") + value

    assert _decode_engine_stream(frame(1, b"out") + frame(2, b"err")) == (b"out", b"err")
    with pytest.raises(SandboxError, match="truncated"):
        _decode_engine_stream(b"\x01\x00")


@pytest.mark.asyncio
async def test_api_adapter_runs_container_and_exec_with_closed_network(tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        path = request.url.path
        if path.endswith("/info"):
            return httpx.Response(200, json={"OSType": "linux"})
        if path.endswith("/images/create"):
            return httpx.Response(200, content=b"{}\n")
        if "/images/" in path and path.endswith("/json"):
            return httpx.Response(200, json={"Os": "linux"})
        if path.endswith("/containers/create"):
            payload = __import__("json").loads(request.content)
            assert payload["User"] == "1200:1300"
            host = payload["HostConfig"]
            assert host["NetworkMode"] == "none"
            assert host["PidsLimit"] == 256
            assert host["ReadonlyRootfs"] is True
            return httpx.Response(201, json={"Id": "api-container"})
        if path.endswith("/containers/api-container/start"):
            return httpx.Response(204)
        if path.endswith("/containers/api-container/exec"):
            return httpx.Response(201, json={"Id": "exec-id"})
        if path.endswith("/exec/exec-id/start"):
            return httpx.Response(200, content=b"\x01\x00\x00\x00\x00\x00\x00\x02ok")
        if path.endswith("/exec/exec-id/json"):
            return httpx.Response(200, json={"ExitCode": 0})
        if path.endswith("/containers/api-container"):
            return httpx.Response(204)
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    assert await adapter.daemon_os(timeout=2) == "linux"
    assert await adapter.image_os("example/test:latest", timeout=2) == "linux"
    await adapter.pull_image("example/test:latest", timeout=2)
    spec = ContainerSpec(
        "abc123",
        "example/test:latest",
        "linux",
        "linux",
        ("sleep", "infinity"),
        str(tmp_path),
        "/workspace",
        False,
        2,
        2 * 1024**3,
        256,
        user="1200:1300",
    )
    container_id = await adapter.start_container(spec, timeout=2)
    result = await adapter.exec(container_id, ["echo", "ok"], timeout=2)
    await adapter.remove_container(container_id, timeout=2)
    assert result == SandboxCommandResult(0, b"ok", b"")
    assert any(method == "POST" and path.endswith("/images/create") for method, path, _ in calls)
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_transports_custom_tool_input_and_output(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    tool_input_dir = tmp_path / "tool-input"
    tool_input_dir.mkdir()

    def frame(stream: int, value: bytes) -> bytes:
        return bytes([stream, 0, 0, 0]) + len(value).to_bytes(4, "big") + value

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/containers/container/exec"):
            body = __import__("json").loads(request.content)
            commands.append(body["Cmd"])
            return httpx.Response(201, json={"Id": f"exec-{len(commands)}"})
        if path.endswith("/exec/exec-1/start"):
            return httpx.Response(200, content=frame(1, b'{"label":"invoice"}'))
        if path.endswith("/exec/exec-1/json"):
            return httpx.Response(200, json={"ExitCode": 0})
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    session = SandboxSession(
        adapter,
        "container",
        "linux",
        None,
        asyncio.get_running_loop().time() + 5,
        tool_input_host_path=str(tool_input_dir),
        tool_input_container_path="/tmp/gabby-tool-input",
    )

    result = await session.invoke_tool(
        ("classify-invoice",), {"text": "Invoice 123"}, max_input_bytes=1024
    )

    assert result == {"label": "invoice"}
    assert len(commands) == 1
    assert commands[0][0] == "classify-invoice"
    assert commands[0][-1].startswith("/tmp/gabby-tool-input/gabby-tool-")
    assert list(tool_input_dir.iterdir()) == []
    await client.aclose()


@pytest.mark.asyncio
async def test_podman_api_adapter_reads_host_os_field(tmp_path: Path) -> None:
    client = httpx.AsyncClient(
        base_url="http://container-engine",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"host": {"os": "linux"}})
        ),
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, engine="podman", adapter="api"), client=client)
    assert await adapter.daemon_os(timeout=1) == "linux"
    requests: list[tuple[str, str, dict[str, str]]] = []

    async def remove_handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(204)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(remove_handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, engine="podman", adapter="api"), client=client)
    await adapter.remove_container("podman-container", timeout=1)
    assert requests == [
        ("POST", "/v1.40/containers/podman-container/stop", {"t": "1"}),
        (
            "DELETE",
            "/v1.40/containers/podman-container",
            {"force": "false", "v": "true"},
        ),
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_handles_missing_images_archive_io_and_hyperv(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/images/missing/json"):
            return httpx.Response(404)
        if path.endswith("/containers/create"):
            body = __import__("json").loads(request.content)
            assert "User" not in body
            host = body["HostConfig"]
            assert host["Isolation"] == "hyperv"
            assert host["NetworkMode"] == "none"
            assert host["Memory"] == 4096
            assert host["CpuCount"] == 2
            assert "NanoCpus" not in host
            assert "PidsLimit" not in host
            assert host["RestartPolicy"] == {"Name": "no"}
            assert host["Mounts"][0]["ReadOnly"] is True
            assert "ReadonlyRootfs" not in host
            assert "CapDrop" not in host
            assert "SecurityOpt" not in host
            assert "Tmpfs" not in host
            return httpx.Response(201, json={"Id": "windows-container"})
        if path.endswith("/start"):
            return httpx.Response(204)
        if path.endswith("/archive") and request.method == "GET":
            return httpx.Response(200, content=b"tar-bytes")
        if path.endswith("/archive") and request.method == "PUT":
            assert request.headers["content-type"] == "application/x-tar"
            return httpx.Response(200)
        if request.method == "DELETE":
            return httpx.Response(404)
        return httpx.Response(500)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    config = _config(tmp_path, adapter="api")
    adapter = ContainerAPIAdapter(config, client=client)
    assert await adapter.image_os("missing", timeout=1) is None
    spec = ContainerSpec(
        "run",
        "windows-image",
        "windows",
        "windows",
        ("powershell.exe", "-Command", "sleep"),
        str(tmp_path),
        r"C:\workspace",
        True,
        2,
        4096,
        None,
    )
    container = await adapter.start_container(spec, timeout=1)
    assert await adapter.read_archive(container, r"C:\workspace\a", timeout=1) == b"tar-bytes"
    await adapter.write_archive(container, r"C:\workspace", b"archive", timeout=1)
    await adapter.remove_container(container, timeout=1)
    assert any(req.method == "PUT" for req in requests)
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_cleans_created_container_when_start_fails(tmp_path: Path) -> None:
    paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        if request.url.path.endswith("/containers/create"):
            return httpx.Response(201, json={"Id": "created"})
        if request.url.path.endswith("/start"):
            return httpx.Response(500, text="start failed")
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    spec = ContainerSpec("run", "img", "linux", "linux", ("sleep", "1"), None, None, True, 1, 10, 1)
    with pytest.raises(SandboxUnavailable, match="HTTP 500"):
        await adapter.start_container(spec, timeout=1)
    assert ("DELETE", "/v1.40/containers/created") in paths
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_cleans_container_by_name_when_create_response_is_lost(
    tmp_path: Path,
) -> None:
    paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/containers/create"):
            raise httpx.ReadTimeout("response lost")
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    spec = ContainerSpec(
        "run-123", "image", "linux", "linux", ("sleep", "1"), None, None, True, 1, 10, 1
    )

    with pytest.raises(SandboxTimeout, match="deadline"):
        await adapter.start_container(spec, timeout=1)

    assert ("DELETE", "/v1.40/containers/gabby-run-123") in paths
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_enforces_response_and_archive_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/archive"):
            return httpx.Response(200, content=b"0123456789")
        return httpx.Response(403, text="denied")

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    with pytest.raises(SandboxOutputLimit):
        await adapter._stream_request("GET", "/containers/c/archive", timeout=1, limit=4)
    with pytest.raises(SandboxUnavailable, match="HTTP 403"):
        await adapter.daemon_os(timeout=1)
    monkeypatch.setattr("gabby.sandbox_engines._ARCHIVE_OUTPUT_LIMIT", 4)
    with pytest.raises(SandboxOutputLimit, match="archive"):
        await adapter.write_archive("c", "/tmp", b"12345", timeout=1)
    await client.aclose()


def test_api_adapter_requires_endpoint_and_rejects_cleartext_remote_api(tmp_path: Path) -> None:
    config = SandboxDefinition(
        engine="docker",
        image="img",
        keepalive_argv=("sleep", "1"),
        adapter="api",
        api_base_url="http://remote.example",
    )
    with pytest.raises(SandboxError, match="HTTPS"):
        ContainerAPIAdapter(config)


def test_api_adapter_rejects_named_pipe_endpoint_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SandboxDefinition(
        engine="docker",
        image="windows/servercore:ltsc2022",
        keepalive_argv=("powershell.exe",),
        adapter="api",
        api_named_pipe=r"\\.\pipe\docker_engine",
    )
    monkeypatch.setattr("gabby.sandbox_engines.sys.platform", "linux")

    with pytest.raises(SandboxUnavailable, match="require a Windows host"):
        ContainerAPIAdapter(config)


@pytest.mark.asyncio
async def test_named_pipe_http_transport_streams_requests_and_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClientError(Exception):
        pass

    class FakeContent:
        async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
            assert chunk_size == 64 * 1024
            yield b"ok"

    class FakeResponse:
        status = 200
        reason = "OK"
        raw_headers = [(b"content-type", b"text/plain"), (b"content-length", b"2")]

        def __init__(self) -> None:
            self.content = FakeContent()
            self.released = False

        def release(self) -> None:
            self.released = True

    class FakeSession:
        closed = False
        request_body = b""
        response: FakeResponse | None = None

        async def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            assert method == "POST"
            assert url.endswith("/engine-check")
            request_data = kwargs["data"]
            self.request_body = b"".join([chunk async for chunk in request_data])
            self.response = FakeResponse()
            return self.response

        async def close(self) -> None:
            self.closed = True

    session = FakeSession()
    connector_options: dict[str, Any] = {}
    session_options: dict[str, Any] = {}

    def create_session(**kwargs: Any) -> FakeSession:
        session_options.update(kwargs)
        return session

    def create_connector(**kwargs: Any) -> object:
        connector_options.update(kwargs)
        return object()

    fake_aiohttp = SimpleNamespace(
        ClientError=FakeClientError,
        ClientSession=create_session,
        ClientTimeout=lambda **kwargs: kwargs,
        NamedPipeConnector=create_connector,
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    transport = _NamedPipeHTTPTransport(r"\\.\pipe\docker_engine")

    async with httpx.AsyncClient(transport=transport, base_url="http://container-engine") as client:
        response = await client.post("/engine-check", content=b"request")

    assert response.status_code == 200
    assert response.text == "ok"
    assert session.request_body == b"request"
    assert connector_options == {"path": r"\\.\pipe\docker_engine"}
    assert session_options["auto_decompress"] is False
    assert session.response is not None and session.response.released
    assert session.closed


@pytest.mark.asyncio
async def test_named_pipe_response_stream_translates_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClientError(Exception):
        pass

    class FailingContent:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
            raise self.error
            yield b""

    class FakeResponse:
        def __init__(self, error: Exception) -> None:
            self.content = FailingContent(error)

        def release(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "aiohttp", SimpleNamespace(ClientError=FakeClientError))

    timed_out = _NamedPipeResponseStream(FakeResponse(TimeoutError()))
    with pytest.raises(httpx.ReadTimeout):
        async for _ in timed_out:
            pass
    await timed_out.aclose()

    failed = _NamedPipeResponseStream(FakeResponse(FakeClientError("private detail")))
    with pytest.raises(httpx.ReadError, match="Could not read"):
        async for _ in failed:
            pass
    await failed.aclose()


@pytest.mark.asyncio
async def test_api_adapter_accepts_unix_socket_and_closes_its_transport(tmp_path: Path) -> None:
    config = SandboxDefinition(
        engine="podman",
        image="img",
        keepalive_argv=("sleep", "1"),
        adapter="api",
        api_unix_socket=tmp_path / "podman.sock",
    )
    adapter = ContainerAPIAdapter(config)
    await adapter.aclose()


@pytest.mark.asyncio
async def test_cli_adapter_reports_engine_errors_and_empty_container_id() -> None:
    class FailedCLI(ContainerCLIAdapter):
        async def _run(self, *args: str, timeout: float, **kwargs: Any) -> SandboxCommandResult:
            if args[0] == "run":
                return SandboxCommandResult(0, b"", b"")
            if args[0] == "info":
                return SandboxCommandResult(0, b"freebsd", b"")
            return SandboxCommandResult(1, b"", b"permission denied")

    adapter = FailedCLI("docker")
    with pytest.raises(SandboxUnavailable, match="unsupported host OS"):
        await adapter.daemon_os(timeout=1)
    with pytest.raises(SandboxUnavailable, match="image pull failed"):
        await adapter.pull_image("img", timeout=1)
    spec = ContainerSpec("run", "img", "linux", "linux", ("sleep", "1"), None, None, True, 1, 10, 1)
    with pytest.raises(SandboxUnavailable, match="no container ID"):
        await adapter.start_container(spec, timeout=1)
    with pytest.raises(SandboxUnavailable, match="could not remove"):
        await adapter.remove_container("id", timeout=1)


@pytest.mark.asyncio
async def test_api_adapter_maps_timeout_and_closes_owned_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, adapter="api")
    adapter = ContainerAPIAdapter(config)
    owned_client = adapter._client

    class ReadTimeoutContext:
        async def __aenter__(self) -> Any:
            raise httpx.ReadTimeout("slow")

        async def __aexit__(self, *args: Any) -> None:
            return None

    def raise_timeout(*args: Any, **kwargs: Any) -> ReadTimeoutContext:
        return ReadTimeoutContext()

    monkeypatch.setattr(adapter._client, "stream", raise_timeout)
    with pytest.raises(SandboxTimeout):
        await adapter.daemon_os(timeout=0.01)
    await adapter.aclose()
    assert owned_client.is_closed


@pytest.mark.asyncio
async def test_api_adapter_maps_http_errors_bad_json_and_invalid_engine_data(
    tmp_path: Path,
) -> None:
    mode = "http"

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "http":
            return httpx.Response(500, text="broken")
        if request.url.path.endswith("/info"):
            return httpx.Response(200, content=b"not json")
        if "/images/" in request.url.path:
            return httpx.Response(200, content=b"[]")
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    with pytest.raises(SandboxUnavailable, match="HTTP 500"):
        await adapter.daemon_os(timeout=1)
    mode = "json"
    with pytest.raises(SandboxUnavailable, match="invalid engine"):
        await adapter.daemon_os(timeout=1)
    with pytest.raises((AttributeError, SandboxUnavailable)):
        await adapter.image_os("img", timeout=1)
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_bounds_pull_progress_and_maps_transport_errors(tmp_path: Path) -> None:
    client = httpx.AsyncClient(
        base_url="http://container-engine",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"123456")),
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("gabby.sandbox_engines._ARCHIVE_OUTPUT_LIMIT", 4)
    with pytest.raises(SandboxOutputLimit, match="pull"):
        await adapter.pull_image("img", timeout=1)
    await client.aclose()
    monkeypatch.undo()

    client = httpx.AsyncClient(
        base_url="http://container-engine",
        transport=httpx.MockTransport(
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))
        ),
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    with pytest.raises(SandboxUnavailable, match="reach"):
        await adapter.daemon_os(timeout=1)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["stream", "pull"])
async def test_api_adapter_reads_only_bounded_preview_for_large_error_body(
    tmp_path: Path, operation: str
) -> None:
    class LargeErrorBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.bytes_yielded = 0
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            chunk = b"E" * 512
            for _ in range(2048):
                self.bytes_yielded += len(chunk)
                yield chunk

        async def aclose(self) -> None:
            self.closed = True

    body = LargeErrorBody()
    client = httpx.AsyncClient(
        base_url="http://container-engine",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, stream=body)),
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    try:
        with pytest.raises(SandboxUnavailable, match="HTTP 500") as error:
            if operation == "stream":
                await adapter._stream_request("GET", "/exec/id/start", timeout=1, limit=1024)
            else:
                await adapter.pull_image("example/test:latest", timeout=1)
        assert str(error.value).endswith("E" * 500)
        assert body.bytes_yielded == 512
        assert body.closed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_rejects_invalid_payloads_and_non_success_pull(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/images/create"):
            return httpx.Response(401, text="unauthorized")
        if request.url.path.endswith("/info"):
            return httpx.Response(200, json={"OSType": "freebsd"})
        if "/images/" in request.url.path:
            return httpx.Response(200, json={"Os": 42})
        if request.url.path.endswith("/containers/create"):
            return httpx.Response(201, json={"Id": ""})
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    with pytest.raises(SandboxUnavailable, match="unsupported host OS"):
        await adapter.daemon_os(timeout=1)
    assert await adapter.image_os("img", timeout=1) is None
    with pytest.raises(SandboxUnavailable, match="HTTP 401"):
        await adapter.pull_image("img", timeout=1)
    spec = ContainerSpec("run", "img", "linux", "linux", ("sleep", "1"), None, None, True, 1, 10, 1)
    with pytest.raises(SandboxUnavailable, match="container ID"):
        await adapter.start_container(spec, timeout=1)
    await client.aclose()


@pytest.mark.asyncio
async def test_api_adapter_exec_validates_results_and_decodes_errors(tmp_path: Path) -> None:
    mode = "bad_id"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/containers/c/exec"):
            if mode == "bad_id":
                return httpx.Response(201, json={})
            return httpx.Response(201, json={"Id": "exec"})
        if request.url.path.endswith("/exec/exec/start"):
            if mode == "bad_stream":
                return httpx.Response(200, content=b"bad")
            return httpx.Response(200, content=b"\x01\x00\x00\x00\x00\x00\x00\x00")
        if request.url.path.endswith("/exec/exec/json"):
            return httpx.Response(200, json={"ExitCode": None})
        return httpx.Response(404)

    client = httpx.AsyncClient(
        base_url="http://container-engine", transport=httpx.MockTransport(handler)
    )
    adapter = ContainerAPIAdapter(_config(tmp_path, adapter="api"), client=client)
    with pytest.raises(SandboxUnavailable, match="exec ID"):
        await adapter.exec("c", ["true"], timeout=1)
    mode = "bad_stream"
    with pytest.raises(SandboxError, match="truncated"):
        await adapter.exec("c", ["true"], timeout=1)
    mode = "bad_exit"
    with pytest.raises(SandboxUnavailable, match="exit code"):
        await adapter.exec("c", ["true"], timeout=1)
    await client.aclose()


def test_decode_engine_stream_rejects_unknown_and_truncated_frames() -> None:
    with pytest.raises(SandboxError, match="unknown"):
        _decode_engine_stream(b"\x03\x00\x00\x00\x00\x00\x00\x00")
    with pytest.raises(SandboxError, match="truncated exec frame"):
        _decode_engine_stream(b"\x01\x00\x00\x00\x00\x00\x00\x02x")


@pytest.mark.asyncio
async def test_api_adapter_remaining_deadline_is_strict() -> None:
    with pytest.raises(SandboxTimeout, match="deadline"):
        ContainerAPIAdapter._remaining(0)
