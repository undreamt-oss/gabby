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
"""Opt-in acceptance checks for native Windows Docker sandboxes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from gabby.config import SandboxDefinition, WorkspaceMount
from gabby.sandbox import SandboxError, open_sandbox
from gabby.sandbox_engines import _run_process


def _windows_sandbox_config(
    image: str, workspace: Path, access: str, adapter: str
) -> SandboxDefinition:
    return SandboxDefinition(
        engine="docker",
        image=image,
        keepalive_argv=(
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "while ($true) { Start-Sleep -Seconds 30 }",
        ),
        adapter=adapter,
        workspace=WorkspaceMount(workspace, access, r"C:\workspace"),
        cpus=2,
        memory_bytes=2 * 1024**3,
        process_limit=None,
        api_named_pipe=(
            os.environ.get("GABBY_WINDOWS_DOCKER_NAMED_PIPE", r"\\.\pipe\docker_engine")
            if adapter == "api"
            else None
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_access", ["read_write", "read_only"])
@pytest.mark.parametrize("adapter", ["cli", "api"])
async def test_native_windows_docker_sandbox_contract_live(
    workspace_access: str, adapter: str, tmp_path: Path
) -> None:
    if os.environ.get("GABBY_RUN_WINDOWS_DOCKER_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_WINDOWS_DOCKER_INTEGRATION=1 on Windows to use Docker")
    if sys.platform != "win32":
        pytest.skip("native Windows Docker acceptance must run on a Windows host")
    image = os.environ.get("GABBY_WINDOWS_DOCKER_IMAGE")
    if not image:
        pytest.fail("GABBY_WINDOWS_DOCKER_IMAGE must name a Windows image usable on this host")

    config = _windows_sandbox_config(image, tmp_path, workspace_access, adapter)
    deadline = asyncio.get_running_loop().time() + 180
    container_id = ""
    async with open_sandbox(config, deadline=deadline) as session:
        container_id = session.container_id
        inspect = await _run_process(
            ["docker", "inspect", "--format", "{{json .}}", container_id], timeout=10
        )
        assert inspect.exit_code == 0, inspect.stderr.decode("utf-8", errors="replace")
        inspection = json.loads(inspect.stdout)
        host_config = inspection["HostConfig"]
        assert host_config["NetworkMode"] == "none"
        assert host_config["Isolation"].casefold() == "hyperv"
        assert host_config["Memory"] == 2 * 1024**3
        assert host_config["CpuCount"] == 2
        assert host_config.get("PidsLimit") in (None, 0)
        workspace_mount = next(
            mount for mount in inspection["Mounts"] if mount["Destination"] == r"C:\workspace"
        )
        assert workspace_mount["RW"] is (workspace_access == "read_write")

        command = await session.exec(["cmd.exe", "/c", "echo sandbox-ready"], timeout=10)
        assert command.exit_code == 0
        assert command.stdout.strip().casefold() == b"sandbox-ready"

        blocked_network = await session.exec(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "$c = New-Object Net.Sockets.TcpClient; try { "
                    "$r = $c.BeginConnect('1.1.1.1', 443, $null, $null); "
                    "if (-not $r.AsyncWaitHandle.WaitOne(3000)) { 'network-disabled'; exit 0 }; "
                    "$c.EndConnect($r); exit 42 } catch { 'network-disabled' } "
                    "finally { $c.Close() }"
                ),
            ],
            timeout=8,
        )
        assert blocked_network.exit_code == 0, blocked_network.stdout.decode(
            "utf-8", errors="replace"
        )
        assert blocked_network.stdout.strip().casefold() == b"network-disabled"

        if workspace_access == "read_write":
            await session.invoke(
                "filesystem.write_file", {"path": "acceptance.txt", "content": "mounted"}
            )
            assert await session.invoke("filesystem.read_file", {"path": "acceptance.txt"}) == {
                "path": "acceptance.txt",
                "content": "mounted",
            }
        else:
            with pytest.raises(SandboxError, match="read-only"):
                await session.invoke(
                    "filesystem.write_file", {"path": "acceptance.txt", "content": "blocked"}
                )

    if workspace_access == "read_write":
        assert (tmp_path / "acceptance.txt").read_text(encoding="utf-8") == "mounted"
    else:
        assert not (tmp_path / "acceptance.txt").exists()
    removed = await _run_process(["docker", "inspect", container_id], timeout=10)
    assert removed.exit_code != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["cli", "api"])
async def test_native_windows_docker_sandbox_is_removed_after_cancellation_live(
    adapter: str,
) -> None:
    if os.environ.get("GABBY_RUN_WINDOWS_DOCKER_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_WINDOWS_DOCKER_INTEGRATION=1 on Windows to use Docker")
    if sys.platform != "win32":
        pytest.skip("native Windows Docker acceptance must run on a Windows host")
    image = os.environ.get("GABBY_WINDOWS_DOCKER_IMAGE")
    if not image:
        pytest.fail("GABBY_WINDOWS_DOCKER_IMAGE must name a Windows image usable on this host")

    config = _windows_sandbox_config(image, Path.cwd(), "read_only", adapter)
    started = asyncio.Event()
    container_id: str | None = None

    async def run_until_cancelled() -> None:
        nonlocal container_id
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 180
        ) as session:
            container_id = session.container_id
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run_until_cancelled())
    try:
        await asyncio.wait_for(started.wait(), timeout=180)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert container_id is not None
        removed = await _run_process(["docker", "inspect", container_id], timeout=10)
        assert removed.exit_code != 0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
