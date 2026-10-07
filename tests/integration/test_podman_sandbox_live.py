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
"""Opt-in acceptance checks against a real local Podman runtime."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gabby.config import SandboxDefinition, WorkspaceMount
from gabby.sandbox import SandboxError, open_sandbox
from gabby.sandbox_engines import _run_process


async def _start_podman_api(socket_path: Path) -> asyncio.subprocess.Process:
    """Start a temporary Podman API service bound only to a test-owned Unix socket."""
    process = await asyncio.create_subprocess_exec(
        "podman",
        "system",
        "service",
        "--time=0",
        f"unix://{socket_path}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    for _ in range(100):
        if socket_path.exists():
            return process
        if process.returncode is not None:
            pytest.fail("Podman API service exited before creating its Unix socket")
        await asyncio.sleep(0.05)
    process.terminate()
    await process.wait()
    pytest.fail("Podman API service did not create its Unix socket within five seconds")


async def _stop_service(process: asyncio.subprocess.Process | None) -> None:
    """Stop the temporary API service after adapter and container cleanup."""
    if process is None or process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["cli", "api"])
@pytest.mark.parametrize("workspace_access", ["read_write", "read_only"])
async def test_podman_run_sandbox_contract_live(
    adapter: str, workspace_access: str, tmp_path: Path
) -> None:
    if os.environ.get("GABBY_RUN_PODMAN_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_PODMAN_INTEGRATION=1 to use the live Podman runtime")
    if os.name == "posix":
        # The default sandbox UID is intentionally unprivileged.
        tmp_path.chmod(0o777)
    socket_path = tmp_path / "podman-api.sock"
    api_process: asyncio.subprocess.Process | None = None
    if adapter == "api":
        api_process = await _start_podman_api(socket_path)

    config = SandboxDefinition(
        engine="podman",
        image="python:3.14-slim",
        keepalive_argv=("python", "-c", "import time; time.sleep(3600)"),
        adapter=adapter,
        workspace=WorkspaceMount(tmp_path, workspace_access, "/workspace"),
        api_unix_socket=socket_path if adapter == "api" else None,
    )
    deadline = asyncio.get_running_loop().time() + 120
    container_id = ""

    try:
        async with open_sandbox(config, deadline=deadline, enable_tool_input=True) as session:
            container_id = session.container_id
            inspect = await _run_process(
                ["podman", "inspect", "--format", "{{json .}}", container_id],
                timeout=10,
            )
            assert inspect.exit_code == 0
            inspection: dict[str, Any] = json.loads(inspect.stdout)
            host_config: dict[str, Any] = inspection["HostConfig"]
            assert host_config["NetworkMode"] == "none"
            assert host_config["Memory"] == 2 * 1024**3
            assert host_config["NanoCpus"] == 2_000_000_000
            assert host_config["PidsLimit"] == 256
            assert host_config["ReadonlyRootfs"] is True
            assert inspection["Config"]["User"] == "65532:65532"
            workspace_mount = next(
                mount for mount in inspection["Mounts"] if mount["Destination"] == "/workspace"
            )
            assert workspace_mount["RW"] is (workspace_access == "read_write")
            tool_input_mount = next(
                mount
                for mount in inspection["Mounts"]
                if mount["Destination"] == "/tmp/gabby-tool-input"
            )
            assert tool_input_mount["RW"] is False

            command = await session.exec(["python", "-c", "print('sandbox-ready')"], timeout=10)
            assert command.exit_code == 0
            assert command.stdout.strip() == b"sandbox-ready"
            custom_tool = await session.invoke_tool(
                (
                    "python",
                    "-c",
                    "import json,sys; "
                    "data=json.load(open(sys.argv[1], encoding='utf-8')); "
                    "print(json.dumps({'length': len(data['text'])}))",
                ),
                {"text": "Invoice overdue"},
                max_input_bytes=1024,
                timeout=10,
            )
            assert custom_tool == {"length": len("Invoice overdue")}
            uid = await session.exec(["python", "-c", "import os; print(os.getuid())"], timeout=10)
            assert uid.exit_code == 0
            assert uid.stdout.strip() == b"65532"

            network = await session.exec(
                [
                    "python",
                    "-c",
                    "import socket; "
                    "s=socket.socket(); s.settimeout(1); "
                    "\ntry: s.connect(('1.1.1.1', 443))\n"
                    "except OSError: print('network-disabled')\n"
                    "else: raise SystemExit('unexpected network access')\n"
                    "finally: s.close()",
                ],
                timeout=5,
            )
            assert network.exit_code == 0
            assert network.stdout.strip() == b"network-disabled"

            if workspace_access == "read_write":
                await session.invoke(
                    "filesystem.write_file", {"path": "acceptance.txt", "content": "mounted"}
                )
                read = await session.invoke("filesystem.read_file", {"path": "acceptance.txt"})
                assert read == {"path": "acceptance.txt", "content": "mounted"}
            else:
                with pytest.raises(SandboxError, match="read-only"):
                    await session.invoke(
                        "filesystem.write_file",
                        {"path": "acceptance.txt", "content": "blocked"},
                    )

        if workspace_access == "read_write":
            assert (tmp_path / "acceptance.txt").read_text(encoding="utf-8") == "mounted"
        else:
            assert not (tmp_path / "acceptance.txt").exists()
        removed = await _run_process(["podman", "inspect", container_id], timeout=10)
        assert removed.exit_code != 0
    finally:
        await _stop_service(api_process)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["cli", "api"])
async def test_podman_sandbox_is_removed_after_cancellation_live(
    adapter: str, tmp_path: Path
) -> None:
    if os.environ.get("GABBY_RUN_PODMAN_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_PODMAN_INTEGRATION=1 to use the live Podman runtime")
    socket_path = tmp_path / "podman-api.sock"
    api_process: asyncio.subprocess.Process | None = None
    if adapter == "api":
        api_process = await _start_podman_api(socket_path)

    config = SandboxDefinition(
        engine="podman",
        image="python:3.14-slim",
        keepalive_argv=("python", "-c", "import time; time.sleep(3600)"),
        adapter=adapter,
        api_unix_socket=socket_path if adapter == "api" else None,
    )
    started = asyncio.Event()
    container_id: str | None = None

    async def run_until_cancelled() -> None:
        nonlocal container_id
        async with open_sandbox(
            config, deadline=asyncio.get_running_loop().time() + 120
        ) as session:
            container_id = session.container_id
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run_until_cancelled())
    try:
        await asyncio.wait_for(started.wait(), timeout=120)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert container_id is not None
        removed = await _run_process(["podman", "inspect", container_id], timeout=10)
        assert removed.exit_code != 0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await _stop_service(api_process)
