# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in acceptance checks against a real local Docker daemon."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from gabby.config import SandboxDefinition, WorkspaceMount
from gabby.sandbox import SandboxError, open_sandbox
from gabby.sandbox_engines import _run_process


def _docker_api_socket() -> Path:
    """Resolve a local Docker API socket for Linux and Docker Desktop hosts."""
    configured = os.environ.get("GABBY_DOCKER_API_SOCKET")
    if configured:
        return Path(configured).expanduser().resolve()

    docker_host = os.environ.get("DOCKER_HOST", "")
    if docker_host.startswith("unix://"):
        return Path(docker_host.removeprefix("unix://")).expanduser().resolve()

    candidates = (Path("/var/run/docker.sock"), Path.home() / ".docker/run/docker.sock")
    return next((path for path in candidates if path.exists()), candidates[0])


def test_docker_api_socket_uses_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    socket_path = tmp_path / "docker.sock"
    monkeypatch.setenv("GABBY_DOCKER_API_SOCKET", str(socket_path))
    monkeypatch.setenv("DOCKER_HOST", "unix:///ignored/docker.sock")

    assert _docker_api_socket() == socket_path.resolve()


def test_docker_api_socket_uses_unix_docker_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GABBY_DOCKER_API_SOCKET", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")

    assert _docker_api_socket() == Path("/run/user/1000/docker.sock")


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["cli", "api"])
@pytest.mark.parametrize("workspace_access", ["read_write", "read_only"])
async def test_docker_run_sandbox_contract_live(
    adapter: str, workspace_access: str, tmp_path: Path
) -> None:
    if os.environ.get("GABBY_RUN_DOCKER_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_DOCKER_INTEGRATION=1 to use the live Docker daemon")
    docker_socket = _docker_api_socket()
    if adapter == "api" and not docker_socket.exists():
        pytest.skip("Docker API unix socket is unavailable")
    if os.name == "posix":
        # The default sandbox UID is intentionally unprivileged.
        tmp_path.chmod(0o777)

    config = SandboxDefinition(
        engine="docker",
        image="python:3.14-slim",
        keepalive_argv=("python", "-c", "import time; time.sleep(3600)"),
        adapter=adapter,
        workspace=WorkspaceMount(tmp_path, workspace_access, "/workspace"),
        api_unix_socket=docker_socket if adapter == "api" else None,
    )
    deadline = asyncio.get_running_loop().time() + 120
    container_id = ""

    async with open_sandbox(config, deadline=deadline, enable_tool_input=True) as session:
        container_id = session.container_id
        inspect = await _run_process(
            ["docker", "inspect", "--format", "{{json .}}", container_id],
            timeout=10,
        )
        assert inspect.exit_code == 0
        inspection = json.loads(inspect.stdout)
        host_config = inspection["HostConfig"]
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
        python_result = await session.invoke(
            "python.run",
            {"code": "print(sum([3, 7, 8]))"},
            command=("python",),
            max_input_bytes=1024,
            timeout=10,
        )
        assert python_result == {"exit_code": 0, "stdout": "18\n", "stderr": ""}
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
async def test_docker_sandbox_is_removed_after_cancellation_live(adapter: str) -> None:
    if os.environ.get("GABBY_RUN_DOCKER_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_DOCKER_INTEGRATION=1 to use the live Docker daemon")
    docker_socket = _docker_api_socket()
    if adapter == "api" and not docker_socket.exists():
        pytest.skip("Docker API unix socket is unavailable")

    config = SandboxDefinition(
        engine="docker",
        image="python:3.14-slim",
        keepalive_argv=("python", "-c", "import time; time.sleep(3600)"),
        adapter=adapter,
        api_unix_socket=docker_socket if adapter == "api" else None,
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
        removed = await _run_process(["docker", "inspect", container_id], timeout=10)
        assert removed.exit_code != 0
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
