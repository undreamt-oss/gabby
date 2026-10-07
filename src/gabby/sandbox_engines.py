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
"""Docker and Podman CLI and Docker-compatible API sandbox adapters."""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any
from urllib.parse import quote

import httpx

from .config import ConfigError, SandboxDefinition, validate_sandbox_api_endpoint
from .sandbox import (
    ContainerSpec,
    SandboxCommandResult,
    SandboxError,
    SandboxOutputLimit,
    SandboxResourceUsage,
    SandboxTimeout,
    SandboxUnavailable,
)

_COMMAND_OUTPUT_LIMIT = 1024 * 1024
_ARCHIVE_OUTPUT_LIMIT = 64 * 1024 * 1024
_CONTROL_RESPONSE_LIMIT = 1024 * 1024
_API_VERSION = "v1.40"
_MIN_WINDOWS_NETWORK_ISOLATION_VERSION = (29, 1, 4)
_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:([-+])([0-9A-Za-z.-]+))?$")
_CPU_PERCENT_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)%$")
_MEMORY_SIZE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(B|kB|MB|GB|TB|KiB|MiB|GiB|TiB)$")
_MEMORY_UNITS = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _cli_memory_bytes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = _MEMORY_SIZE_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    size = float(match.group(1)) * _MEMORY_UNITS[match.group(2)]
    return int(size) if math.isfinite(size) and size >= 0 else None


def _cli_resource_usage(payload: object) -> SandboxResourceUsage | None:
    if not isinstance(payload, dict):
        return None
    cpu_percent: float | None = None
    cpu_value = payload.get("CPUPerc")
    if isinstance(cpu_value, str):
        match = _CPU_PERCENT_PATTERN.fullmatch(cpu_value.strip())
        if match is not None:
            parsed = float(match.group(1))
            if math.isfinite(parsed) and parsed >= 0:
                cpu_percent = parsed
    memory_value = payload.get("MemUsage")
    memory_current: int | None = None
    if isinstance(memory_value, str):
        memory_current = _cli_memory_bytes(memory_value.partition("/")[0].strip())
    pids_value = payload.get("PIDs")
    process_count: int | None = None
    if isinstance(pids_value, str) and pids_value.isdecimal():
        process_count = int(pids_value)
    usage = SandboxResourceUsage(
        cpu_percent=cpu_percent,
        memory_current_bytes=memory_current,
        process_count=process_count,
    )
    return usage if usage.as_dict() else None


def _api_resource_usage(payload: object) -> SandboxResourceUsage | None:
    if not isinstance(payload, dict):
        return None
    cpu_stats = payload.get("cpu_stats")
    cpu_usage = cpu_stats.get("cpu_usage") if isinstance(cpu_stats, dict) else None
    cpu_time = (
        _nonnegative_int(cpu_usage.get("total_usage")) if isinstance(cpu_usage, dict) else None
    )
    memory_stats = payload.get("memory_stats")
    memory_current = memory_peak = None
    if isinstance(memory_stats, dict):
        memory_current = _nonnegative_int(memory_stats.get("usage"))
        memory_peak = _nonnegative_int(memory_stats.get("max_usage"))
    pids_stats = payload.get("pids_stats")
    process_count = (
        _nonnegative_int(pids_stats.get("current")) if isinstance(pids_stats, dict) else None
    )
    usage = SandboxResourceUsage(
        cpu_time_ns=cpu_time,
        memory_current_bytes=memory_current,
        memory_peak_bytes=memory_peak,
        process_count=process_count,
    )
    return usage if usage.as_dict() else None


class _NamedPipeResponseStream(httpx.AsyncByteStream):
    """Adapt an aiohttp named-pipe response to HTTPX's streaming contract."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.content.iter_chunked(64 * 1024):
                yield chunk
        except TimeoutError as exc:
            raise httpx.ReadTimeout("Container API response timed out") from exc
        except Exception as exc:
            # aiohttp is optional and its exception types must not leak through
            # the HTTPX-based adapter contract.
            try:
                import aiohttp
            except ImportError:
                raise httpx.ReadError("Could not read the container API response") from exc
            if isinstance(exc, aiohttp.ClientError):
                raise httpx.ReadError("Could not read the container API response") from exc
            raise

    async def aclose(self) -> None:
        self._response.release()


class _NamedPipeHTTPTransport(httpx.AsyncBaseTransport):
    """HTTPX transport backed by aiohttp's Windows named-pipe connector."""

    _aiohttp: Any
    _session: Any

    def __init__(self, pipe_path: str) -> None:
        if sys.platform != "win32":
            raise SandboxUnavailable("Windows named-pipe API endpoints require a Windows host")
        try:
            import aiohttp
        except ImportError as exc:
            raise SandboxUnavailable(
                "Windows named-pipe API support requires the optional 'windows-sandbox' extra"
            ) from exc
        try:
            connector = aiohttp.NamedPipeConnector(path=pipe_path)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=None),
                auto_decompress=False,
            )
            self._aiohttp = aiohttp
        except (RuntimeError, OSError, ValueError) as exc:
            raise SandboxUnavailable(
                "Could not initialize the configured Windows container API named pipe"
            ) from exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timeout_values = request.extensions.get("timeout", {})
        timeout = self._aiohttp.ClientTimeout(
            total=timeout_values.get("read"),
            connect=timeout_values.get("connect"),
            sock_connect=timeout_values.get("connect"),
            sock_read=timeout_values.get("read"),
        )

        request_stream = request.stream
        if not isinstance(request_stream, httpx.AsyncByteStream):
            raise httpx.RequestError(
                "Container API request body is not asynchronous", request=request
            )

        async def request_body() -> AsyncIterator[bytes]:
            async for chunk in request_stream:
                yield chunk

        body = request_body() if request.method in {"POST", "PUT", "PATCH"} else None
        try:
            response = await self._session.request(
                request.method,
                str(request.url),
                headers=dict(request.headers),
                data=body,
                timeout=timeout,
                allow_redirects=False,
            )
        except TimeoutError as exc:
            raise httpx.ConnectTimeout("Container API request timed out", request=request) from exc
        except self._aiohttp.ClientError as exc:
            raise httpx.ConnectError(
                "Could not reach the configured container API", request=request
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise httpx.ConnectError(
                "Could not reach the configured container API", request=request
            ) from exc

        headers = [
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in response.raw_headers
        ]
        return httpx.Response(
            status_code=response.status,
            headers=headers,
            stream=_NamedPipeResponseStream(response),
            request=request,
            extensions={"reason_phrase": response.reason.encode("ascii", errors="replace")},
        )

    async def aclose(self) -> None:
        await self._session.close()


def _validate_windows_docker_version(value: object) -> None:
    if not isinstance(value, str):
        raise SandboxUnavailable("Could not determine the Docker server version")
    match = _VERSION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise SandboxUnavailable("Docker returned an invalid server version")
    version = tuple(int(part) for part in match.groups()[:3])
    prerelease_at_minimum = (
        version == _MIN_WINDOWS_NETWORK_ISOLATION_VERSION and match.group(4) == "-"
    )
    if version < _MIN_WINDOWS_NETWORK_ISOLATION_VERSION or prerelease_at_minimum:
        minimum = ".".join(map(str, _MIN_WINDOWS_NETWORK_ISOLATION_VERSION))
        raise SandboxUnavailable(
            "Native Windows containers with disabled networking require Docker "
            f"Engine {minimum} or newer"
        )


async def _read_error_preview(response: httpx.Response, *, limit: int = 500) -> str:
    """Read a bounded error preview without buffering the remainder of the response."""
    output = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=limit):
        remaining = limit - len(output)
        output.extend(chunk[:remaining])
        if len(chunk) > remaining or len(output) >= limit:
            break
    return output.decode("utf-8", errors="replace")


class _OutputBudget:
    """Share one retained-byte budget between a command's stdout and stderr readers."""

    def __init__(self, limit: int) -> None:
        self.remaining = limit
        self.exceeded = asyncio.Event()
        self._lock = asyncio.Lock()

    async def retain(self, chunk: bytes) -> tuple[bytes, bool]:
        """Retain only bytes within the shared limit and signal the first overflow."""
        async with self._lock:
            retained_size = min(self.remaining, len(chunk))
            self.remaining -= retained_size
            did_exceed = retained_size < len(chunk)
            if did_exceed:
                self.exceeded.set()
            return chunk[:retained_size], did_exceed


async def _drain(stream: asyncio.StreamReader, budget: _OutputBudget) -> tuple[bytes, bool]:
    output = bytearray()
    while chunk := await stream.read(64 * 1024):
        retained, exceeded = await budget.retain(chunk)
        output.extend(retained)
        if exceeded:
            return bytes(output), True
    return bytes(output), False


async def _run_process(
    argv: Sequence[str],
    *,
    timeout: float,
    input_data: bytes | None = None,
    output_limit: int = _COMMAND_OUTPUT_LIMIT,
) -> SandboxCommandResult:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def remaining() -> float:
        value = deadline - loop.time()
        if value <= 0:
            raise TimeoutError
        return value

    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *argv,
                stdin=(
                    asyncio.subprocess.PIPE
                    if input_data is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=remaining(),
        )
    except TimeoutError as exc:
        raise SandboxTimeout("Container engine command exceeded its deadline") from exc
    except FileNotFoundError as exc:
        raise SandboxUnavailable(
            f"Container engine executable is not available: {argv[0]}"
        ) from exc
    except OSError as exc:
        raise SandboxUnavailable(f"Could not start container engine executable: {argv[0]}") from exc

    assert process.stdout is not None and process.stderr is not None
    output_budget = _OutputBudget(output_limit)
    stdout_task = asyncio.create_task(_drain(process.stdout, output_budget))
    stderr_task = asyncio.create_task(_drain(process.stderr, output_budget))
    process_wait_task: asyncio.Task[int] | None = None
    output_wait_task: asyncio.Task[bool] | None = None

    async def cleanup() -> None:
        nonlocal process_wait_task, output_wait_task
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        if process_wait_task is not None and not process_wait_task.done():
            with suppress(TimeoutError):
                await asyncio.wait_for(process_wait_task, timeout=1.0)
        elif process_wait_task is None and process.returncode is None:
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=1.0)
        if output_wait_task is not None and not output_wait_task.done():
            output_wait_task.cancel()
            await asyncio.gather(output_wait_task, return_exceptions=True)
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    try:
        if input_data is not None:
            assert process.stdin is not None
            process.stdin.write(input_data)
            await asyncio.wait_for(process.stdin.drain(), timeout=remaining())
            process.stdin.close()
            await asyncio.wait_for(process.stdin.wait_closed(), timeout=remaining())
        process_wait_task = asyncio.create_task(process.wait())
        output_wait_task = asyncio.create_task(output_budget.exceeded.wait())
        completed, _ = await asyncio.wait(
            {process_wait_task, output_wait_task},
            timeout=remaining(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not completed:
            raise TimeoutError
        if output_budget.exceeded.is_set():
            raise SandboxOutputLimit(f"Container engine output exceeded {output_limit} bytes")
        await process_wait_task
        output_wait_task.cancel()
        await asyncio.gather(output_wait_task, return_exceptions=True)
        stdout_result, stderr_result = await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task), timeout=remaining()
        )
    except TimeoutError as exc:
        await cleanup()
        raise SandboxTimeout("Container engine command exceeded its deadline") from exc
    except BaseException:
        await cleanup()
        raise
    stdout, stdout_exceeded = stdout_result
    stderr, stderr_exceeded = stderr_result
    if stdout_exceeded or stderr_exceeded or output_budget.exceeded.is_set():
        raise SandboxOutputLimit(f"Container engine output exceeded {output_limit} bytes")
    return SandboxCommandResult(process.returncode or 0, stdout, stderr)


class ContainerCLIAdapter:
    """CLI integration shared by the Docker and Podman command-line clients."""

    def __init__(self, engine: str, executable: str | None = None) -> None:
        if engine not in ("docker", "podman"):
            raise ValueError("engine must be 'docker' or 'podman'")
        self.engine = engine
        self.executable = executable or engine

    async def _run(
        self,
        *args: str,
        timeout: float,
        input_data: bytes | None = None,
        output_limit: int = _COMMAND_OUTPUT_LIMIT,
    ) -> SandboxCommandResult:
        return await _run_process(
            (self.executable, *args),
            timeout=timeout,
            input_data=input_data,
            output_limit=output_limit,
        )

    async def _checked(
        self, operation: str, *args: str, timeout: float, **kwargs: Any
    ) -> SandboxCommandResult:
        result = await self._run(*args, timeout=timeout, **kwargs)
        if result.exit_code != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise SandboxUnavailable(
                f"{self.engine} {operation} failed (exit {result.exit_code}): {message[:500]}"
            )
        return result

    async def daemon_os(self, *, timeout: float) -> str:
        if self.engine == "docker":
            result = await self._checked("info", "info", "--format", "{{.OSType}}", timeout=timeout)
        else:
            result = await self._checked("info", "info", "--format", "json", timeout=timeout)
            try:
                info = json.loads(result.stdout)
                value = info["host"]["os"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SandboxUnavailable("podman returned invalid engine information") from exc
        if self.engine == "docker":
            value = result.stdout.decode("utf-8", errors="replace").strip()
        if not isinstance(value, str) or value not in ("linux", "windows"):
            raise SandboxUnavailable(f"{self.engine} returned an unsupported host OS")
        return value

    async def image_os(self, image: str, *, timeout: float) -> str | None:
        result = await self._run("image", "inspect", "--format", "{{.Os}}", image, timeout=timeout)
        if result.exit_code != 0:
            return None
        value = result.stdout.decode("utf-8", errors="replace").strip()
        return value or None

    async def validate_windows_sandbox_support(self, *, timeout: float) -> None:
        if self.engine != "docker":
            raise SandboxUnavailable("Native Windows containers require the Docker engine")
        result = await self._checked(
            "version", "version", "--format", "{{.Server.Version}}", timeout=timeout
        )
        _validate_windows_docker_version(result.stdout.decode("utf-8", errors="replace"))

    async def pull_image(self, image: str, *, timeout: float) -> None:
        await self._checked("image pull", "image", "pull", image, timeout=timeout)

    async def start_container(self, spec: ContainerSpec, *, timeout: float) -> str:
        name = f"gabby-{spec.run_id}"
        args = [
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            f"io.gabby.run={spec.run_id}",
            "--network",
            "none",
            "--memory",
            str(spec.memory_bytes),
        ]
        if spec.image_os == "windows":
            args.extend(["--isolation", "hyperv", "--cpu-count", str(int(spec.cpus))])
        else:
            args.extend(
                [
                    "--cpus",
                    str(spec.cpus),
                    "--pids-limit",
                    str(spec.process_limit),
                    "--user",
                    spec.user,
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges=true",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                    "--init",
                ]
            )
        if spec.tool_input_host_path is not None and spec.tool_input_container_path is not None:
            if "," in spec.tool_input_host_path or "," in spec.tool_input_container_path:
                raise SandboxError("Tool input mount paths containing commas are not supported")
            args.extend(
                [
                    "--mount",
                    "type=bind,source="
                    + spec.tool_input_host_path
                    + ",target="
                    + spec.tool_input_container_path
                    + ",readonly",
                ]
            )
        if spec.workspace_host_path is not None and spec.workspace_container_path is not None:
            if "," in spec.workspace_host_path or "," in spec.workspace_container_path:
                raise SandboxError("Workspace paths containing commas are not supported")
            mount = (
                "type=bind,source="
                + spec.workspace_host_path
                + ",target="
                + spec.workspace_container_path
                + (",readonly" if spec.workspace_read_only else "")
            )
            args.extend(["--mount", mount])
        args.extend(["--pull=never", spec.image, *spec.keepalive_argv])
        try:
            result = await self._checked("container start", *args, timeout=timeout)
            container_id = result.stdout.decode("utf-8", errors="replace").strip()
            if not container_id:
                raise SandboxUnavailable(f"{self.engine} returned no container ID")
            return container_id
        except BaseException as start_error:
            # The daemon may have created the detached container even when the
            # CLI timed out before returning its ID. The run-specific name is
            # known before creation, so use it for best-effort cleanup.
            try:
                await self.remove_container(name, timeout=min(timeout, 10.0))
            except Exception as cleanup_error:
                start_error.add_note(
                    f"Could not remove the container after startup failed: {cleanup_error}"
                )
            raise

    async def exec(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        timeout: float,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        args = ["exec"]
        if working_directory is not None:
            args.extend(["--workdir", working_directory])
        for key, value in (environment or {}).items():
            if not key or "=" in key or "\x00" in key or "\x00" in value:
                raise SandboxError("Sandbox environment variables contain an invalid value")
            args.extend(["--env", f"{key}={value}"])
        args.extend([container_id, *argv])
        return await self._run(*args, timeout=timeout)

    async def read_archive(self, container_id: str, path: str, *, timeout: float) -> bytes:
        result = await self._checked(
            "archive read",
            "cp",
            f"{container_id}:{path}",
            "-",
            timeout=timeout,
            output_limit=_ARCHIVE_OUTPUT_LIMIT,
        )
        return result.stdout

    async def write_archive(
        self, container_id: str, path: str, archive: bytes, *, timeout: float
    ) -> None:
        await self._checked(
            "archive write",
            "cp",
            "-",
            f"{container_id}:{path}",
            timeout=timeout,
            input_data=archive,
        )

    async def remove_container(self, container_id: str, *, timeout: float) -> None:
        result = await self._run("rm", "--force", container_id, timeout=timeout)
        if result.exit_code != 0:
            detail = result.stderr.decode("utf-8", errors="replace").casefold()
            if "no such container" not in detail and "not found" not in detail:
                raise SandboxUnavailable(f"{self.engine} could not remove the run container")

    async def resource_usage(
        self, container_id: str, *, timeout: float
    ) -> SandboxResourceUsage | None:
        result = await self._checked(
            "stats",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            container_id,
            timeout=timeout,
            output_limit=16 * 1024,
        )
        try:
            return _cli_resource_usage(json.loads(result.stdout))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SandboxUnavailable(
                f"{self.engine} returned invalid container statistics"
            ) from None

    async def aclose(self) -> None:
        return None


class ContainerAPIAdapter:
    """Docker-compatible Engine API client for Docker and Podman."""

    def __init__(
        self, config: SandboxDefinition, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.engine = config.engine
        self._client: httpx.AsyncClient
        self._owns_client = client is None
        if config.api_base_url is not None:
            try:
                validate_sandbox_api_endpoint(config.api_base_url)
            except ConfigError as exc:
                raise SandboxError(str(exc)) from exc
        if client is not None:
            self._client = client
        elif config.api_unix_socket is not None:
            transport: httpx.AsyncBaseTransport = httpx.AsyncHTTPTransport(
                uds=str(config.api_unix_socket)
            )
            self._client = httpx.AsyncClient(
                base_url="http://container-engine", transport=transport
            )
        elif config.api_named_pipe is not None:
            transport = _NamedPipeHTTPTransport(config.api_named_pipe)
            self._client = httpx.AsyncClient(
                base_url="http://container-engine", transport=transport
            )
        elif config.api_base_url is not None:
            self._client = httpx.AsyncClient(base_url=config.api_base_url)
        else:
            raise SandboxError("Container API adapter requires an API endpoint")

    def _path(self, path: str) -> str:
        if path == "/version":
            return path
        return f"/{_API_VERSION}{path}"

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise SandboxTimeout("Container API operation exceeded its deadline")
        return remaining

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        expected: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> httpx.Response:
        body = bytearray()
        try:
            async with self._client.stream(
                method, self._path(path), timeout=timeout, **kwargs
            ) as response:
                status_code = response.status_code
                request = response.request
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _CONTROL_RESPONSE_LIMIT:
                        raise SandboxOutputLimit(
                            "Container API control response exceeded "
                            f"{_CONTROL_RESPONSE_LIMIT} bytes"
                        )
                    body.extend(chunk)
        except httpx.TimeoutException as exc:
            raise SandboxTimeout("Container API request exceeded its deadline") from exc
        except httpx.HTTPError as exc:
            raise SandboxUnavailable("Could not reach the configured container API") from exc
        response = httpx.Response(status_code, content=bytes(body), request=request)
        if status_code not in expected:
            detail = response.text[:500]
            raise SandboxUnavailable(
                f"Container API {method} {path} returned HTTP {status_code}: {detail}"
            )
        return response

    async def _stream_request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        limit: int,
        expected: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> bytes:
        output = bytearray()
        try:
            async with self._client.stream(
                method, self._path(path), timeout=timeout, **kwargs
            ) as response:
                if response.status_code not in expected:
                    detail = await _read_error_preview(response)
                    raise SandboxUnavailable(
                        f"Container API {method} {path} returned HTTP "
                        f"{response.status_code}: {detail}"
                    )
                async for chunk in response.aiter_bytes():
                    if len(output) + len(chunk) > limit:
                        raise SandboxOutputLimit(f"Container API response exceeded {limit} bytes")
                    output.extend(chunk)
        except httpx.TimeoutException as exc:
            raise SandboxTimeout("Container API request exceeded its deadline") from exc
        except httpx.HTTPError as exc:
            raise SandboxUnavailable(
                "Could not read the configured container API response"
            ) from exc
        return bytes(output)

    async def daemon_os(self, *, timeout: float) -> str:
        response = await self._request("GET", "/info", timeout=timeout)
        try:
            info = response.json()
            value = info.get("OSType")
            if value is None and self.engine == "podman":
                value = info.get("host", {}).get("os")
        except (ValueError, AttributeError) as exc:
            raise SandboxUnavailable("Container API returned invalid engine information") from exc
        if not isinstance(value, str) or value.lower() not in ("linux", "windows"):
            raise SandboxUnavailable("Container API returned an unsupported host OS")
        return value.lower()

    async def validate_windows_sandbox_support(self, *, timeout: float) -> None:
        if self.engine != "docker":
            raise SandboxUnavailable("Native Windows containers require the Docker engine")
        response = await self._request("GET", "/version", timeout=timeout)
        try:
            version = response.json().get("Version")
        except (ValueError, AttributeError) as exc:
            raise SandboxUnavailable(
                "Docker API returned invalid server version information"
            ) from exc
        _validate_windows_docker_version(version)

    async def image_os(self, image: str, *, timeout: float) -> str | None:
        path = "/images/" + quote(image, safe="") + "/json"
        response = await self._request("GET", path, timeout=timeout, expected=(200, 404))
        if response.status_code == 404:
            return None
        try:
            value = response.json().get("Os")
        except (ValueError, AttributeError) as exc:
            raise SandboxUnavailable("Container API returned invalid image information") from exc
        return value if isinstance(value, str) else None

    async def pull_image(self, image: str, *, timeout: float) -> None:
        # Drain bounded progress output so large layer pulls cannot grow memory use.
        path = self._path("/images/create")
        try:
            async with self._client.stream(
                "POST", path, params={"fromImage": image}, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    body = await _read_error_preview(response)
                    raise SandboxUnavailable(
                        f"Container API image pull returned HTTP {response.status_code}: {body}"
                    )
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _ARCHIVE_OUTPUT_LIMIT:
                        raise SandboxOutputLimit("Container image pull output exceeded its limit")
        except httpx.TimeoutException as exc:
            raise SandboxTimeout("Container image pull exceeded its deadline") from exc
        except httpx.HTTPError as exc:
            raise SandboxUnavailable("Could not pull the configured container image") from exc

    async def start_container(self, spec: ContainerSpec, *, timeout: float) -> str:
        deadline = asyncio.get_running_loop().time() + timeout
        name = f"gabby-{spec.run_id}"
        host_config: dict[str, Any] = {
            "NetworkMode": "none",
            "Memory": spec.memory_bytes,
            "RestartPolicy": {"Name": "no"},
        }
        container_config: dict[str, Any] = {}
        if spec.image_os == "linux":
            container_config["User"] = spec.user
            host_config.update(
                {
                    "ReadonlyRootfs": True,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                    "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
                }
            )
        else:
            host_config["Isolation"] = "hyperv"
            host_config["CpuCount"] = int(spec.cpus)
        if spec.image_os == "linux":
            host_config["NanoCpus"] = int(spec.cpus * 1_000_000_000)
            host_config["PidsLimit"] = spec.process_limit
        mounts: list[dict[str, Any]] = []
        if spec.tool_input_host_path is not None and spec.tool_input_container_path is not None:
            mounts.append(
                {
                    "Type": "bind",
                    "Source": spec.tool_input_host_path,
                    "Target": spec.tool_input_container_path,
                    "ReadOnly": True,
                }
            )
        if spec.workspace_host_path is not None and spec.workspace_container_path is not None:
            mounts.append(
                {
                    "Type": "bind",
                    "Source": spec.workspace_host_path,
                    "Target": spec.workspace_container_path,
                    "ReadOnly": spec.workspace_read_only,
                }
            )
        if mounts:
            host_config["Mounts"] = mounts
        container_id: str | None = None
        try:
            response = await self._request(
                "POST",
                "/containers/create",
                timeout=self._remaining(deadline),
                expected=(201,),
                params={"name": name},
                json={
                    "Image": spec.image,
                    "Cmd": list(spec.keepalive_argv),
                    **container_config,
                    "Labels": {"io.gabby.run": spec.run_id},
                    "HostConfig": host_config,
                },
            )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SandboxUnavailable("Container API returned invalid create response") from exc
            if not isinstance(payload, dict):
                raise SandboxUnavailable("Container API returned invalid create response")
            container_id = payload.get("Id")
            if not isinstance(container_id, str) or not container_id:
                raise SandboxUnavailable("Container API did not return a container ID")
            await self._request(
                "POST",
                f"/containers/{quote(container_id, safe='')}/start",
                timeout=self._remaining(deadline),
                expected=(204,),
            )
        except BaseException as start_error:
            try:
                # A timed-out create request can succeed at the daemon while
                # its response is lost. Clean up by the unique name in that
                # case; once known, prefer the returned container ID.
                await self.remove_container(container_id or name, timeout=min(timeout, 10.0))
            except Exception as cleanup_error:
                start_error.add_note(
                    f"Could not remove the container after startup failed: {cleanup_error}"
                )
            raise
        return container_id

    async def exec(
        self,
        container_id: str,
        argv: Sequence[str],
        *,
        timeout: float,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        deadline = asyncio.get_running_loop().time() + timeout
        create_body: dict[str, Any] = {
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "Cmd": list(argv),
            "Env": [f"{key}={value}" for key, value in (environment or {}).items()],
        }
        if working_directory is not None:
            create_body["WorkingDir"] = working_directory
        response = await self._request(
            "POST",
            f"/containers/{container_id}/exec",
            timeout=self._remaining(deadline),
            expected=(201,),
            json=create_body,
        )
        try:
            exec_id = response.json().get("Id")
        except (ValueError, AttributeError) as exc:
            raise SandboxUnavailable("Container API returned invalid exec response") from exc
        if not isinstance(exec_id, str) or not exec_id:
            raise SandboxUnavailable("Container API did not return an exec ID")
        started = await self._stream_request(
            "POST",
            f"/exec/{exec_id}/start",
            timeout=self._remaining(deadline),
            limit=_COMMAND_OUTPUT_LIMIT,
            json={"Detach": False, "Tty": False},
        )
        stdout, stderr = _decode_engine_stream(started)
        inspect = await self._request(
            "GET", f"/exec/{exec_id}/json", timeout=self._remaining(deadline)
        )
        try:
            exit_code = inspect.json().get("ExitCode")
        except (ValueError, AttributeError) as exc:
            raise SandboxUnavailable("Container API returned invalid exec status") from exc
        if not isinstance(exit_code, int):
            raise SandboxUnavailable("Container API did not return the command exit code")
        return SandboxCommandResult(exit_code, stdout, stderr)

    async def read_archive(self, container_id: str, path: str, *, timeout: float) -> bytes:
        response = await self._stream_request(
            "GET",
            f"/containers/{container_id}/archive",
            timeout=timeout,
            limit=_ARCHIVE_OUTPUT_LIMIT,
            params={"path": path},
        )
        return response

    async def write_archive(
        self, container_id: str, path: str, archive: bytes, *, timeout: float
    ) -> None:
        if len(archive) > _ARCHIVE_OUTPUT_LIMIT:
            raise SandboxOutputLimit("Sandbox archive exceeded its size limit")
        await self._request(
            "PUT",
            f"/containers/{container_id}/archive",
            timeout=timeout,
            params={"path": path},
            content=archive,
            headers={"Content-Type": "application/x-tar"},
        )

    async def remove_container(self, container_id: str, *, timeout: float) -> None:
        if self.engine == "podman":
            await self._request(
                "POST",
                f"/containers/{container_id}/stop",
                timeout=timeout,
                expected=(204, 304, 404),
                params={"t": "1"},
            )
        await self._request(
            "DELETE",
            f"/containers/{container_id}",
            timeout=timeout,
            expected=(204, 404),
            params={"force": "false" if self.engine == "podman" else "true", "v": "true"},
        )

    async def resource_usage(
        self, container_id: str, *, timeout: float
    ) -> SandboxResourceUsage | None:
        response = await self._request(
            "GET",
            f"/containers/{quote(container_id, safe='')}/stats",
            timeout=timeout,
            params={"stream": "false"},
        )
        try:
            return _api_resource_usage(response.json())
        except ValueError:
            raise SandboxUnavailable(
                "Container API returned invalid container statistics"
            ) from None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _decode_engine_stream(value: bytes) -> tuple[bytes, bytes]:
    """Decode the multiplexed stdout/stderr format used by Docker-compatible exec."""
    if not value:
        return b"", b""
    stdout = bytearray()
    stderr = bytearray()
    offset = 0
    while offset < len(value):
        if len(value) - offset < 8:
            raise SandboxError("Container API returned a truncated exec stream")
        stream_type = value[offset]
        size = int.from_bytes(value[offset + 4 : offset + 8], "big")
        offset += 8
        end = offset + size
        if end > len(value):
            raise SandboxError("Container API returned a truncated exec frame")
        if stream_type == 1:
            stdout.extend(value[offset:end])
        elif stream_type == 2:
            stderr.extend(value[offset:end])
        else:
            raise SandboxError("Container API returned an unknown exec stream type")
        offset = end
    return bytes(stdout), bytes(stderr)


def cli_adapter(engine: str) -> ContainerCLIAdapter:
    """Create the built-in CLI adapter for a configured engine."""
    return ContainerCLIAdapter(engine)


def api_adapter(config: SandboxDefinition) -> ContainerAPIAdapter:
    """Create the built-in Docker-compatible API adapter."""
    return ContainerAPIAdapter(config)
