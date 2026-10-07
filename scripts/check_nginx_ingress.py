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
"""Exercise the Nginx ingress reference against a running local Gabby service.

Requires Docker on Linux and a healthy Gabby service published at 127.0.0.1:8787.
"""

from __future__ import annotations

import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "deploy" / "nginx.conf"
NGINX_IMAGE = os.environ.get(
    "GABBY_NGINX_TEST_IMAGE",
    "nginx:alpine@sha256:df221db836e1754089190208cee7eeda94f233197056426eda74a43ab1abeac2",
)
HEALTH_URL = "http://127.0.0.1:8787/health"
RUN_PATH = "/v1/agents/research-synthesizer/run"
MAX_REQUEST_BYTES = 1_000_000
BURST_REQUESTS = 35


def main() -> int:
    """Validate syntax, request-size forwarding, and ingress rate enforcement."""
    if shutil.which("docker") is None:
        return _fail("Docker CLI is required")
    if not NGINX_IMAGE or len(NGINX_IMAGE) > 256:
        return _fail("GABBY_NGINX_TEST_IMAGE must be a bounded image reference")
    try:
        _assert_upstream_health()
        with tempfile.TemporaryDirectory(prefix="gabby-nginx-check-") as temporary:
            root = Path(temporary)
            cert_dir = root / "certs"
            cert_dir.mkdir(mode=0o700)
            _create_certificate(cert_dir)
            port = _available_loopback_port()
            rendered_config = _render_test_config(port)
            test_config = root / "nginx.conf"
            test_config.write_text(rendered_config, encoding="utf-8")
            container_id = _start_nginx(test_config, cert_dir)
            try:
                ingress_url = f"https://127.0.0.1:{port}"
                opener = _local_https_opener()
                _wait_for_ingress(opener, ingress_url, container_id)
                _assert_security_headers(opener, ingress_url + "/health")
                _assert_status(
                    opener,
                    ingress_url + RUN_PATH,
                    status=413,
                    body=b"x" * (MAX_REQUEST_BYTES + 1),
                )
                statuses = [
                    _request_status(opener, ingress_url + RUN_PATH, body=b'{"input":"smoke"}')
                    for _ in range(BURST_REQUESTS)
                ]
                if 429 not in statuses or 401 not in statuses:
                    raise RuntimeError(
                        "rate-limit smoke failed: expected both upstream 401 and edge 429; "
                        f"received {statuses}"
                    )
            finally:
                _stop_nginx(container_id)
        print(
            "Nginx ingress passed: health forwarded, oversized body rejected with 413, "
            "security headers are present, and burst traffic received 429 after upstream "
            "authentication responses."
        )
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError, urllib.error.URLError) as exc:
        return _fail(str(exc))


def _assert_upstream_health() -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(HEALTH_URL, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"Gabby health endpoint returned HTTP {response.status}")


def _create_certificate(cert_dir: Path) -> None:
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-nodes",
            "-days",
            "1",
            "-newkey",
            "rsa:2048",
            "-subj",
            "/CN=agents.example.com",
            "-keyout",
            str(cert_dir / "privkey.pem"),
            "-out",
            str(cert_dir / "fullchain.pem"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )


def _available_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _render_test_config(port: int) -> str:
    config = CONFIG.read_text(encoding="utf-8")
    production_listener = "listen 443 ssl;"
    if config.count(production_listener) != 1:
        raise RuntimeError("Nginx config must define exactly one TLS listener")
    return config.replace(
        production_listener,
        f"listen 127.0.0.1:{port} ssl;",
        1,
    )


def _start_nginx(test_config: Path, cert_dir: Path) -> str:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--detach",
            "--network",
            "host",
            "--mount",
            f"type=bind,src={test_config},dst=/etc/nginx/nginx.conf,readonly",
            "--mount",
            f"type=bind,src={cert_dir},dst=/etc/letsencrypt/live/agents.example.com,readonly",
            NGINX_IMAGE,
            "nginx",
            "-g",
            "daemon off;",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    container_id = result.stdout.strip()
    if len(container_id) < 12 or any(char not in "0123456789abcdef" for char in container_id):
        raise RuntimeError("Docker returned an invalid Nginx container ID")
    return container_id


def _local_https_opener() -> urllib.request.OpenerDirector:
    context = ssl._create_unverified_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )


def _wait_for_ingress(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    container_id: str,
) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with opener.open(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        state = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if state.returncode != 0 or state.stdout.strip() != "true":
            logs = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            raise RuntimeError(f"Nginx exited before becoming ready: {logs.stderr[-2000:]}")
        time.sleep(0.1)
    raise RuntimeError("Nginx did not become ready within 30 seconds")


def _assert_security_headers(opener: urllib.request.OpenerDirector, url: str) -> None:
    with opener.open(url, timeout=5) as response:
        expected = {
            "strict-transport-security": "max-age=31536000",
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
            "referrer-policy": "no-referrer",
        }
        actual = {name.lower(): value for name, value in response.headers.items()}
        for name, value in expected.items():
            if actual.get(name) != value:
                raise RuntimeError(f"Nginx security header {name!r} was missing or incorrect")
        server = actual.get("server", "")
        if "/" in server:
            raise RuntimeError("Nginx Server header exposes its version")


def _request_status(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    body: bytes,
) -> int:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as response:
        return int(response.code)


def _assert_status(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    status: int,
    body: bytes,
) -> None:
    actual = _request_status(opener, url, body=body)
    if actual != status:
        raise RuntimeError(f"Nginx ingress returned HTTP {actual}, expected {status}")


def _stop_nginx(container_id: str) -> None:
    subprocess.run(
        ["docker", "stop", "--time", "5", container_id],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _fail(message: str) -> int:
    print(f"Nginx ingress check failed: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
