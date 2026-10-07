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
"""Unit checks for the Nginx ingress response security contract."""

from __future__ import annotations

from typing import cast
from urllib.request import OpenerDirector

import pytest

from scripts.check_nginx_ingress import _assert_security_headers

_EXPECTED_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Server": "nginx",
}


class _Response:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Opener:
    def __init__(self, headers: dict[str, str]) -> None:
        self.response = _Response(headers)

    def open(self, _url: str, *, timeout: float) -> _Response:
        assert timeout == 5
        return self.response


def _opener_with_headers(headers: dict[str, str]) -> OpenerDirector:
    return cast(OpenerDirector, _Opener(headers))


def test_nginx_ingress_checks_security_headers() -> None:
    _assert_security_headers(_opener_with_headers(_EXPECTED_HEADERS), "https://example.test/health")


@pytest.mark.parametrize(
    "missing_header",
    [
        "Strict-Transport-Security",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
    ],
)
def test_nginx_ingress_rejects_missing_security_headers(missing_header: str) -> None:
    headers = {key: value for key, value in _EXPECTED_HEADERS.items() if key != missing_header}
    with pytest.raises(RuntimeError, match="missing or incorrect"):
        _assert_security_headers(_opener_with_headers(headers), "https://example.test/health")


def test_nginx_ingress_rejects_server_version_disclosure() -> None:
    headers = {**_EXPECTED_HEADERS, "Server": "nginx/1.29.0"}
    with pytest.raises(RuntimeError, match="exposes its version"):
        _assert_security_headers(_opener_with_headers(headers), "https://example.test/health")
