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
"""Create a single-agent FastAPI service from deployment-owned configuration."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from gabby import Agent, BearerTokenAuthenticator
from gabby import create_app as create_gabby_app


def create_app() -> FastAPI:
    """Build the service from environment configuration without embedding secrets."""
    config_value = os.environ.get("GABBY_AGENT_CONFIG")
    if not config_value:
        raise RuntimeError("GABBY_AGENT_CONFIG must point to an agent YAML file")
    config_path = Path(config_value).expanduser()
    authenticator = BearerTokenAuthenticator.from_env("GABBY_API_TOKEN")
    agent = Agent.from_file(config_path)
    return create_gabby_app(
        agent,
        authenticator=authenticator,
        max_concurrent_runs=int(os.environ.get("GABBY_MAX_CONCURRENT_RUNS", "8")),
        max_request_bytes=int(os.environ.get("GABBY_MAX_REQUEST_BYTES", "1000000")),
        max_response_bytes=int(os.environ.get("GABBY_MAX_RESPONSE_BYTES", "4194304")),
    )
