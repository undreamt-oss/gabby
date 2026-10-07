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

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.13-python3.14-trixie-slim
ARG RUNTIME_IMAGE=python:3.14-slim-trixie
FROM ${UV_IMAGE} AS build

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra server --no-editable

COPY examples/hosted-agent.yaml ./agent.yaml
COPY examples/skills ./skills

FROM ${RUNTIME_IMAGE} AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 gabby \
    && useradd --system --uid 10001 --gid gabby --home-dir /nonexistent \
        --shell /usr/sbin/nologin gabby

WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
COPY --from=build --chown=10001:10001 /app/agent.yaml /app/agent.yaml
COPY --from=build --chown=10001:10001 /app/skills /app/skills

USER 10001:10001
EXPOSE 8787
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2).read()"]

ENTRYPOINT ["gabby", "serve", "/app/agent.yaml", "--host", "0.0.0.0", "--port", "8787", "--max-concurrent-runs", "8", "--bearer-token-env", "GABBY_API_TOKEN", "--bearer-scope", "agent:run", "--bearer-scope", "agent:stream", "--run-scope", "agent:run", "--stream-scope", "agent:stream"]
