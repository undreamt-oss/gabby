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
"""Consume Gabby's stateless SSE API from an application-owned client."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote, urlsplit

import httpx


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Decode Gabby's JSON SSE envelopes and reject incomplete or mismatched frames."""
    event_name: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                payload = json.loads("\n".join(data_lines))
                if not isinstance(payload, dict):
                    raise ValueError("Gabby SSE event must contain a JSON object")
                event_type = payload.get("type")
                data = payload.get("data")
                if not isinstance(event_type, str) or not isinstance(data, dict):
                    raise ValueError("Gabby SSE event has an invalid type or data object")
                if event_name is not None and event_name != event_type:
                    raise ValueError("Gabby SSE event name does not match its JSON envelope")
                yield {"type": event_type, "data": data}
            event_name = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    if data_lines:
        raise ValueError("Gabby SSE stream ended before the final event delimiter")


async def stream_agent(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    agent_name: str,
    api_token: str,
    task: str,
    context: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield typed events from one stateless run; the caller owns context and memory."""
    parsed_url = urlsplit(base_url)
    hostname = parsed_url.hostname
    if (
        parsed_url.scheme not in {"http", "https"}
        or hostname is None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("Gabby base URL must be an HTTP(S) origin without credentials or query")
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() in {"localhost", "localhost."}
    if parsed_url.scheme == "http" and not loopback:
        raise ValueError("Gabby base URL must use HTTPS outside loopback")
    url = f"{base_url.rstrip('/')}/v1/agents/{quote(agent_name, safe='')}/stream"
    payload = {
        "input": task,
        "context": context or {},
        "memory": memory or {},
        "include_trace": False,
    }
    async with client.stream(
        "POST",
        url,
        headers={"Authorization": f"Bearer {api_token}"},
        json=payload,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "text/event-stream":
            raise ValueError("Gabby stream endpoint did not return text/event-stream")
        async for event in iter_sse_events(response):
            yield event


def _json_object(value: str) -> dict[str, Any]:
    """Parse one command-line JSON object for caller-owned context or memory."""
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be valid JSON") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return result


async def _run(args: argparse.Namespace, *, api_token: str) -> None:
    timeout = httpx.Timeout(connect=10, read=180, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        streamed_text = False
        async for event in stream_agent(
            client,
            base_url=args.base_url,
            agent_name=args.agent,
            api_token=api_token,
            task=args.task,
            context=args.context,
            memory=args.memory,
        ):
            event_type = event["type"]
            data = event["data"]
            if event_type == "text_delta":
                text = data.get("text")
                if isinstance(text, str):
                    print(text, end="", flush=True)
                    streamed_text = True
            elif event_type == "completed":
                result = data.get("result")
                trace_id = result.get("trace_id") if isinstance(result, dict) else None
                if streamed_text:
                    print()
                elif isinstance(result, dict) and isinstance(result.get("output"), str):
                    print(result["output"])
                if isinstance(trace_id, str):
                    print(f"trace_id={trace_id}", file=sys.stderr)
            elif event_type == "error":
                error = data.get("error", "agent execution failed")
                error_type = data.get("error_type", "AgentExecutionError")
                raise RuntimeError(f"{error_type}: {error}")
            else:
                print(f"[{event_type}]", file=sys.stderr)


def main() -> int:
    """Run one streamed task against a Gabby service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="one task to send to the agent")
    parser.add_argument("--base-url", default=os.environ.get("GABBY_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--agent", required=True, help="configured Gabby agent route name")
    parser.add_argument(
        "--context-json", type=_json_object, default=_json_object("{}"), help="caller context JSON"
    )
    parser.add_argument(
        "--memory-json", type=_json_object, default=_json_object("{}"), help="caller memory JSON"
    )
    args = parser.parse_args()
    api_token = os.environ.get("GABBY_API_TOKEN")
    if not api_token:
        parser.error("GABBY_API_TOKEN must be provided by the host environment")
    try:
        asyncio.run(_run(args, api_token=api_token))
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"Gabby request failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
