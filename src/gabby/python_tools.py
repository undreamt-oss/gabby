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
"""Sandboxed Python execution tool for data-oriented agents."""

from __future__ import annotations

import math

from .tools import Tool

_MAX_CODE_BYTES = 1024 * 1024


def python_run_tool(
    *,
    executable: str = "python3",
    name: str = "python_run",
    description: str = "Run Python code in the configured per-run sandbox.",
    max_code_bytes: int = 256 * 1024,
    max_result_bytes: int = 1024 * 1024,
    timeout_seconds: float = 30,
) -> Tool:
    """Create a Python tool whose source and process execute inside the agent container.

    The selected sandbox image must contain the configured interpreter. Gabby stages each source
    file in the sandbox's private, read-only tool-input mount, runs it with Python isolated mode,
    and removes the staged source after execution. The container's workspace, network, deadline,
    CPU, memory, and process policies remain in force.
    """
    if not isinstance(executable, str) or not executable or "\x00" in executable:
        raise ValueError("executable must be a non-empty interpreter command")
    for key, value, maximum in (
        ("max_code_bytes", max_code_bytes, _MAX_CODE_BYTES),
        ("max_result_bytes", max_result_bytes, 1024 * 1024),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be a positive integer")
        if not 1 <= value <= maximum:
            raise ValueError(f"{key} must be from 1 through {maximum}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0.01 <= timeout_seconds <= 3600
    ):
        raise ValueError("timeout_seconds must be finite and between 0.01 and 3600 seconds")

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string", "minLength": 1}},
            "required": ["code"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "exit_code": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
            "required": ["exit_code", "stdout", "stderr"],
            "additionalProperties": False,
        },
        timeout_seconds=float(timeout_seconds),
        max_input_bytes=max_code_bytes,
        max_result_bytes=max_result_bytes,
        permissions=("sandbox:python",),
        sandbox_action="python.run",
        execution="sandboxed",
        sandbox_command=(executable,),
    )
