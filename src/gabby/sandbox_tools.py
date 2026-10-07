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
"""Built-in tool declarations whose effects stay inside a run-scoped sandbox."""

from __future__ import annotations

from .tools import Tool


def sandbox_tools() -> list[Tool]:
    """Return core shell and workspace tools backed by a per-run container."""
    return [
        Tool(
            name="shell_run",
            description=(
                "Run one executable inside the agent's isolated container. Provide argv as an "
                "array; shell expansion, pipelines, and redirection are not implicit."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {"type": "string", "description": "Relative path in the workspace."},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            permissions=["shell.execute"],
            sandbox_action="shell.run",
            execution="sandboxed",
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
        ),
        Tool(
            name="filesystem_read_file",
            description="Read a UTF-8 text file inside the configured workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1}},
                "required": ["path"],
                "additionalProperties": False,
            },
            permissions=["filesystem.read"],
            sandbox_action="filesystem.read_file",
            execution="sandboxed",
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="filesystem_write_file",
            description=(
                "Write a UTF-8 text file inside the configured workspace. The workspace mount "
                "must be configured read/write."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            permissions=["filesystem.write"],
            sandbox_action="filesystem.write_file",
            execution="sandboxed",
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "bytes_written": {"type": "integer", "minimum": 0},
                },
                "required": ["path", "bytes_written"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="filesystem_list_dir",
            description="List entries in a directory inside the configured workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path; use '.' for the workspace root.",
                    }
                },
                "additionalProperties": False,
            },
            permissions=["filesystem.read"],
            sandbox_action="filesystem.list_dir",
            execution="sandboxed",
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "entries": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["path", "entries"],
                "additionalProperties": False,
            },
        ),
    ]
