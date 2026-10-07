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
"""Read-only, bounded SQLite tools for data-oriented agents."""

from __future__ import annotations

import base64
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from .tools import Tool, ToolContext, ToolError, ToolErrorCode

_DEFAULT_MAX_ROWS = 500
_DEFAULT_MAX_OUTPUT_BYTES = 768 * 1024
_DEFAULT_MAX_CELL_BYTES = 256 * 1024
_DEFAULT_QUERY_TIMEOUT_SECONDS = 2.0
_MAX_SQLITE_COLUMNS = 256
_MAX_SQLITE_BIND_PARAMETERS = 256


def sqlite_query_tool(
    *,
    resource_name: str = "sqlite_database",
    name: str = "sqlite_query",
    description: str = "Run a read-only SQL query against the approved SQLite database.",
    max_rows: int = _DEFAULT_MAX_ROWS,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    max_cell_bytes: int = _DEFAULT_MAX_CELL_BYTES,
    timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> Tool:
    """Build a read-only SQLite query tool using a named host environment resource.

    The environment resource must be a database file path. The connection opens in SQLite
    read-only mode, an authorizer permits only query/read/function operations, and SQLite's
    progress handler enforces the query deadline. Returned rows, individual values, and query
    work are bounded. Grant the agent the ``database:read`` permission explicitly.
    """
    if not isinstance(resource_name, str) or not resource_name:
        raise ValueError("resource_name must be a non-empty string")
    _positive_integer(max_rows, "max_rows", maximum=100_000)
    _positive_integer(max_output_bytes, "max_output_bytes", minimum=128, maximum=16 * 1024 * 1024)
    _positive_integer(max_cell_bytes, "max_cell_bytes", minimum=1024, maximum=16 * 1024 * 1024)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0.01 <= timeout_seconds <= 300
    ):
        raise ValueError("timeout_seconds must be finite and between 0.01 and 300 seconds")

    def query(
        sql: str,
        parameters: list[Any] | None = None,
        *,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        if not isinstance(sql, str):
            raise ToolError("SQL query must be text", code=ToolErrorCode.INVALID_ARGUMENTS)
        try:
            query_size = len(sql.encode("utf-8"))
        except UnicodeEncodeError:
            raise ToolError(
                "SQL query must be valid UTF-8", code=ToolErrorCode.INVALID_ARGUMENTS
            ) from None
        if query_size > 64 * 1024:
            raise ToolError(
                "SQL query exceeds its input limit", code=ToolErrorCode.INVALID_ARGUMENTS
            )
        if parameters is None:
            parameters = []
        if len(parameters) > _MAX_SQLITE_BIND_PARAMETERS:
            raise ToolError("Too many SQL parameters", code=ToolErrorCode.INVALID_ARGUMENTS)
        database = tool_context.resources.get(resource_name)
        if not isinstance(database, (str, Path)):
            raise ToolError(
                "Configured SQLite resource is unavailable", code=ToolErrorCode.TOOL_UNAVAILABLE
            )

        try:
            path = Path(database).expanduser().resolve(strict=True)
            if not path.is_file():
                raise OSError
            connection = sqlite3.connect(
                f"{path.as_uri()}?mode=ro", uri=True, timeout=min(float(timeout_seconds), 0.25)
            )
        except (OSError, RuntimeError, sqlite3.Error, ValueError):
            raise ToolError(
                "Configured SQLite resource is unavailable", code=ToolErrorCode.TOOL_UNAVAILABLE
            ) from None

        deadline = time.monotonic() + float(timeout_seconds)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_cell_bytes)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, _MAX_SQLITE_COLUMNS)
            connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _MAX_SQLITE_BIND_PARAMETERS)

            def authorize(
                action: int,
                first: str | None,
                second: str | None,
                _database: str | None,
                _trigger: str | None,
            ) -> int:
                if action == sqlite3.SQLITE_FUNCTION:
                    function_name = (second or first or "").lower()
                    if function_name == "load_extension":
                        return sqlite3.SQLITE_DENY
                if action in {
                    sqlite3.SQLITE_SELECT,
                    sqlite3.SQLITE_READ,
                    sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE,
                }:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            connection.set_authorizer(authorize)
            connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = connection.execute(sql, parameters)
            if cursor.description is None:
                raise ToolError("SQL must be a read-only query", code=ToolErrorCode.POLICY_DENIED)
            columns = [str(column[0]) for column in cursor.description]
            rows: list[list[Any]] = []
            result = {"columns": columns, "rows": rows, "truncated": False}
            used_bytes = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            if used_bytes > max_output_bytes:
                raise ToolError(
                    "SQL result column labels exceed the output limit",
                    code=ToolErrorCode.RESULT_TOO_LARGE,
                )
            truncated = False
            for row in cursor:
                converted = [_json_value(value) for value in row]
                row_size = len(json.dumps(converted, ensure_ascii=False).encode("utf-8"))
                row_delta = row_size - 2 if not rows else row_size + 2
                if len(rows) >= max_rows or used_bytes + row_delta > max_output_bytes:
                    truncated = True
                    break
                rows.append(converted)
                used_bytes += row_delta
            result["truncated"] = truncated
            return result
        except ToolError:
            raise
        except sqlite3.Error:
            if time.monotonic() >= deadline:
                raise ToolError(
                    "SQL query exceeded its time limit", code=ToolErrorCode.DEADLINE_EXCEEDED
                ) from None
            raise ToolError(
                "SQL query was rejected or failed", code=ToolErrorCode.EXECUTION_FAILED
            ) from None
        finally:
            connection.close()

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "minLength": 1, "maxLength": 65536},
                "parameters": {
                    "type": "array",
                    "items": {"type": ["string", "number", "integer", "boolean", "null"]},
                    "maxItems": _MAX_SQLITE_BIND_PARAMETERS,
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        handler=query,
        timeout_seconds=float(timeout_seconds) + 1,
        max_input_bytes=128 * 1024,
        max_result_bytes=max_output_bytes + 16 * 1024,
        output_schema={
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": ["string", "number", "integer", "boolean", "null", "object"]
                        },
                    },
                },
                "truncated": {"type": "boolean"},
            },
            "required": ["columns", "rows", "truncated"],
            "additionalProperties": False,
        },
        permissions=("database:read",),
        context_parameter="tool_context",
        context_resources=(resource_name,),
    )


def _positive_integer(value: object, name: str, *, maximum: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
