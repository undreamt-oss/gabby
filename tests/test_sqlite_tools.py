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
"""Read-only and bounded behavior for the built-in SQLite data tool."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from gabby import ToolContext, ToolError, ToolErrorCode, sqlite_query_tool
from gabby import sqlite_tools as sqlite_tools_module


def _database(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE sales (region TEXT, amount REAL);"
            "INSERT INTO sales VALUES ('north', 10), ('north', 20), ('south', 7);"
        )
    return path


def _invoke(tool: Any, path: Path, sql: str, parameters: list[Any] | None = None) -> Any:
    assert tool.handler is not None
    context = ToolContext(
        agent_name="data-agent",
        run_id="run-1",
        environment_type="data",
        environment_description="approved reporting database",
        capabilities=("sqlite_query",),
        resources={"sqlite_database": path},
    )
    return tool.handler(sql, parameters, tool_context=context)


def test_sqlite_tool_runs_parameterized_reads_and_returns_bounded_rows(tmp_path: Path) -> None:
    path = _database(tmp_path / "sales.sqlite")
    tool = sqlite_query_tool(max_rows=1)

    result = _invoke(
        tool,
        path,
        "SELECT region, amount FROM sales WHERE region = ? ORDER BY amount",
        ["north"],
    )

    assert result == {
        "columns": ["region", "amount"],
        "rows": [["north", 10.0]],
        "truncated": True,
    }
    assert tool.permissions == ("database:read",)
    assert tool.context_resources == ("sqlite_database",)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sales",
        "PRAGMA table_info(sales)",
        "ATTACH DATABASE ':memory:' AS extra",
        "SELECT load_extension('example')",
    ],
)
def test_sqlite_tool_rejects_mutation_pragma_attach_and_extension_loading(
    tmp_path: Path, sql: str
) -> None:
    path = _database(tmp_path / "sales.sqlite")
    tool = sqlite_query_tool()

    with pytest.raises(ToolError) as caught:
        _invoke(tool, path, sql)

    assert caught.value.code in {ToolErrorCode.POLICY_DENIED, ToolErrorCode.EXECUTION_FAILED}
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM sales").fetchone() == (3,)


def test_sqlite_tool_enforces_result_bytes_and_execution_deadline(tmp_path: Path) -> None:
    path = _database(tmp_path / "sales.sqlite")
    bounded = sqlite_query_tool(max_output_bytes=128)
    output = _invoke(bounded, path, "SELECT printf('%0100d', 1) AS value FROM sales")
    assert output["truncated"] is True
    assert len(output["rows"]) < 3
    assert len(json.dumps(output).encode("utf-8")) <= 128

    deadline = sqlite_query_tool(timeout_seconds=0.01)
    with pytest.raises(ToolError) as caught:
        _invoke(
            deadline,
            path,
            "WITH RECURSIVE count(n) AS (VALUES(1) UNION ALL SELECT n + 1 FROM count "
            "WHERE n < 100000000) SELECT max(n) FROM count",
        )
    assert caught.value.code == ToolErrorCode.DEADLINE_EXCEEDED


def test_sqlite_tool_requires_named_database_resource_and_valid_limits(tmp_path: Path) -> None:
    tool = sqlite_query_tool()
    context = ToolContext(
        agent_name="data-agent",
        run_id="run-1",
        environment_type="data",
        environment_description="missing resource",
        capabilities=(),
        resources={},
    )
    assert tool.handler is not None
    with pytest.raises(ToolError) as caught:
        tool.handler("SELECT 1", None, tool_context=context)
    assert caught.value.code == ToolErrorCode.TOOL_UNAVAILABLE

    with pytest.raises(ValueError, match="max_rows"):
        sqlite_query_tool(max_rows=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource_name": ""},
        {"resource_name": 1},
        {"max_rows": 0},
        {"max_output_bytes": 127},
        {"max_cell_bytes": 1023},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": True},
    ],
)
def test_sqlite_tool_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        sqlite_query_tool(**kwargs)


def test_sqlite_tool_bounds_query_parameters_and_resource_resolution(tmp_path: Path) -> None:
    path = _database(tmp_path / "sales.sqlite")
    tool = sqlite_query_tool()
    with pytest.raises(ToolError) as oversized_query:
        _invoke(tool, path, "SELECT 1 " + (" " * (64 * 1024)))
    assert oversized_query.value.code == ToolErrorCode.INVALID_ARGUMENTS

    with pytest.raises(ToolError) as too_many_parameters:
        _invoke(tool, path, "SELECT ?", [None] * 257)
    assert too_many_parameters.value.code == ToolErrorCode.INVALID_ARGUMENTS

    with pytest.raises(ToolError) as invalid_unicode:
        _invoke(tool, path, "SELECT '\ud800'")
    assert invalid_unicode.value.code == ToolErrorCode.INVALID_ARGUMENTS

    with pytest.raises(ToolError) as unavailable:
        _invoke(tool, tmp_path / "missing.sqlite", "SELECT 1")
    assert unavailable.value.code == ToolErrorCode.TOOL_UNAVAILABLE


def test_sqlite_tool_rejects_oversized_column_labels_and_normalizes_values(
    tmp_path: Path,
) -> None:
    path = _database(tmp_path / "sales.sqlite")
    tool = sqlite_query_tool(max_output_bytes=128)
    with pytest.raises(ToolError) as oversized_labels:
        _invoke(tool, path, f'SELECT 1 AS "{"x" * 256}"')
    assert oversized_labels.value.code == ToolErrorCode.RESULT_TOO_LARGE

    normalized = _invoke(sqlite_query_tool(), path, "SELECT x'00ff', 1e999")
    assert normalized["rows"] == [[{"base64": "AP8="}, None]]
    assert sqlite_tools_module._json_value(float("nan")) is None
