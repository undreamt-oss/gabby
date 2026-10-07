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
"""Construction contract for the sandboxed Python execution tool."""

from __future__ import annotations

import pytest

from gabby import python_run_tool


def test_python_run_tool_is_sandboxed_and_bounded() -> None:
    tool = python_run_tool(executable="python.exe", max_code_bytes=2048, timeout_seconds=12)

    assert tool.sandbox_action == "python.run"
    assert tool.execution == "sandboxed"
    assert tool.sandbox_command == ("python.exe",)
    assert tool.max_input_bytes == 2048
    assert tool.timeout_seconds == 12
    assert tool.permissions == ("sandbox:python",)
    assert tool.validate_arguments({"code": "print(40 + 2)"}) == {"code": "print(40 + 2)"}
    assert tool.validate_result({"exit_code": 0, "stdout": "42\n", "stderr": ""})["exit_code"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"executable": ""}, "executable"),
        ({"max_code_bytes": 0}, "max_code_bytes"),
        ({"max_code_bytes": 1024 * 1024 + 1}, "max_code_bytes"),
        ({"max_result_bytes": True}, "max_result_bytes"),
        ({"max_result_bytes": 1024 * 1024 + 1}, "max_result_bytes"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
    ],
)
def test_python_run_tool_rejects_invalid_limits(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        python_run_tool(**kwargs)  # type: ignore[arg-type]
