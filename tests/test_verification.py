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
"""Validation and serialization contracts for verifier results."""

from __future__ import annotations

from typing import Any

import pytest

from gabby.verification import VerificationResult


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"passed": 1, "method": "fixture"}, TypeError),
        ({"passed": True, "method": 1}, ValueError),
        ({"passed": True, "method": " "}, ValueError),
        ({"passed": True, "method": "fixture", "details": []}, TypeError),
        ({"passed": True, "method": "fixture", "details": {1: "bad"}}, TypeError),
        (
            {"passed": True, "method": "fixture", "details": {"bad": object()}},
            TypeError,
        ),
        (
            {"passed": True, "method": "fixture", "details": {"bad": float("nan")}},
            TypeError,
        ),
        ({"passed": True, "method": "fixture", "evidence": ["not-a-tuple"]}, TypeError),
        ({"passed": True, "method": "fixture", "evidence": (1,)}, TypeError),
    ],
)
def test_verification_result_rejects_invalid_fields(
    values: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        VerificationResult(**values)


def test_verification_result_snapshots_and_serializes_details() -> None:
    source = {"checks": ["schema", "policy"]}
    result = VerificationResult(
        passed=True,
        method="fixture",
        details=source,
        evidence=("tests/test_contract.py",),
    )
    source["checks"].append("changed")

    assert result.as_dict() == {
        "passed": True,
        "method": "fixture",
        "details": {"checks": ["schema", "policy"]},
        "evidence": ["tests/test_contract.py"],
    }
