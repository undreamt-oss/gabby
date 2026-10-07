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
"""Repository scorecard arithmetic and report synchronization checks."""

from __future__ import annotations

from copy import deepcopy

from scripts.check_scorecard import (
    load_scorecard,
    report_errors,
    validate_scorecard,
)


def test_scorecard_is_valid_and_matches_human_report() -> None:
    data = load_scorecard()

    assert validate_scorecard(data) == []
    assert report_errors(data) == []


def test_scorecard_rejects_weight_or_evidence_drift() -> None:
    data = deepcopy(load_scorecard())
    data["categories"][0]["weight"] = 16
    data["categories"][1]["evidence"] = " "

    errors = validate_scorecard(data)

    assert "architecture_contracts.weight must be 15" in errors
    assert "runtime_skills.evidence must be nonempty text" in errors


def test_scorecard_detects_human_report_drift() -> None:
    data = deepcopy(load_scorecard())
    data["categories"][0]["evidence"] = "A stale report would not include this text."

    errors = report_errors(data)

    assert "report row does not match the TOML source: architecture_contracts" in errors
