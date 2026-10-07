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
"""Validate and print Gabby's fixed, evidence-based project scorecard."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCORECARD_PATH = ROOT / "docs" / "project-scorecard.toml"
REPORT_PATH = ROOT / "docs" / "PROJECT_SCORECARD.md"
EXPECTED_CATEGORIES = (
    ("architecture_contracts", "Architecture and contracts", 15),
    ("runtime_skills", "Agent runtime and skills", 20),
    ("model_knowledge", "Model and knowledge providers", 15),
    ("security_sandbox", "Security and sandboxing", 20),
    ("api_deployment", "API, CLI, and deployment", 10),
    ("oss_ci", "OSS quality and CI", 15),
    ("docs_examples", "Docs and examples", 5),
)


def load_scorecard(path: Path = SCORECARD_PATH) -> dict[str, Any]:
    """Load the TOML scorecard and validate that its root is a table."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Scorecard root must be a TOML table")
    return data


def validate_scorecard(data: dict[str, Any]) -> list[str]:
    """Return structural, evidence, or arithmetic violations in the scorecard."""
    errors: list[str] = []
    if data.get("version") != 1:
        errors.append("version must be 1")
    categories = data.get("categories")
    if not isinstance(categories, list):
        return [*errors, "categories must be an array of tables"]
    if data.get("category_order") != [key for key, _, _ in EXPECTED_CATEGORIES]:
        errors.append("category_order must match the fixed scorecard category order")
    if len(categories) != len(EXPECTED_CATEGORIES):
        errors.append("categories must contain exactly the fixed scorecard categories")

    total_weight = 0
    for index, expected in enumerate(EXPECTED_CATEGORIES):
        if index >= len(categories):
            break
        category = categories[index]
        key, label, weight = expected
        if not isinstance(category, dict):
            errors.append(f"categories[{index}] must be a table")
            continue
        if category.get("key") != key:
            errors.append(f"categories[{index}].key must be {key!r}")
        if category.get("label") != label:
            errors.append(f"{key}.label must be {label!r}")
        if category.get("weight") != weight:
            errors.append(f"{key}.weight must be {weight}")
        score = category.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
            errors.append(f"{key}.score must be an integer from 0 through 100")
        evidence = category.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{key}.evidence must be nonempty text")
        remaining = category.get("remaining")
        if not isinstance(remaining, str) or not remaining.strip():
            errors.append(f"{key}.remaining must be nonempty text")
        total_weight += weight
    if total_weight != 100:
        errors.append(f"fixed category weights must total 100, got {total_weight}")
    return errors


def weighted_score(data: dict[str, Any]) -> float:
    """Calculate the weighted percentage without rounding intermediate values."""
    categories = data.get("categories")
    if not isinstance(categories, list):
        raise ValueError("categories must be an array of tables")
    total = 0
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("each category must be a table")
        weight = category.get("weight")
        score = category.get("score")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or isinstance(score, bool)
            or not isinstance(score, int)
        ):
            raise ValueError("category weight and score must be integers")
        total += weight * score
    return total / 100


def report_errors(data: dict[str, Any], report_path: Path = REPORT_PATH) -> list[str]:
    """Check that the human-readable report matches the TOML source of truth."""
    text = report_path.read_text(encoding="utf-8")
    score = weighted_score(data)
    whole_score = int(score + 0.5)
    errors: list[str] = []
    if not re.search(rf"\*\*Weighted progress: {whole_score}%\.\*\*", text):
        errors.append("reported whole-number progress does not match the weighted score")
    if not re.search(rf"= {score:.2f}%", text):
        errors.append("reported weighted calculation does not match the TOML scorecard")
    for category in data["categories"]:
        expected_row = (
            f"| {category['label']} | {category['weight']}% | {category['score']}% | "
            f"{category['evidence']} {category['remaining']} |"
        )
        if expected_row not in text:
            errors.append(f"report row does not match the TOML source: {category['key']}")
    return errors


def main() -> int:
    """Validate and print the weighted score and fixed category breakdown."""
    try:
        data = load_scorecard()
        errors = validate_scorecard(data)
        if not errors:
            errors.extend(report_errors(data))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"Scorecard invalid: {exc}")
        return 1
    if errors:
        print("Scorecard invalid:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    score = weighted_score(data)
    print(f"Gabby project score: {score:.2f}%")
    for category in data["categories"]:
        print(f"- {category['label']}: {category['score']}% (weight {category['weight']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
