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
"""Tests for repository documentation policy checks."""

from pathlib import Path

from scripts.check_documentation import architecture_decision_errors


def test_architecture_decisions_require_unique_numbered_index_entries(tmp_path: Path) -> None:
    (tmp_path / "0001-first.md").write_text("# ADR 0001: First\n", encoding="utf-8")
    (tmp_path / "0001-second.md").write_text("# ADR 0001: Second\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Architecture Decision Records\n\n- [First](0001-first.md)\n- [Second](0001-second.md)\n",
        encoding="utf-8",
    )

    errors = architecture_decision_errors(tmp_path)

    assert any("duplicate ADR ID 0001" in error for error in errors)


def test_architecture_decisions_require_filename_heading_and_index_agreement(
    tmp_path: Path,
) -> None:
    (tmp_path / "0001-first.md").write_text("# ADR 0002: First\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Architecture Decision Records\n\n- [Stale](0002-stale.md)\n",
        encoding="utf-8",
    )

    errors = architecture_decision_errors(tmp_path)

    assert any("heading ID must match filename ID 0001" in error for error in errors)
    assert any("missing from index: 0001-first.md" in error for error in errors)
    assert any("index references missing ADR: 0002-stale.md" in error for error in errors)


def test_architecture_decisions_accept_consistent_index(tmp_path: Path) -> None:
    (tmp_path / "0001-first.md").write_text("# ADR 0001: First\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# Architecture Decision Records\n\n- [First](0001-first.md)\n",
        encoding="utf-8",
    )

    assert architecture_decision_errors(tmp_path) == []
