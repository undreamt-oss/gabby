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
"""Check that maintained Python files retain the Apache license header."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LINES = (
    "# Copyright ",
    'Licensed under the Apache License, Version 2.0 (the "License");',
    "https://www.apache.org/licenses/LICENSE-2.0",
    "limitations under the License.",
)


def main() -> int:
    missing: list[Path] = []
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "examples"):
        for path in sorted(directory.rglob("*.py")):
            contents = path.read_text(encoding="utf-8")
            first_lines = contents.splitlines()[:15]
            if not all(
                any(required in line for line in first_lines) for required in REQUIRED_LINES
            ):
                missing.append(path)
    if missing:
        for path in missing:
            print(f"Missing Apache-2.0 header: {path.relative_to(ROOT)}")
        return 1
    print("All maintained Python files under src/, scripts/, tests/, and examples/ have headers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
