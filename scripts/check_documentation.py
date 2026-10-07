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
"""Validate public Python documentation, comment hygiene, and Markdown links."""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "src" / "gabby", ROOT / "scripts", ROOT / "tests", ROOT / "examples")
LOCAL_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
FORBIDDEN_COMMENT_MARKERS = re.compile(r"\b(?:TODO|FIXME|XXX|HACK|WIP)\b", re.IGNORECASE)
ADR_FILENAME = re.compile(r"^(\d{4})-.+\.md$")
ADR_HEADING = re.compile(r"^# ADR (\d{4}): .+$", re.MULTILINE)
ADR_INDEX_LINK = re.compile(r"\]\((\d{4}-[^)]+\.md)\)")


def python_files() -> list[Path]:
    """Return maintained Python files covered by this policy."""
    return sorted(
        path
        for base in PYTHON_ROOTS
        for path in base.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )


def markdown_files() -> list[Path]:
    """Return repository Markdown files, excluding generated files and environments."""
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not {".git", ".venv", "dist", "build"}.intersection(path.parts)
    )


def package_exports() -> set[str]:
    """Read the root package's declared public exports."""
    path = ROOT / "src" / "gabby" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, (ast.List, ast.Tuple))
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        ):
            return {
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
    raise ValueError("src/gabby/__init__.py must declare __all__ as a list or tuple")


def public_documentation_errors(path: Path, tree: ast.Module, exports: set[str]) -> list[str]:
    """Find undocumented root exports and methods on exported public classes."""
    errors: list[str] = []
    if not ast.get_docstring(tree):
        errors.append(f"{path}: missing module docstring")

    def check_class(node: ast.ClassDef) -> None:
        if node.name in exports and not ast.get_docstring(node):
            errors.append(f"{path}:{node.lineno}: public class {node.name} lacks a docstring")
        if node.name not in exports:
            return
        for child in node.body:
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not child.name.startswith("_")
                and not ast.get_docstring(child)
            ):
                errors.append(
                    f"{path}:{child.lineno}: public method {child.name} lacks a docstring"
                )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in exports and not ast.get_docstring(node):
                errors.append(
                    f"{path}:{node.lineno}: public function {node.name} lacks a docstring"
                )
        elif isinstance(node, ast.ClassDef):
            check_class(node)
    return errors


def comment_errors(path: Path) -> list[str]:
    """Find stale-work markers in Python comments, excluding strings and code."""
    errors: list[str] = []
    with path.open("rb") as stream:
        for token in tokenize.tokenize(stream.readline):
            if token.type == tokenize.COMMENT and FORBIDDEN_COMMENT_MARKERS.search(token.string):
                errors.append(f"{path}:{token.start[0]}: unresolved marker in comment")
    return errors


def markdown_errors(path: Path) -> list[str]:
    """Find missing document headings, unbalanced fences, and broken local links."""
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    first = next((line.strip() for line in lines if line.strip()), "")
    if not first.startswith("# "):
        errors.append(f"{path}: document must begin with an H1 heading")
    if sum(line.strip().startswith("```") for line in lines) % 2:
        errors.append(f"{path}: unmatched fenced code block")
    content = "\n".join(lines)
    for target in LOCAL_LINK.findall(content):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if not (path.parent / target).resolve().exists():
            errors.append(f"{path}: local link does not exist: {target}")
    return errors


def architecture_decision_errors(directory: Path | None = None) -> list[str]:
    """Check ADR numbering, headings, and index coverage for one ADR directory."""
    adr_directory = directory or ROOT / "docs" / "architecture" / "adr"
    index_path = adr_directory / "README.md"
    errors: list[str] = []
    ids: dict[str, Path] = {}
    filenames: set[str] = set()

    for path in sorted(adr_directory.glob("*.md")):
        if path == index_path:
            continue
        match = ADR_FILENAME.fullmatch(path.name)
        if match is None:
            errors.append(f"{path}: ADR filename must begin with a four-digit ID")
            continue
        adr_id = match.group(1)
        if adr_id in ids:
            errors.append(f"{path}: duplicate ADR ID {adr_id} also used by {ids[adr_id]}")
        else:
            ids[adr_id] = path
        filenames.add(path.name)
        heading = ADR_HEADING.search(path.read_text(encoding="utf-8"))
        if heading is None or heading.group(1) != adr_id:
            errors.append(f"{path}: ADR heading ID must match filename ID {adr_id}")

    if not index_path.exists():
        return [*errors, f"{index_path}: ADR index is missing"]
    indexed = ADR_INDEX_LINK.findall(index_path.read_text(encoding="utf-8"))
    if len(indexed) != len(set(indexed)):
        errors.append(f"{index_path}: ADR files must be indexed only once")
    for filename in sorted(filenames - set(indexed)):
        errors.append(f"{index_path}: ADR is missing from index: {filename}")
    for filename in sorted(set(indexed) - filenames):
        errors.append(f"{index_path}: index references missing ADR: {filename}")
    return errors


def main() -> int:
    """Check source documentation and links across repository Markdown."""
    errors: list[str] = []
    try:
        exports = package_exports()
    except (OSError, SyntaxError, ValueError) as exc:
        print(
            f"Documentation policy violations:\n  Unable to read public exports: {exc}",
            file=sys.stderr,
        )
        return 1
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}: invalid Python syntax")
            continue
        if path.is_relative_to(ROOT / "src" / "gabby"):
            errors.extend(public_documentation_errors(path, tree, exports))
        elif (
            path.is_relative_to(ROOT / "scripts") or path.is_relative_to(ROOT / "examples")
        ) and not ast.get_docstring(tree):
            errors.append(f"{path}: missing module docstring")
        errors.extend(comment_errors(path))
    for path in markdown_files():
        errors.extend(markdown_errors(path))
    errors.extend(architecture_decision_errors())
    if errors:
        print("Documentation policy violations:", file=sys.stderr)
        print("\n".join(f"  {error}" for error in errors), file=sys.stderr)
        return 1
    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
