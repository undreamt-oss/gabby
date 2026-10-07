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
"""Validate the contents, paths, and metadata of built Gabby distributions."""

from __future__ import annotations

import argparse
import ast
import email.message
import email.parser
import posixpath
import stat
import sys
import tarfile
import tomllib
from pathlib import Path, PureWindowsPath
from zipfile import BadZipFile, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIST = ROOT / "dist"
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
EXPECTED_NAME = PROJECT["name"]
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_REQUIRES_PYTHON = {">=3.11,<3.15", "<3.15,>=3.11"}
EXPECTED_VERSION = PROJECT["version"]
REQUIRED_PUBLIC_EXPORTS = frozenset(
    {
        "Agent",
        "CSVTextParser",
        "JSONTextParser",
        "MarkupTextParser",
        "NvidiaReranker",
        "PDFTextParser",
        "PostgresStreamJournal",
        "TransformersProvider",
    }
)


def _safe_member(name: str) -> bool:
    """Return whether an archive member has a safe relative POSIX path."""
    if not name or "\\" in name or "\x00" in name:
        return False
    normalized = posixpath.normpath(name)
    windows_path = PureWindowsPath(name)
    return (
        not normalized.startswith("/")
        and ".." not in normalized.split("/")
        and not windows_path.is_absolute()
        and not windows_path.drive
    )


def _safe_tar_member(member: tarfile.TarInfo) -> bool:
    """Allow only regular files and directories with safe relative names."""
    return _safe_member(member.name) and (member.isfile() or member.isdir())


def _safe_zip_member(member: ZipInfo) -> bool:
    """Reject ZIP symlinks while allowing normal wheel entries without Unix mode bits."""
    mode = (member.external_attr >> 16) & 0o170000
    return _safe_member(member.filename) and mode != stat.S_IFLNK


def _metadata_errors(metadata: email.message.Message, source: Path) -> list[str]:
    """Return errors for the release metadata carried by one archive."""
    errors: list[str] = []
    for key, value in (("Name", EXPECTED_NAME), ("License-Expression", EXPECTED_LICENSE)):
        if metadata.get(key) != value:
            errors.append(f"{source}: metadata {key!r} must be {value!r}")
    if metadata.get("Requires-Python") not in EXPECTED_REQUIRES_PYTHON:
        errors.append(f"{source}: metadata 'Requires-Python' must describe Python 3.11–3.14")
    version = metadata.get("Version")
    if not version:
        errors.append(f"{source}: metadata must contain a version")
    elif version != EXPECTED_VERSION:
        errors.append(
            f"{source}: metadata version must match the declared project version "
            f"({EXPECTED_VERSION!r})"
        )
    if not metadata.get("License-File"):
        errors.append(f"{source}: metadata must declare the license file")
    return errors


def _read_metadata(raw: bytes, source: Path) -> list[str]:
    """Parse metadata bytes and return validation errors without raising parser failures."""
    metadata = email.parser.BytesParser().parsebytes(raw)
    return _metadata_errors(metadata, source)


def _public_api_errors(raw: bytes, source: Path) -> list[str]:
    """Ensure the wheel package root contains Gabby's required public parser exports."""
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(source))
    except (SyntaxError, UnicodeDecodeError):
        return [f"{source}: package __init__.py is not valid UTF-8 Python"]
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not isinstance(
            statement.value, (ast.List, ast.Tuple)
        ):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            continue
        exports = {
            item.value
            for item in statement.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        missing = sorted(REQUIRED_PUBLIC_EXPORTS - exports)
        return (
            [f"{source}: __all__ is missing required exports: {', '.join(missing)}"]
            if missing
            else []
        )
    return [f"{source}: package __init__.py must declare __all__ as a list or tuple"]


def _check_wheel(path: Path) -> list[str]:
    """Validate package modules, typing marker, license, paths, and wheel metadata."""
    errors: list[str] = []
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        errors.extend(
            f"{path}: unsafe archive member {member.filename!r}"
            for member in members
            if not _safe_zip_member(member)
        )
        if len(names) != len(set(names)):
            errors.append(f"{path}: duplicate archive member names are not allowed")
        metadata_names = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_names) != 1:
            errors.append(f"{path}: expected exactly one wheel METADATA file")
        else:
            errors.extend(_read_metadata(archive.read(metadata_names[0]), path))

        for name, description in (
            ("gabby/__init__.py", "the importable Gabby package"),
            ("gabby/ingestion.py", "the ingestion implementation"),
            ("gabby/training.py", "the optional adapter training implementation"),
            ("gabby/py.typed", "the PEP 561 typing marker"),
            ("gabby/sql/postgres_stream_journal.sql", "the PostgreSQL SSE journal migration"),
            ("gabby/sql/postgres_knowledge.sql", "the PostgreSQL knowledge migration"),
            ("gabby/sql/postgres_vector.sql", "the PostgreSQL vector migration"),
        ):
            if name not in names:
                errors.append(f"{path}: missing {description}: {name}")
        migration_name = "gabby/sql/postgres_stream_journal.sql"
        if migration_name in names:
            packaged = archive.read(migration_name)
            source = (ROOT / "sql" / "postgres_stream_journal.sql").read_bytes()
            if packaged != source:
                errors.append(f"{path}: packaged PostgreSQL SSE journal migration is stale")
        knowledge_migration = "gabby/sql/postgres_knowledge.sql"
        if knowledge_migration in names:
            packaged = archive.read(knowledge_migration)
            source = (ROOT / "sql" / "postgres_knowledge.sql").read_bytes()
            if packaged != source:
                errors.append(f"{path}: packaged PostgreSQL knowledge migration is stale")
        vector_migration = "gabby/sql/postgres_vector.sql"
        if vector_migration in names:
            packaged = archive.read(vector_migration)
            source = (ROOT / "sql" / "postgres_vector.sql").read_bytes()
            if packaged != source:
                errors.append(f"{path}: packaged PostgreSQL vector migration is stale")
        if "gabby/__init__.py" in names:
            errors.extend(_public_api_errors(archive.read("gabby/__init__.py"), path))
        if not any(name.endswith(".dist-info/WHEEL") for name in names):
            errors.append(f"{path}: missing wheel metadata")
        if not any(name.endswith(".dist-info/RECORD") for name in names):
            errors.append(f"{path}: missing wheel record")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
            errors.append(f"{path}: missing packaged LICENSE file")
    return errors


def _check_sdist(path: Path) -> list[str]:
    """Validate source modules, typing marker, license, paths, and source metadata."""
    errors: list[str] = []
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        errors.extend(
            f"{path}: unsafe archive member {member.name!r}"
            for member in members
            if not _safe_tar_member(member)
        )
        if len(names) != len(set(names)):
            errors.append(f"{path}: duplicate archive member names are not allowed")
        pkg_info = sorted(
            name for name in names if name.endswith("/PKG-INFO") and name.count("/") == 1
        )
        if len(pkg_info) != 1:
            errors.append(f"{path}: expected exactly one source PKG-INFO file")
        else:
            member = archive.extractfile(pkg_info[0])
            if member is None:
                errors.append(f"{path}: unable to read {pkg_info[0]}")
            else:
                errors.extend(_read_metadata(member.read(), path))
        for suffix, description in (
            ("/src/gabby/__init__.py", "the importable Gabby package"),
            ("/src/gabby/ingestion.py", "the ingestion implementation"),
            ("/src/gabby/training.py", "the optional adapter training implementation"),
            ("/src/gabby/py.typed", "the PEP 561 typing marker"),
            (
                "/src/gabby/sql/postgres_stream_journal.sql",
                "the PostgreSQL SSE journal migration",
            ),
            (
                "/src/gabby/sql/postgres_knowledge.sql",
                "the PostgreSQL knowledge migration",
            ),
            ("/src/gabby/sql/postgres_vector.sql", "the PostgreSQL vector migration"),
            ("/LICENSE", "the project license"),
        ):
            if not any(name.endswith(suffix) for name in names):
                errors.append(f"{path}: missing {description} ({suffix})")
        packaged_migrations = [
            name for name in names if name.endswith("/src/gabby/sql/postgres_stream_journal.sql")
        ]
        if len(packaged_migrations) == 1:
            member = archive.extractfile(packaged_migrations[0])
            if (
                member is None
                or member.read() != (ROOT / "sql" / "postgres_stream_journal.sql").read_bytes()
            ):
                errors.append(f"{path}: packaged PostgreSQL SSE journal migration is stale")
        packaged_knowledge_migrations = [
            name for name in names if name.endswith("/src/gabby/sql/postgres_knowledge.sql")
        ]
        if len(packaged_knowledge_migrations) == 1:
            member = archive.extractfile(packaged_knowledge_migrations[0])
            if (
                member is None
                or member.read() != (ROOT / "sql" / "postgres_knowledge.sql").read_bytes()
            ):
                errors.append(f"{path}: packaged PostgreSQL knowledge migration is stale")
        packaged_vector_migrations = [
            name for name in names if name.endswith("/src/gabby/sql/postgres_vector.sql")
        ]
        if len(packaged_vector_migrations) == 1:
            member = archive.extractfile(packaged_vector_migrations[0])
            if (
                member is None
                or member.read() != (ROOT / "sql" / "postgres_vector.sql").read_bytes()
            ):
                errors.append(f"{path}: packaged PostgreSQL vector migration is stale")
    return errors


def check_distribution_directory(directory: Path) -> list[str]:
    """Return package-integrity errors for the wheel and source archive in ``directory``."""
    if not directory.is_dir():
        return [f"distribution directory does not exist: {directory}"]
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    errors: list[str] = []
    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel in {directory}, found {len(wheels)}")
    if len(sdists) != 1:
        errors.append(f"expected exactly one sdist in {directory}, found {len(sdists)}")
    name_prefix = f"{EXPECTED_NAME.replace('-', '_')}-{EXPECTED_VERSION}-"
    if len(wheels) == 1:
        if not wheels[0].name.startswith(name_prefix):
            errors.append(f"{wheels[0]}: filename must start with {name_prefix!r}")
        try:
            errors.extend(_check_wheel(wheels[0]))
        except (OSError, BadZipFile, UnicodeDecodeError) as exc:
            errors.append(f"{wheels[0]}: unable to inspect archive ({type(exc).__name__})")
    if len(sdists) == 1:
        expected_sdist = f"{EXPECTED_NAME.replace('-', '_')}-{EXPECTED_VERSION}.tar.gz"
        if sdists[0].name != expected_sdist:
            errors.append(f"{sdists[0]}: filename must be {expected_sdist!r}")
        try:
            errors.extend(_check_sdist(sdists[0]))
        except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
            errors.append(f"{sdists[0]}: unable to inspect archive ({type(exc).__name__})")
    return errors


def main() -> int:
    """Validate the requested distribution directory and return a shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_DIST,
        help="directory containing exactly one wheel and one source archive (default: dist)",
    )
    arguments = parser.parse_args()
    errors = check_distribution_directory(arguments.directory)
    if errors:
        print("Package integrity checks failed:", file=sys.stderr)
        print("\n".join(f"  {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Package integrity checks passed for {arguments.directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
