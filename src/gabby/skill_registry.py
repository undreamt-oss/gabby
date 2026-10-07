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
"""Async client for static, signed Gabby skill registries."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .config import ConfigError, SkillDefinition, load_skill, validate_skill_definition
from .skill_packages import (
    _INSTALL_PROVENANCE,
    _MAX_PACKAGE_ARCHIVE_BYTES,
    SkillPackageError,
    SkillPackageInfo,
    _copy_package_snapshot,
    _read_install_provenance,
    install_skill,
    verify_installed_skill,
)
from .skill_signing import _read_bounded_signature_file

_CATALOG_PATH = "/v1/catalog.json"
_CATALOG_FORMAT = "gabby-skill-catalog"
_CATALOG_VERSION = 1
_MAX_CATALOG_BYTES = 4 * 1024 * 1024
_MAX_SKILLS = 10_000
_MAX_VERSIONS_PER_SKILL = 1_000
_MAX_DESCRIPTION_BYTES = 16 * 1024
_MAX_SIGNATURE_BYTES = 16 * 1024
_MAX_BUILD_PACKAGES = 10_000
_MAX_BUILD_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_DEPENDENCY_PACKAGES = 256
_MAX_DEPENDENCY_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MAX_DEPENDENCY_RESOLUTION_SECONDS = 300
_MAX_CATALOG_FUTURE_SKEW_SECONDS = 300
_MAX_CATALOG_AGE_SECONDS = 10 * 365 * 24 * 60 * 60


class SkillRegistryError(RuntimeError):
    """A registry response is unavailable, malformed, or cannot be installed safely."""


@dataclass(frozen=True)
class SkillCatalogEntry:
    """Untrusted discovery metadata for one package name and its exact versions."""

    name: str
    description: str
    versions: tuple[str, ...]


@dataclass(frozen=True)
class StaticSkillRegistryBuild:
    """Summary of one validated static registry directory build."""

    path: Path
    skill_count: int
    package_count: int
    catalog_sha256: str


def build_static_skill_registry(
    packages: Iterable[str | Path],
    output: str | Path,
    *,
    trusted_keys: Mapping[str, bytes],
    existing_registry: str | Path | None = None,
) -> StaticSkillRegistryBuild:
    """Build a new static registry directory from publisher-signed skill packages.

    The output is created from a private sibling staging directory and is never merged into an
    existing path. Each archive and detached signature is copied into staging, verified there,
    installed into a temporary validation directory, and checked against its declared skill
    identity before it is included in the catalog. The caller supplies the publisher trust map;
    catalog contents never establish trust.
    """
    if not isinstance(trusted_keys, Mapping) or not trusted_keys:
        raise SkillRegistryError("Trusted signing keys are required to build a registry")
    trusted_key_snapshot = dict(trusted_keys)
    if any(
        not isinstance(key_id, str) or not isinstance(public_key, bytes) or len(public_key) != 32
        for key_id, public_key in trusted_key_snapshot.items()
    ):
        raise SkillRegistryError("Trusted signing keys are invalid")

    if isinstance(packages, (str, Path)):
        raise SkillRegistryError("Provide a sequence of skill package paths")
    package_paths: list[str | Path] = []
    try:
        for package in packages:
            if len(package_paths) == _MAX_BUILD_PACKAGES:
                raise SkillRegistryError("Registry build exceeds the 10000 package limit")
            package_paths.append(package)
    except TypeError:
        raise SkillRegistryError("Skill packages must be an iterable of paths") from None
    existing_identities: dict[Path, tuple[str, str]] = {}
    if existing_registry is not None:
        existing_packages = _existing_static_registry_packages(
            existing_registry,
            remaining_package_capacity=_MAX_BUILD_PACKAGES - len(package_paths),
        )
        existing_identities = {
            archive: (skill_name, skill_version)
            for archive, skill_name, skill_version in existing_packages
        }
        package_paths[:0] = [archive for archive, _, _ in existing_packages]
    if not package_paths:
        raise SkillRegistryError("Provide between 1 and 10000 skill packages")

    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError:
        raise SkillRegistryError("Registry output parent directory is unavailable") from None
    if not parent.is_dir():
        raise SkillRegistryError("Registry output parent must be a directory")
    destination = parent / destination.name
    if not destination.name or destination.exists() or destination.is_symlink():
        raise SkillRegistryError("Registry output must be a new directory path")

    stage = Path(tempfile.mkdtemp(prefix=".gabby-registry-build-", dir=parent))
    website = stage / "website"
    artifact_root = website / "v1" / "skills"
    catalog_entries: dict[str, dict[str, Any]] = {}
    seen_versions: set[tuple[str, str]] = set()
    total_archive_bytes = 0
    package_count = 0
    try:
        for index, package_value in enumerate(package_paths):
            source = Path(package_value).expanduser()
            source_snapshot = stage / f"package-{index}.gabskill"
            try:
                _copy_package_snapshot(source, source_snapshot)
                source_signature = Path(f"{source}.sig")
                signature_bytes = _read_bounded_signature_file(source_signature)
            except (OSError, SkillPackageError):
                raise SkillRegistryError(
                    "A skill package or its detached signature is unavailable or invalid"
                ) from None

            archive_size = source_snapshot.stat().st_size
            total_archive_bytes += archive_size
            if total_archive_bytes > _MAX_BUILD_ARCHIVE_BYTES:
                raise SkillRegistryError("Registry package archives exceed the 1 GiB build limit")

            signature_snapshot = Path(f"{source_snapshot}.sig")
            try:
                with signature_snapshot.open("xb") as signature_file:
                    signature_file.write(signature_bytes)
            except OSError:
                raise SkillRegistryError("Could not stage a skill package signature") from None

            validation_root = stage / f"validation-{index}"
            try:
                installed = install_skill(
                    source_snapshot,
                    validation_root,
                    require_signature=True,
                    trusted_keys=trusted_key_snapshot,
                    signature_path=signature_snapshot,
                )
                skill = load_skill(installed.path / "skill.yaml")
            except (SkillPackageError, ConfigError, OSError):
                raise SkillRegistryError(
                    "A signed skill package failed identity or content validation"
                ) from None
            finally:
                shutil.rmtree(validation_root, ignore_errors=True)

            expected_identity = existing_identities.get(source)
            if expected_identity is not None and expected_identity != (
                installed.name,
                installed.version,
            ):
                raise SkillRegistryError(
                    "Existing registry artifact identity does not match its catalog path"
                )
            identity = (installed.name, installed.version)
            if identity in seen_versions:
                raise SkillRegistryError("Registry build contains a duplicate skill version")
            seen_versions.add(identity)
            description = skill.description
            try:
                if len(description.encode("utf-8")) > _MAX_DESCRIPTION_BYTES:
                    raise SkillRegistryError("Skill description exceeds the registry size limit")
            except UnicodeError:
                raise SkillRegistryError("Skill description is not valid UTF-8") from None

            entry = catalog_entries.setdefault(
                installed.name,
                {
                    "name": installed.name,
                    "description": description,
                    "versions": [],
                    "_description_version": installed.version,
                },
            )
            entry["versions"].append(installed.version)
            # Keep the catalog description aligned with the highest version in the package set.
            if _semver_sort_key(installed.version) > _semver_sort_key(
                entry["_description_version"]
            ):
                entry["description"] = description
                entry["_description_version"] = installed.version

            skill_parts = installed.name.split("/")
            artifact_directory = artifact_root.joinpath(*skill_parts, "versions", installed.version)
            try:
                artifact_directory.mkdir(parents=True, exist_ok=False)
                os.replace(source_snapshot, artifact_directory / "package.gabskill")
                os.replace(signature_snapshot, artifact_directory / "package.gabskill.sig")
            except OSError:
                raise SkillRegistryError("Could not write the static registry artifacts") from None

            # Verify the exact bytes that the static host will serve, not only the input snapshot.
            published_archive = artifact_directory / "package.gabskill"
            published_signature = artifact_directory / "package.gabskill.sig"
            try:
                verified = install_skill(
                    published_archive,
                    stage / f"published-validation-{index}",
                    require_signature=True,
                    trusted_keys=trusted_key_snapshot,
                    signature_path=published_signature,
                )
            except SkillPackageError:
                raise SkillRegistryError(
                    "Staged static registry artifact failed verification"
                ) from None
            finally:
                shutil.rmtree(stage / f"published-validation-{index}", ignore_errors=True)
            if verified.name != installed.name or verified.version != installed.version:
                raise SkillRegistryError("Staged registry artifact identity changed")
            package_count += 1

        skills = []
        for name in sorted(catalog_entries):
            entry = catalog_entries[name]
            entry["versions"].sort(key=_semver_sort_key)
            entry.pop("_description_version", None)
            skills.append(entry)
        catalog_document = {
            "format": _CATALOG_FORMAT,
            "format_version": _CATALOG_VERSION,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "skills": skills,
        }
        catalog_bytes = json.dumps(
            catalog_document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(catalog_bytes) > _MAX_CATALOG_BYTES:
            raise SkillRegistryError("Generated registry catalog exceeds the 4 MiB limit")
        try:
            (website / "v1" / "catalog.json").write_bytes(catalog_bytes)
            # Validate our output with the same strict parser used for remote catalogs.
            _parse_catalog(json.loads(catalog_bytes, object_pairs_hook=_unique_json_object))
            if destination.exists() or destination.is_symlink():
                raise SkillRegistryError("Registry output must be a new directory path")
            os.rename(website, destination)
        except SkillRegistryError:
            raise
        except (OSError, json.JSONDecodeError, UnicodeError):
            raise SkillRegistryError("Could not finalize the static skill registry") from None
        return StaticSkillRegistryBuild(
            path=destination,
            skill_count=len(catalog_entries),
            package_count=package_count,
            catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _existing_static_registry_packages(
    registry: str | Path,
    *,
    remaining_package_capacity: int,
) -> list[tuple[Path, str, str]]:
    """Resolve bounded archive paths referenced by an existing static registry catalog."""
    root = Path(registry).expanduser()
    if root.is_symlink():
        raise SkillRegistryError("Existing skill registry root must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError:
        raise SkillRegistryError("Existing skill registry is unavailable") from None
    if not root.is_dir():
        raise SkillRegistryError("Existing skill registry must be a directory")

    catalog_path = root / "v1" / "catalog.json"
    _ensure_existing_registry_path(root, catalog_path, expect_directory=False)
    try:
        catalog_bytes = _read_bounded_registry_file(catalog_path, limit=_MAX_CATALOG_BYTES)
        catalog_document = json.loads(
            catalog_bytes,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, SkillRegistryError):
        raise SkillRegistryError(
            "Existing skill registry catalog is unavailable or invalid"
        ) from None
    entries = _parse_catalog(catalog_document)
    package_count = sum(len(entry.versions) for entry in entries)
    if package_count > remaining_package_capacity:
        raise SkillRegistryError("Existing registry exceeds the 10000 package build limit")

    packages: list[tuple[Path, str, str]] = []
    for entry in entries:
        for version in entry.versions:
            relative = Path("v1") / "skills" / Path(*entry.name.split("/"))
            artifact_directory = root / relative / "versions" / version
            archive_path = artifact_directory / "package.gabskill"
            signature_path = Path(f"{archive_path}.sig")
            _ensure_existing_registry_path(root, archive_path, expect_directory=False)
            _ensure_existing_registry_path(root, signature_path, expect_directory=False)
            if archive_path.stat().st_size > _MAX_PACKAGE_ARCHIVE_BYTES:
                raise SkillRegistryError("Existing skill package exceeds the archive size limit")
            try:
                _read_bounded_signature_file(signature_path)
            except SkillPackageError:
                raise SkillRegistryError(
                    "Existing skill signature is unavailable or invalid"
                ) from None
            packages.append((archive_path, entry.name, version))
    return packages


def _ensure_existing_registry_path(
    root: Path,
    path: Path,
    *,
    expect_directory: bool,
) -> None:
    """Reject symlinks and special files in paths imported from a static registry."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise SkillRegistryError("Existing skill registry artifact escaped its root") from None
    cursor = root
    parts = relative.parts if expect_directory else relative.parts[:-1]
    for part in parts:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except OSError:
            raise SkillRegistryError("Existing skill registry artifact is unavailable") from None
        if not stat.S_ISDIR(metadata.st_mode):
            raise SkillRegistryError("Existing skill registry path contains a symlink or file")
    if expect_directory:
        return
    try:
        metadata = path.lstat()
    except OSError:
        raise SkillRegistryError("Existing skill registry artifact is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise SkillRegistryError("Existing skill registry artifact must be a regular file")


def _read_bounded_registry_file(path: Path, *, limit: int) -> bytes:
    """Read one unchanged regular file from a registry without following symlinks."""
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise SkillRegistryError("Existing registry file is not a bounded regular file")
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            opened = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or opened.st_size > limit
            ):
                raise SkillRegistryError("Existing registry file changed while it was opened")
            payload = source.read(limit + 1)
    except SkillRegistryError:
        raise
    except OSError:
        raise SkillRegistryError("Existing registry file is unavailable") from None
    if len(payload) > limit:
        raise SkillRegistryError("Existing registry file exceeds its size limit")
    return payload


def _semver_sort_key(version: str) -> tuple[Any, ...]:
    """Return a comparison key following SemVer precedence for validated versions."""
    core_and_build = version.split("+", 1)[0]
    core, separator, prerelease = core_and_build.partition("-")
    major, minor, patch = (int(part) for part in core.split("."))
    if not separator:
        return major, minor, patch, 1, (), version
    identifiers: list[tuple[int, int | str]] = []
    for identifier in prerelease.split("."):
        identifiers.append((0, int(identifier)) if identifier.isdigit() else (1, identifier))
    return major, minor, patch, 0, tuple(identifiers), version


class SkillRegistryClient:
    """Discover and install publisher-signed packages from a static HTTPS registry.

    A registry serves ``/v1/catalog.json`` and package artifacts at fixed paths beneath
    ``/v1/skills/{name}/versions/{version}/``. The catalog is discovery metadata, not a trust
    document: every installed package must have a valid detached signature from a host-provided
    trust map. Artifact URLs are derived from the configured origin so catalog data cannot redirect
    credentials or package requests to another host.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 15.0,
        max_catalog_age_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
    ) -> None:
        normalized_url = _validate_base_url(base_url)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 300
        ):
            raise ValueError("timeout_seconds must be greater than 0 and at most 300")
        if not isinstance(trust_env, bool):
            raise ValueError("trust_env must be a boolean")
        if max_catalog_age_seconds is not None and (
            isinstance(max_catalog_age_seconds, bool)
            or not isinstance(max_catalog_age_seconds, (int, float))
            or not math.isfinite(max_catalog_age_seconds)
            or not 1 <= max_catalog_age_seconds <= _MAX_CATALOG_AGE_SECONDS
        ):
            raise ValueError("max_catalog_age_seconds must be from 1 through 315360000")
        if headers is not None and (
            not isinstance(headers, Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in headers.items()
            )
        ):
            raise ValueError("registry headers must be a string-to-string mapping")
        self.base_url = normalized_url
        self._headers = dict(headers or {})
        self._timeout_seconds = timeout_seconds
        self._max_catalog_age_seconds = max_catalog_age_seconds
        self._transport = transport
        self._trust_env = trust_env
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SkillRegistryClient:
        self._get_client()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the owned HTTP connection pool, if it has been opened."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def catalog(self) -> tuple[SkillCatalogEntry, ...]:
        """Fetch and validate the bounded registry catalog."""
        payload = await self._get_bytes(_CATALOG_PATH, limit=_MAX_CATALOG_BYTES)
        try:
            document = json.loads(payload, object_pairs_hook=_unique_json_object)
        except (UnicodeError, json.JSONDecodeError, SkillRegistryError):
            raise SkillRegistryError("Registry catalog is invalid JSON") from None
        return _parse_catalog(document, max_age_seconds=self._max_catalog_age_seconds)

    async def search(self, query: str) -> tuple[SkillCatalogEntry, ...]:
        """Return catalog entries whose name or description contains the query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        term = query.strip().casefold()
        entries = await self.catalog()
        return tuple(
            entry
            for entry in entries
            if term in entry.name.casefold() or term in entry.description.casefold()
        )

    async def versions(self, name: str) -> tuple[str, ...]:
        """Return available exact SemVer versions for one skill ID."""
        _validate_skill_identity(name, "0.0.0")
        for entry in await self.catalog():
            if entry.name == name:
                return entry.versions
        raise SkillRegistryError("Skill was not found in the configured registry")

    async def install(
        self,
        name: str,
        version: str,
        registry: str | Path,
        *,
        trusted_keys: Mapping[str, bytes],
    ) -> SkillPackageInfo:
        """Fetch one exact package, authenticate it, and atomically install it locally.

        Exact versions are required to avoid mutable ``latest`` aliases. The response package and
        signature are bounded while streaming. A validation install in a private scratch registry
        checks that authenticated package identity matches the requested catalog entry before the
        package can modify the caller's registry.
        """
        _validate_skill_identity(name, version)
        if not isinstance(trusted_keys, Mapping) or not trusted_keys:
            raise SkillRegistryError("Trusted signing keys are required for remote skill installs")
        encoded_name = "/".join(quote(part, safe="") for part in name.split("/"))
        encoded_version = quote(version, safe="")
        package_path = f"/v1/skills/{encoded_name}/versions/{encoded_version}/package.gabskill"
        temporary_directory = Path(tempfile.mkdtemp(prefix=".gabby-registry-"))
        package_path_local = temporary_directory / "package.gabskill"
        signature_path_local = Path(f"{package_path_local}.sig")
        try:
            await self._download(package_path, package_path_local, limit=_MAX_PACKAGE_ARCHIVE_BYTES)
            await self._download(
                f"{package_path}.sig", signature_path_local, limit=_MAX_SIGNATURE_BYTES
            )
            candidate_registry = temporary_directory / "validated"
            try:
                candidate = install_skill(
                    package_path_local,
                    candidate_registry,
                    require_signature=True,
                    trusted_keys=trusted_keys,
                )
            except SkillPackageError as exc:
                raise SkillRegistryError(
                    "Registry package failed signature or package validation"
                ) from exc
            if candidate.name != name or candidate.version != version:
                raise SkillRegistryError(
                    "Registry package identity does not match the requested version"
                )
            try:
                installed = install_skill(
                    package_path_local,
                    registry,
                    require_signature=True,
                    trusted_keys=trusted_keys,
                )
            except SkillPackageError as exc:
                raise SkillRegistryError("Verified skill package could not be installed") from exc
            if (
                installed.name != name
                or installed.version != version
                or installed.sha256 != candidate.sha256
            ):
                shutil.rmtree(installed.path, ignore_errors=True)
                raise SkillRegistryError("Registry package changed during installation")
            return installed
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    async def install_with_dependencies(
        self,
        name: str,
        version: str,
        registry: str | Path,
        *,
        trusted_keys: Mapping[str, bytes],
    ) -> tuple[SkillPackageInfo, ...]:
        """Fetch and install one exact skill plus its pinned dependency closure.

        All catalog references, package signatures, package identities, and dependency edges are
        validated before the local registry is modified. Dependencies are installed before their
        dependents. Each package install is atomic and the operation is retryable: an already
        installed exact version is reused only if its trusted signature and archive digest match.
        A process crash or filesystem failure during the final sequence may leave a valid prefix of
        the dependency closure installed; rerunning the same request safely resumes it.
        """
        _validate_skill_identity(name, version)
        if not isinstance(trusted_keys, Mapping) or not trusted_keys:
            raise SkillRegistryError("Trusted signing keys are required for remote skill installs")
        key_snapshot = dict(trusted_keys)
        if any(
            not isinstance(key_id, str)
            or not isinstance(public_key, bytes)
            or len(public_key) != 32
            for key_id, public_key in key_snapshot.items()
        ):
            raise SkillRegistryError("Trusted signing keys are invalid")

        registry_root = Path(registry).expanduser()
        if registry_root.is_symlink():
            raise SkillRegistryError("Local skill registry must not be a symlink")
        try:
            registry_root = registry_root.resolve(strict=False)
        except OSError:
            raise SkillRegistryError("Local skill registry path is unavailable") from None

        temporary_directory = Path(tempfile.mkdtemp(prefix=".gabby-registry-graph-"))
        plan: list[tuple[str, str, Path, str, str]] = []
        resolved_versions: dict[str, str] = {}
        visiting: set[str] = set()
        total_archive_bytes = 0
        try:
            async with asyncio.timeout(_MAX_DEPENDENCY_RESOLUTION_SECONDS):
                catalog = await self.catalog()
                catalog_by_name = {entry.name: entry for entry in catalog}

                async def resolve(skill_name: str, skill_version: str) -> None:
                    nonlocal total_archive_bytes
                    _validate_skill_identity(skill_name, skill_version)
                    if skill_name in visiting:
                        raise SkillRegistryError("Skill dependency graph contains a cycle")
                    prior_version = resolved_versions.get(skill_name)
                    if prior_version is not None:
                        if prior_version != skill_version:
                            raise SkillRegistryError(
                                "Skill dependency graph requires conflicting versions of one skill"
                            )
                        return
                    entry = catalog_by_name.get(skill_name)
                    if entry is None or skill_version not in entry.versions:
                        raise SkillRegistryError(
                            "A requested skill or dependency version is absent from the catalog"
                        )
                    if len(plan) + len(visiting) >= _MAX_DEPENDENCY_PACKAGES:
                        raise SkillRegistryError("Skill dependency graph exceeds 256 packages")
                    visiting.add(skill_name)
                    resolved_versions[skill_name] = skill_version

                    encoded_name = "/".join(quote(part, safe="") for part in skill_name.split("/"))
                    encoded_version = quote(skill_version, safe="")
                    artifact_path = (
                        f"/v1/skills/{encoded_name}/versions/{encoded_version}/package.gabskill"
                    )
                    index = len(resolved_versions) - 1
                    package_path = temporary_directory / f"package-{index}.gabskill"
                    signature_path = Path(f"{package_path}.sig")
                    await self._download(
                        artifact_path, package_path, limit=_MAX_PACKAGE_ARCHIVE_BYTES
                    )
                    await self._download(
                        f"{artifact_path}.sig", signature_path, limit=_MAX_SIGNATURE_BYTES
                    )
                    total_archive_bytes += package_path.stat().st_size
                    if total_archive_bytes > _MAX_DEPENDENCY_ARCHIVE_BYTES:
                        raise SkillRegistryError(
                            "Skill dependency archives exceed the 1 GiB total limit"
                        )

                    validation_root = temporary_directory / f"validated-{index}"
                    try:
                        candidate = install_skill(
                            package_path,
                            validation_root,
                            require_signature=True,
                            trusted_keys=key_snapshot,
                        )
                        skill = load_skill(candidate.path / "skill.yaml")
                    except (SkillPackageError, ConfigError, OSError):
                        raise SkillRegistryError(
                            "A skill dependency failed signature or package validation"
                        ) from None
                    finally:
                        shutil.rmtree(validation_root, ignore_errors=True)
                    if candidate.name != skill_name or candidate.version != skill_version:
                        raise SkillRegistryError(
                            "A skill package identity does not match its catalog reference"
                        )

                    for dependency in skill.dependencies:
                        if "@" in dependency:
                            dependency_name, _, dependency_version = dependency.rpartition("@")
                        else:
                            raise SkillRegistryError(
                                "Remote dependency fetch requires exact skill-id@version pins"
                            )
                        await resolve(dependency_name, dependency_version)

                    visiting.remove(skill_name)
                    plan.append(
                        (
                            skill_name,
                            skill_version,
                            package_path,
                            candidate.sha256,
                            candidate.signing_key_id or "",
                        )
                    )

                await resolve(name, version)
        except TimeoutError:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise SkillRegistryError(
                "Skill dependency resolution exceeded its five-minute deadline"
            ) from None
        except BaseException:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

        try:
            existing: dict[tuple[str, str], str] = {}
            for skill_name, skill_version, _archive, archive_digest, _key_id in plan:
                target_parent = registry_root
                if target_parent.exists() and not target_parent.is_dir():
                    raise SkillRegistryError("Local skill registry conflicts with a file")
                for part in skill_name.split("/"):
                    target_parent = target_parent / part
                    if target_parent.is_symlink():
                        raise SkillRegistryError("Local skill registry path contains a symlink")
                    if target_parent.exists() and not target_parent.is_dir():
                        raise SkillRegistryError("Local skill registry path conflicts with a file")
                target = target_parent / skill_version
                if target.is_symlink():
                    raise SkillRegistryError("Installed skill version must not be a symlink")
                if not target.exists():
                    continue
                if not target.is_dir():
                    raise SkillRegistryError("Installed skill version path is not a directory")
                try:
                    signing_key_id = verify_installed_skill(
                        target,
                        trusted_keys=key_snapshot,
                        expected_name=skill_name,
                        expected_version=skill_version,
                    )
                    provenance = _read_install_provenance(target / _INSTALL_PROVENANCE)
                except SkillPackageError:
                    raise SkillRegistryError(
                        "An existing skill version is not valid under the supplied trust keys"
                    ) from None
                if provenance["archive_sha256"] != archive_digest:
                    raise SkillRegistryError(
                        "An existing exact skill version has different signed package bytes"
                    )
                existing[(skill_name, skill_version)] = signing_key_id

            installed_packages: list[SkillPackageInfo] = []
            for skill_name, skill_version, package_path, archive_digest, _signing_key_id in plan:
                if (skill_name, skill_version) in existing:
                    destination = registry_root.joinpath(*skill_name.split("/"), skill_version)
                    installed_packages.append(
                        SkillPackageInfo(
                            skill_name,
                            skill_version,
                            destination,
                            archive_digest,
                            existing[(skill_name, skill_version)],
                        )
                    )
                    continue
                try:
                    installed = install_skill(
                        package_path,
                        registry_root,
                        require_signature=True,
                        trusted_keys=key_snapshot,
                    )
                except SkillPackageError:
                    raise SkillRegistryError(
                        "A validated skill package could not be installed; rerun to resume"
                    ) from None
                installed_packages.append(installed)
            return tuple(installed_packages)
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout_seconds),
                follow_redirects=False,
                transport=self._transport,
                trust_env=self._trust_env,
            )
        return self._client

    async def _get_bytes(self, path: str, *, limit: int) -> bytes:
        client = self._get_client()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with client.stream("GET", path) as response:
                    _ensure_success(response)
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > limit:
                            raise SkillRegistryError("Registry response exceeded its size limit")
                        content.extend(chunk)
                    return bytes(content)
        except SkillRegistryError:
            raise
        except TimeoutError:
            raise SkillRegistryError("Registry request exceeded its deadline") from None
        except (httpx.HTTPError, OSError):
            raise SkillRegistryError("Registry request failed") from None

    async def _download(self, path: str, destination: Path, *, limit: int) -> None:
        client = self._get_client()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with client.stream("GET", path) as response:
                    _ensure_success(response)
                    with destination.open("xb") as output:
                        received = 0
                        async for chunk in response.aiter_bytes():
                            received += len(chunk)
                            if received > limit:
                                raise SkillRegistryError(
                                    "Registry artifact exceeded its size limit"
                                )
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
        except SkillRegistryError:
            raise
        except TimeoutError:
            raise SkillRegistryError("Registry artifact download exceeded its deadline") from None
        except (httpx.HTTPError, OSError):
            raise SkillRegistryError("Registry artifact download failed") from None


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 2048:
        raise ValueError("registry base URL must be a string no longer than 2048 bytes")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("registry base URL is invalid") from None
    if (
        parsed.scheme not in {"https", "http"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("registry base URL must be an HTTPS origin without credentials or query")
    if parsed.scheme == "http" and not _is_loopback(host):
        raise ValueError("remote skill registries require HTTPS; HTTP is allowed only on loopback")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("registry base URL port is invalid")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_skill_identity(name: Any, version: Any) -> None:
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError("skill name and exact SemVer version are required")
    try:
        validate_skill_definition(SkillDefinition(name=name, version=version))
    except ConfigError as exc:
        raise ValueError("skill name or exact SemVer version is invalid") from exc


def _parse_catalog(
    value: Any,
    *,
    max_age_seconds: float | None = None,
    now: datetime | None = None,
) -> tuple[SkillCatalogEntry, ...]:
    if not isinstance(value, dict) or set(value) not in (
        {"format", "format_version", "skills"},
        {"format", "format_version", "generated_at", "skills"},
    ):
        raise SkillRegistryError("Registry catalog has an invalid structure")
    if (
        value["format"] != _CATALOG_FORMAT
        or isinstance(value["format_version"], bool)
        or value["format_version"] != _CATALOG_VERSION
    ):
        raise SkillRegistryError("Unsupported registry catalog format")
    generated_at = value.get("generated_at")
    if generated_at is None:
        if max_age_seconds is not None:
            raise SkillRegistryError("Registry catalog has no freshness timestamp")
    else:
        if not isinstance(generated_at, str) or not 19 <= len(generated_at) <= 40:
            raise SkillRegistryError("Registry catalog freshness timestamp is invalid")
        try:
            timestamp = datetime.fromisoformat(
                generated_at[:-1] + "+00:00" if generated_at.endswith("Z") else generated_at
            )
        except ValueError:
            raise SkillRegistryError("Registry catalog freshness timestamp is invalid") from None
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise SkillRegistryError("Registry catalog freshness timestamp must include a timezone")
        current_time = now or datetime.now(UTC)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("now must include a timezone")
        age_seconds = (current_time - timestamp).total_seconds()
        if age_seconds < -_MAX_CATALOG_FUTURE_SKEW_SECONDS:
            raise SkillRegistryError("Registry catalog freshness timestamp is in the future")
        if max_age_seconds is not None and age_seconds > max_age_seconds:
            raise SkillRegistryError("Registry catalog is stale")
    skills = value["skills"]
    if not isinstance(skills, list) or len(skills) > _MAX_SKILLS:
        raise SkillRegistryError("Registry catalog has an invalid skill list")
    entries: list[SkillCatalogEntry] = []
    seen_names: set[str] = set()
    for item in skills:
        if not isinstance(item, dict) or set(item) != {"name", "description", "versions"}:
            raise SkillRegistryError("Registry catalog contains an invalid skill entry")
        name = item["name"]
        description = item["description"]
        versions = item["versions"]
        if not isinstance(name, str) or not isinstance(description, str):
            raise SkillRegistryError("Registry catalog contains an invalid skill entry")
        try:
            if len(description.encode("utf-8")) > _MAX_DESCRIPTION_BYTES:
                raise SkillRegistryError("Registry skill description exceeds its size limit")
            _validate_skill_identity(name, "0.0.0")
        except (UnicodeError, ValueError) as exc:
            raise SkillRegistryError("Registry catalog contains an invalid skill identity") from exc
        if name in seen_names:
            raise SkillRegistryError("Registry catalog contains duplicate skill IDs")
        seen_names.add(name)
        if not isinstance(versions, list) or not 1 <= len(versions) <= _MAX_VERSIONS_PER_SKILL:
            raise SkillRegistryError("Registry skill entry has an invalid version list")
        if any(not isinstance(version, str) for version in versions) or len(set(versions)) != len(
            versions
        ):
            raise SkillRegistryError("Registry skill entry has invalid or duplicate versions")
        try:
            for version in versions:
                _validate_skill_identity(name, version)
        except ValueError as exc:
            raise SkillRegistryError(
                "Registry skill entry has invalid or duplicate versions"
            ) from exc
        entries.append(SkillCatalogEntry(name, description, tuple(versions)))
    return tuple(entries)


def _ensure_success(response: httpx.Response) -> None:
    if response.is_redirect:
        raise SkillRegistryError("Registry redirects are not allowed")
    if not 200 <= response.status_code < 300:
        raise SkillRegistryError(f"Registry returned HTTP status {response.status_code}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillRegistryError("Registry catalog contains duplicate JSON keys")
        result[key] = value
    return result
