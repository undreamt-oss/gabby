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
"""Bounded, checksummed local skill package creation and installation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

from .config import _SEMVER_PATTERN, ConfigError, load_skill

_PACKAGE_MANIFEST = ".gabby-skill.json"
_INSTALL_PROVENANCE = ".gabby-install.json"
_INSTALL_PROVENANCE_FORMAT = "gabby-skill-install"
_MAX_INSTALL_PROVENANCE_BYTES = 16 * 1024
_MAX_REGISTRY_AUDIT_ENTRIES = 100_000
_INSTALL_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_PACKAGE_FORMAT = "gabby-skill-package"
_PACKAGE_FORMAT_VERSION = 1
_MAX_PACKAGE_FILES = 1000
_MAX_PACKAGE_TREE_ENTRIES = 10_000
_MAX_PACKAGE_FILE_BYTES = 10 * 1024 * 1024
_MAX_PACKAGE_TOTAL_BYTES = 100 * 1024 * 1024
_MAX_PACKAGE_ARCHIVE_BYTES = 100 * 1024 * 1024
_MAX_PACKAGE_MANIFEST_BYTES = 1024 * 1024
_COPY_CHUNK_BYTES = 64 * 1024


class SkillPackageError(ValueError):
    """A local skill package is malformed, unsafe, or cannot be installed."""


@dataclass(frozen=True)
class SkillPackageInfo:
    """Identity and content digest for a skill package operation."""

    name: str
    version: str
    path: Path
    sha256: str
    signing_key_id: str | None = None


@dataclass(frozen=True)
class SkillPackageFile:
    """One validated file entry inside a portable skill package."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SkillPackageInspection:
    """Validated package metadata suitable for review without installing the skill."""

    name: str
    version: str
    description: str
    tools: tuple[str, ...]
    knowledge: tuple[str, ...]
    dependencies: tuple[str, ...]
    verification: tuple[str, ...]
    files: tuple[SkillPackageFile, ...]
    archive_sha256: str
    signature_present: bool
    signature_verified: bool
    signing_key_id: str | None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class SkillRegistryAuditRecord:
    """Best-effort installation provenance for one skill in a local registry.

    The metadata is useful for inventory, but a writable registry owner can alter it. It is not
    proof of current package integrity or a runtime authorization decision.
    """

    name: str
    version: str
    path: Path
    archive_sha256: str | None
    signing_key_id: str | None
    signature_verified: bool | None
    status: str


def inspect_skill_package(
    package: str | Path,
    *,
    trusted_keys: Mapping[str, bytes] | None = None,
    require_signature: bool = False,
) -> SkillPackageInspection:
    """Validate and describe a skill package without installing it into a caller registry.

    Skill instructions and examples are intentionally omitted from the result. The package archive
    is copied to private temporary storage, fully checked against its checksummed manifest, and
    extracted only into a temporary directory. If a detached signature is present and trusted keys
    are supplied, it is verified before metadata is returned.
    """
    if not isinstance(require_signature, bool):
        raise SkillPackageError("require_signature must be a boolean")
    if trusted_keys is not None and not isinstance(trusted_keys, Mapping):
        raise SkillPackageError("trusted_keys must be a key-to-public-key mapping")
    if require_signature and (not isinstance(trusted_keys, Mapping) or not trusted_keys):
        raise SkillPackageError("Trusted signing keys are required for signature verification")

    temporary_directory = Path(tempfile.mkdtemp(prefix=".gabby-inspect-"))
    archive_snapshot = temporary_directory / "package.gabskill"
    signature_snapshot = Path(f"{archive_snapshot}.sig")
    extracted = temporary_directory / "contents"
    try:
        from .skill_signing import _read_bounded_signature_file

        source = Path(package).expanduser()
        _copy_package_snapshot(source, archive_snapshot)
        source_signature = Path(f"{source}.sig")
        signature_bytes: bytes | None
        try:
            signature_bytes = _read_bounded_signature_file(source_signature)
        except FileNotFoundError:
            signature_bytes = None
        except OSError:
            raise SkillPackageError("Skill signature file is unavailable") from None
        except SkillPackageError:
            if source_signature.is_symlink() or source_signature.exists():
                raise
            signature_bytes = None

        if signature_bytes is None and require_signature:
            raise SkillPackageError("Skill package signature is required")
        if signature_bytes is not None:
            with signature_snapshot.open("xb") as signature_file:
                signature_file.write(signature_bytes)

        signing_key_id: str | None = None
        if trusted_keys is not None and signature_bytes is not None:
            from .skill_signing import verify_skill_package_signature

            signing_key_id = verify_skill_package_signature(
                archive_snapshot,
                trusted_keys=trusted_keys,
                signature_path=signature_snapshot,
            )
        elif require_signature:
            raise SkillPackageError("Skill package signature could not be verified")

        manifest = _extract_and_verify(archive_snapshot, extracted)
        skill = load_skill(extracted / "skill.yaml")
        declared_identity = manifest["skill"]
        if skill.name != declared_identity["name"] or skill.version != declared_identity["version"]:
            raise SkillPackageError("Package identity does not match skill.yaml")
        files = tuple(
            SkillPackageFile(item["path"], item["size"], item["sha256"])
            for item in sorted(manifest["files"], key=lambda record: record["path"])
        )
        return SkillPackageInspection(
            name=skill.name,
            version=skill.version,
            description=skill.description,
            tools=tuple(skill.tools),
            knowledge=tuple(skill.knowledge),
            dependencies=tuple(skill.dependencies),
            verification=tuple(skill.verification),
            files=files,
            archive_sha256=_hash_file(archive_snapshot),
            signature_present=signature_bytes is not None,
            signature_verified=signing_key_id is not None,
            signing_key_id=signing_key_id,
            input_schema=skill.input_schema,
            output_schema=skill.output_schema,
        )
    except SkillPackageError:
        raise
    except (ConfigError, OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise SkillPackageError("Skill package could not be inspected") from exc
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def pack_skill(source: str | Path, output: str | Path) -> SkillPackageInfo:
    """Create a deterministic checksummed archive from one validated skill directory.

    Checksums detect corruption but do not authenticate the publisher. Detached signatures can be
    added and required during installation through the optional skill-signing feature. Skill
    instructions and included resources remain untrusted content and must be reviewed before use.
    """
    root = Path(source).expanduser()
    if root.is_symlink():
        raise SkillPackageError("Skill package source must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise SkillPackageError("Skill package source directory is unavailable") from exc
    if not root.is_dir():
        raise SkillPackageError("Skill package source must be a directory")
    manifest_path = root / "skill.yaml"
    try:
        skill = load_skill(manifest_path)
    except (OSError, ConfigError) as exc:
        raise SkillPackageError("Skill package must contain a valid skill.yaml") from exc

    _validate_skill_package_name(skill.name)
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    try:
        if destination.resolve(strict=False).is_relative_to(root):
            raise SkillPackageError("Output archive must be outside the skill source directory")
    except OSError as exc:
        raise SkillPackageError("Output archive path is unavailable") from exc
    if not destination.parent.is_dir():
        raise SkillPackageError("Output directory must already exist")

    files = _collect_source_files(root)
    records: list[dict[str, Any]] = []
    payloads: list[tuple[str, Path, str]] = []
    for relative, path, size in files:
        digest = _hash_file(path)
        records.append({"path": relative, "size": size, "sha256": digest})
        payloads.append((relative, path, digest))
    package_manifest = {
        "format": _PACKAGE_FORMAT,
        "format_version": _PACKAGE_FORMAT_VERSION,
        "skill": {"name": skill.name, "version": skill.version},
        "files": records,
    }
    encoded_manifest = json.dumps(
        package_manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded_manifest) > _MAX_PACKAGE_MANIFEST_BYTES:
        raise SkillPackageError("Skill package manifest exceeds the 1 MiB limit")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".gabby-skill-", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(
                temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                _write_zip_entry(archive, _PACKAGE_MANIFEST, encoded_manifest)
                for relative, path, digest in payloads:
                    _write_source_entry(archive, relative, path, digest)
        archive_size = temporary_path.stat().st_size
        if archive_size > _MAX_PACKAGE_ARCHIVE_BYTES:
            raise SkillPackageError("Skill package archive exceeds the 100 MiB limit")
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise SkillPackageError(f"Output archive already exists: {destination}") from exc
    except SkillPackageError:
        raise
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise SkillPackageError("Could not create skill package archive") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    digest = _hash_file(destination)
    return SkillPackageInfo(skill.name, skill.version, destination, digest)


def install_skill(
    package: str | Path,
    registry: str | Path,
    *,
    require_signature: bool = False,
    trusted_keys: Mapping[str, bytes] | None = None,
    signature_path: str | Path | None = None,
) -> SkillPackageInfo:
    """Validate and atomically install a package, optionally requiring publisher authentication."""
    archive_path = Path(package).expanduser()
    try:
        archive_path = archive_path.resolve(strict=True)
    except OSError as exc:
        raise SkillPackageError("Skill package archive is unavailable") from exc
    if not isinstance(require_signature, bool):
        raise SkillPackageError("require_signature must be a boolean")
    snapshot_directory = Path(tempfile.mkdtemp(prefix=".gabby-package-"))
    try:
        snapshot = snapshot_directory / "package.gabskill"
        _copy_package_snapshot(archive_path, snapshot)
        signing_key_id: str | None = None
        signature_document: dict[str, Any] | None = None
        if require_signature or trusted_keys is not None or signature_path is not None:
            if trusted_keys is None:
                raise SkillPackageError("Trusted signing keys are required to verify this package")
            from .skill_signing import _verify_skill_package_signature_details

            # The verifier hashes the private snapshot but reads the detached sidecar from
            # the caller's package path by default. Resolving the sidecar relative to the
            # snapshot would look for a nonexistent ``package.gabskill.sig`` file.
            source_signature = (
                Path(signature_path).expanduser()
                if signature_path is not None
                else Path(f"{archive_path}.sig")
            )
            signing_key_id, signature_document = _verify_skill_package_signature_details(
                snapshot,
                trusted_keys=trusted_keys,
                signature_path=source_signature,
            )
        return _install_skill_snapshot(snapshot, registry, signing_key_id, signature_document)
    finally:
        shutil.rmtree(snapshot_directory, ignore_errors=True)


def uninstall_skill(registry: str | Path, name: str, version: str) -> None:
    """Remove one exact versioned skill after validating its path and declared identity."""
    try:
        _validate_skill_package_name(name)
    except SkillPackageError as exc:
        raise SkillPackageError("Skill name is invalid") from exc
    if not isinstance(version, str) or not _SEMVER_PATTERN.fullmatch(version):
        raise SkillPackageError("Skill version must be an exact semantic version")

    root = Path(registry).expanduser()
    if root.is_symlink():
        raise SkillPackageError("Skill registry must not be a symlink")
    try:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise SkillPackageError("Skill registry must be a directory")
    except OSError:
        raise SkillPackageError("Skill registry is unavailable") from None

    target = root
    for component in (*name.split("/"), version):
        target = target / component
        if target.is_symlink():
            raise SkillPackageError("Installed skill path must not contain symlinks")
    try:
        resolved_target = target.resolve(strict=True)
        if not resolved_target.is_relative_to(root) or not resolved_target.is_dir():
            raise SkillPackageError("Installed skill path is invalid")
        skill_path = resolved_target / "skill.yaml"
        if skill_path.is_symlink() or not skill_path.is_file():
            raise SkillPackageError("Installed skill manifest is unavailable")
        skill = load_skill(skill_path)
        if skill.name != name or skill.version != version:
            raise SkillPackageError("Installed skill identity does not match the requested path")
    except SkillPackageError:
        raise
    except (ConfigError, OSError):
        raise SkillPackageError("Installed skill could not be validated") from None
    try:
        shutil.rmtree(resolved_target)
    except OSError:
        raise SkillPackageError("Could not remove the selected skill version") from None


def _copy_package_snapshot(source: Path, destination: Path) -> None:
    """Copy one bounded archive from a single open file into a private temporary directory."""
    try:
        # The preliminary check gives useful errors for directories and prevents opening
        # special files. Non-blocking open plus fstat closes the path-swap gap for FIFOs.
        metadata = source.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_PACKAGE_ARCHIVE_BYTES:
            raise SkillPackageError("Skill package archive is not a bounded regular file")
        descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0),
        )
        with os.fdopen(descriptor, "rb") as input_file:
            metadata = os.fstat(input_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_PACKAGE_ARCHIVE_BYTES:
                raise SkillPackageError("Skill package archive is not a bounded regular file")
            copied = 0
            with destination.open("xb") as output_file:
                while chunk := input_file.read(_COPY_CHUNK_BYTES):
                    copied += len(chunk)
                    if copied > _MAX_PACKAGE_ARCHIVE_BYTES:
                        raise SkillPackageError("Skill package archive exceeds the 100 MiB limit")
                    output_file.write(chunk)
    except SkillPackageError:
        raise
    except OSError:
        raise SkillPackageError("Skill package archive is unavailable") from None


def _install_skill_snapshot(
    archive_path: Path,
    registry: str | Path,
    signing_key_id: str | None,
    signature_document: dict[str, Any] | None,
) -> SkillPackageInfo:
    """Extract a verified private archive snapshot into the versioned local registry."""
    root = Path(registry).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
    except OSError as exc:
        raise SkillPackageError("Skill registry directory is unavailable") from exc
    if not root.is_dir():
        raise SkillPackageError("Skill registry must be a directory")

    staging = Path(tempfile.mkdtemp(prefix=".gabby-install-", dir=root))
    try:
        manifest = _extract_and_verify(archive_path, staging)
        try:
            skill = load_skill(staging / "skill.yaml")
        except (OSError, ConfigError) as exc:
            raise SkillPackageError("Packaged skill.yaml failed Gabby validation") from exc
        expected_skill = manifest["skill"]
        if skill.name != expected_skill["name"] or skill.version != expected_skill["version"]:
            raise SkillPackageError("Package identity does not match skill.yaml")

        target_parent = root
        for part in skill.name.split("/"):
            target_parent = target_parent / part
            if target_parent.is_symlink():
                raise SkillPackageError("Skill registry path contains a symlink")
            target_parent.mkdir(exist_ok=True)
            if not target_parent.is_dir():
                raise SkillPackageError("Skill registry path conflicts with a non-directory")
        target = target_parent / skill.version
        if target.exists() or target.is_symlink():
            raise SkillPackageError(f"Skill {skill.name}@{skill.version} is already installed")
        digest = _hash_file(archive_path)
        _write_install_provenance(staging, digest, signing_key_id, signature_document)
        os.rename(staging, target)
        return SkillPackageInfo(skill.name, skill.version, target, digest, signing_key_id)
    except SkillPackageError:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeError, json.JSONDecodeError) as exc:
        raise SkillPackageError("Could not install skill package") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_install_provenance(
    installation: Path,
    archive_sha256: str,
    signing_key_id: str | None,
    signature_document: dict[str, Any] | None,
) -> None:
    """Write informational signer metadata before atomically publishing an installation."""
    document = {
        "format": _INSTALL_PROVENANCE_FORMAT,
        "format_version": 2,
        "archive_sha256": archive_sha256,
        "signing_key_id": signing_key_id,
        "signature_verified": signing_key_id is not None,
        "signature": signature_document,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(encoded) > _MAX_INSTALL_PROVENANCE_BYTES:
        raise SkillPackageError("Skill install provenance exceeds its size limit")
    destination = installation / _INSTALL_PROVENANCE
    try:
        with destination.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except OSError:
        raise SkillPackageError("Could not write skill install provenance") from None


def audit_skill_registry(
    registry: str | Path,
    *,
    revoked_key_ids: Iterable[str] = (),
    trusted_keys: Mapping[str, bytes] | None = None,
) -> tuple[SkillRegistryAuditRecord, ...]:
    """Inventory skills and verify signed installed content against host trust and revocation.

    V1 package signatures remain identifiable as install-time-only evidence. V2 signatures bind
    the package manifest and can be checked against the files currently installed. Directories
    without metadata, including older installations, are reported as ``unknown``.
    """
    revoked = frozenset(revoked_key_ids)
    if any(
        not isinstance(key_id, str) or not _INSTALL_KEY_ID.fullmatch(key_id) for key_id in revoked
    ):
        raise SkillPackageError("Revoked skill key IDs must use the supported key-ID format")
    root = Path(registry).expanduser()
    if root.is_symlink():
        raise SkillPackageError("Skill registry audit root must not be a symlink")
    try:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise SkillPackageError("Skill registry audit root must be a directory")
    except OSError:
        raise SkillPackageError("Skill registry audit root is unavailable") from None

    records: list[SkillRegistryAuditRecord] = []
    pending = [root]
    inspected = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as scan:
                entries = []
                for entry in scan:
                    inspected += 1
                    if inspected > _MAX_REGISTRY_AUDIT_ENTRIES:
                        raise SkillPackageError("Skill registry exceeds the audit entry limit")
                    entries.append(entry)
        except OSError:
            records.append(_invalid_audit_record(root, directory))
            continue
        entries.sort(key=lambda entry: entry.name)

        has_skill_manifest = False
        for entry in entries:
            path = Path(entry.path)
            if entry.name == "skill.yaml":
                has_skill_manifest = not entry.is_symlink() and entry.is_file(follow_symlinks=False)
            if entry.is_symlink():
                records.append(_invalid_audit_record(root, path))
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)

        if has_skill_manifest:
            record = _audit_install_directory(root, directory, revoked, trusted_keys)
            if record is not None:
                records.append(record)

    return tuple(
        sorted(records, key=lambda record: (record.name, record.version, str(record.path)))
    )


def verify_installed_skill(
    directory: str | Path,
    *,
    trusted_keys: Mapping[str, bytes],
    revoked_key_ids: Iterable[str] = (),
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> str:
    """Verify a v2 signed skill's current files and return its trusted signer ID.

    This is suitable for host admission checks during agent construction. Trust keys and revoked
    IDs must come from host-owned configuration. Legacy and unsigned skill directories are rejected.
    """
    if not isinstance(trusted_keys, Mapping) or not trusted_keys:
        raise SkillPackageError("At least one trusted skill signing key is required")
    revoked = frozenset(revoked_key_ids)
    if any(
        not isinstance(key_id, str) or not _INSTALL_KEY_ID.fullmatch(key_id) for key_id in revoked
    ):
        raise SkillPackageError("Revoked skill key IDs must use the supported key-ID format")
    path = Path(directory).expanduser()
    if path.is_symlink():
        raise SkillPackageError("Installed skill directory must not be a symlink")
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise SkillPackageError("Installed skill path must be a directory")
        skill = load_skill(path / "skill.yaml")
        provenance = _read_install_provenance(path / _INSTALL_PROVENANCE)
    except SkillPackageError:
        raise
    except (ConfigError, OSError, UnicodeError, json.JSONDecodeError):
        raise SkillPackageError("Installed skill could not be validated") from None
    if (expected_name is not None and skill.name != expected_name) or (
        expected_version is not None and skill.version != expected_version
    ):
        raise SkillPackageError("Installed skill identity does not match its resolved definition")
    if provenance["signing_key_id"] is None or not provenance["signature_verified"]:
        raise SkillPackageError("Installed skill has no verified publisher signature")
    signature = provenance["signature"]
    key_id = provenance["signing_key_id"]
    if signature is None or signature["format_version"] != 2:
        raise SkillPackageError("Installed skill requires an auditable v2 publisher signature")
    if key_id in revoked:
        raise SkillPackageError("Installed skill signer is revoked")
    if key_id not in trusted_keys:
        raise SkillPackageError("Installed skill signer is not trusted")
    if signature["content_sha256"] != _installed_content_digest(path, skill.name, skill.version):
        raise SkillPackageError("Installed skill content does not match its signed manifest")
    from .skill_signing import _verify_content_signature

    _verify_content_signature(signature, trusted_keys)
    return cast(str, key_id)


def _audit_install_directory(
    root: Path,
    directory: Path,
    revoked_key_ids: frozenset[str],
    trusted_keys: Mapping[str, bytes] | None,
) -> SkillRegistryAuditRecord | None:
    relative = directory.relative_to(root)
    inferred_name = "/".join(relative.parts[:-1]) if len(relative.parts) > 1 else ""
    inferred_version = relative.parts[-1] if relative.parts else ""
    try:
        skill = load_skill(directory / "skill.yaml")
        relative_name = "/".join(relative.parts)
        if relative_name == skill.name:
            return SkillRegistryAuditRecord(
                skill.name, skill.version, directory, None, None, None, "unknown"
            )
        if skill.name != inferred_name or skill.version != inferred_version:
            # Ignore nested resource files that happen to be named skill.yaml.
            return None
        provenance_path = directory / _INSTALL_PROVENANCE
        if provenance_path.is_symlink():
            raise SkillPackageError("Skill install provenance must not be a symlink")
        if not provenance_path.exists():
            return SkillRegistryAuditRecord(
                skill.name, skill.version, directory, None, None, None, "unknown"
            )
        provenance = _read_install_provenance(provenance_path)
        key_id = provenance["signing_key_id"]
        signature = provenance["signature"]
        if key_id in revoked_key_ids:
            status = "revoked"
        elif provenance["signature_verified"]:
            if signature is None or signature["format_version"] == 1 or trusted_keys is None:
                status = "recorded-signed"
            elif key_id not in trusted_keys:
                status = "unknown"
            else:
                try:
                    verify_installed_skill(
                        directory,
                        trusted_keys=trusted_keys,
                        expected_name=skill.name,
                        expected_version=skill.version,
                    )
                except SkillPackageError:
                    status = "invalid"
                else:
                    status = "verified"
        else:
            status = "unsigned"
        return SkillRegistryAuditRecord(
            skill.name,
            skill.version,
            directory,
            provenance["archive_sha256"],
            provenance["signing_key_id"],
            provenance["signature_verified"],
            status,
        )
    except (
        ConfigError,
        OSError,
        SkillPackageError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ):
        return SkillRegistryAuditRecord(
            inferred_name or relative.as_posix(),
            inferred_version,
            directory,
            None,
            None,
            None,
            "invalid",
        )


def _read_install_provenance(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise SkillPackageError("Skill install provenance must not be a symlink")
    try:
        with path.open("rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > (
                _MAX_INSTALL_PROVENANCE_BYTES
            ):
                raise SkillPackageError("Skill install provenance is not a bounded regular file")
            encoded = source.read(_MAX_INSTALL_PROVENANCE_BYTES + 1)
    except SkillPackageError:
        raise
    except OSError:
        raise SkillPackageError("Skill install provenance is unavailable") from None
    if len(encoded) > _MAX_INSTALL_PROVENANCE_BYTES:
        raise SkillPackageError("Skill install provenance exceeds its size limit")
    value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict) or set(value) not in (
        {
            "format",
            "format_version",
            "archive_sha256",
            "signing_key_id",
            "signature_verified",
        },
        {
            "format",
            "format_version",
            "archive_sha256",
            "signing_key_id",
            "signature_verified",
            "signature",
        },
    ):
        raise SkillPackageError("Skill install provenance has an invalid structure")
    digest = value["archive_sha256"]
    key_id = value["signing_key_id"]
    verified = value["signature_verified"]
    if (
        value["format"] != _INSTALL_PROVENANCE_FORMAT
        or isinstance(value["format_version"], bool)
        or value["format_version"] not in (1, 2)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or (
            key_id is not None
            and (not isinstance(key_id, str) or not _INSTALL_KEY_ID.fullmatch(key_id))
        )
        or not isinstance(verified, bool)
        or verified != (key_id is not None)
    ):
        raise SkillPackageError("Skill install provenance is invalid")
    if value["format_version"] == 1:
        value["signature"] = None
    elif not _valid_install_signature(value["signature"], key_id) or (
        value["signature"] is not None and value["signature"]["archive_sha256"] != digest
    ):
        raise SkillPackageError("Skill install signature evidence is invalid")
    return value


def _valid_install_signature(value: Any, key_id: str | None) -> bool:
    if value is None:
        return key_id is None
    v1_fields = {"format", "format_version", "key_id", "archive_sha256", "signature"}
    v2_fields = v1_fields | {"content_sha256"}
    if not isinstance(value, dict) or value.get("format") != "gabby-skill-signature":
        return False
    if type(value.get("format_version")) is int and value.get("format_version") == 1:
        return (
            set(value) == v1_fields
            and value.get("key_id") == key_id
            and _is_sha256(value.get("archive_sha256"))
            and isinstance(value.get("signature"), str)
            and len(value["signature"]) == 88
        )
    return (
        set(value) == v2_fields
        and type(value.get("format_version")) is int
        and value.get("format_version") == 2
        and value.get("key_id") == key_id
        and _is_sha256(value.get("archive_sha256"))
        and _is_sha256(value.get("content_sha256"))
        and isinstance(value.get("signature"), str)
        and len(value["signature"]) == 88
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _installed_content_digest(directory: Path, name: str, version: str) -> str:
    files = _collect_source_files(directory, allow_install_provenance=True)
    manifest = {
        "format": _PACKAGE_FORMAT,
        "format_version": _PACKAGE_FORMAT_VERSION,
        "skill": {"name": name, "version": version},
        "files": [
            {"path": relative, "size": size, "sha256": _hash_file(path)}
            for relative, path, size in files
        ],
    }
    encoded = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid_audit_record(root: Path, path: Path) -> SkillRegistryAuditRecord:
    relative = path.relative_to(root)
    return SkillRegistryAuditRecord(
        "/".join(relative.parts[:-1]) or relative.as_posix(),
        relative.parts[-1] if relative.parts else "",
        path,
        None,
        None,
        None,
        "invalid",
    )


def _collect_source_files(
    root: Path, *, allow_install_provenance: bool = False
) -> list[tuple[str, Path, int]]:
    files: list[tuple[str, Path, int]] = []
    total_bytes = 0
    try:
        paths: list[Path] = []
        for path in root.rglob("*"):
            paths.append(path)
            if len(paths) > _MAX_PACKAGE_TREE_ENTRIES:
                raise SkillPackageError("Skill package directory tree exceeds 10000 entries")
        paths.sort()
    except OSError as exc:
        raise SkillPackageError("Could not read skill package source") from exc
    for path in paths:
        if path.is_symlink():
            raise SkillPackageError("Skill package source must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SkillPackageError("Skill package source must contain only regular files")
        relative = path.relative_to(root).as_posix()
        _validate_archive_path(relative)
        if relative == _INSTALL_PROVENANCE and allow_install_provenance:
            continue
        if relative in {_PACKAGE_MANIFEST, _INSTALL_PROVENANCE}:
            raise SkillPackageError(f"Skill source uses reserved file name {relative}")
        size = path.stat().st_size
        if size > _MAX_PACKAGE_FILE_BYTES:
            raise SkillPackageError("Skill package files are limited to 10 MiB each")
        total_bytes += size
        if total_bytes > _MAX_PACKAGE_TOTAL_BYTES:
            raise SkillPackageError("Skill package contents exceed the 100 MiB total limit")
        files.append((relative, path, size))
        if len(files) > _MAX_PACKAGE_FILES:
            raise SkillPackageError("Skill packages are limited to 1000 files")
    if not any(relative == "skill.yaml" for relative, _, _ in files):
        raise SkillPackageError("Skill package source must contain skill.yaml")
    _validate_portable_path_collisions(
        [_PACKAGE_MANIFEST, _INSTALL_PROVENANCE, *(relative for relative, _, _ in files)]
    )
    return files


def _extract_and_verify(archive_path: Path, staging: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        entries = archive.infolist()
        if not 1 <= len(entries) <= _MAX_PACKAGE_FILES + 1:
            raise SkillPackageError("Skill package has an invalid file count")
        names: set[str] = set()
        entry_map: dict[str, zipfile.ZipInfo] = {}
        total_size = 0
        for info in entries:
            name = info.filename
            _validate_archive_path(name)
            if name == _INSTALL_PROVENANCE:
                raise SkillPackageError("Skill package contains a reserved install-provenance path")
            if name in names:
                raise SkillPackageError("Skill package contains duplicate paths")
            names.add(name)
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if info.is_dir() or stat.S_ISLNK(mode) or kind not in (0, stat.S_IFREG):
                raise SkillPackageError("Skill packages may contain regular files only")
            if info.flag_bits & 0x1:
                raise SkillPackageError("Encrypted skill packages are not supported")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise SkillPackageError("Skill package uses an unsupported compression method")
            if info.file_size > _MAX_PACKAGE_FILE_BYTES:
                raise SkillPackageError("Skill package files are limited to 10 MiB each")
            total_size += info.file_size
            if total_size > _MAX_PACKAGE_TOTAL_BYTES:
                raise SkillPackageError("Skill package contents exceed the 100 MiB total limit")
            entry_map[name] = info

        _validate_portable_path_collisions([*names, _INSTALL_PROVENANCE])

        manifest_info = entry_map.get(_PACKAGE_MANIFEST)
        if manifest_info is None or manifest_info.file_size > _MAX_PACKAGE_MANIFEST_BYTES:
            raise SkillPackageError("Skill package manifest is missing or oversized")
        try:
            manifest_data = json.loads(
                archive.read(manifest_info).decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise SkillPackageError("Skill package manifest is invalid") from exc
        manifest = _validate_package_manifest(manifest_data)
        listed_files = {record["path"] for record in manifest["files"]}
        if listed_files != names - {_PACKAGE_MANIFEST}:
            raise SkillPackageError("Package manifest does not match archive contents")

        for record in manifest["files"]:
            name = record["path"]
            info = entry_map[name]
            if info.file_size != record["size"]:
                raise SkillPackageError(f"Package file size mismatch: {name}")
            target = staging.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    written += len(chunk)
                    if written > _MAX_PACKAGE_FILE_BYTES or written > record["size"]:
                        raise SkillPackageError(f"Package file exceeded its declared size: {name}")
                    digest.update(chunk)
                    output.write(chunk)
            if written != record["size"] or digest.hexdigest() != record["sha256"]:
                raise SkillPackageError(f"Package checksum mismatch: {name}")
    return manifest


def _validate_package_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "format",
        "format_version",
        "skill",
        "files",
    }:
        raise SkillPackageError("Skill package manifest has an invalid structure")
    if (
        value["format"] != _PACKAGE_FORMAT
        or isinstance(value["format_version"], bool)
        or value["format_version"] != _PACKAGE_FORMAT_VERSION
    ):
        raise SkillPackageError("Unsupported skill package format")
    skill = value["skill"]
    if not isinstance(skill, dict) or set(skill) != {"name", "version"}:
        raise SkillPackageError("Skill package identity is invalid")
    if not isinstance(skill["name"], str) or not isinstance(skill["version"], str):
        raise SkillPackageError("Skill package identity is invalid")
    _validate_skill_package_name(skill["name"])
    files = value["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= _MAX_PACKAGE_FILES:
        raise SkillPackageError("Skill package manifest has an invalid file list")
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise SkillPackageError("Skill package file record is invalid")
        path = record["path"]
        _validate_archive_path(path)
        if path == _PACKAGE_MANIFEST or path in seen:
            raise SkillPackageError("Skill package manifest contains duplicate or reserved paths")
        seen.add(path)
        if (
            isinstance(record["size"], bool)
            or not isinstance(record["size"], int)
            or not 0 <= record["size"] <= _MAX_PACKAGE_FILE_BYTES
        ):
            raise SkillPackageError("Skill package file size is invalid")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise SkillPackageError("Skill package file checksum is invalid")
    if "skill.yaml" not in seen:
        raise SkillPackageError("Skill package manifest omits skill.yaml")
    return value


def _validate_archive_path(name: Any) -> None:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise SkillPackageError("Skill package contains an invalid file path")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows_path.drive
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != name
    ):
        raise SkillPackageError("Skill package file paths must be safe relative paths")
    _validate_portable_path_parts(path.parts)


def _validate_skill_package_name(name: str) -> None:
    _validate_archive_path(f"{name}/package-version/skill.yaml")


def _validate_portable_path_parts(parts: tuple[str, ...]) -> None:
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    for part in parts:
        if (
            part.endswith((".", " "))
            or any(character in '<>|?*"' for character in part)
            or part.split(".", 1)[0].upper() in reserved
        ):
            raise SkillPackageError("Skill package paths must be portable across host platforms")


def _validate_portable_path_collisions(names: Iterable[str]) -> None:
    """Reject case-folded and Unicode-normalized path collisions across host filesystems."""
    path_kinds: dict[tuple[str, ...], bool] = {}
    for name in names:
        parts = PurePosixPath(name).parts
        normalized = tuple(unicodedata.normalize("NFC", part).casefold() for part in parts)
        for index in range(1, len(normalized) + 1):
            key = normalized[:index]
            is_directory = index < len(normalized)
            existing_kind = path_kinds.get(key)
            if key in path_kinds and (existing_kind != is_directory or not is_directory):
                raise SkillPackageError(
                    "Skill package paths collide on case-insensitive or "
                    "Unicode-normalizing filesystems"
                )
            path_kinds[key] = is_directory


def _write_zip_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillPackageError("Skill package manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _write_source_entry(
    archive: zipfile.ZipFile,
    name: str,
    path: Path,
    expected_digest: str,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    digest = hashlib.sha256()
    written = 0
    try:
        with path.open("rb") as source, archive.open(info, mode="w") as output:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > _MAX_PACKAGE_FILE_BYTES:
                    raise SkillPackageError("Skill package files are limited to 10 MiB each")
                digest.update(chunk)
                output.write(chunk)
    except OSError as exc:
        raise SkillPackageError(
            "Skill source changed or became unavailable during packaging"
        ) from exc
    if digest.hexdigest() != expected_digest:
        raise SkillPackageError("Skill source changed during packaging; package creation aborted")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
