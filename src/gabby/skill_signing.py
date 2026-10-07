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
"""Detached Ed25519 signatures for portable Gabby skill packages."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .skill_packages import (
    _MAX_PACKAGE_ARCHIVE_BYTES,
    _PACKAGE_MANIFEST,
    SkillPackageError,
    _hash_file,
    _validate_package_manifest,
)

_SIGNATURE_FORMAT = "gabby-skill-signature"
_SIGNATURE_VERSION = 2
_SIGNATURE_DOMAIN_V1 = b"gabby-skill-package-signature-v1\0"
_SIGNATURE_DOMAIN_V2 = b"gabby-skill-package-signature-v2\0"
_MAX_SIGNATURE_BYTES = 16 * 1024
_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)


def sign_skill_package(
    package: str | Path,
    *,
    key_id: str,
    private_key: bytes,
    signature_path: str | Path | None = None,
) -> Path:
    """Sign an archive digest using a raw Ed25519 private key and write a detached sidecar.

    Cryptographic support is optional. Install ``gabby-agent-runtime[skill-signing]`` to use it.
    The private key is used in memory only and is never serialized into the signature document.
    """
    _validate_key_id(key_id)
    if not isinstance(private_key, bytes) or len(private_key) != 32:
        raise SkillPackageError("Ed25519 private keys must contain exactly 32 raw bytes")
    archive_path = _bounded_package_path(package)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        raise SkillPackageError(
            "Install gabby-agent-runtime[skill-signing] to sign skill packages"
        ) from None
    try:
        signer = Ed25519PrivateKey.from_private_bytes(private_key)
        digest = _hash_file(archive_path)
        content_digest = _package_content_digest(archive_path)
        signature = signer.sign(_signature_message(key_id, digest, content_digest))
    except (OSError, ValueError):
        raise SkillPackageError("Could not sign skill package") from None

    document = {
        "format": _SIGNATURE_FORMAT,
        "format_version": _SIGNATURE_VERSION,
        "key_id": key_id,
        "archive_sha256": digest,
        "content_sha256": content_digest,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if len(encoded) > _MAX_SIGNATURE_BYTES:
        raise SkillPackageError("Skill signature document exceeds its size limit")

    destination = (
        Path(signature_path).expanduser()
        if signature_path is not None
        else Path(f"{archive_path}.sig")
    )
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    if destination.exists() or destination.is_symlink():
        raise SkillPackageError("Skill signature output already exists")
    if not destination.parent.is_dir():
        raise SkillPackageError("Skill signature output directory must already exist")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".gabby-signature-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, destination)
    except FileExistsError as exc:
        raise SkillPackageError("Skill signature output already exists") from exc
    except OSError:
        raise SkillPackageError("Could not write skill signature") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def verify_skill_package_signature(
    package: str | Path,
    *,
    trusted_keys: Mapping[str, bytes],
    signature_path: str | Path | None = None,
) -> str:
    """Verify a detached signature against a host-owned key-ID-to-public-key trust map.

    Returns the trusted key ID after verification. The signature and package are untrusted input;
    trusted public keys must come from a host-managed trust source.
    """
    key_id, _ = _verify_skill_package_signature_details(
        package,
        trusted_keys=trusted_keys,
        signature_path=signature_path,
    )
    return key_id


def _verify_skill_package_signature_details(
    package: str | Path,
    *,
    trusted_keys: Mapping[str, bytes],
    signature_path: str | Path | None = None,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(trusted_keys, Mapping) or not trusted_keys:
        raise SkillPackageError("At least one trusted skill signing key is required")
    archive_path = _bounded_package_path(package)
    signature_file = (
        Path(signature_path).expanduser()
        if signature_path is not None
        else Path(f"{archive_path}.sig")
    )
    try:
        encoded = _read_bounded_signature_file(signature_file)
    except SkillPackageError:
        raise
    except OSError:
        raise SkillPackageError("Skill signature file is unavailable") from None
    document = _parse_signature(encoded)
    key_id = _validate_key_id(document["key_id"])
    public_key = trusted_keys.get(key_id)
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise SkillPackageError("Skill signature key is not trusted")
    try:
        digest = _hash_file(archive_path)
    except OSError:
        raise SkillPackageError("Skill package archive became unavailable") from None
    if document["archive_sha256"] != digest:
        raise SkillPackageError("Skill signature does not match the package digest")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise SkillPackageError(
            "Install gabby-agent-runtime[skill-signing] to verify skill signatures"
        ) from None
    try:
        verifier = Ed25519PublicKey.from_public_bytes(public_key)
        if document["format_version"] == 2:
            content_digest = _package_content_digest(archive_path)
            if document["content_sha256"] != content_digest:
                raise SkillPackageError("Skill signature does not match the package content")
            message = _signature_message(key_id, digest, content_digest)
        else:
            message = _signature_message(key_id, digest)
        verifier.verify(base64.b64decode(document["signature"], validate=True), message)
    except SkillPackageError:
        raise
    except (InvalidSignature, ValueError):
        raise SkillPackageError("Skill package signature verification failed") from None
    return key_id, document


def _read_bounded_signature_file(path: Path) -> bytes:
    """Open and read one unchanged regular sidecar without trusting path metadata alone."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise SkillPackageError("Skill signature file must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_SIGNATURE_BYTES:
        raise SkillPackageError("Skill signature file is not a bounded regular file")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened_metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or opened_metadata.st_size > _MAX_SIGNATURE_BYTES
            or opened_metadata.st_dev != metadata.st_dev
            or opened_metadata.st_ino != metadata.st_ino
        ):
            raise SkillPackageError("Skill signature file changed while it was opened")
        encoded = handle.read(_MAX_SIGNATURE_BYTES + 1)
    if len(encoded) > _MAX_SIGNATURE_BYTES:
        raise SkillPackageError("Skill signature file exceeds its size limit")
    return encoded


def _bounded_package_path(package: str | Path) -> Path:
    try:
        path = Path(package).expanduser().resolve(strict=True)
        metadata = path.stat()
    except OSError:
        raise SkillPackageError("Skill package archive is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_PACKAGE_ARCHIVE_BYTES:
        raise SkillPackageError("Skill package archive is not a bounded regular file")
    return path


def _parse_signature(encoded: bytes) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("ascii"), object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError):
        raise SkillPackageError("Skill signature document is invalid") from None
    v1_fields = {
        "format",
        "format_version",
        "key_id",
        "archive_sha256",
        "signature",
    }
    v2_fields = v1_fields | {"content_sha256"}
    if not isinstance(value, dict) or set(value) not in (v1_fields, v2_fields):
        raise SkillPackageError("Skill signature document has an invalid structure")
    if (
        value["format"] != _SIGNATURE_FORMAT
        or isinstance(value["format_version"], bool)
        or value["format_version"] not in (1, 2)
        or (value["format_version"] == 1 and set(value) != v1_fields)
        or (value["format_version"] == 2 and set(value) != v2_fields)
    ):
        raise SkillPackageError("Unsupported skill signature format")
    key_id = _validate_key_id(value["key_id"])
    value["key_id"] = key_id
    digest = value["archive_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise SkillPackageError("Skill signature package digest is invalid")
    if value["format_version"] == 2:
        content_digest = value["content_sha256"]
        if (
            not isinstance(content_digest, str)
            or len(content_digest) != 64
            or any(char not in "0123456789abcdef" for char in content_digest)
        ):
            raise SkillPackageError("Skill signature content digest is invalid")
    signature = value["signature"]
    if not isinstance(signature, str) or len(signature) != 88:
        raise SkillPackageError("Skill signature value is invalid")
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (ValueError, binascii.Error):
        raise SkillPackageError("Skill signature value is invalid") from None
    if len(decoded) != 64:
        raise SkillPackageError("Skill signature value is invalid")
    return value


def _validate_key_id(value: Any) -> str:
    if not isinstance(value, str) or _KEY_ID.fullmatch(value) is None:
        raise SkillPackageError(
            "Skill signing key IDs must be 1–128 ASCII letters, digits, '.', '_' or '-'"
        )
    return value


def _signature_message(key_id: str, digest: str, content_digest: str | None = None) -> bytes:
    if content_digest is None:
        return _SIGNATURE_DOMAIN_V1 + key_id.encode("ascii") + b"\0" + bytes.fromhex(digest)
    return (
        _SIGNATURE_DOMAIN_V2
        + key_id.encode("ascii")
        + b"\0"
        + bytes.fromhex(digest)
        + bytes.fromhex(content_digest)
    )


def _package_content_digest(archive_path: Path) -> str:
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest = next(
                (entry for entry in archive.infolist() if entry.filename == _PACKAGE_MANIFEST),
                None,
            )
            if manifest is None or manifest.file_size > 1024 * 1024:
                raise SkillPackageError("Skill package manifest is missing or oversized")
            value = _validate_package_manifest(
                json.loads(
                    archive.read(manifest).decode("utf-8"), object_pairs_hook=_unique_json_object
                )
            )
    except SkillPackageError:
        raise
    except (OSError, zipfile.BadZipFile, UnicodeError, json.JSONDecodeError):
        raise SkillPackageError("Skill package manifest is invalid") from None
    value["files"] = sorted(value["files"], key=lambda record: record["path"])
    canonical = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verify_content_signature(
    document: Mapping[str, Any], trusted_keys: Mapping[str, bytes]
) -> None:
    """Verify persisted v2 evidence after the installed tree digest has been checked."""
    key_id = _validate_key_id(document.get("key_id"))
    public_key = trusted_keys.get(key_id)
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise SkillPackageError("Skill signature key is not trusted")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(public_key).verify(
            base64.b64decode(document["signature"], validate=True),
            _signature_message(key_id, document["archive_sha256"], document["content_sha256"]),
        )
    except ImportError:
        raise SkillPackageError(
            "Install gabby-agent-runtime[skill-signing] to verify skill signatures"
        ) from None
    except (InvalidSignature, ValueError, binascii.Error):
        raise SkillPackageError("Skill package signature verification failed") from None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillPackageError("Skill signature contains duplicate JSON keys")
        result[key] = value
    return result
