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
"""Host-owned skill signature admission policy."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .config import ConfigError, SkillDefinition
from .skill_packages import SkillPackageError, verify_installed_skill

_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_MAX_TRUSTED_SKILL_KEYS = 256


class SkillRevokedError(PermissionError):
    """At least one publisher used by this agent has been revoked."""


class SkillRevocationUnavailable(RuntimeError):
    """The configured revocation service could not make a reliable decision."""


class SkillIntegrityError(PermissionError):
    """A trusted installed skill no longer matches its verified package signature."""


class SkillRevocationChecker(Protocol):
    """Async host-owned check for revoked signers before each agent execution."""

    async def check_not_revoked(self, key_ids: frozenset[str]) -> None:
        """Return only when all signer IDs are permitted; raise on revoke or backend failure."""


@runtime_checkable
class SkillRevocationStore(SkillRevocationChecker, Protocol):
    """Host-owned management contract for durable skill signer revocations.

    A shared implementation may back multiple Gabby processes. It must define its own write
    durability and read-after-write consistency; `Agent` uses only
    :class:`SkillRevocationChecker` and never assumes a particular storage topology.
    """

    async def revoke(self, key_id: str) -> None:
        """Persist a signer revocation for subsequent checks."""

    async def reinstate(self, key_id: str) -> bool:
        """Remove a signer revocation and return whether one existed."""

    async def revoked_key_ids(self) -> frozenset[str]:
        """Return the complete revocation set for bounded operator inspection."""


def load_skill_trust_keys(directory: str | Path) -> dict[str, bytes]:
    """Load raw Ed25519 public keys from a host-managed ``KEY_ID.pub`` directory.

    The directory is a deployment-owned trust source, separate from agent configuration. It may
    contain at most 256 regular key files; symlinks, subdirectories, and unexpected files fail
    closed. Each public key file must contain exactly 32 bytes.
    """
    root = Path(directory).expanduser()
    if root.is_symlink():
        raise ConfigError("Skill trust directory must not be a symlink")
    try:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ConfigError("Skill trust path must be a directory")
    except OSError:
        raise ConfigError("Skill trust directory is unavailable") from None

    keys: dict[str, bytes] = {}
    try:
        with os.scandir(root) as entries:
            for index, entry in enumerate(entries, start=1):
                if index > _MAX_TRUSTED_SKILL_KEYS:
                    raise ConfigError("Skill trust directory exceeds the 256-key limit")
                if entry.is_symlink():
                    raise ConfigError("Skill trust directory entries must not be symlinks")
                if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".pub"):
                    raise ConfigError("Skill trust directory may contain only KEY_ID.pub files")
                key_id = entry.name[:-4]
                if _KEY_ID.fullmatch(key_id) is None:
                    raise ConfigError("Skill trust key filenames must use KEY_ID.pub syntax")
                path = root / entry.name
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 32:
                    raise ConfigError(f"Trusted Ed25519 key {key_id!r} must be a 32-byte file")
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
                        or opened_metadata.st_dev != metadata.st_dev
                        or opened_metadata.st_ino != metadata.st_ino
                    ):
                        raise ConfigError(f"Trusted Ed25519 key {key_id!r} changed while loading")
                    key = handle.read(33)
                if len(key) != 32:
                    raise ConfigError(f"Trusted Ed25519 key {key_id!r} must contain 32 raw bytes")
                keys[key_id] = key
    except ConfigError:
        raise
    except OSError:
        raise ConfigError("Skill trust directory or key file is unavailable") from None
    if not keys:
        raise ConfigError("Skill trust directory must contain at least one KEY_ID.pub file")
    return keys


@dataclass(frozen=True)
class SkillTrustPolicy:
    """Require trusted v2 publisher signatures for every resolved filesystem skill.

    This policy is supplied by the host, not agent YAML. Its public keys and revocation set are
    snapshotted at construction, so edits to a caller-owned mapping cannot change an existing
    agent's trust decision. Local unsigned skills remain available when no policy is supplied.
    """

    trusted_keys: Mapping[str, bytes]
    revoked_key_ids: frozenset[str] = frozenset()

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        revoked_key_ids: frozenset[str] = frozenset(),
    ) -> SkillTrustPolicy:
        """Construct a policy from host-managed ``KEY_ID.pub`` files."""
        return cls(load_skill_trust_keys(directory), revoked_key_ids)

    def __post_init__(self) -> None:
        if not isinstance(self.trusted_keys, Mapping) or not self.trusted_keys:
            raise ConfigError("skill trust policy requires at least one trusted public key")
        keys = dict(self.trusted_keys)
        if any(
            not isinstance(key_id, str)
            or _KEY_ID.fullmatch(key_id) is None
            or not isinstance(key, bytes)
            or len(key) != 32
            for key_id, key in keys.items()
        ):
            raise ConfigError("skill trust keys must map valid key IDs to raw 32-byte public keys")
        if isinstance(self.revoked_key_ids, (str, bytes)):
            raise ConfigError("revoked skill key IDs must be a collection of IDs")
        try:
            revoked = frozenset(self.revoked_key_ids)
        except TypeError:
            raise ConfigError("revoked skill key IDs must be an iterable of IDs") from None
        if any(
            not isinstance(key_id, str) or _KEY_ID.fullmatch(key_id) is None for key_id in revoked
        ):
            raise ConfigError("revoked skill key IDs use an invalid format")
        object.__setattr__(self, "trusted_keys", MappingProxyType(keys))
        object.__setattr__(self, "revoked_key_ids", revoked)

    def verify(self, skill: SkillDefinition) -> str:
        """Verify the on-disk package that supplied a resolved skill definition."""
        if skill.source_path is None:
            raise SkillPackageError(
                f"Skill {skill.name!r} has no installed package provenance under the trust policy"
            )
        return verify_installed_skill(
            skill.source_path.parent,
            trusted_keys=self.trusted_keys,
            revoked_key_ids=self.revoked_key_ids,
            expected_name=skill.name,
            expected_version=skill.version,
        )
