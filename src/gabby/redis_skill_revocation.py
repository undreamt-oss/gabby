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
"""Optional Redis-backed shared skill publisher revocation store."""

from __future__ import annotations

import re
from collections.abc import Awaitable
from typing import Any, Protocol

from .skill_trust import (
    SkillRevocationStore,
    SkillRevocationUnavailable,
    SkillRevokedError,
)

_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_MAX_CHECK_KEY_IDS = 1024
_MAX_ENTRY_LIMIT = 1_000_000
_MAX_SCAN_BATCHES = 1024


class AsyncRedisCommands(Protocol):
    """Minimal async Redis command surface used by the revocation adapter."""

    def execute_command(self, *args: str) -> Awaitable[Any]:
        """Execute a Redis command and return its awaitable response."""


class RedisSkillRevocationStore(SkillRevocationStore):
    """Shared revocation store over a host-owned async Redis client.

    Redis client construction, authentication, TLS, pool sizing, lifecycle, persistence, and
    replica-routing policy remain host-owned. Reads use ``SMISMEMBER`` against the configured
    client's target, so deployments must route checks to an authority that meets their required
    read-after-write consistency. Redis 6.2 or newer is required.
    """

    def __init__(
        self,
        client: AsyncRedisCommands,
        *,
        key: str = "gabby:skill-revocations",
        max_entries: int = 100_000,
        scan_batch_size: int = 256,
    ) -> None:
        if not callable(getattr(client, "execute_command", None)):
            raise TypeError("client must provide execute_command()")
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 256
            or any(ord(char) < 0x21 or ord(char) > 0x7E for char in key)
        ):
            raise ValueError("key must contain 1 to 256 visible ASCII characters")
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= _MAX_ENTRY_LIMIT
        ):
            raise ValueError("max_entries must be from 1 through 1000000")
        if (
            isinstance(scan_batch_size, bool)
            or not isinstance(scan_batch_size, int)
            or not 1 <= scan_batch_size <= 10_000
        ):
            raise ValueError("scan_batch_size must be from 1 through 10000")
        self._client = client
        self._key = key
        self._max_entries = max_entries
        self._scan_batch_size = scan_batch_size

    async def check_not_revoked(self, key_ids: frozenset[str]) -> None:
        """Fail closed when any requested signer is present in the shared Redis set."""
        keys = _validate_key_ids(key_ids)
        if not keys:
            return
        ordered = sorted(keys)
        try:
            response = await self._execute("SMISMEMBER", self._key, *ordered)
            if (
                not isinstance(response, (list, tuple))
                or len(response) != len(ordered)
                or any(not _is_redis_boolean(value) for value in response)
            ):
                raise ValueError("invalid SMISMEMBER response")
        except Exception:
            raise SkillRevocationUnavailable("Skill revocation check is unavailable") from None
        if any(_redis_boolean(value) for value in response):
            raise SkillRevokedError("A skill publisher has been revoked")

    async def revoke(self, key_id: str) -> None:
        """Add a signer ID to the shared revocation set idempotently."""
        _validate_key_id(key_id)
        try:
            result = await self._execute("SADD", self._key, key_id)
        except Exception:
            raise SkillRevocationUnavailable("Skill revocation update is unavailable") from None
        if not _is_redis_boolean(result):
            raise SkillRevocationUnavailable("Skill revocation update is unavailable")

    async def reinstate(self, key_id: str) -> bool:
        """Remove one signer from the shared revocation set."""
        _validate_key_id(key_id)
        try:
            result = await self._execute("SREM", self._key, key_id)
        except Exception:
            raise SkillRevocationUnavailable("Skill revocation update is unavailable") from None
        if not _is_redis_boolean(result):
            raise SkillRevocationUnavailable("Skill revocation update is unavailable")
        return _redis_boolean(result)

    async def revoked_key_ids(self) -> frozenset[str]:
        """List revoked key IDs through bounded ``SSCAN`` batches."""
        cursor = 0
        revoked: set[str] = set()
        scanned = 0
        scans = 0
        max_scans = min(
            _MAX_SCAN_BATCHES,
            max(
                32,
                ((self._max_entries + self._scan_batch_size - 1) // self._scan_batch_size) * 4 + 16,
            ),
        )
        try:
            while True:
                scans += 1
                if scans > max_scans:
                    raise ValueError("revocation scan exceeded its batch bound")
                response = await self._execute(
                    "SSCAN",
                    self._key,
                    str(cursor),
                    "COUNT",
                    str(self._scan_batch_size),
                )
                if not isinstance(response, (list, tuple)) or len(response) != 2:
                    raise ValueError("invalid SSCAN response")
                cursor = _decode_cursor(response[0])
                members = response[1]
                if not isinstance(members, (list, tuple)):
                    raise ValueError("invalid SSCAN members")
                scanned += len(members)
                if scanned > self._max_entries * 2:
                    raise ValueError("revocation set exceeds scan bound")
                for member in members:
                    key_id = _decode_key_id(member)
                    revoked.add(key_id)
                    if len(revoked) > self._max_entries:
                        raise ValueError("revocation set exceeds configured entry bound")
                if cursor == 0:
                    return frozenset(revoked)
        except Exception:
            raise SkillRevocationUnavailable("Skill revocation listing is unavailable") from None

    async def _execute(self, *args: str) -> Any:
        """Await the host client's command result without assuming client ownership."""
        return await self._client.execute_command(*args)


def _validate_key_ids(value: frozenset[str]) -> frozenset[str]:
    if not isinstance(value, (set, frozenset)) or len(value) > _MAX_CHECK_KEY_IDS:
        raise ValueError("skill signer checks must contain at most 1024 key IDs")
    for key_id in value:
        _validate_key_id(key_id)
    return value


def _validate_key_id(value: object) -> None:
    if not isinstance(value, str) or _KEY_ID.fullmatch(value) is None:
        raise ValueError("skill signer key IDs use an invalid format")


def _is_redis_boolean(value: object) -> bool:
    return value in (0, 1, False, True, b"0", b"1")


def _redis_boolean(value: object) -> bool:
    if value in (1, True, b"1"):
        return True
    if value in (0, False, b"0"):
        return False
    raise ValueError("invalid Redis boolean response")


def _decode_cursor(value: object) -> int:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("invalid Redis scan cursor") from None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError("invalid Redis scan cursor")


def _decode_key_id(value: object) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("invalid Redis revocation member") from None
    if not isinstance(value, str):
        raise ValueError("invalid Redis revocation member")
    _validate_key_id(value)
    return value
