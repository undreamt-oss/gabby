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
"""Contract coverage for the host-client Redis skill revocation adapter."""

from __future__ import annotations

from typing import Any

import pytest

from gabby import RedisSkillRevocationStore, SkillRevocationUnavailable, SkillRevokedError


class FakeRedis:
    def __init__(self) -> None:
        self.values: set[str] = set()
        self.commands: list[tuple[str, ...]] = []
        self.failure: Exception | None = None
        self.invalid_response: Any = None

    async def execute_command(self, *args: str) -> Any:
        self.commands.append(args)
        if self.failure is not None:
            raise self.failure
        command, *parameters = args
        if self.invalid_response is not None:
            return self.invalid_response
        if command == "SMISMEMBER":
            return [int(key_id in self.values) for key_id in parameters[1:]]
        if command == "SADD":
            before = len(self.values)
            self.values.add(parameters[1])
            return int(len(self.values) != before)
        if command == "SREM":
            before = len(self.values)
            self.values.discard(parameters[1])
            return int(len(self.values) != before)
        if command == "SSCAN":
            _, cursor, _, count = parameters
            ordered = sorted(self.values)
            start = int(cursor)
            batch = ordered[start : start + int(count)]
            next_cursor = start + len(batch)
            if next_cursor >= len(ordered):
                next_cursor = 0
            return next_cursor, [key_id.encode("ascii") for key_id in batch]
        raise AssertionError(f"unexpected Redis command: {command}")


@pytest.mark.asyncio
async def test_redis_skill_revocation_store_is_shared_and_idempotent() -> None:
    redis = FakeRedis()
    writer = RedisSkillRevocationStore(redis)
    reader = RedisSkillRevocationStore(redis)

    await writer.revoke("publisher-a")
    await writer.revoke("publisher-a")
    assert await reader.revoked_key_ids() == frozenset({"publisher-a"})
    with pytest.raises(SkillRevokedError):
        await reader.check_not_revoked(frozenset({"publisher-a", "publisher-b"}))
    assert await reader.reinstate("publisher-a") is True
    assert await writer.reinstate("publisher-a") is False
    await reader.check_not_revoked(frozenset({"publisher-a"}))

    commands = [entry[0] for entry in redis.commands]
    assert commands.count("SADD") == 2
    assert "SMISMEMBER" in commands
    assert "SSCAN" in commands


@pytest.mark.asyncio
async def test_redis_skill_revocation_listing_is_bounded() -> None:
    redis = FakeRedis()
    redis.values.update({"publisher-a", "publisher-b"})
    store = RedisSkillRevocationStore(redis, max_entries=1, scan_batch_size=1)

    with pytest.raises(SkillRevocationUnavailable, match="listing is unavailable"):
        await store.revoked_key_ids()


@pytest.mark.asyncio
async def test_redis_skill_revocation_failures_are_sanitized() -> None:
    redis = FakeRedis()
    redis.failure = RuntimeError("redis://user:secret@host/private")
    store = RedisSkillRevocationStore(redis)

    with pytest.raises(SkillRevocationUnavailable) as caught:
        await store.check_not_revoked(frozenset({"publisher-a"}))
    assert "secret" not in str(caught.value)

    with pytest.raises(SkillRevocationUnavailable):
        await store.revoke("publisher-a")
    with pytest.raises(SkillRevocationUnavailable):
        await store.reinstate("publisher-a")
    with pytest.raises(SkillRevocationUnavailable):
        await store.revoked_key_ids()


@pytest.mark.asyncio
async def test_redis_skill_revocation_rejects_malformed_backend_responses() -> None:
    redis = FakeRedis()
    redis.invalid_response = [2]
    store = RedisSkillRevocationStore(redis)

    with pytest.raises(SkillRevocationUnavailable):
        await store.check_not_revoked(frozenset({"publisher-a"}))
    with pytest.raises(SkillRevocationUnavailable):
        await store.revoke("publisher-a")

    with pytest.raises(SkillRevocationUnavailable):
        await store.reinstate("publisher-a")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"key": ""}, "visible ASCII"),
        ({"max_entries": 0}, "max_entries"),
        ({"max_entries": True}, "max_entries"),
        ({"scan_batch_size": 0}, "scan_batch_size"),
        ({"scan_batch_size": 10_001}, "scan_batch_size"),
    ],
)
def test_redis_skill_revocation_configuration_is_validated(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RedisSkillRevocationStore(FakeRedis(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_redis_skill_revocation_rejects_invalid_key_ids() -> None:
    store = RedisSkillRevocationStore(FakeRedis())
    with pytest.raises(ValueError, match="invalid format"):
        await store.revoke("../publisher")
    with pytest.raises(ValueError, match="at most 1024"):
        await store.check_not_revoked(frozenset(f"publisher-{index}" for index in range(1025)))


@pytest.mark.asyncio
async def test_redis_skill_revocation_empty_check_is_a_noop() -> None:
    redis = FakeRedis()
    await RedisSkillRevocationStore(redis).check_not_revoked(frozenset())
    assert redis.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scan_response",
    [
        [],
        ("invalid-cursor", []),
        (0, "not-a-list"),
        (0, [b"\xff"]),
        (0, ["../invalid-key"]),
        (0, [1]),
    ],
)
async def test_redis_skill_revocation_listing_rejects_malformed_scan_data(
    scan_response: Any,
) -> None:
    redis = FakeRedis()
    redis.invalid_response = scan_response
    with pytest.raises(SkillRevocationUnavailable):
        await RedisSkillRevocationStore(redis, max_entries=10).revoked_key_ids()


@pytest.mark.asyncio
async def test_redis_skill_revocation_listing_rejects_unbounded_and_repeated_scans() -> None:
    class OversizedRedis(FakeRedis):
        async def execute_command(self, *args: str) -> Any:
            self.commands.append(args)
            return 0, [b"publisher-a", b"publisher-b"]

    class RepeatingRedis(FakeRedis):
        async def execute_command(self, *args: str) -> Any:
            self.commands.append(args)
            return 1, []

    for client in (OversizedRedis(), RepeatingRedis()):
        with pytest.raises(SkillRevocationUnavailable, match="listing is unavailable"):
            await RedisSkillRevocationStore(
                client, max_entries=1, scan_batch_size=1
            ).revoked_key_ids()


def test_redis_skill_revocation_requires_an_async_command_client() -> None:
    with pytest.raises(TypeError, match="execute_command"):
        RedisSkillRevocationStore(object())  # type: ignore[arg-type]
