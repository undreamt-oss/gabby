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
"""Use one host-owned Redis authority to share skill revocations across Gabby processes.

Install ``redis>=5`` in the host environment. Gabby itself has no Redis dependency; it consumes
the async client's ``execute_command`` method through ``RedisSkillRevocationStore``.
"""

from __future__ import annotations

import asyncio
import os

from redis.asyncio import Redis

from gabby import RedisSkillRevocationStore, SkillRevokedError


async def main() -> None:
    redis = Redis.from_url(
        os.environ["GABBY_REDIS_URL"],
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )
    try:
        revocations = RedisSkillRevocationStore(
            redis,
            key=os.environ.get("GABBY_SKILL_REVOCATION_KEY", "production:skill-revocations"),
        )
        await revocations.revoke("compromised-publisher")
        try:
            await revocations.check_not_revoked(frozenset({"compromised-publisher"}))
        except SkillRevokedError:
            print("The publisher is revoked across every process using this Redis set.")
        print(sorted(await revocations.revoked_key_ids()))
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
