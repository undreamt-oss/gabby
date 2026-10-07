# ADR 0101: Redis skill revocation adapter

## Status

Accepted

## Context

Gabby defines `SkillRevocationStore` so multi-instance deployments can check and manage publisher
revocations through one authority. The standard-library SQLite implementation is intended for one
host and local filesystems with SQLite locking semantics. Hosts can supply their own shared store,
but there was no concrete adapter demonstrating the protocol against a common shared service.

## Decision

Provide `RedisSkillRevocationStore` as an optional adapter that accepts a host-owned async Redis
client through the minimal `AsyncRedisCommands.execute_command` protocol. Use Redis 6.2+ `SMISMEMBER`
for fresh signer checks, `SADD`/`SREM` for management, and bounded `SSCAN` for operator listings.
Do not add a Redis runtime dependency to the core package or construct, configure, or close the
client in Gabby. Validate key IDs and backend responses, cap the number of IDs per check and the
management listing size, and convert backend failures to sanitized `SkillRevocationUnavailable`
errors so agent executions fail closed.

## Consequences

- Applications can use a shared Redis set from multiple Gabby processes without writing a basic
  storage adapter.
- Redis credentials, TLS, client timeouts, persistence, backups, namespacing, replica routing, and
  read-after-write consistency remain host responsibilities.
- Checks are not cached. Active execution revocation latency is bounded by each agent's poll
  interval and the Redis client's response time, subject to the host's routing consistency.
- The adapter distributes signer revocations only. Trusted public-key provisioning and rotation
  remain separate deployment configuration.
- Redis 6.2 or newer is required for `SMISMEMBER`; other compatible Redis services can be injected
  by the host if they satisfy the same command and consistency contract.

## References

- [`RedisSkillRevocationStore` extension contract](../../EXTENSION_CONTRACTS.md)
- [Skill revocation operations](../../OPERATIONS.md#enforce-skill-publisher-revocations)
- [Redis command reference: `SMISMEMBER`](https://redis.io/docs/latest/commands/smismember/)
