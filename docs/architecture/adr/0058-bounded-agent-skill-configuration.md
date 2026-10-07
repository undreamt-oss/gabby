# ADR 0058: Bounded agent and skill configuration text

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-01.

## Context

Agent and skill loaders read YAML and file-backed skill instructions and examples into memory
without byte limits. Model-request bounds reject an oversized prompt only after these files have
already been read and resolved. Programmatic definitions also bypassed any text-size limit, so
file-loaded and in-memory construction had different resource boundaries.

## Decision

- Read agent YAML through a bounded binary read, with a 10 MiB maximum, before UTF-8 decoding or
  YAML parsing.
- Read `skill.yaml` manifests through a bounded binary read, with a 1 MiB maximum matching the
  portable skill-package manifest limit.
- Read file-backed skill instruction and example resources through a bounded binary read, with a
  10 MiB maximum per file matching the portable skill-package file limit.
- Apply the same UTF-8 byte limits to the description/instruction fields on `AgentDefinition` and
  the description/instruction/example fields on `SkillDefinition`.
- Limit an agent to 256 distinct resolved skills, counting transitive dependencies.
- Bound combined agent description/instructions, global instructions, and resolved skill
  descriptions/instructions/examples by the smaller of `policies.max_model_request_bytes` and a
  16 MiB hard ceiling. Enforce the budget incrementally while resolving dependencies.
- Reject oversized or invalid UTF-8 input with `ConfigError`; do not truncate authored content.

## Consequences

Configuration and skill resources have deterministic per-file bounds before parsing, and skill
composition has a bounded entry count and aggregate text budget. Oversized definitions fail during
load or construction instead of later during model request serialization. Hosts that need larger
reference material should put it in a bounded knowledge store. These limits do not bound memory
inside custom parsers, host-provided extensions, or the host-owned registry itself.

## Alternatives considered

- **Check file size with `stat()` and then call `read_text()`:** rejected because the file can grow
  between the size check and the read.
- **Rely on provider request limits:** rejected because the provider limit runs after configuration
  and skill text have already been allocated and assembled.
- **Truncate oversized instructions:** rejected because silent truncation changes agent behavior;
  construction should fail clearly.

## Compatibility and evidence

This pre-1.0 tightening applies equally to YAML-loaded and programmatic definitions. Config and
agent tests cover oversized YAML, manifests, file-backed text, Unicode byte counting, matching
in-memory validation, aggregate skill text, and transitive skill-count limits. Skill package limits
align with the existing 1 MiB manifest and 10 MiB per-file caps.
