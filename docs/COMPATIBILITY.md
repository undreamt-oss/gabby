# Compatibility and deprecation policy

This document records the compatibility contract Gabby intends to ship with v1.0. It takes effect
when the maintainers publish v1.0 and identify this contract in that release's notes. While the
package version is `0.x`, changes follow [the current extension policy](EXTENSIONS.md) and may be
incompatible.

## Proposed v1.0 supported surface

The v1.0 release will support these contracts:

- **Python construction and execution:** `Agent`, `AgentDefinition`, `ResolvedAgentDefinition`,
  `SkillDefinition`, `ResolvedSkillDefinition`, `RunRequest`, `ExecutionResult`, `AgentStreamEvent`,
  `ExecutionTrace`, `TraceEvent`, and `AgentTool` for explicit stateless agent composition.
  `Agent.arun`, `Agent.astream`, `Agent.run`, `Agent.aclose`, and `Agent.close` retain the documented
  statelessness, cancellation, and ownership behavior.
- **Python extension contracts:** `ModelProvider`, `StreamingModelProvider`, `Tool`, `ToolRegistry`,
  `AsyncRedisCommands`, `RedisSkillRevocationStore`, `AsyncPostgresExecutor`, `PostgresApprovalAudit`,
  `AsyncPostgresPool`, `PostgresGenerationManifestStore`, `PostgresKnowledgeStore`,
  `PostgresStreamJournal`, `PostgresVectorStore`,
  `ToolContext`, `ToolError`, `ToolErrorCode`, `CancellationToken`, `Environment`, `FileParser`,
  `ParsedPage`, `MarkupTextParser`, `TOMLTextParser`, `YAMLTextParser`, `ParquetTextParser`, `VCardTextParser`, `Retriever`,
  `KnowledgeStore`, `EmbeddingProvider`, `AsymmetricEmbeddingProvider`, `VectorStore`, `Reranker`, `ICalendarTextParser`,
  `MboxTextParser`, `SkillSelector`, `Planner`,
  `Verifier`, `Tracer`, `Authenticator`, `ApprovalHandler`, `ApprovalAuditSink`, and `EngineAdapter`, plus the
  content-minimizing `SQLiteApprovalAudit`, shared-backend `PostgresApprovalAudit`,
  `AuditedApprovalHandler`, `ApprovalAuditRecord`, and
  sanitized `ApprovalAuditError`, plus their documented result types and lifecycle behavior in
  [the extension contracts reference](EXTENSION_CONTRACTS.md).
- **Built-in rerankers:** `CohereReranker`, `JinaReranker`, `NvidiaReranker`,
  `VoyageReranker`, and the optional local `TransformersReranker` use the documented `Reranker`
  contract, bounded hosted requests, and host-owned credentials.
- **Sandboxed data execution:** `python_run_tool` executes bounded source using a configured
  interpreter inside the agent's per-run container. Its tool schema, output, timeout, and sandbox
  policy are part of the Python API contract.
- **Built-in knowledge ingestion:** UTF-8 `.log` files use the same bounded text parser as Markdown
  and plain text and are not interpreted as a particular log format. `MarkupTextParser`, `TOMLTextParser`, and `YAMLTextParser` parse bounded source and structured
  configuration files into searchable text; `ParquetTextParser` parses bounded flat tables into
  row-labeled pages when the optional `parquet` extra is installed; `ICalendarTextParser` parses
  bounded UTF-8 iCalendar event, task, and journal records into cited pages with filterable metadata;
  `RSSAtomTextParser` parses bounded
  RSS 1.0/2.0 and Atom files into one cited, filterable page per feed entry; `VCardTextParser`
  parses bounded UTF-8 vCard 2.1, 3.0, and 4.0 records into one cited, filterable page per contact;
  `MboxTextParser` indexes separator-framed mail archives using the attachment-excluding MIME parser;
  `ODPTextParser` reads OpenDocument Presentation slides into cited pages with slide-name metadata.
- **Skill publisher trust:** `SkillTrustPolicy`, `load_skill_trust_keys`, host-owned
  `KEY_ID.pub` trust-directory validation, `SkillRevocationChecker`, `SkillRevocationStore`, and
  the documented construction-time snapshot and revocation contracts. `RedisSkillRevocationStore`
  is a host-client adapter for a Redis 6.2+ shared revocation set; the host owns client lifecycle and
  consistency configuration.
- **Agent and skill configuration:** documented YAML keys and their validation, precedence,
  policy-enforcement, and unknown-key behavior. `AgentDefinition` and `SkillDefinition` fields
  documented by the reference API are included.
- **Tool concurrency:** `Tool.parallel_safe` and `policies.max_parallel_tool_calls` are opt-in;
  the default remains sequential, and only eligible host handlers can overlap.
- **HTTP:** `create_app`, `StreamJournal`, `StreamJournalSnapshot`, `POST /v1/agents/{agent_name}/run`,
  `POST /v1/agents/{agent_name}/stream`, and `GET /health`, with their documented schemas, status
  codes, authorization rules, response limits, and SSE event names and payloads.
- **Local MCP integration:** `create_mcp_server` and `gabby mcp` over stdio when the optional `mcp`
  extra is installed; remote deployments continue to use the authenticated HTTP API.
- **MCP client bridge:** `register_mcp_tools` imports tools from a host-connected MCP client into a
  `ToolRegistry`. The host owns the connection and its transport security; only text or structured
  JSON results enter Gabby tool context.
- **Anthropic provider:** `AnthropicProvider` is available as a built-in optional provider using the
  native Messages API. Its live model and hosted endpoint support is supported only for the exact
  combinations verified and named in each release's notes.
- **Gemini provider:** `GeminiProvider` uses the native GenerateContent REST API, preserves function
  identifiers and thought-signature parts across stateless tool turns, and supports bounded text
  streaming. Live model support is limited to combinations verified and named in release notes.
- **CLI:** commands and flags shown by `gabby --help` and documented in the user guides.
  Undocumented internal flags are excluded.
- **Python package imports:** the named Python contracts above are importable from `gabby`; their
  module paths are internal and may change.

Optional model providers, storage engines, platform-specific sandbox capabilities, and other
integrations are supported only where the v1.0 release notes identify their tested versions and
platforms. Internal modules, underscored names, undocumented fields, trace implementation details
beyond the documented schema, and features explicitly marked experimental are outside the contract.

## Proposed rules for 1.x

- Follow Semantic Versioning. Compatible additions and deprecations ship in minor releases; fixes
  ship in patch releases; intentional breaking changes to a supported surface require a new major
  release.
- Before removing or changing a supported Python interface, emit `DeprecationWarning`, document a
  migration, and keep the old behavior for at least two minor releases and six months after the
  first deprecation notice, whichever period is longer.
- Remove a deprecated supported contract only in a major release after that window has elapsed.
- For HTTP, YAML, and CLI behavior where Python warnings cannot reach the caller, announce the
  deprecation in release notes and the relevant guide, and retain the old behavior for the same
  window. When feasible, accept both old and replacement forms during the window.
- A deprecated interface remains covered by tests until removal. The release that removes it must
  include the migration instructions and the earliest supported replacement.
- Security, privacy, legal, or severe data-integrity issues may require an earlier incompatible
  change. The release notes and security advisory will explain the exception and provide the
  safest available migration.
- Python version support is the exact range declared by package metadata and CI for each release;
  support for a new Python version is a compatible addition, while dropping a previously supported
  version requires a major release.

The rules apply to maintained 1.x releases. Gabby will maintain the latest 1.x minor release for
bug fixes and security fixes. The immediately preceding minor release receives security and severe
data-integrity fixes for six months after its successor is released. Only the latest patch release
of each supported minor receives updates. Older minor releases are end-of-life and receive no
promised updates; users should upgrade to a supported release. Each release's declared Python range
and CI matrix define its supported Python versions.

## Change records and review

User-visible changes belong in `CHANGELOG.md`. A public-contract change also requires an ADR when it
changes lifecycle, security, concurrency, wire behavior, or compatibility guarantees. Pull requests
that change a supported contract should update the contract reference and add or adjust contract
tests. See [CONTRIBUTING.md](../CONTRIBUTING.md) and [ADR 0009](architecture/adr/0009-pre-1-0-extension-api-policy.md).
