# Extension API policy

Gabby is pre-1.0. Public Python extension APIs may change incompatibly before v1.0, including
signatures, lifecycle behavior, error types, and supported extension points. Gabby does not promise
SemVer compatibility for `0.x` releases. The project will record user-visible changes in the
changelog and architecture-level changes in an ADR.

The proposed v1.0 compatibility contract and deprecation window are in
[`COMPATIBILITY.md`](COMPATIBILITY.md). They become binding only when a v1.0 release is declared;
this pre-1.0 policy does not promise that today's interfaces will all be included. A Python name
being importable from `gabby` means it is an intentional public entry point, but does not create a
compatibility promise before v1.0. Internal modules and underscored names are implementation details.

## Current extension model

The [extension contracts reference](EXTENSION_CONTRACTS.md) lists the public interfaces, method
results, lifecycle ownership, concurrency assumptions, and timeout/failure behavior in one place.

`AgentTool` adapts an async child `Agent` into a parent `Tool`. The parent must declare the tool and
grant its `agent:invoke` permission. Delegation sends only the tool's explicit task string: caller
context, memory, metadata, and principal are not forwarded by default. Set `forward_principal=True`
only when the child's approval handler should receive the authenticated caller identity. Each child
keeps its own tools and policies; the host owns and closes both agents. Calls are bounded by the
parent tool timeout and byte caps, and cycles or delegation deeper than eight agents fail closed.
The child trace carries the parent run ID in its `parent_trace_id` metadata and request event so
independently exported traces can be correlated without merging their event contents.
See the [agent composition example](../README.md#compose-specialized-agents).

Inject extension implementations when constructing an `Agent` or a FastAPI app. Current examples
include `ModelProvider`, `Environment`, `PolicyEngineFactory`, `Retriever`, `KnowledgeStore`,
`EngineAdapter`, `Authenticator`, `SkillSelector`, `Planner`, `Verifier`, `Tracer`, and the FastAPI
`StreamJournal` storage contract.
A stream journal receives principal-bound, fingerprinted session keys and ordered bounded SSE frames;
the host owns the lifecycle of injected implementations. A tracer receives ordered `TraceEvent`
snapshots during a run; events from concurrent runs may interleave, so it must support concurrent
calls. The regular `ExecutionTrace` remains in the result. Tracer errors are
best-effort: after a callback times out or raises, Gabby records a sanitized `tracer_error`,
disables that tracer for the run, and continues. The host owns the tracer lifecycle and must treat
event details as potentially sensitive. A planner returns a structured `PlanningResult` with an
`ExecutionPlan`; the runtime validates and records it, then supplies it to the reasoning model as
advisory context. `ModelPlanner` is opt-in and forwards task, caller context, memory, active skill
descriptions, and policy-approved tool descriptions to its injected provider. `Planner.plan()` can
be called again when `policies.max_replans` is set from 1 through 3. Revised calls receive
`previous_plan` and bounded `PlanningObservation` values for the latest tool batch. `ModelPlanner`
labels that material untrusted; the runtime records a `replanning` trace and emits `plan_updated`.
Custom planners must accept the revision keywords to use this option. Calls share the run deadline
and model retry policy. Each included tool observation excerpt is sent to the configured planner
provider; hosts should account for that data transfer when selecting the planning model. The
runtime does not put these excerpts in the plan trace. See
[ADR 0122](architecture/adr/0122-bounded-plan-revisions.md). A verifier returns a
`VerificationResult`; enabled verification
fails the run if a verifier raises, returns another value, or reports `passed=False`. Check the
public exports and protocol docstrings in `gabby` for the current signatures. Agent definitions are
resolved into immutable snapshots; injected service handles remain host-owned references.
`Agent.arun()`, `Agent.astream()`, and direct `RunRequest` construction validate a non-empty task of
at most 100,000 characters and snapshot JSON-normalized `context`, `memory`, and `metadata` before
runtime awaits. Those values share the configured `max_model_request_bytes` input-data budget; request
maps require string top-level keys. This is the same task and map-shape boundary used by FastAPI.
Gabby rejects unknown keys in its owned agent, skill, policy, knowledge, verification, environment,
and sandbox mappings, so misspelled controls fail during construction. Provider-specific `model`
options and named `environment.resources` remain extensible. The complete declarative agent
configuration, including model options, lists, text, and sandbox settings, must be finite
JSON-compatible data and is bounded to 10 MiB, 100,000 values, and 128 nesting levels before the
immutable plan is built. Extensions must validate their own option names and semantics. Host resource
handles supplied through an injected `Environment` remain host-owned references and are not
serialized into this configuration snapshot.

An `Agent` accepts an optional `policy_engine_factory`. Gabby asynchronously creates a policy engine
for each run, after skill activation and before any tools are exposed to the model. The factory
receives the active declared tool names, immutable agent policies, the environment tool allowlist,
and the authenticated principal. Return an object with async `authorize_tool(name)` and
`authorize_permissions(name, permissions)` methods; raise `ToolError` with
`ToolErrorCode.POLICY_DENIED` to deny access. The default `DefaultPolicyEngineFactory` creates the
built-in `PolicyEngine`, which intersects declared tools, agent grants, and environment grants.
Custom factories are host-trusted code and must fail closed. Gabby sanitizes their failure details;
tool schema checks, approval requirements, sandbox requirements, and execution bounds remain
runtime-enforced. The factory and its dependencies are host-owned and must tolerate concurrent runs.
See [ADR 0124](architecture/adr/0124-injectable-policy-engine.md).
Agent YAML files are limited to 10 MiB, `skill.yaml` manifests to 1 MiB, and each file-backed skill
text resource to 10 MiB. Agent and skill text fields in programmatic definitions receive the same
UTF-8 byte limits; programmatic skill metadata arrays are capped at 1 MiB and 10,000 values in
aggregate. Oversized inputs fail during loading or construction; Gabby never truncates instructions
or examples. An agent can resolve at most 256 skills, including dependencies, and
the combined UTF-8 text in agent/global instructions and resolved skill descriptions, instructions,
and examples is bounded by the smaller of `policies.max_model_request_bytes` and 16 MiB.

Every `Tool` requires both an input `parameters` schema and an `output_schema`. The output schema is
validated during tool construction and checked against each handler result before the observation is
added to model context. Use specific result properties and `additionalProperties: false` for
host-facing contracts; Gabby rejects a missing output schema. This is a pre-1.0 contract tightening
recorded in [ADR 0053](architecture/adr/0053-required-tool-output-schemas.md).

Tool failures use the public `ToolErrorCode` string enum. Gabby places `error_code` in the model's
tool observation, the `tool_error` trace event, and the `tool_failed` SSE event. A custom handler may
raise `ToolError(message, code=...)` to select a safe category; Gabby preserves the code and replaces
the handler message with a generic message before exposing the failure to the model. The stable
codes cover invalid arguments, policy and approval denials, deadlines, unavailable tools or
sandboxes, oversized or invalid results, and execution failures. Provider and extension authors
should branch on the code rather than Python exception type or message. See
[ADR 0054](architecture/adr/0054-stable-tool-error-codes.md).

Model failures are not retried unless the agent sets `policies.max_model_retries` from 1 through 3.
Built-in chat providers classify temporary HTTP statuses and network failures as retryable. Custom
providers and model-backed selector/planner extensions can opt in by raising the public
`RetryableModelError`; other exceptions are not retried. Retry waits share the run deadline. This
can result in another billable provider call. Streaming extensions are retried only before the
runtime emits the first text delta. See [ADR 0055](architecture/adr/0055-bounded-transient-model-retries.md).

The built-in `BearerTokenAuthenticator` is for a simple single-tenant service and can attach
configured scopes to its principal. The optional `JWTBearerAuthenticator` verifies a signed token
against a configured issuer, audience, HTTPS JWKS endpoint, and asymmetric algorithm allowlist;
install `gabby-agent-runtime[auth]` to use it. JWT authentication returns the validated subject and
reads the standard space-delimited `scope` claim or list-valued `scp` claim. Set its optional
`scope_mapping` to translate exact issuer scopes to one or more Gabby capabilities; unmapped issuer
scopes are dropped when a mapping is configured. With no mapping, scopes pass through unchanged.
Embedded hosts can pass `run_scopes` and `stream_scopes` to `create_app` for route-level
authorization. The host still owns role/group claim mapping, per-user policy, issuer metadata
discovery, revocation storage, and tenant routing. To enable request-time JWT revocation checks,
inject a `TokenRevocationChecker`; Gabby then requires a bounded ASCII `jti` claim and checks it on
every authenticated request, passing the configured issuer, token ID, and expiry to the checker. A
revoked token is rejected. Checker failures and timeouts fail closed as authentication-unavailable
responses; the lookup timeout defaults to one second and no result is cached. The host owns checker
lifecycle, and a distributed store's consistency and propagation. `SQLiteTokenRevocationStore`
provides a durable local implementation without another dependency. It supports idempotent
`revoke()` calls and explicit `cleanup_expired()` maintenance; configure a local filesystem path and
protect existing files with deployment-appropriate permissions (new database files are owner-only
where supported). Use a different checker for network filesystems
or distributed deployments. See
[ADR 0035](architecture/adr/0035-route-scoped-api-authorization.md),
[ADR 0042](architecture/adr/0042-jwt-scope-to-capability-mapping.md), and
[ADR 0043](architecture/adr/0043-pluggable-jwt-revocation-check.md) and
[ADR 0046](architecture/adr/0046-sqlite-jwt-revocation-store.md).

For shared skill publisher revocations, `RedisSkillRevocationStore` accepts a host-owned async
Redis client through `AsyncRedisCommands`; the core package does not import or construct Redis.
The adapter uses `SMISMEMBER`, `SADD`, `SREM`, and bounded `SSCAN`, and requires Redis 6.2+. The host
owns client lifecycle, credentials, TLS, persistence, routing, and the read-after-write consistency
guarantee. Configure a namespaced set key per trust domain. The store defaults to a 100,000-entry
management-list bound; the runtime checker validates at most 1,024 signer IDs per call and reads
fresh state without caching. Its live propagation behavior depends on which Redis authority the host
client queries. See the [shared revocation example](../examples/redis_skill_revocation.py) and
[ADR 0101](architecture/adr/0101-redis-skill-revocation-adapter.md).

Tools that opt into `ToolContext` receive a read-only `CancellationToken`. A synchronous handler can
call `context.cancellation.wait(timeout)` from its callback worker thread, or poll
`context.cancellation.is_cancelled`; asynchronous handlers are cancelled at their await points and
can inspect the same property during cleanup. The token is signaled when Gabby times out or cancels
the invocation. This is cooperative: handlers that ignore the token may continue after Gabby has
stopped awaiting them. Keep side effects idempotent where possible, and use a sandboxed tool for
work that must be forcibly terminated. See
[ADR 0037](architecture/adr/0037-cooperative-tool-cancellation.md).

`ToolContext.agent_instance_id` is an opaque, process-local identity for the current `Agent` object.
Gabby uses it to detect delegation cycles without assuming configured agent names are unique. It is
not stable across process restarts and must not be persisted or treated as an authorization token.
It remains host-only and is never added to model context.

For example, a host can record event metadata without logging event contents:

```python
import logging

from gabby import Agent, TraceEvent

logger = logging.getLogger("myapp.agent_trace")


class LoggingTracer:
    async def on_event(self, *, trace_id: str, agent_name: str, event: TraceEvent) -> None:
        logger.info(
            "agent trace event",
            extra={"trace_id": trace_id, "agent": agent_name, "kind": event.kind},
        )


agent = Agent(definition, tracer=LoggingTracer())
```

Gabby defaults to a 4 MiB UTF-8 serialized request cap per provider call, configurable through
`policies.max_model_request_bytes`. The runtime and built-in `ModelPlanner` and `ModelSkillSelector`
check the canonical request before invoking their provider. Custom selectors and planners that
make provider calls must apply `ensure_model_request_size(messages=..., tools=..., model=...,
temperature=..., max_bytes=...)` themselves using the agent's policy limit; the core cannot
inspect an extension's private outbound request construction.

`FileParser` is a synchronous protocol with lowercase dotted `extensions` and a `parse(bytes)` method
returning ordered `ParsedPage` values. `ParsedPage.metadata` is a mapping of JSON-compatible
page-level fields copied onto each chunk from that page. `FileIngestor` owns root-confined path checks,
per-file source replacement, deterministic chunking, and file, directory, page, and extracted-text
limits. The core UTF-8 parser handles Markdown, plain text, and `.log` files as text without
interpreting a log-specific format; `YAMLTextParser` handles bounded
standalone `.yaml` and `.yml` data files, rejecting duplicate keys, alias cycles, multiple documents,
merge keys, unsafe tags, non-JSON values, and excessive depth or node count. It renders nested values as stable
dotted key paths and list indexes to make structured values searchable, with configurable input and
output byte caps. An optional leading YAML front matter block is removed from indexed body text and
exposed as page metadata. The block is capped at 16 KiB,
uses safe YAML loading, rejects duplicate keys and cyclic or non-JSON values, and is limited to 10,000
nodes and 32 levels. YAML dates and times become ISO-8601 strings. Caller-supplied file metadata
overrides front matter; Gabby assigns reserved chunk and page fields afterward. `TOMLTextParser`
uses the standard-library TOML parser, rejects malformed, excessively nested, or oversized inputs,
normalizes date/time values to ISO-8601 strings, and renders stable dotted paths and list indexes
within an output byte cap. Duplicate TOML keys are rejected by the strict parser. `EmailTextParser`
handles bounded RFC 5322/MIME
messages, indexes selected headers and text bodies, skips attachments, and prefers plain-text
alternatives over duplicate HTML. HTML-only mail uses the bounded visible-text HTML parser.
It indexes From, To, Cc, Date, Subject, and Message-ID; default limits are 10 MiB input, 1,000
MIME parts, 32 nesting levels, and 10 million extracted characters. Set limits on `EmailTextParser`
and pass it through `FileIngestor(parsers=...)` when you need different parser-specific values.
`ICalendarTextParser` turns each direct-child `VEVENT`, `VTODO`, and `VJOURNAL` into a source-ordered cited page,
unfolds content lines, and decodes RFC 5545 text escapes. Event metadata retains the existing
`calendar_event_index`, UID, start/end, location, organizer, and attendee fields. Task pages expose
`calendar_task_index`, UID, start, due, completed, status, priority, and attendee fields when present.
Task text includes summary, due/completed times, description, status, priority, and selected common
fields. Journal pages expose `calendar_journal_index`, UID, start, and status fields when present;
journal text includes summary, start, description, status, and selected common fields. Nested alarms
and other subcomponents are excluded. It requires UTF-8 and valid component
boundaries, and defaults to 10 MiB input, 1,000 events, 1,000 tasks, 1,000 journals, 1,000
attendees per record, 100,000 lines, and 10 million extracted characters. Configure its limits directly and pass it through
`FileIngestor(parsers=...)`
for parser-specific bounds.
`RSSAtomTextParser` turns RSS 1.0/2.0 and Atom entries into one-based cited pages and exposes feed
title, entry ID, link, date, author, and categories as metadata. It accepts `.rss` and `.atom`; the
default `XMLTextParser` detects RSS, RDF, and Atom roots in `.xml` files and delegates with the same
input, output, XML structure, and feed-item bounds. Entry HTML is reduced to visible text. It does
not fetch feeds or follow links; only absolute HTTP(S) citations without embedded credentials are
retained. DTDs, entities, malformed XML, and oversized feeds are rejected.
Configure `RSSAtomTextParser` or `XMLTextParser(max_feed_items=...)` directly and pass the parser
through `FileIngestor(parsers=...)` when you need parser-specific limits.
`OPMLTextParser` indexes OPML 1.0/1.1/2.0 outlines as individual cited pages, preserving the
outline hierarchy as category metadata. It supports `.opml` directly, and the default XML parser
also detects OPML roots in `.xml` files. Feed, website, and outline URLs are retained only as safe
absolute HTTP(S) citations; the parser never follows them. Input, output, outline count, XML element
count, depth, and attributes are bounded, and DTD/entity declarations are rejected. Configure
`OPMLTextParser` directly or pass it through `FileIngestor(parsers=...)` for parser-specific limits.
`JSONTextParser` validates bounded strict JSON, rejecting duplicate keys, non-standard constants,
and excessive nesting or token counts. It preserves ordinary JSON source text; when the top-level
`version` identifies JSON Feed 1.0 or 1.1, it instead returns one cited page per item, with feed
title, ID, link, publication date, author, and tags in searchable text or filter metadata. It accepts
plain text and HTML item content, strips HTML to visible text, and applies feed item and aggregate
output limits. Only absolute HTTP(S) item links without embedded credentials become citations; feeds
and links are never fetched. Configure `JSONTextParser` directly or pass it through
`FileIngestor(parsers=...)` for parser-specific limits.
`HTMLTextParser` handles UTF-8 HTML and omits script,
style, template, SVG, and hidden subtrees while retaining titles, headings, and paragraph
boundaries. `JSONLinesTextParser` applies strict JSON duplicate-key, nesting, and token checks to each
non-blank line, labels records with their physical line numbers, and caps record count, record bytes,
and rendered output size. `XMLTextParser` streams UTF-8 XML into path-labeled text and attributes, rejects DTD/entity declarations, and limits input, output, depth, elements, and attributes without an additional package. `RTFTextParser` uses a bounded standard-library parser to extract visible text and Unicode escapes, skips non-body destinations and binary payloads, and limits input, nesting, control words, and output. It does not render layout, images, or embedded objects. `DOCXTextParser` and `PPTXTextParser` use only the standard library to read bounded OpenXML archives
without extracting entries to disk. DOCX preserves paragraph and table order, tabs, and line breaks;
it does not extract images, deleted revision text, page layout, or page attribution. PPTX preserves
slide order and returns one-based slide numbers as `page_number` citations; it does not extract notes,
images, or chart data. Both reject encrypted archives, unsafe or duplicate member paths, unsupported
XML encodings, DTD/entity declarations, malformed packages, and oversized inputs. DOCX defaults cap
the archive at 10 MiB, uncompressed entries at 32 MiB, package metadata XML at 1 MiB per part, the
main XML part at 16 MiB, XML elements at 250,000, paragraphs at 100,000, and extracted text at 10
million characters. PPTX defaults cap the archive at 10 MiB, uncompressed entries at 32 MiB, each
slide XML part at 4 MiB, slides at 100, and extracted text at 10 million characters.
`EPUBTextParser` reads XHTML/HTML chapters in OPF spine order and returns one-based chapter-order
`page_number` citations. It rejects unsafe archive paths, remote chapter references, encrypted
archives, duplicate members, DTD/entity declarations, malformed packages, and oversized content;
defaults cap the archive at 10 MiB, expanded members at 32 MiB, package XML at 1 MiB per part,
chapters at 4 MiB, chapter count at 1,000, and extracted text at 10 million characters. It does not
extract images, stylesheets, or other embedded resources. `ODTTextParser` reads document-ordered
headings, paragraphs, tables, tabs, line breaks, and repeated spaces from bounded OpenDocument Text
archives. It rejects encrypted archives, unsafe or duplicate member paths, DTD/entity declarations,
malformed packages, and oversized input; defaults cap the archive at 10 MiB, expanded content at
32 MiB, content XML at 16 MiB, XML elements at 250,000, paragraphs at 100,000, and extracted text
at 10 million characters. `ODSTextParser` emits sheet- and row-labeled column values for non-empty
rows and handles bounded repeated rows and columns. It rejects encrypted archives, unsafe or duplicate
paths, DTD/entity declarations, malformed packages, and oversized input; defaults cap the archive at
10 MiB, expanded content at 32 MiB, content XML at 16 MiB, XML elements at 250,000, rows at
100,000, columns at 1,000, cells at 1 million, sheets at 1,000, and extracted text at 10 million
characters. It does not calculate formulas or extract embedded media. `XLSXTextParser` follows workbook sheet
order and renders cached values, inline strings, and shared strings with sheet, row, and column
labels. It rejects external or unsafe relationships, encrypted archives, duplicate paths, DTD/entity
declarations, malformed packages, and oversized input; defaults cap the archive at 10 MiB, expanded
content at 32 MiB, worksheet XML at 4 MiB per sheet, shared-string XML at 16 MiB, sheets at 1,000,
rows at 100,000, columns at 1,000, cells and shared strings at 1 million, and extracted text at 10
million characters. It does not recalculate formulas. Custom parsers execute as trusted in-process Python; use a
separate process or service when parsing untrusted complex formats. PDF support uses the optional `pdf` extra
(`pip install 'gabby-agent-runtime[pdf]'`) and records one-based `page_number` metadata for citations.
The built-in pypdf parser applies a 4 MiB decoded-output limit to each supported PDF stream filter
and a 32 MiB aggregate decoded page-content limit per PDF by default. Configure
`max_content_stream_bytes` and `max_total_content_stream_bytes` on `PDFTextParser` or `FileIngestor`
to change them. These limits bound decoded page-content streams, while parser memory and CPU can still depend on PDF
structure and library internals.
`ParquetTextParser` uses the optional `parquet` extra to stream flat scalar tables in bounded
batches. It emits row-labeled pages with row-range metadata and caps input, rows, columns, row
groups, metadata-reported uncompressed bytes, cell and rendered row size, page count, and total
output. Nested and
binary columns are rejected. See [ADR 0106](architecture/adr/0106-bounded-parquet-ingestion.md).
`ImageOCRParser` adds bounded OCR for raster image formats through the optional `ocr` extra and a
host-installed Tesseract executable. Its `OCRBackend` protocol lets hosts provide another engine.
The default backend limits frame count, total decoded pixels, extracted characters, and per-frame
execution time, and returns one-based page attribution for multi-frame images. Tesseract runs as a
child process; install it from the operating system package manager and keep it patched. The optional
`pdf-ocr` extra enables scanned and vector-only PDF OCR through
`FileIngestor(pdf_ocr_backend=TesseractOCRBackend())`. Blank-text pages use their first embedded
image when available; otherwise `PDFiumPageRenderer` rasterizes the page in memory. Both paths keep
original PDF page numbers and share the configured page, pixel, text, and Tesseract timeout limits.
Rendered images are also bounded by `max_ocr_image_bytes`, defaulting to 16 MiB. Hosts can
inject a `PDFPageRenderer` implementation through `pdf_page_renderer`. The PDFium renderer is
synchronous and cannot be interrupted by the OCR timeout; isolate parsing when processing hostile
PDFs. Multiple embedded images are not composited into one page; inject a custom renderer for those
layouts. Pass `TesseractOCRBackend(language="eng")` with a language model installed on the host; the
default is English.
`TextFileIngestor` remains an alias for `FileIngestor` for existing callers. See
[ADR 0032](architecture/adr/0032-pluggable-page-aware-file-ingestion.md) and
[ADR 0045](architecture/adr/0045-stdlib-html-text-ingestion.md).

Provider responses default to 4 MiB per call through `policies.max_model_response_bytes`. The
runtime checks normalized responses and streamed deltas returned by custom providers. An extension
that reads from a remote provider must enforce a transport-level cap before buffering the response;
the runtime cannot undo memory already allocated by the extension. Use the public
`ensure_model_response_size` helper for normalized `ModelResponse` values.

Provider credentials must stay outside agent configuration. Built-in provider `from_config`
factories reject inline `api_key`, credential-bearing model headers, URL userinfo, and query
parameters with credential-like names; use the configured environment variable or inject a provider
constructed by the host from its secret manager. YAML-loaded and programmatic definitions both
pass through `validate_agent_definition` during agent construction; hosts can also call it before
constructing an agent. See [ADR 0017](architecture/adr/0017-provider-credentials-outside-agent-config.md)
and [ADR 0021](architecture/adr/0021-shared-agent-definition-validation.md).

Built-in OpenAI-compatible, Ollama, Hugging Face, Anthropic, and Gemini providers require HTTPS for remote
`base_url` values and permit HTTP only for loopback local inference. Agent validation and direct
construction of built-in adapters enforce this boundary. Custom `ModelProvider` implementations
are responsible for securing their own transport. See
[ADR 0019](architecture/adr/0019-https-for-remote-model-providers.md).

`AnthropicProvider` targets the native Messages API without adding an Anthropic SDK dependency.
It maps Gabby's system messages, function schemas, assistant tool calls, and tool observations into
Anthropic's request format, then maps text, tool-use, usage, and streaming input deltas back to the
shared provider contract. Configure `model.max_tokens` from 1 through 200,000 (default 1,024); the
provider requires this output limit. Credentials come from `ANTHROPIC_API_KEY` or a configured
`model.api_key_env`. See [ADR 0090](architecture/adr/0090-anthropic-messages-provider.md).

`GeminiProvider` uses the native GenerateContent REST API and the `GEMINI_API_KEY` environment
variable by default. It sets a bounded `maxOutputTokens`, sends function declarations with
`parametersJsonSchema`, and preserves the exact returned content parts, thought signatures, and
function-call IDs needed to complete stateless tool turns. Streaming function calls arrive as
complete calls; text arrives incrementally. Its opaque provider metadata is bounded, passed back
only to the model provider, and excluded from traces and tool arguments. See
[ADR 0096](architecture/adr/0096-native-gemini-provider.md), the official
[GenerateContent API](https://ai.google.dev/api/generate-content), and the
[thought-signature guide](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).

`TransformersProvider` is the optional in-process local chat adapter. Install the `transformers`
extra and a PyTorch 2.5+ build appropriate for the host; Gabby does not choose a CUDA or accelerator
wheel. Loading is lazy. Set `revision` to a fixed Hub commit for reproducible model selection and
`local_files_only: true` for offline loading. Remote weights use the configured `api_key_env`
(default `HF_TOKEN`); custom model code is disabled and only safetensors checkpoints are accepted.
`max_input_tokens` defaults to 32,768 and `max_new_tokens` to 1,024; both can be configured within
validated bounds. Local generation runs in Gabby's bounded daemon callback pool and is serialized
per provider instance. Cancellation requests stop generation between tokens; model kernels that do
not return to the stopping check cannot be forcibly interrupted. This adapter currently implements
completion without incremental token streaming. Models must provide a compatible chat template;
when the agent exposes tools, the template must include the tool schemas and the tokenizer must
provide `parse_response` with a response template. A model-specific `model.tool_response_template`
can supply that parser schema when the tokenizer has no default; it is snapshotted as bounded JSON
and limited to 64 KiB. The provider removes one complete Markdown JSON fence before parsing when an
explicit schema is configured, then validates the parsed call against the registered tool names and
argument shapes. It never infers calls from arbitrary prose. A real CPU structured-call acceptance
passed with Transformers 5.18.0 and a pinned Qwen2.5-Coder-0.5B-Instruct revision using a custom
response template. This verifies parser/provider compatibility and one model's tool-call output;
model selection quality, full multi-step behavior, other templates, devices, throughput, and
accelerator combinations still need separate acceptance. See [ADR 0071](architecture/adr/0071-local-transformers-provider.md),
the [Transformers chat-template guide](https://huggingface.co/docs/transformers/main/chat_templating),
and the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

File-backed skills are portable directories rooted at `skill.yaml`. Long procedures can live in a
UTF-8 `instructions.md` beside the manifest or in a path selected with `instructions_file`; authored
examples can live in `examples.md` or a path selected with `examples_file`. Resolved text paths must
stay within the skill directory. Skill examples enter model context only when that skill activates;
the `examples/` and `tests/` subdirectories may accompany the package as authoring resources without
being loaded into model context. Agent definitions can resolve packages from their adjacent
`skills/` directory or configured `skill_paths`. Create a deterministic local archive with
`mkdir -p ./dist`, then pack and install it with:

```sh
gabby skill pack ./skills/debugging --output ./dist/debugging.gabskill
gabby skill install ./dist/debugging.gabskill --registry ./skills
```

Optional `input_schema` validates the JSON object `{"task": ..., "context": ..., "memory": ...}`
after selection and dependency expansion, before the main reasoning model or any tool executes.
Optional `output_schema` validates the final response as JSON while the skill is active. When
multiple active skills define output schemas, the same response must satisfy every schema as well
as any agent-level output schema. Gabby buffers streamed text until all active output contracts
pass. `gabby inspect` and `gabby skill inspect` report these contracts; skill package checksums and
publisher signatures cover them as part of `skill.yaml`. These schemas describe data shape and do
not grant tools or permissions. See [ADR 0123](architecture/adr/0123-validated-skill-io-schemas.md).

Installation writes
`<registry>/<skill-id>/<version>/` and refuses to replace an installed version. Packaging and
installation reject case-folded or Unicode-normalized path collisions so an archive has consistent
contents on case-insensitive hosts. Per-file SHA-256 checksums detect corruption. The optional
`skill-signing` extra provides detached Ed25519 signatures over the exact archive digest and key ID.
`sign_skill_package` accepts a raw 32-byte private key; `verify_skill_package_signature` requires a
host-owned mapping from key IDs to raw 32-byte public keys. Pass `require_signature=True` and
`trusted_keys=...` to `install_skill` when installation must authenticate a publisher. The CLI
provides `gabby skill sign`, `gabby skill verify`, and `gabby skill install --require-signature`.
Unsigned local installation remains available by default. Signature verification establishes that
the archive was signed by one of the configured keys; skill instructions remain untrusted model
input and must be reviewed. For remote discovery and signature-required downloads, see the
[static skill registry format](SKILL_REGISTRY.md); Python hosts can use the async
`SkillRegistryClient` with host-injected headers and public keys.

```python
async with SkillRegistryClient(
    "https://skills.example.org", headers=host_secret_headers
) as registry:
    matches = await registry.search("support")
    installed = await registry.install(
        "support/triage",
        "1.2.0",
        "./skills",
        trusted_keys=trusted_publisher_keys,
    )
```

The client does not choose a version on the host's behalf; the consumer selects an exact version
from the discovery results. See the registry guide for the static file layout and publisher workflow.
Programmatic `SkillDefinition` entries supplied through `skill_registry` receive the same name,
field, dependency, and source-path validation as file-loaded manifests; call the public
`validate_skill_definition` helper to preflight a package before composing it into an agent.
Manifests carry a SemVer version (default `0.1.0` for older packages). Agent skill references and
skill dependencies can pin exact versions with `skill-id@MAJOR.MINOR.PATCH`. Registry mappings may
use stable IDs for one package or exact versioned IDs to hold multiple versions. A single agent
resolves at most one version of each stable ID; use exact pins when reproducibility matters. A
filesystem registry may keep a legacy package at `<root>/<id>/skill.yaml` or multiple packages at
`<root>/<id>/<version>/skill.yaml`; unpinned references fail if the versioned layout is ambiguous.

`HybridRetriever` can use any `Retriever` and base `VectorStore` for ordinary rank fusion. Wrap any
retriever with `RerankingRetriever` to add an injected `Reranker` as a second-stage ranker. The
wrapper fetches at most 100 candidates (20 by default), asks the reranker for no more than the
requested result count, and accepts only unique documents already present in the candidate set.
Returned values are copies of the original candidates; a reranker may reorder or omit evidence but
cannot replace its contents. See
[ADR 0027](architecture/adr/0027-composable-reranking.md).

`CohereReranker` is a concrete optional hosted implementation of `Reranker`. It sends candidate
text, the query, model name, and result limit to Cohere's v2 rerank API, and maps validated response
indexes back to the original documents; source metadata and IDs stay in Gabby. Set `COHERE_API_KEY`
in the host environment, or select another environment variable through `api_key_env`. Its default
model is `rerank-v4.0-fast`; `base_url` defaults to `https://api.cohere.com/v2` and remote URLs must
use HTTPS. Per-call serialized request and streamed response caps default to 4 MiB, candidate count
is capped at 100, and `max_tokens_per_doc` defaults to 4096. Reranking sends document text to the
configured service, so select it only when that data flow is appropriate for the application.
Install and run the opt-in live acceptance in `tests/integration/README.md`. See the
[Cohere Rerank API reference](https://docs.cohere.com/v2/reference/rerank).

```python
from gabby import CohereReranker, RerankingRetriever

reranker = CohereReranker.from_config({"model": "rerank-v4.0-fast"})
retriever = RerankingRetriever(base_retriever, reranker, candidate_limit=20)
try:
    documents = await retriever.retrieve("account recovery", limit=5)
finally:
    await reranker.aclose()
```

`JinaReranker` is another hosted `Reranker` implementation. It defaults to Jina's
`https://api.jina.ai/v1/rerank` endpoint, the `jina-reranker-v3.5` model, and the `JINA_API_KEY`
host environment variable. Configure a different model or `api_key_env` through `from_config`; inline
credentials in config are rejected. Gabby sends only query and candidate text and asks for indexes and
scores without duplicate document bodies, then maps validated indexes back to the original local
documents. HTTPS is required for remote endpoints. Serialized requests and streamed responses are
bounded to 4 MiB by default, with a 100-candidate maximum. Since candidate text leaves the process,
use this adapter only where that data flow is appropriate. Hosted API terms govern API use; Jina's
current reranker weight licenses differ by model, so check the [Jina reranker terms and model
licenses](https://jina.ai/en-US/reranker/) before self-hosting weights.

```python
from gabby import JinaReranker, RerankingRetriever

reranker = JinaReranker.from_config({"model": "jina-reranker-v3.5"})
retriever = RerankingRetriever(base_retriever, reranker, candidate_limit=20)
try:
    documents = await retriever.retrieve("account recovery", limit=5)
finally:
    await reranker.aclose()
```

`VoyageReranker` is a hosted adapter for Voyage's `/v1/rerank` endpoint. It defaults to the
`rerank-2.5-lite` model and reads `VOYAGE_API_KEY` from the host environment; `from_config` accepts
non-secret model, endpoint, truncation, and bound settings. Requests set `return_documents` to false
and `truncation` to false by default, so returned document text is not duplicated and oversized
inputs are reported by the provider instead of silently truncated. Gabby requests no more than 100
candidates, validates response indexes and relevance scores, and maps results to original documents.
Remote endpoints must use HTTPS, requests and streamed responses are capped at 4 MiB by default,
and the provider call defaults to a 120-second timeout. Candidate text is sent to Voyage; review its
[API reference](https://docs.voyageai.com/reference/reranker-api) and data terms for your deployment.
An opt-in live API contract check is documented in
[`tests/integration/README.md`](../tests/integration/README.md).

```python
from gabby import RerankingRetriever, VoyageReranker

reranker = VoyageReranker.from_config({"model": "rerank-2.5-lite"})
retriever = RerankingRetriever(base_retriever, reranker, candidate_limit=20)
try:
    documents = await retriever.retrieve("account recovery", limit=5)
finally:
    await reranker.aclose()
```

`NvidiaReranker` is a hosted adapter for NVIDIA NeMo's reranking endpoint. It defaults to
`nvidia/rerank-qa-mistral-4b`, reads `NVIDIA_API_KEY` from the host environment, and posts query
and passage text to `https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking`. It requests
`truncate: "NONE"`, validates returned passage indexes and finite logits, and maps ranked results
back to the original documents. At most 100 candidates are sent; request and response bodies are
bounded to 4 MiB by default and the request timeout defaults to 120 seconds. Candidate text leaves
Gabby when this hosted adapter is used. See [NVIDIA's API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nim-rerankqa-mistral-4b-v3-infer)
and review the applicable model and service terms.

```python
from gabby import NvidiaReranker, RerankingRetriever

reranker = NvidiaReranker.from_config({"model": "nvidia/rerank-qa-mistral-4b"})
retriever = RerankingRetriever(base_retriever, reranker, candidate_limit=20)
try:
    documents = await retriever.retrieve("account recovery", limit=5)
finally:
    await reranker.aclose()
```

`TransformersReranker` is an optional local implementation for Hugging Face sequence-classification
checkpoints. Install the `transformers` extra and a host-compatible PyTorch build. The model loads on
first use; custom model code and pickle-based weights are disabled. Pin `revision` for repeatable
model selection and set `local_files_only: true` for offline operation. Private checkpoints use
`api_key_env` (default `HF_TOKEN`). Inputs are bounded to 4 MiB of UTF-8 text by default, tokenized
to at most 512 tokens per query/document pair, and scored in batches of 8 through Gabby's bounded
worker pool. The default relevance score is the sole logit for one-output regression models or the
positive-class logit at index 1 for two-class models; other classifiers require an explicit
`relevance_label_index`. Ranking ties preserve candidate order. These controls bound request shape
and execution time but do not establish a model's relevance quality; evaluate a chosen checkpoint
on representative queries before deployment. See the local-model acceptance harness in
[`tests/integration/README.md`](../tests/integration/README.md).

```python
from gabby import RerankingRetriever, TransformersReranker

reranker = TransformersReranker.from_config(
    {
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "revision": "<reviewed-commit>",
        "local_files_only": True,
    }
)
retriever = RerankingRetriever(base_retriever, reranker, candidate_limit=20)
documents = await retriever.retrieve("account recovery", limit=5)
```

`SQLiteVectorStore` is the built-in persistent implementation of `GenerationAwareVectorStore`. It
uses normalized double-precision vectors and exact cosine search; it scans the index linearly while
retaining only the requested top results in memory. This is a dependency-free local option for small
and moderate corpora, not an approximate or distributed index. It fixes the vector dimension on the
first write and cannot identify an embedding model from its output, so hosts must create a new index
when switching models. The host supplies the `EmbeddingProvider`. See
[ADR 0033](architecture/adr/0033-sqlite-exact-cosine-vector-store.md).

`OpenAICompatibleEmbeddingProvider` implements `EmbeddingProvider` for services exposing
`POST /embeddings`. It accepts a fixed model name, splits inputs into bounded batches (64 by default,
100 maximum), preserves indexed response order, validates finite consistent vectors, and enforces
4 MiB request and response caps per call by default. The response cap is applied while reading the
HTTP stream. Remote endpoints must use HTTPS; HTTP is accepted only on loopback. Store credentials
outside configuration and use `api_key_env` or a provider instance built by the host. This adapter
does not count tokens or identify embedding-model revisions: the provider owns model token limits,
and hosts must reindex into a new database when changing embedding models. See
[ADR 0034](architecture/adr/0034-openai-compatible-embedding-provider.md).

`HuggingFaceFeatureExtractionProvider` targets Hugging Face Inference Providers' feature-extraction
task API. It batches inputs and handles sentence vectors or token features; token features are
mean-pooled by default, with CLS pooling available when the model contract calls for it. The adapter
bounds request and response bytes, validates vector shapes and finite values, and reads credentials
from `HF_TOKEN` by default:

```python
from gabby import HuggingFaceFeatureExtractionProvider

embeddings = HuggingFaceFeatureExtractionProvider.from_config(
    {"model": "BAAI/bge-small-en-v1.5", "pooling": "mean"}
)
vectors = await embeddings.embed(["Gabby supports reusable agents."])
await embeddings.aclose()
```

Set `HF_TOKEN` in the host environment before calling the provider. The selected model and hosted
Inference Provider receive indexed text. Model-specific pooling,
prompt names, and truncation settings must match the chosen embedding model. See [Hugging Face's
feature-extraction task documentation](https://huggingface.co/docs/inference-providers/en/tasks/feature-extraction).

`GeminiEmbeddingProvider` implements `EmbeddingProvider` through Google's native
`batchEmbedContents` API. It defaults to `GEMINI_API_KEY`, batches at most 100 inputs per request,
and incrementally enforces configurable request and response byte caps. It supports dimensions from
128 through 3072. The `gemini-embedding-001` model accepts supported `task_type` values; the newer
`gemini-embedding-2` model rejects that option because Google recommends putting task instructions
in the text input. A document `title` is accepted only with `RETRIEVAL_DOCUMENT`. Provider output is
validated for item count, vector shape, dimensions, and finite values. Changing model or task usage
requires an appropriately rebuilt index. Role-specific `query_task_type` and `document_task_type`
are applied by hybrid retrieval and indexing through the optional `AsymmetricEmbeddingProvider`
contract. `query_prefix` and `document_prefix` support embedding-2 task instructions; the labeled
evaluator delegates separate query/document calls to that contract. See [ADR 0097](architecture/adr/0097-native-gemini-embeddings.md),
[Google's API reference](https://ai.google.dev/api/embeddings), and [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings).

`HybridIndexCoordinator` requires `GenerationAwareRetriever`, `GenerationAwareVectorStore`, and a
`GenerationManifestStore`. Generation-aware backends must durably persist a source's highest fencing
token, reject staging or incomplete-generation deletion with an older token, stage each generation
atomically, advance a fence without changing data when asked, filter searches by the active generation
map, and make retired-generation deletion idempotent. Fence advancement is required when recovery
claims a fully staged generation: it prevents a delayed former owner from modifying that generation
after activation. These rules protect concurrent processes and recovery after a crash; an adapter
that ignores fences is not safe to use with the coordinator.
`SQLiteGenerationManifestStore` is for processes sharing a local host filesystem. Multi-host
deployments can use `PostgresGenerationManifestStore` with the optional `postgres` extra and a
host-owned asyncpg-compatible pool. Apply `sql/postgres_generation_manifest.sql` through the host's
migration system before use. The adapter uses PostgreSQL transactions and the server clock for
leases and fencing. The host owns TLS, credentials, routing, pool lifecycle, migrations, backups,
and consistency. Every Gabby instance must also use the same shared lexical and vector backends
that implement the generation-aware fencing contract; a shared manifest does not make SQLite
knowledge files safe to share across hosts. Other database systems can implement
`GenerationManifestStore` with atomic lease claims and an authoritative clock.

`PostgresKnowledgeStore` provides a host-pooled shared lexical store through the
`GenerationAwareRetriever` and `KnowledgeStore` contracts. Apply `sql/postgres_knowledge.sql`
externally and inject the store into `FileIngestor`, `Agent`, or `HybridIndexCoordinator`. It uses
per-source advisory transaction locks and durable fencing rows for staged generations. Multi-host
hybrid indexing still requires a shared generation-aware vector store and
`PostgresGenerationManifestStore`; keep their records in the same PostgreSQL consistency domain.

Custom `EngineAdapter` implementations used with native Windows images must implement
`validate_windows_sandbox_support()` and fail unless they can enforce the required isolation. The
built-in Docker CLI/API adapters require Docker Engine 29.1.4 or newer before starting a Windows
container with networking disabled. An adapter without this check is rejected before container
startup; Linux adapter behavior does not use this Windows-specific check. See
[ADR 0031](architecture/adr/0031-windows-network-isolation-engine-floor.md).

An engine adapter may additionally implement the `SandboxResourceMonitor` capability and its
`resource_usage(container_id, timeout=...)` method. It returns a `SandboxResourceUsage` snapshot or
`None`. The snapshot may report cumulative CPU time in
nanoseconds, final CPU percentage, current/peak memory bytes, and current process count. Fields
that an engine cannot report should be omitted. Gabby collects the snapshot near the end of a run
and records it in the trace; collection errors are ignored so resource telemetry cannot fail the
agent execution. Existing adapters without this method remain usable.

The runtime is async-first. Async extension methods are awaited directly. Supported synchronous
callbacks run through Gabby's bounded worker bridge, but a timeout cannot forcibly stop synchronous
code that has already started. Extensions execute as trusted Python in the Gabby process unless the
extension itself delegates work to an isolation boundary. Tool policy does not sandbox arbitrary
provider, retriever, verifier, authenticator, or selector code.

## Extension author guidance

- Pin Gabby to a tested version range in applications and extensions.
- Avoid depending on private modules or undocumented object fields.
- Treat request data, retrieved content, model output, and tool results as untrusted input.
- Keep credentials in the host's secret manager or environment, not agent or skill YAML.
- Test timeout, cancellation, error, and cleanup behavior for external resources you own.
- Review the changelog and ADRs when upgrading between pre-1.0 versions.

See [ADR 0009](architecture/adr/0009-pre-1-0-extension-api-policy.md) for the accepted stability
policy and [CONTRIBUTING.md](../CONTRIBUTING.md) for repository practices.
