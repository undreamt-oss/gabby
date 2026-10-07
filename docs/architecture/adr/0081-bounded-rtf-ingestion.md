# ADR 0081: Bounded RTF ingestion

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-10-03.

## Context

Gabby ingests modern OpenXML and OpenDocument formats, but rejects Rich Text Format files that
remain common in archived business, support, and research documents. RTF is a control-word and
group-based text format; binary objects and formatting destinations must not be mistaken for body
text or allowed to bypass parser bounds. The implementation follows the Microsoft
[RTF 1.9.1 specification](https://interoperability.blob.core.windows.net/files/Archive_References/%5BMSFT-RTF%5D.pdf)
for control words, Unicode fallback, groups, and binary payloads.

## Decision

- Add a dependency-free `RTFTextParser` to the default `FileIngestor` parser set.
- Preserve visible body text, common paragraph and tab controls, escaped literals, ANSI code-page
  bytes, and signed 16-bit `\\uN` values with `\\ucN` fallback skipping.
- Skip known non-body destinations, unknown starred destinations, hidden text, and exact-length
  `\\binN` data without interpreting binary bytes as group syntax.
- Bound input bytes, group depth, control-word count, and extracted characters. Reject malformed
  group structure, invalid escapes, unsupported declared code pages, and oversized control values.
- Do not render layout, fonts, images, embedded objects, or document metadata.

## Consequences

Developers can index visible text from common RTF files without an optional package. RTF files remain
untrusted input; parser limits cap syntax work and text output, but the parser does not reproduce
formatting or object rendering. Hosts can replace it through `FileIngestor(parsers=...)`.

## Alternatives considered

- **Add a full RTF rendering dependency:** rejected because Gabby needs retrieval text, not document
  layout, and a renderer would expand the core's dependency and attack surface.
- **Treat RTF bytes as plain text:** rejected because control words, groups, code pages, Unicode
  fallback, and binary payloads would pollute results and could make structural bytes appear as text.
- **Reject all unknown control words:** rejected because formatting control words are routinely added
  by different writers; unknown starred destinations are skipped, while unknown formatting words
  do not alter extracted text.

## Evidence

Parser tests cover visible text, Unicode and fallback, code-page escapes, destination and binary
skipping, malformed input, bounds, and default SQLite ingestion. See the [ingestion guide](../../EXTENSIONS.md).
