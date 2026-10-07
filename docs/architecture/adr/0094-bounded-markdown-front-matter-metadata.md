# ADR 0094: Bounded Markdown front matter metadata

## Status

Accepted

## Context

File ingestion can create source-attributed chunks and exact metadata filters, but Markdown front
matter is currently indexed as ordinary body text. Callers need a portable way to attach per-file
labels such as title, team, or classification without separately maintaining a metadata manifest.

## Decision

Extend `ParsedPage` with optional JSON-compatible metadata. `Utf8TextParser` recognizes an optional
leading YAML block, removes it from the indexed body, and returns validated metadata with the page.
The parser uses `SafeLoader`, rejects duplicate mapping keys, cycles, unsupported values, excessive
depth or node count, and front matter larger than 16 KiB. YAML date and time scalars become ISO-8601
strings. Caller-supplied `FileIngestor` metadata overrides parsed front matter. Gabby adds reserved
chunk and page fields after both sources are merged.

## Consequences

Markdown metadata is queryable through existing retrieval filters and remains separate from body
text. The parser contract gains an additive `ParsedPage.metadata` field. Plain UTF-8 input without a
front matter opener remains unchanged. Invalid front matter fails ingestion instead of silently
indexing potentially ambiguous metadata.

## Alternatives considered

- Index YAML front matter as body text and require hosts to attach metadata separately. This duplicates
  classification data and makes exact filtering harder to use.
- Parse YAML in `FileIngestor` based on filename. This moves format-specific parsing out of the
  replaceable parser interface and complicates custom parser behavior.
