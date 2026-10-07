# ADR 0076: Bounded OpenDocument Text ingestion

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Research, policy, and business agents need to index OpenDocument Text documents as well as Office
Open XML and EPUB files. ODT is a ZIP package containing XML content. Reading it as raw text loses
paragraph order and table structure; adding an external parser would increase the core dependency
surface.

## Decision

- Add a standard-library `ODTTextParser` to the default `FileIngestor` parser set.
- Require an ODT package whose first ZIP member is an uncompressed `mimetype` with the expected
  OpenDocument Text media type and that contains `content.xml`.
- Read `content.xml` without extracting any archive members to disk. Preserve document order for
  headings, paragraphs, and table cells, including repeated spaces, tabs, and line breaks.
- Reject encrypted archives, unsafe or duplicate paths, DTD/entity declarations, malformed XML,
  incorrect package markers, and oversized input.
- Bound archive bytes to 10 MiB, expanded members to 32 MiB, content XML to 16 MiB, XML elements to
  250,000, paragraphs to 100,000, and extracted text to 10 million characters by default. Expose
  parser-specific limits through the public parser constructor.
- Keep the parser in process and document that hosts handling hostile documents should use process
  isolation for stronger resource control.

## Consequences

ODT documents become searchable through the existing file-ingestion and knowledge-store interfaces
without an added dependency. Text is indexed in source order, but Gabby does not render or extract
embedded images, charts, or other package resources. Bounds limit input and output size, but they do
not provide a hard CPU deadline or process isolation.

## Alternatives considered

- **Index the archive as plain text:** rejected because XML markup obscures document text and order.
- **Add a third-party parser dependency:** deferred to keep core dependency-free and retain direct
  control of package, XML, and output limits.
- **Extract embedded media:** deferred because it requires an independent conversion/OCR pipeline.

## Compatibility and evidence

This is an additive pre-1.0 API feature. Tests cover document and table order, inline whitespace,
SQLite indexing, malformed packages, entity declarations, unsafe paths, and archive, XML, paragraph,
and output bounds. See [ADR 0032](0032-pluggable-page-aware-file-ingestion.md), the
[file-ingestion guide](../../../README.md#knowledge-retrieval), and the
[extension contract](../../EXTENSIONS.md).
