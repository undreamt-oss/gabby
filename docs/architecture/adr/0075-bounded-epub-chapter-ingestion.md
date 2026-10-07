# ADR 0075: Bounded EPUB chapter ingestion

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Research, policy, and documentation workflows commonly receive EPUB books. EPUB is a ZIP package
whose OPF manifest and spine define the ordered reading content. Treating the package as plain text
loses chapter boundaries and citations, while extracting all members to disk or loading every asset
would add avoidable risk and resource use.

## Decision

- Add a standard-library `EPUBTextParser` to the default `FileIngestor` parser set; do not add an
  EPUB dependency to the core package.
- Require the first, uncompressed `mimetype` member, a valid `META-INF/container.xml`, an OPF package,
  and non-empty spine entries that resolve to XHTML or HTML manifest items.
- Read only the package metadata and spine chapters, in spine order, without writing archive members
  to disk. Use the existing visible-HTML tokenizer to omit executable and non-visible subtrees.
- Return one `ParsedPage` per spine item and use its one-based reading-order chapter ordinal as
  `page_number` metadata. The field is a chapter citation for EPUB sources; Gabby does not interpret
  it as a printed page number.
- Reject encrypted archives, unsafe or duplicate member paths, remote or escaping package
  references, unsupported spine items, invalid encodings, DTD/entity declarations, and malformed
  package structure.
- Bound archive bytes to 10 MiB, expanded members to 32 MiB, metadata parts to 1 MiB each, chapter
  bytes to 4 MiB each, chapter count to 1,000, and extracted text to 10 million characters by
  default. Expose parser-specific limits through the public parser constructor.
- Document that images, stylesheets, and embedded media are not extracted and that in-process
  parsing should be isolated by hosts processing hostile complex documents.

## Consequences

EPUB books become searchable through the existing file-ingestion and knowledge-store interfaces,
with chapter-level source attribution. The feature adds no runtime dependency. EPUB is a ZIP and XML
format with complex metadata; strict input bounds and validation reduce exposure but do not provide
process isolation or a hard CPU deadline.

## Alternatives considered

- **Index each archive as one raw document:** rejected because it discards spine order and meaningful
  chapter citations.
- **Add a third-party EPUB parser:** deferred to keep core dependency-free and retain explicit
  bounds on archive expansion and chapter processing.
- **Extract all package resources:** rejected because images and styling are not required for text
  retrieval and would increase work and attack surface.

## Compatibility and evidence

This is an additive pre-1.0 API feature. Tests cover spine-order extraction, visible-text handling,
chapter citations, ingestion into SQLite FTS5, malformed and unsafe references, entity rejection,
and archive, chapter, and output bounds. See [ADR 0032](0032-pluggable-page-aware-file-ingestion.md),
the [file-ingestion guide](../../../README.md#knowledge-retrieval), and the
[extension contract](../../EXTENSIONS.md).
