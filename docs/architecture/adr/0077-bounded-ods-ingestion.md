# ADR 0077: Bounded OpenDocument Spreadsheet ingestion

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Research and data-analysis agents need to search small spreadsheets alongside CSV and document
sources. OpenDocument Spreadsheet files are ZIP packages containing XML workbook data. Indexing the
package as raw text loses sheet and row context, while expanding repeated rows or columns without
limits can create much more output than the source file contains.

## Decision

- Add a standard-library `ODSTextParser` to the default `FileIngestor` parser set.
- Require an ODS package whose first ZIP member is an uncompressed `mimetype` with the expected
  spreadsheet media type and that contains `content.xml`.
- Read workbook XML without extracting members to disk. Render non-empty cell values in sheet and row
  order, prefixing each row with its sheet name and each cell value with its spreadsheet column.
  Prefer text paragraphs, then typed cached cell values. Do not recalculate formulas.
- Expand repeated rows and columns only within configured limits. Empty rows are omitted from text,
  while their position still contributes to source row numbers.
- Reject encrypted archives, unsafe or duplicate paths, DTD/entity declarations, malformed XML,
  incorrect package markers, and oversized input.
- Bound archive bytes to 10 MiB, expanded members to 32 MiB, content XML to 16 MiB, XML elements to
  250,000, rows to 100,000, columns to 1,000, cells to 1,000,000, sheets to 1,000, and extracted
  text to 10 million characters by default. Expose parser-specific limits through the public parser
  constructor.
- Keep the parser in process. Hosts handling hostile documents should use process isolation for
  stronger CPU and memory control.

## Consequences

ODS spreadsheets become searchable through the existing file-ingestion and knowledge-store
interfaces without adding a dependency. Row labels preserve sheet and coordinate context during
retrieval. The parser does not calculate formulas, infer header semantics, or extract charts and
embedded media. Size limits bound expansion but do not create a hard CPU deadline or process
isolation.

## Alternatives considered

- **Index raw `content.xml`:** rejected because cell values lose their sheet, row, and column
  relationship in retrieved text.
- **Expand every repeated cell without limits:** rejected because a small XML attribute can request a
  very large logical table.
- **Add a spreadsheet library dependency:** deferred to keep core dependency-free and maintain
  explicit expansion limits.

## Compatibility and evidence

This is an additive pre-1.0 API feature. Tests cover sheet and row labeling, typed values, default
SQLite ingestion, malformed and unsafe packages, and archive, row, column, cell, and output bounds.
See [ADR 0032](0032-pluggable-page-aware-file-ingestion.md), the
[file-ingestion guide](../../../README.md#knowledge-retrieval), and the
[extension contract](../../EXTENSIONS.md).
