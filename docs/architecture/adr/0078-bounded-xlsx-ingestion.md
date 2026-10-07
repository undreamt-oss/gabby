# ADR 0078: Bounded XLSX ingestion

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

Spreadsheet support for data-analysis and research agents needs to cover OOXML workbooks as well as
OpenDocument spreadsheets. XLSX stores workbook, worksheet, relationship, and optional shared-string
parts in a ZIP package. SpreadsheetML cells may refer to strings by index, store inline values, or
contain cached formula values. Reading raw XML loses sheet and coordinate context.

## Decision

- Add a standard-library `XLSXTextParser` to the default `FileIngestor` parser set.
- Resolve the workbook from package relationships, then follow workbook relationships in sheet
  order to worksheet and optional shared-string parts. Do not extract package members to disk or
  follow external relationships.
- Render supported values from shared strings, rich-text runs, inline strings, numeric/string/date
  cached values, and booleans. Label each non-empty row with its sheet name, row number, and column
  letters. Do not calculate formulas; consume only stored cached values.
- Reject encrypted archives, unsafe or duplicate ZIP paths, external or unsafe package relationships,
  DTD/entity declarations, invalid cell references, malformed package parts, and oversized input.
- Bound archive bytes to 10 MiB, expanded members to 32 MiB, worksheet XML to 4 MiB per sheet,
  shared-string XML to 16 MiB, metadata XML to 1 MiB per part, XML elements to 250,000, sheets to
  1,000, rows to 100,000, columns to 1,000, cells and shared strings to 1,000,000, and extracted
  text to 10 million characters by default. Expose parser-specific limits through the public parser
  constructor.
- Keep the parser dependency-free and in process. Hosts processing hostile documents should isolate
  ingestion for stronger CPU and memory controls.

## Consequences

XLSX files become searchable through the existing ingestion and knowledge-store interfaces without
adding a package dependency. Workbook ordering and cell coordinates are preserved in retrieved text.
Charts, macros, external data, formula calculation, and workbook rendering are out of scope. Bounds
constrain package and output work but do not provide process isolation or a hard CPU deadline.

## Alternatives considered

- **Index raw XML parts:** rejected because values lose worksheet and coordinate context.
- **Load through a full spreadsheet library:** deferred to avoid a core dependency and keep explicit
  package and expansion bounds.
- **Recalculate formulas during ingestion:** rejected because it would require a calculation engine
  and could change application semantics; only values already stored in the workbook are indexed.

## Compatibility and evidence

This is an additive pre-1.0 API feature. Tests cover sheet order, shared and inline strings, typed
values, default SQLite ingestion, malformed and unsafe relationships, DTD/entity rejection, and
archive, part, row, column, cell, shared-string, and output bounds. The workbook/worksheet/shared
string relationship model follows the [Microsoft SpreadsheetML structure documentation](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/structure-of-a-spreadsheetml-document)
and its [shared string table guide](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-the-shared-string-table).
See [ADR 0032](0032-pluggable-page-aware-file-ingestion.md), the
[file-ingestion guide](../../../README.md#knowledge-retrieval), and the
[extension contract](../../EXTENSIONS.md).
