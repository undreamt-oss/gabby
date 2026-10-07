# ADR 0106: Bounded Parquet ingestion

## Status

Accepted

## Context

Data-oriented agents need to retrieve facts from common columnar datasets. Parquet is widely used
for analytics, but compressed files can expand substantially and nested values do not render well
as retrieval text. A reader must preserve field-to-value relationships while keeping resource use
bounded and must not add a heavy analytics library to the core runtime.

## Decision

Provide `ParquetTextParser` through a separate `parquet` optional extra backed by PyArrow. The parser
accepts flat scalar columns, streams bounded record batches, and emits row-labeled pages with
one-based page numbers and row-range metadata. It enforces limits for file bytes, rows, columns,
row groups, aggregate metadata-reported uncompressed column bytes, individual text cells and rows,
pages, and total rendered output. PyArrow Thrift metadata string and container limits are set before
metadata is parsed. Nested and binary fields are rejected and must be flattened or transformed by
the application. The parser does not execute dataset-defined code or materialize a full table.

## Consequences

- Data agents can index tabular Parquet records without requiring pandas.
- Core installs do not acquire PyArrow or its native runtime dependency.
- Each file uses explicit limits, but an optional native parser remains part of the trusted Gabby
  process and should be updated with its host environment.
- Nested and binary schema support can be added only with a bounded, searchable representation.

## Validation

Contract tests build real Parquet fixtures using PyArrow and cover typed values, citations,
ingestion, malformed input, unsupported types, and configured resource limits. CI installs the
optional extra on supported Python and host-platform jobs.
