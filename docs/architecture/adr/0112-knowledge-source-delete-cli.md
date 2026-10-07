# ADR 0112: Root-confined knowledge source cleanup from the CLI

## Status

Accepted

## Context

`FileIngestor.delete_file()` can remove indexed chunks for a source that has already been deleted
from disk, but the CLI had no corresponding operation. A stale document could therefore remain
retrievable unless the caller wrote a Python cleanup script.

## Decision

- Add `gabby knowledge delete ROOT SOURCE --database DB`.
- Require an existing knowledge database and an existing ingestion root; `SOURCE` is a relative path
  resolved using `FileIngestor`'s existing traversal and symlink checks.
- Allow cleanup after the source file is absent while rejecting unsafe paths.
- Delete only indexed chunks; never remove or modify source files.
- Return a machine-readable JSON result with the normalized source and deleted document count.

## Consequences

CLI users can keep a local FTS5 index aligned with file removals without embedding cleanup code in
their applications. The operation is source-scoped and is not a whole-directory synchronization;
callers decide when a missing source should be removed.

## Verification

CLI contract tests ingest a source, remove the file, delete its index entry through the command,
and confirm later searches no longer return its text. The shared `FileIngestor` path checks remain
covered by the ingestion suite.
