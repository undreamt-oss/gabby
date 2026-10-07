# ADR 0015: Bounded text directory ingestion

**Status:** Accepted by the project architect.  
**Date:** 2026-09-29.

## Context

Text file ingestion already limited each file to 10 MiB, but recursive directory ingestion had no
bound on file count or total bytes. A large tree could allocate a large path list and spend
unbounded time reading, chunking, and indexing files. Filesystem discovery and path checks also ran
synchronously from async methods.

## Decision

- Keep a 10 MiB default per-file limit and add per-ingestion defaults of 10,000 files and 1 GiB of
  raw file bytes.
- Expose `max_file_bytes`, `max_files`, and `max_total_bytes` as `TextFileIngestor` construction
  options; each must be a positive integer. `max_total_bytes` applies to a single-file import too
  when configured below `max_file_bytes`.
- Discover matching paths and their sizes off the event loop. Reject count, total-byte, or per-file
  overages before indexing any source.
- Enforce byte limits again while reading each file so changes after discovery cannot push actual
  bytes read past the per-file or total-byte limit.
- Keep source replacement atomic, but do not promise that a directory import is an all-or-nothing
  transaction. File changes or I/O failures during processing can leave earlier sources indexed.
- Run directory traversal, path checks, and file reads through Gabby's bounded synchronous
  callback workers.

## Consequences

The default bounds cap the discovered path list and total raw input for a normal import while
allowing applications to tune the limits or split larger imports into batches. The preflight scan is
not a filesystem snapshot; concurrent file changes can still cause an import to fail after earlier
sources were replaced. Applications needing whole-corpus atomicity should use generation-aware
indexing and coordinate a complete source set outside this convenience ingestor.

## Alternatives considered

- Keep only the per-file limit: simple, but a directory with an arbitrary number of individually
  small files could still consume excessive indexing time and resources.
- Require the caller to batch files: provides host control but leaves the bundled recursive ingestor
  without a bounded default.
- Make an entire directory import atomic: useful for snapshots, but requires a corpus-level
  generation transaction beyond per-source replacement and would make this operation significantly
  more complex.

## Validation

Implementation and static quality gates are present. Runtime acceptance still needs to cover exact
limits, over-limit preflight rejection before writes, file changes after discovery, and cancellation
while traversing or reading. Directory-level transactional rollback is not promised.
