# ADR 0088: Bounded Jupyter Notebook ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

Research and data-analysis agents need to retrieve notebook explanations and source code as
knowledge. A notebook is JSON with cell structure, but indexing the whole JSON would mix useful
source with execution state and potentially large or sensitive outputs.

## Decision

- Add a dependency-free `NotebookTextParser` for Jupyter notebook format version 4 (`.ipynb`).
- Validate strict UTF-8 JSON with duplicate-key, non-standard numeric constant, depth, and token
  checks before interpreting cells; require the v4 top-level fields and per-cell metadata/source
  structure, and require valid unique cell IDs for minor version 5 or later.
- Index non-empty markdown, code, and raw cell source in order, with one-based cell citations.
- Exclude output payloads, execution counts, and metadata from indexed text.
- Bound input bytes, cell count, source bytes per cell, total rendered output, JSON depth, and token
  count. Reject malformed cells and unsupported notebook versions.

## Consequences

Notebook source becomes searchable through the existing `FileIngestor` and knowledge stores without
executing code or installing notebook tooling. Results do not include rendered charts, tables,
execution outputs, attachments, or rich MIME data. Hosts that need those formats can inject another
parser under a separate extension or use the existing parser replacement interface.

## Alternatives considered

- Index the original JSON: rejected because it mixes source with execution state and produces poor
  retrieval text.
- Include cell outputs: rejected because they may be large, stale, or contain sensitive data.
- Execute notebooks during ingestion: rejected because indexing must not run source code.
