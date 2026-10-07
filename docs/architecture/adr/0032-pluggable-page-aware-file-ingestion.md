# ADR 0032: Pluggable page-aware file ingestion

**Status:** Accepted under the architect's delegated implementation authority.  
**Date:** 2026-09-30.

## Context

The existing ingestor accepted only Markdown and plain text. Research and knowledge workflows also
need searchable text from PDFs, and PDF chunks need page citations. Requiring a parser dependency
for every Gabby installation would unnecessarily enlarge the core runtime.

## Decision

- Add a generic `FileParser` contract that receives the bounded file bytes and returns ordered
  `ParsedPage` values. Parsers run through the existing bounded synchronous callback workers.
- Make `FileIngestor` the general public API. Retain `TextFileIngestor` as a compatibility alias.
- Keep UTF-8 Markdown/plain-text parsing in core. Provide `PDFTextParser` through an optional
  `pypdf` dependency and the `pdf` install extra.
- Preserve a one-based `page_number` in each PDF chunk's metadata for citation. Treat that metadata
  as reserved and not caller-supplied.
- Bound input bytes, page count, and total extracted characters, then atomically replace each
  source only after parsing and chunking succeeds. Keep directory limits for file count and bytes.
- Keep OCR optional and extension-based. [ADR 0074](0074-optional-bounded-pdfium-vector-page-ocr.md)
  adds the opt-in built-in OCR path for scanned and vector-only pages.

## Consequences

Text-only users retain the lightweight default dependency set. PDF users install the optional
extra. The interface supports additional parsers without changing source replacement or chunking.
OCR remains separately optional and is loaded only when the host configures an OCR backend.
The pypdf parser runs trusted Python in the Gabby process. It limits decoded output for supported
content stream filters to 4 MiB per stream and 32 MiB of aggregate page content per PDF by default, in addition to
input-byte, page-count, and extracted-character limits. These controls bound common content-stream
decompression attacks, but do not guarantee a total memory or CPU ceiling for every PDF structure
or pypdf internal representation.

## Alternatives considered

- **Make pypdf a required dependency:** rejected because PDF parsing is not needed by every agent.
- **Leave PDF support entirely to host code:** rejected because page-attributed text extraction is a
  common research workflow and belongs behind the replaceable parser contract.
- **Add OCR to the built-in parser:** deferred because OCR requires a separate model/runtime and
  resource policy. The parser interface allows hosts to supply that capability.

## Compatibility and evidence

`FileIngestor` and the parser protocol are pre-1.0 API additions. `TextFileIngestor` remains an
alias. Tests cover page citation metadata, mixed directory ingestion, deterministic replacement,
malformed PDF rejection, and page/output bounds. Install `gabby-agent-runtime[pdf]` to use the
built-in PDF parser. See the [file-ingestion guide](../../../README.md#knowledge-retrieval) and
[threat model](../../THREAT_MODEL.md).
